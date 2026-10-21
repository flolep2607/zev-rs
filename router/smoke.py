"""End-to-end check against the real binary, stdlib only: one healthy fake zev pod and one wedged one.

The wedged pod is 2026-10-21's failure: it answers every call with 429 at once, so a least-busy client kept feeding
it, and its /metrics show a full queue with a pass counter that never moves. Through the router every request must
still succeed, and the wedged pod must end up skipped as stalled.

    cargo build && python3 smoke.py [path/to/zev-router]
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BIN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CARGO_TARGET_DIR", "target") + "/debug/zev-router"
served = {"healthy": 0, "wedged": 0}
passes = {"healthy": 0}
# Phase 1 the wedged pod looks idle (in_flight 0, so a frozen counter is not a stall yet) and attractive: only the
# 429 retry keeps requests alive. Phase 2 its queue reads full with the counter still frozen: it must be skipped.
wedged_queue = {"in_flight": 0}


def fake(kind):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if kind == "wedged":  # step counter frozen
                m = f'zev_in_flight{{model="kev"}} {wedged_queue["in_flight"]}\nzev_passes_total{{model="kev"}} 19\n'
            else:
                passes["healthy"] += 1
                m = f'zev_in_flight{{model="kev"}} 1\nzev_passes_total{{model="kev"}} {passes["healthy"]}\n'
            self.reply(200, m.encode(), "text/plain")

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            served[kind] += 1
            if kind == "wedged":
                self.reply(429, b'{"detail":"inference queue saturated"}')
            else:
                self.reply(200, json.dumps({"answers": {"echo": json.loads(body)}}).encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


healthy, wedged = fake("healthy"), fake("wedged")
cfg = {
    "listen": "127.0.0.1:18089",
    "metrics_interval_s": 1,
    "stall_s": 2,
    "queue_timeout_s": 5,
    "pool_paths": {"/v1/systemone": "kev"},
    "upstreams": [
        # The wedged pod weighs 10x the healthy one: it would get ~90% of first picks if the router believed it.
        {"name": "healthy", "pool": "kev", "addrs": [f"127.0.0.1:{healthy}"], "port": 0, "gpu_multiplier": 1, "max_inflight": 64},
        {"name": "wedged", "pool": "kev", "addrs": [f"127.0.0.1:{wedged}"], "port": 0, "gpu_multiplier": 10, "max_inflight": 64},
    ],
}
path = tempfile.mktemp(suffix=".json")
json.dump(cfg, open(path, "w"))
router = subprocess.Popen([BIN, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    url = "http://127.0.0.1:18089"
    for _ in range(50):
        try:
            urllib.request.urlopen(url + "/ready", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    time.sleep(1.5)  # first scrape: both pods sampled, the wedged one not yet stalled

    def call(i):
        req = urllib.request.Request(url + "/v1/systemone", data=json.dumps({"state": f"line {i}"}).encode(), method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())["answers"]["echo"]["state"]

    results = [call(i) for i in range(30)]
    assert all(s == 200 for s, _ in results), results
    assert [e for _, e in results] == [f"line {i}" for i in range(30)], "the body must survive the retry intact"
    assert served["healthy"] == 30, served
    assert served["wedged"] > 0, "the wedged pod must have been tried, or the 429 retry was not exercised"

    wedged_queue["in_flight"] = 1024
    time.sleep(3)  # past stall_s with the counter frozen and work queued
    status = {p["name"]: p for p in json.load(urllib.request.urlopen(url + "/balancer-status"))["pods"]}
    assert status["wedged"]["stalled"] and status["wedged"]["score"] == 0, status["wedged"]
    before = served["wedged"]
    for i in range(10):
        call(i)
    assert served["wedged"] == before, "a stalled pod must not be offered at all"
    print(f"smoke ok: 40/40 answered by the healthy pod; {before} 429s from the wedged one retried, then it was skipped as stalled")
finally:
    router.terminate()
    os.unlink(path)
