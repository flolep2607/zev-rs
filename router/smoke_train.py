"""End-to-end check of the zstd dictionary auto-training: two router replicas, a stub Kubernetes API (one Lease, one
ConfigMap; on a ConfigMap write it writes the file the way the kubelet refreshes a mounted ConfigMap), and one rented
box running the real vast/inflate_mw.py.

    cargo build && python3 smoke_train.py [path/to/inference_router_rs]

Checks: only the Lease holder trains; its first dictionary beats plain zstd, goes to the ConfigMap, both replicas
load it and the box is pushed it; a later training on the same traffic is rejected as not better; when the leader
dies the other replica takes the Lease over; each pod's node and GPU come from the API.
"""

import base64
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zstandard

BIN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CARGO_TARGET_DIR", "target") + "/debug/inference_router_rs"
INFLATE_MW = os.environ.get("INFLATE_MW", "/data/vllm/vast/inflate_mw.py")
tmp = tempfile.mkdtemp()
DICT_FILE = os.path.join(tmp, "prompts.zdict")  # absent at first: plain zstd until a dictionary is trained

# --- stub API server -------------------------------------------------------------------------------------------------
objs, version, cm_writes = {}, [0], []


class Api(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        return json.loads(self.rfile.read(int(self.headers["content-length"])))

    def do_GET(self):
        o = objs.get(self.path)
        self.reply(200, o) if o else self.reply(404, {"reason": "NotFound"})

    def store(self, key, o):
        version[0] += 1
        o["metadata"]["resourceVersion"] = str(version[0])
        objs[key] = o
        if o.get("kind") == "ConfigMap" or "/configmaps/" in key:  # the kubelet's refresh of the mounted file
            open(DICT_FILE + ".tmp", "wb").write(base64.b64decode(o["binaryData"]["prompts.zdict"]))
            os.replace(DICT_FILE + ".tmp", DICT_FILE)
            holder = objs.get("/apis/coordination.k8s.io/v1/namespaces/t/leases/inference-router-rs-zdict", {}).get("spec", {})
            cm_writes.append(holder.get("holderIdentity"))
        self.reply(200, o)

    def do_POST(self):
        o = self.body()
        key = f"{self.path}/{o['metadata']['name']}"
        if key in objs:
            return self.reply(409, {"reason": "AlreadyExists"})
        self.store(key, o)

    def do_PUT(self):
        o = self.body()
        if self.path not in objs or objs[self.path]["metadata"]["resourceVersion"] != o["metadata"].get("resourceVersion"):
            return self.reply(409, {"reason": "Conflict"})
        self.store(self.path, o)


# Where the pods run: every fake is on 127.0.0.1, here a pod on a 4090 node.
objs["/api/v1/namespaces/t/pods?fieldSelector=status.phase%3DRunning"] = {
    "items": [{"status": {"podIP": "127.0.0.1"}, "spec": {"nodeName": "gpuworker-t"}}]}
objs["/api/v1/nodes/gpuworker-t"] = {"metadata": {"labels": {"nvidia.com/gpu.product": "NVIDIA-GeForce-RTX-4090"}}}
api = ThreadingHTTPServer(("127.0.0.1", 0), Api)
threading.Thread(target=api.serve_forever, daemon=True).start()

# --- one rented box: the real inflate_mw ------------------------------------------------------------------------------
BOX_APP = r'''
import importlib.util, os, zstandard
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
spec = importlib.util.spec_from_file_location("inflate_mw", os.environ["INFLATE_MW"])
mw = importlib.util.module_from_spec(spec); spec.loader.exec_module(mw)
app = FastAPI()
app.middleware("http")(mw.inflate)
RECORD, PASSES = [], [0]

@app.post("/v1/chat/completions")
async def chat(request: Request):
    return {"len": len((await request.json())["messages"][0]["content"])}

@app.get("/metrics")
async def metrics():
    PASSES[0] += 1
    return PlainTextResponse(f'zev_in_flight{{model="b"}} 0\nzev_passes_total{{model="b"}} {PASSES[0]}\n')

@app.get("/_seen")
async def _seen():
    return RECORD

async def recorder(scope, receive, send):
    if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
        return await app(scope, receive, send)
    chunks = []
    async def rcv():
        m = await receive()
        chunks.append(m.get("body", b""))
        return m
    await app(scope, rcv, send)
    raw = b"".join(chunks)
    enc = dict(scope["headers"]).get(b"content-encoding", b"").decode() or None
    RECORD.append({"enc": enc, "dict": zstandard.get_frame_parameters(raw).dict_id if enc == "zstd" else None})
'''
open(os.path.join(tmp, "boxapp.py"), "w").write(BOX_APP)
BOX = 18391
box = subprocess.Popen([sys.executable, "-m", "uvicorn", "boxapp:recorder", "--app-dir", tmp, "--port", str(BOX), "--log-level", "warning"],
                       env={**os.environ, "INFLATE_MW": INFLATE_MW, "VLLM_API_KEY": "s3cret", "INFLATE_ZSTD_DICT": DICT_FILE + ".none"})
ext = os.path.join(tmp, "endpoints.json")
json.dump({"api_key": "s3cret", "endpoints": [{"name": "zbox", "ip": "127.0.0.1", "port": BOX, "gpu_multiplier": 1, "zstd": True}]},
          open(ext, "w"))

# --- two replicas ---------------------------------------------------------------------------------------------------
routers, PORTS = {}, {"r1": 18389, "r2": 18390}
for name, port in PORTS.items():
    cfg = {"listen": f"127.0.0.1:{port}", "metrics_interval_s": 1, "compress_min": 1024, "external_file": ext,
           "zstd_dict": DICT_FILE, "upstreams": [], "kube_api": f"http://127.0.0.1:{api.server_address[1]}", "kube_namespace": "t",
           "autotrain": {"min_samples": 60,
                         "max_samples": 400, "every_s": 6, "min_interval_s": 3, "margin": 0.1, "dict_bytes": 16384, "lease_s": 4}}
    path = os.path.join(tmp, f"{name}.json")
    json.dump(cfg, open(path, "w"))
    routers[name] = subprocess.Popen([BIN, path], env={**os.environ, "HOSTNAME": name, "RUST_LOG": "warn"},
                                     stdout=subprocess.DEVNULL, stderr=open(os.path.join(tmp, f"{name}.log"), "w"))

SYSTEM = "You are a shopping agent on a retailer's site. Use the tools to find the product, its price and stock. " * 15
WORDS = "toner cartridge black hp laser 85a printer paper a4 ream office chair mesh desk lamp led usb cable stapler".split()
rng = random.Random(7)


def prompt():
    page = " ".join(rng.choice(WORDS) + str(rng.randint(0, 999)) for _ in range(300))
    return {"model": "holo", "messages": [{"role": "user", "content": SYSTEM + page}], "tools": ["click", "type", "scroll", "extract"]}


def post(name):
    req = urllib.request.Request(f"http://127.0.0.1:{PORTS[name]}/v1/chat/completions", data=json.dumps(prompt()).encode(),
                                 method="POST", headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def status(name):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORTS[name]}/balancer-status", timeout=5))


def until(cond, what, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if cond():
                return
        except OSError:
            pass
        time.sleep(0.3)
    raise AssertionError(f"timed out waiting for {what}")


try:
    until(lambda: all(status(n)["pods"] for n in PORTS), "both replicas to find the box")
    until(lambda: sum(status(n)["autotrain"]["leader"] for n in PORTS) == 1, "one leader")
    leader = next(n for n in PORTS if status(n)["autotrain"]["leader"])
    other = next(n for n in PORTS if n != leader)

    # 1. Traffic through both replicas; the leader trains on what it sent and publishes; the other never trains.
    codes = [post(n) for _ in range(80) for n in PORTS]
    until(lambda: status(leader)["autotrain"]["last_result"].startswith("published"), "the leader to publish a dictionary")
    a = status(leader)["autotrain"]
    new_id = zstandard.ZstdCompressionDict(open(DICT_FILE, "rb").read()).dict_id()
    assert a["last_reason"] == "NoBaseline" and a["last_candidate_ratio"] > 1.1 * a["last_current_ratio"], a
    assert cm_writes == [leader], cm_writes
    until(lambda: all(status(n)["zstd_dict"] == new_id for n in PORTS), "both replicas to load the published dictionary")
    until(lambda: status(leader)["pods"][0]["box_dict"] == new_id, "the box to be pushed the dictionary")
    codes += [post(n) for _ in range(30) for n in PORTS]
    seen = json.load(urllib.request.urlopen(f"http://127.0.0.1:{BOX}/_seen"))
    assert seen[-1]["enc"] == "zstd" and seen[-1]["dict"] == new_id, seen[-3:]
    assert status(other)["autotrain"]["trainings"] == 0 and not status(other)["autotrain"]["leader"]
    print(f"  {leader} trained {a['last_candidate_ratio']:.2f}x vs plain zstd {a['last_current_ratio']:.2f}x, published {new_id}; "
          f"both replicas loaded it, the box holds it; {other} never trained")

    # 2. The next scheduled training sees the same traffic: the candidate is not 10% better, nothing is published.
    codes += [post(n) for _ in range(40) for n in PORTS]  # past the 50 bodies a live ratio needs under the new dictionary
    until(lambda: status(leader)["autotrain"]["trainings"] >= 2, "the second (Age) training")
    b = status(leader)["autotrain"]
    assert b["last_reason"] == "Age" and b["last_result"].startswith("rejected"), b
    assert cm_writes == [leader] and status(leader)["zstd_dict"] == new_id
    assert b["live_ratio"] and b["live_ratio"] > 2, b
    print(f"  Age retrain: {b['last_result']}; live ratio {b['live_ratio']:.2f}x")

    # 3. The leader dies: after lease_s without renewal the other replica takes the Lease.
    routers[leader].kill()
    routers[leader].wait()
    until(lambda: status(other)["autotrain"]["leader"], f"{other} to take the Lease over", timeout=15)
    print(f"  {leader} stopped: {other} took the Lease over")
    assert all(c == 200 for c in codes), codes
    pod = status(other)["pods"][0]
    assert pod["node"] == "gpuworker-t" and pod["gpu"] == "NVIDIA-GeForce-RTX-4090", pod
    metrics = urllib.request.urlopen(f"http://127.0.0.1:{PORTS[other]}/metrics").read().decode()
    assert "inference_router_zdict_leader 1" in metrics and "inference_router_zdict_live_ratio" in metrics
    print(f"smoke_train ok: {len(codes)} requests, 0 failed")
finally:
    for p in routers.values():
        p.kill()  # SIGTERM is Pingora's graceful shutdown, which waits out its grace period
    box.terminate()
