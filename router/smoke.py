"""End-to-end checks against the real binary, stdlib only: fake zev/vLLM pods, a fake capture receiver, a rented-box
Secret file. Each block asserts one behaviour florian-nginx had (or should have had).

    cargo build && python3 smoke.py [path/to/zev-router]
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib

import zstandard  # pip install zstandard: the rented boxes' decoder, mirrored by the fakes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BIN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CARGO_TARGET_DIR", "target") + "/debug/zev-router"
seen = {}  # pod name -> list of (headers, raw body, decoded body)
wedged = {"in_flight": 0}  # 0 at first: a frozen counter is not a stall until work queues
stream_closed = threading.Event()
events = []


def fake(name, mode="ok"):
    passes = [0]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def reply(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if mode == "wedged":
                m = f'zev_in_flight{{model="kev"}} {wedged["in_flight"]}\nzev_passes_total{{model="kev"}} 19\n'
            else:
                passes[0] += 1
                m = f'zev_in_flight{{model="kev"}} 0\nzev_passes_total{{model="kev"}} {passes[0]}\n'
            self.reply(200, m.encode(), "text/plain")

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["content-length"]))
            enc = self.headers.get("content-encoding")
            body = (zlib.decompress(raw) if enc == "deflate"
                    else zstandard.ZstdDecompressor(dict_data=ZDICT).decompress(raw) if enc == "zstd" else raw)
            seen.setdefault(name, []).append(({k.lower(): v for k, v in self.headers.items()}, raw, body))
            if mode == "wedged":
                return self.reply(429, b'{"detail":"inference queue saturated"}')
            if mode == "slow":  # an SSE generation that a drain must cut
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                try:
                    for i in range(100):
                        chunk = f"data: token {i}\n\n".encode()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                        time.sleep(0.1)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    stream_closed.set()
                return
            state = json.loads(body).get("state", "")
            self.reply(200, json.dumps({"pod": name, "len": len(state), "head": state[:20]}).encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"127.0.0.1:{srv.server_address[1]}"


class Receiver(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        events.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
        self.send_response(204)
        self.end_headers()


recv = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
threading.Thread(target=recv.serve_forever, daemon=True).start()

pods = {n: fake(n, m) for n, m in [("healthy", "ok"), ("wedged", "wedged"), ("combo", "ok"), ("mm", "ok"),
                                   ("text", "ok"), ("ext", "ok"), ("aff1", "ok"), ("aff2", "ok"), ("slow", "slow")]}
# A zev-like pod on axum's serve (HTTP/1.1 + h2c), built next to the router by `cargo build --example fake_h2c_pod`.
H2C_BIN = os.path.join(os.path.dirname(BIN), "examples", "fake_h2c_pod")
zev_proc = subprocess.Popen([H2C_BIN, "18190"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pods["zev"] = "127.0.0.1:18190"
# A zstd dictionary trained on look-alike prompts, loaded by the router (zstd_dict) and by the fake zstd box.
ZDICT = zstandard.train_dictionary(16384, [json.dumps({"messages": [{"role": "user", "content": f"page {i} " + "y" * 300}],
                                                       "tools": ["click", "type", "scroll"]}).encode() for i in range(300)])
zdict_file = tempfile.mktemp(suffix=".zdict")
open(zdict_file, "wb").write(ZDICT.as_bytes())
pods["extz"] = fake("extz", "ok")
ext_file = tempfile.mktemp(suffix=".json")
json.dump({"api_key": "s3cret", "endpoints": [
    {"name": "ext", "ip": "127.0.0.1", "port": int(pods["ext"].split(":")[1]), "gpu_multiplier": 1, "deflate": True},
    {"name": "extz", "ip": "127.0.0.1", "port": int(pods["extz"].split(":")[1]), "gpu_multiplier": 1,
     "deflate": True, "zstd": True},
    {"name": "no-weight", "ip": "127.0.0.1", "port": 9}]}, open(ext_file, "w"))


def up(name, pool, mult=1, **kw):
    return {"name": name, "pool": pool, "addrs": [pods[name]], "gpu_multiplier": mult, "max_inflight": 64, **kw}


cfg = {
    "listen": "127.0.0.1:18089", "metrics_interval_s": 1, "stall_s": 2, "queue_timeout_s": 5,
    "affinity_wait_s": 2, "compress_min": 1024, "external_file": ext_file, "zstd_dict": zdict_file,
    "pool_paths": {"/v1/systemone": "kev", "/aff": "aff", "/slow": "slow"},
    "pool_models": {"combo-r3": "kev-combo", "zev-h2": "zev-h2"},
    "capture": {"endpoint": f"http://127.0.0.1:{recv.server_address[1]}/events", "check_interval_s": 1, "max_pending": 64},
    "upstreams": [
        # The wedged pod weighs 10x the healthy one: ~90% of first picks if the router believed it.
        up("healthy", "kev"), up("wedged", "kev", 10), up("combo", "kev-combo"),
        up("mm", "default"), up("text", "default", text_only=True),
        up("aff1", "aff"), up("aff2", "aff"), up("slow", "slow"),
        up("zev", "zev-h2", http2=True),
    ],
}
path = tempfile.mktemp(suffix=".json")
json.dump(cfg, open(path, "w"))
router = subprocess.Popen([BIN, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
URL = "http://127.0.0.1:18089"


def post(p, payload, headers=None, timeout=10):
    req = urllib.request.Request(URL + p, data=json.dumps(payload).encode(), method="POST",
                                 headers={"content-type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def code(p, payload=None, method="POST", headers=None):
    req = urllib.request.Request(URL + p, data=json.dumps(payload or {}).encode(), method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


try:
    for _ in range(50):
        try:
            urllib.request.urlopen(URL + "/ready", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    time.sleep(1.5)  # first scrape and capture health check

    # 1. 429s from the wedged pod are retried, the body (over Pingora's 64 KiB retry buffer) survives intact.
    big = "x" * 200_000
    for i in range(20):
        s, b = post("/v1/systemone", {"state": f"line {i} " + (big if i % 5 == 0 else "")})
        r = json.loads(b)
        assert s == 200 and r["pod"] == "healthy" and r["head"].startswith(f"line {i}"), r
        assert r["len"] == len(f"line {i} ") + (len(big) if i % 5 == 0 else 0), "body intact through the retry"
    assert len(seen.get("wedged", [])) > 0, "the wedged pod must have been tried, or no retry was exercised"

    # 2. Its queue now reads full with the counter frozen: skipped as stalled, not even tried.
    wedged["in_flight"] = 1024
    time.sleep(3)
    st = {p["name"]: p for p in json.load(urllib.request.urlopen(URL + "/balancer-status"))["pods"]}
    assert st["wedged"]["stalled"] and st["wedged"]["score"] == 0, st["wedged"]
    n = len(seen["wedged"])
    for i in range(10):
        post("/v1/systemone", {"state": f"after {i}"})
    assert len(seen["wedged"]) == n, "a stalled pod must not be offered at all"

    # 3. The body's model picks the pool on the same path: kev-combo, never the kev pods, and back.
    for _ in range(5):
        assert json.loads(post("/v1/systemone", {"model": "combo-r3", "state": "s"})[1])["pod"] == "combo"
        assert json.loads(post("/v1/systemone", {"model": "kev-4b", "state": "s"})[1])["pod"] == "healthy"

    # 4. Image requests never reach the text-only pod; a schema that merely names image_url is text.
    img = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}]}
    txt = {"messages": [{"role": "user", "content": "return an 'image_url' field"}], "schema": {"image_url": {"type": "string"}}}
    got_img = {json.loads(post("/v1/chat/completions", img)[1])["pod"] for _ in range(20)}
    got_txt = {json.loads(post("/v1/chat/completions", txt)[1])["pod"] for _ in range(40)}
    assert "text" not in got_img and "text" in got_txt, (got_img, got_txt)

    # 5. The rented box from the Secret: weightless entry refused, its key sent to it alone, bodies deflated.
    st = {p["name"]: p for p in json.load(urllib.request.urlopen(URL + "/balancer-status"))["pods"]}
    assert "ext" in st and st["ext"]["external"] and "no-weight" not in st, sorted(st)
    for _ in range(30):
        post("/v1/chat/completions", {"messages": [{"role": "user", "content": "y" * 4000}]}, {"authorization": "Bearer client"})
    ext = seen.get("ext", [])
    assert ext, "the rented box got no traffic"
    assert all(h.get("authorization") == "Bearer s3cret" for h, _, _ in ext), "a rented box gets the rented key"
    big_ext = [(h, raw, body) for h, raw, body in ext if len(body) >= 1024]  # under compress_min goes as is
    assert big_ext and all(h.get("content-encoding") == "deflate" and len(raw) < len(body) for h, raw, body in big_ext), "deflated"
    assert all("content-encoding" not in h for h, _, body in ext if len(body) < 1024), "small bodies stay plain"
    extz = [(h, raw, body) for h, raw, body in seen.get("extz", []) if len(body) >= 1024]
    assert extz and all(h.get("content-encoding") == "zstd" and h.get("authorization") == "Bearer s3cret"
                        for h, _, _ in extz), "a +zstd box gets zstd with the dictionary, and its key"
    zratio = sum(len(b) for _, _, b in extz) / sum(len(r) for _, r, _ in extz)
    lratio = sum(len(b) for _, _, b in big_ext) / sum(len(r) for _, r, _ in big_ext)
    assert zratio > lratio, f"zstd + dictionary must beat zlib: {zratio:.1f}x vs {lratio:.1f}x"
    lan = [h for name in ("mm", "text") for h, _, body in seen.get(name, []) if len(body) >= 1024]
    assert lan and all("content-encoding" not in h for h in lan), "in-cluster pods are never sent compressed bodies"
    assert all(h.get("authorization") != "Bearer s3cret" for h, _, _ in seen["mm"]), "never leak the key in-cluster"

    # 6. Capture: request + response events joined by capture_id for the vLLM pool, nothing for Kev.
    time.sleep(0.5)  # earlier blocks' events still in delivery
    events.clear()
    post("/v1/chat/completions", txt)
    post("/v1/systemone", {"state": "not captured"})
    time.sleep(0.5)
    reqs = [e for e in events if e["event"] == "request"]
    assert len(reqs) == 1 and "image_url" in reqs[0]["request"]["body"], events
    resps = [e for e in events if e["event"] == "response" and e["capture_id"] == reqs[0]["capture_id"]]
    assert len(resps) == 1 and resps[0]["response"]["status"] == 200, events
    assert not any("not captured" in json.dumps(e) for e in events), "Kev traffic must not be captured"

    # 7. Affinity: a session sticks to one pod; two sessions may differ.
    homes = {json.loads(post("/aff", {"state": "s"}, {"x-session-id": "abc"})[1])["pod"] for _ in range(10)}
    assert len(homes) == 1, homes

    # 8. /drain: refused through an ingress, cuts a live stream, closes admission, reopens.
    assert code("/drain", headers={"x-forwarded-for": "1.2.3.4"}) == 403
    streamed = []

    def stream():
        req = urllib.request.Request(URL + "/slow", data=b"{}", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                for line in r:
                    streamed.append(line)
        except Exception:
            pass

    t = threading.Thread(target=stream)
    t.start()
    time.sleep(0.6)
    assert code("/drain?seconds=30") == 200
    t.join(timeout=5)
    assert not t.is_alive() and 0 < len(streamed) < 100, f"the stream must be cut mid-way, got {len(streamed)} lines"
    assert stream_closed.wait(3), "the upstream connection must close, or the generation runs on"
    assert code("/v1/systemone", {"state": "x"}) == 503, "drained: new requests refused"
    assert code("/drain?seconds=0") == 200 and code("/v1/systemone", {"state": "x"}) == 200, "resume reopens"

    # 9. h2c: 200 requests at 50 concurrent ride ONE long-lived TCP connection (HTTP/1.1 would open ~50).
    from concurrent.futures import ThreadPoolExecutor
    q = {"model": "zev-h2", "state": "hp toner", "questions": {"k": {"type": "choice", "instructions": "kind",
                                                                   "criteria": {"good": "a product", "fee": "a fee"}}}}
    with ThreadPoolExecutor(50) as ex:
        got = list(ex.map(lambda _: post("/v1/systemone", q), range(200)))
    assert all(c == 200 and json.loads(b)["version"] == "HTTP/2.0" for c, b in got), got[:2]
    ss = subprocess.run(["ss", "-tnH", "state", "established", "( dport = :18190 )"], capture_output=True, text=True).stdout
    conns = len([l for l in ss.splitlines() if l.strip()])
    assert conns == 1, f"h2c must multiplex over one connection, saw {conns}:\n{ss}"
    st = {p["name"]: p for p in json.load(urllib.request.urlopen(URL + "/balancer-status"))["pods"]}
    assert st["zev"]["http2"] and st["zev"]["iterations"] is not None, st["zev"]  # scraped over h2 too
    print("  h2c: 200 requests at 50 concurrent, all HTTP/2, over 1 TCP connection")

    print("smoke ok: 429 retry with a 200 KB body, stall skip, model pool, image routing, rented box key+deflate, "
          "capture, affinity, drain cut + reopen, h2c over one connection")
finally:
    router.terminate()
    if zev_proc:
        zev_proc.terminate()
    os.unlink(path)
    os.unlink(ext_file)
    os.unlink(zdict_file)
