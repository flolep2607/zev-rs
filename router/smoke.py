"""End-to-end checks against the real binary: fake zev/vLLM pods, a fake capture receiver, a rented-box Secret file,
and two rented boxes running the real vast/inflate_mw.py under uvicorn. Each block asserts one behaviour.

    cargo build && cargo build --example fake_h2c_pod && python3 smoke.py [path/to/inference_router_rs]

Needs python3 with zstandard, fastapi and uvicorn; INFLATE_MW points at inflate_mw.py (default: the price-compare
checkout at /data/vllm/vast/inflate_mw.py).
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
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zstandard  # the rented boxes' decoder

BIN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CARGO_TARGET_DIR", "target") + "/debug/inference_router_rs"
INFLATE_MW = os.environ.get("INFLATE_MW", "/data/vllm/vast/inflate_mw.py")
seen = {}  # pod name -> list of (headers, raw body, decoded body)
wedged = {"in_flight": 0}  # 0 at first: a frozen counter is not a stall until work queues
stream_closed = threading.Event()
inflate_delay = [0.0]  # the "inflate" pod's answer time, raised mid-test
events = []
# What each fake lists on /v1/models: (dialect, names). Mutable, so a pod can stop listing a model mid-test.
models = {"combo": ("zev", ["combo-r3"]), "mm": ("vllm", ["holo"]), "text": ("vllm", ["holo"]),
          "lora": ("vllm", ["holo", "lora-a"]), "fading": ("vllm", ["fade-1"])}


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
            if self.path == "/v1/models":
                if name not in models:
                    return self.reply(404, b"{}")
                dialect, ids = models[name]
                if dialect == "zev":
                    return self.reply(200, json.dumps({"models": [{"name": i} for i in ids]}).encode())
                data = [{"id": i, "max_model_len": 8192, "parent": None if i == "holo" else "holo"} for i in ids]
                return self.reply(200, json.dumps({"object": "list", "data": data}).encode())
            if mode == "wedged":
                m = f'zev_in_flight{{model="kev"}} {wedged["in_flight"]}\nzev_passes_total{{model="kev"}} 19\n'
            elif mode in ("quick", "lag"):  # busy every scrape; quick serves 4x the tokens per GPU-busy second
                passes[0] += 1
                tok = passes[0] * 0.5 * (4000 if mode == "quick" else 1000)
                m = (f'zev_in_flight{{model="kev"}} 1\nzev_passes_total{{model="kev"}} {passes[0]}\n'
                     f'zev_pass_tokens_total{{model="kev"}} {tok}\nzev_pass_duration_seconds_sum{{model="kev"}} {passes[0] * 0.5}\n')
            else:
                passes[0] += 1
                m = f'zev_in_flight{{model="kev"}} 0\nzev_passes_total{{model="kev"}} {passes[0]}\n'
            self.reply(200, m.encode(), "text/plain")

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["content-length"]))
            enc = self.headers.get("content-encoding")
            body = zlib.decompress(raw) if enc == "deflate" else raw
            seen.setdefault(name, []).append(({k.lower(): v for k, v in self.headers.items()}, raw, body))
            if mode == "inflate":
                time.sleep(inflate_delay[0])
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


# A rented box: the real inflate_mw in front of an echo app, wrapped by a recorder that sees each raw body (its
# encoding and zstd dictionary id) before the middleware inflates it.
BOX_APP = r'''
import importlib.util, json, os, zstandard
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
spec = importlib.util.spec_from_file_location("inflate_mw", os.environ["INFLATE_MW"])
mw = importlib.util.module_from_spec(spec); spec.loader.exec_module(mw)
app = FastAPI()
app.middleware("http")(mw.inflate)
RECORD, PASSES = [], [0]

@app.post("/v1/chat/completions")
async def chat(request: Request):
    m = await request.json()
    return {"pod": os.environ["BOX_NAME"], "len": len(m["messages"][0]["content"])}

@app.get("/metrics")
async def metrics():
    PASSES[0] += 1
    return PlainTextResponse(f'zev_in_flight{{model="b"}} 0\nzev_passes_total{{model="b"}} {PASSES[0]}\n')

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "holo", "max_model_len": 8192, "parent": None},
                                       {"id": os.environ["BOX_NAME"] + "-model", "max_model_len": 8192, "parent": None}]}

if os.environ.get("BOX_LORA"):
    @app.post("/v1/load_lora_adapter")
    async def load_lora():
        return {}

@app.get("/_seen")
async def _seen():
    return RECORD

async def recorder(scope, receive, send):
    if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
        return await app(scope, receive, send)
    chunks = []
    async def rcv():
        m = await receive()
        if m["type"] == "http.request":
            chunks.append(m.get("body", b""))
        return m
    await app(scope, rcv, send)
    raw = b"".join(chunks)
    enc = dict(scope["headers"]).get(b"content-encoding", b"").decode() or None
    did = zstandard.get_frame_parameters(raw).dict_id if enc == "zstd" else None
    RECORD.append({"enc": enc, "dict": did, "len": len(raw)})
'''
box_dir = tempfile.mkdtemp()
open(os.path.join(box_dir, "boxapp.py"), "w").write(BOX_APP)
boxes = {}


def box(name, port, key, lora=False):
    env = {**os.environ, "INFLATE_MW": INFLATE_MW, "VLLM_API_KEY": key, "BOX_NAME": name,
           "INFLATE_ZSTD_DICT": os.path.join(box_dir, "none.zdict"), **({"BOX_LORA": "1"} if lora else {})}
    boxes[name] = subprocess.Popen([sys.executable, "-m", "uvicorn", "boxapp:recorder", "--app-dir", box_dir,
                                    "--port", str(port), "--log-level", "warning"], env=env)
    return port


def box_get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return json.load(r)


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
                                   ("text", "ok"), ("ext", "ok"), ("aff1", "ok"), ("aff2", "ok"), ("slow", "slow"),
                                   ("lora", "ok"), ("fading", "ok"), ("quick", "quick"), ("lag", "lag"),
                                   ("inflate", "inflate")]}
# A zev-like pod on axum's serve (HTTP/1.1 + h2c), built next to the router by `cargo build --example fake_h2c_pod`.
H2C_BIN = os.path.join(os.path.dirname(BIN), "examples", "fake_h2c_pod")
zev_proc = subprocess.Popen([H2C_BIN, "18190"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pods["zev"] = "127.0.0.1:18190"


def train(word):
    return zstandard.train_dictionary(16384, [json.dumps({"messages": [{"role": "user", "content": f"{word} {i} " + "y" * 300}],
                                                          "tools": ["click", "type", "scroll"]}).encode() for i in range(300)])


DICT_A, DICT_B = train("page"), train("site")
zdict_file = tempfile.mktemp(suffix=".zdict")
open(zdict_file, "wb").write(DICT_A.as_bytes())
ZBOX, ZBAD = box("zbox", 18191, "s3cret", lora=True), box("zbad", 18192, "not-the-secret-key")
ext_file = tempfile.mktemp(suffix=".json")
json.dump({"api_key": "s3cret", "endpoints": [
    {"name": "ext", "ip": "127.0.0.1", "port": int(pods["ext"].split(":")[1]), "gpu_multiplier": 1, "deflate": True},
    {"name": "zbox", "ip": "127.0.0.1", "port": ZBOX, "gpu_multiplier": 1, "deflate": True, "zstd": True},
    {"name": "zbad", "ip": "127.0.0.1", "port": ZBAD, "gpu_multiplier": 1, "deflate": True, "zstd": True},
    {"name": "no-weight", "ip": "127.0.0.1", "port": 9}]}, open(ext_file, "w"))


def up(name, pool, mult=1, **kw):
    return {"name": name, "pool": pool, "addrs": [pods[name]], "gpu_multiplier": mult, "max_inflight": 64, **kw}


cfg = {
    "listen": "127.0.0.1:18089", "metrics_interval_s": 1, "stall_s": 2, "queue_timeout_s": 5,
    "affinity_wait_s": 2, "compress_min": 1024, "external_file": ext_file, "zstd_dict": zdict_file,
    "pool_paths": {"/v1/systemone": "kev", "/aff": "aff", "/slow": "slow", "/learn": "learn", "/infl": "infl"},
    "pool_models": {"zev-h2": "zev-h2"},  # combo-r3 is found through /v1/models instead
    "capture": {"endpoint": f"http://127.0.0.1:{recv.server_address[1]}/events", "check_interval_s": 1, "max_pending": 64},
    "upstreams": [
        # The wedged pod weighs 10x the healthy one: ~90% of first picks if the router believed it.
        up("healthy", "kev"), up("wedged", "kev", 10), up("combo", "kev-combo"),
        up("mm", "default"), up("text", "default", text_only=True), up("lora", "lora-pool"), up("fading", "fade-pool"),
        up("aff1", "aff"), up("aff2", "aff"), up("slow", "slow"),
        up("zev", "zev-h2", http2=True),
        up("quick", "learn", 10), up("lag", "learn", 10), up("inflate", "infl"),
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


def status():
    s = json.load(urllib.request.urlopen(URL + "/balancer-status"))
    return s, {p["name"]: p for p in s["pods"]}


def until(cond, what, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


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
    _, st = status()
    assert st["wedged"]["stalled"] and st["wedged"]["score"] == 0, st["wedged"]
    n = len(seen["wedged"])
    for i in range(10):
        post("/v1/systemone", {"state": f"after {i}"})
    assert len(seen["wedged"]) == n, "a stalled pod must not be offered at all"

    # 3. A model a pod lists on /v1/models pins the request to it on the same path (kev-combo, discovered, no
    #    pool_models); a name no pod lists (kev-4b) keeps the path's pool.
    for _ in range(5):
        assert json.loads(post("/v1/systemone", {"model": "combo-r3", "state": "s"})[1])["pod"] == "combo"
        assert json.loads(post("/v1/systemone", {"model": "kev-4b", "state": "s"})[1])["pod"] == "healthy"

    # 4. Image requests never reach the text-only pod; a schema that merely names image_url is text.
    img = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}]}
    txt = {"messages": [{"role": "user", "content": "return an 'image_url' field"}], "schema": {"image_url": {"type": "string"}}}
    got_img = {json.loads(post("/v1/chat/completions", img)[1])["pod"] for _ in range(20)}
    got_txt = {json.loads(post("/v1/chat/completions", txt)[1])["pod"] for _ in range(40)}
    assert "text" not in got_img and "text" in got_txt, (got_img, got_txt)

    # 5. Rented boxes from the Secret: weightless entry refused, the key sent to them alone, bodies deflated for a
    #    deflate box; never compressed for an in-cluster pod.
    _, st = status()
    assert "ext" in st and st["ext"]["external"] and "no-weight" not in st, sorted(st)
    until(lambda: status()[1]["zbox"]["box_dict"] == DICT_A.dict_id(), "zbox to hold dictionary A")
    for _ in range(60):
        post("/v1/chat/completions", {"messages": [{"role": "user", "content": "y" * 4000}]}, {"authorization": "Bearer client"})
    ext = seen.get("ext", [])
    assert ext, "the rented box got no traffic"
    assert all(h.get("authorization") == "Bearer s3cret" for h, _, _ in ext), "a rented box gets the rented key"
    big_ext = [(h, raw, body) for h, raw, body in ext if len(body) >= 1024]  # under compress_min goes as is
    assert big_ext and all(h.get("content-encoding") == "deflate" and len(raw) < len(body) for h, raw, body in big_ext), "deflated"
    assert all("content-encoding" not in h for h, _, body in ext if len(body) < 1024), "small bodies stay plain"
    lan = [h for name in ("mm", "text") for h, _, body in seen.get(name, []) if len(body) >= 1024]
    assert lan and all("content-encoding" not in h for h in lan), "in-cluster pods are never sent compressed bodies"
    assert all(h.get("authorization") != "Bearer s3cret" for h, _, _ in seen["mm"]), "never leak the key in-cluster"

    # 10. Dictionary sync, against the real inflate_mw: the router pushed A to the box with the right key and
    #     compresses with it; the box with another key refused the push, holds nothing, and is sent bodies as is
    #     -- every request still 200.
    for name in ("zbox", "zbad"):  # pinned by the model each box lists, so both surely get traffic
        for _ in range(15):
            r = post("/v1/chat/completions", {"model": f"{name}-model", "messages": [{"role": "user", "content": "y" * 4000}]})
            assert r[0] == 200 and json.loads(r[1])["pod"] == name, r
    zbox_seen = [r for r in box_get(ZBOX, "/_seen") if r["len"] >= 1024 or r["enc"]]
    assert zbox_seen and all(r["enc"] == "zstd" and r["dict"] == DICT_A.dict_id() for r in zbox_seen), zbox_seen[:3]
    zbad_seen = box_get(ZBAD, "/_seen")
    assert zbad_seen and all(r["enc"] is None for r in zbad_seen), ("no dictionary confirmed: sent uncompressed", [r for r in zbad_seen if r["enc"]][:3])
    assert box_get(ZBAD, "/zstd-dict")["dict_id"] is None and status()[1]["zbad"]["box_dict"] is None
    assert status()[1]["zbox"]["lora_runtime"] is True and status()[1]["zbad"]["lora_runtime"] is False, \
        "runtime LoRA loading read from /openapi.json"
    put = urllib.request.Request(f"http://127.0.0.1:{ZBOX}/zstd-dict", data=DICT_B.as_bytes(), method="PUT",
                                 headers={"authorization": "Bearer wrong"})
    try:
        urllib.request.urlopen(put, timeout=5)
        raise AssertionError("a wrong key must not install a dictionary")
    except urllib.error.HTTPError as e:
        assert e.code == 401, e.code
    assert box_get(ZBOX, "/zstd-dict")["dict_id"] == DICT_A.dict_id()

    # 11. Retrain mid-traffic: the router's dictionary file changes under 40 concurrent requests; it reloads, pushes
    #     B, compresses with B once the box confirms. Not one request fails, and bodies compressed with A just
    #     before the swap still decode because the box keeps the previous dictionaries.
    before = len(box_get(ZBOX, "/_seen"))
    results = []

    def traffic():
        with ThreadPoolExecutor(8) as ex:
            for _ in range(15):
                results.extend(ex.map(lambda _: post("/v1/chat/completions", {"model": "zbox-model", "messages": [{"role": "user", "content": "y" * 4000}]})[0], range(8)))
                time.sleep(0.25)

    t = threading.Thread(target=traffic)
    t.start()
    time.sleep(0.8)
    open(zdict_file, "wb").write(DICT_B.as_bytes())
    t.join()
    until(lambda: status()[1]["zbox"]["box_dict"] == DICT_B.dict_id(), "zbox to hold dictionary B")
    assert status()[0]["zstd_dict"] == DICT_B.dict_id()
    for _ in range(10):
        results.append(post("/v1/chat/completions", {"model": "zbox-model", "messages": [{"role": "user", "content": "y" * 4000}]})[0])
    assert results and all(c == 200 for c in results), f"{sum(c != 200 for c in results)} failed during the swap"
    after = box_get(ZBOX, "/_seen")[before:]
    dicts = {r["dict"] for r in after if r["enc"] == "zstd"}
    assert DICT_A.dict_id() in dicts and DICT_B.dict_id() in dicts, dicts
    held = box_get(ZBOX, "/zstd-dict")["held"]
    assert held[-1] == DICT_B.dict_id() and DICT_A.dict_id() in held, held
    print(f"  dictionary sync: push on mismatch, wrong key refused, {len(results)} requests through a swap A -> B, 0 failed")

    # 12. Model routing by discovery: a LoRA adapter goes only to the pod that loaded it; a model that a pod
    #     listed and no longer does fails fast (503 naming it) instead of reaching a pod that would serve another.
    for _ in range(5):
        assert json.loads(post("/v1/chat/completions", {"model": "lora-a", "state": "s"})[1])["pod"] == "lora"
    assert json.loads(post("/v1/chat/completions", {"model": "fade-1", "state": "s"})[1])["pod"] == "fading"
    models["fading"] = ("vllm", [])
    time.sleep(2.5)
    t0 = time.time()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps({"model": "fade-1"}).encode(), method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("a model nobody serves any more must not be routed")
    except urllib.error.HTTPError as e:
        assert e.code == 503 and b"fade-1" in e.read() and time.time() - t0 < 1, "fail fast, naming the model"
    _, st = status()
    assert [m["id"] for m in st["lora"]["models"]] == ["holo", "lora-a"] and st["lora"]["models"][1]["parent"] == "holo"
    metrics = urllib.request.urlopen(URL + "/metrics").read().decode()
    assert 'model="lora-a",parent="holo"' in metrics and "inference_router_zstd_dict_id" in metrics
    print("  models: LoRA pinned to its pod, kev-combo by discovery, vanished model -> 503 fast")

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
    q = {"model": "zev-h2", "state": "hp toner", "questions": {"k": {"type": "choice", "instructions": "kind",
                                                                   "criteria": {"good": "a product", "fee": "a fee"}}}}
    with ThreadPoolExecutor(50) as ex:
        got = list(ex.map(lambda _: post("/v1/systemone", q), range(200)))
    assert all(c == 200 and json.loads(b)["version"] == "HTTP/2.0" for c, b in got), got[:2]
    ss = subprocess.run(["ss", "-tnH", "state", "established", "( dport = :18190 )"], capture_output=True, text=True).stdout
    conns = len([l for l in ss.splitlines() if l.strip()])
    assert conns == 1, f"h2c must multiplex over one connection, saw {conns}:\n{ss}"
    _, st = status()
    assert st["zev"]["http2"] and st["zev"]["iterations"] is not None, st["zev"]  # scraped over h2 too
    print("  h2c: 200 requests at 50 concurrent, all HTTP/2, over 1 TCP connection")

    # 13. Learned weights: same gpu_multiplier, but "quick" serves 4x the tokens per busy second, so it gets ~4x the
    #     traffic. An inflating pod's adaptive limit is cut below its cap.
    _, st = status()
    assert abs(st["quick"]["weight"] - 16) < 1 and abs(st["lag"]["weight"] - 4) < 1, (st["quick"]["weight"], st["lag"]["weight"])
    got = [json.loads(post("/learn", {"state": "s"})[1])["pod"] for _ in range(300)]
    share = got.count("quick") / len(got)
    assert 0.7 < share < 0.9, share
    for _ in range(100):
        post("/infl", {"state": "s"})
    base = status()[1]["inflate"]
    assert base["limit"] == base["cap"] == 64, base
    inflate_delay[0] = 0.15
    for _ in range(25):
        post("/infl", {"state": "s"})
    after = status()[1]["inflate"]
    assert after["limit"] < 32 and after["room"] < 32, after
    print(f"  learned weights: quick {st['quick']['weight']:.1f} / lag {st['lag']['weight']:.1f} (both seeded 10) -> {share:.0%} "
          f"of traffic to quick; inflating pod's limit 64 -> {after['limit']:.1f} (baseline {after['latency_baseline_s'] * 1000:.0f} ms)")

    print("smoke ok: 429 retry with a 200 KB body, stall skip, model pins, image routing, rented box key+deflate, "
          "dictionary sync + swap, capture, affinity, drain cut + reopen, h2c over one connection, learned weights, adaptive limit")
finally:
    router.terminate()
    zev_proc.terminate()
    for p in boxes.values():
        p.terminate()
    os.unlink(path)
    os.unlink(ext_file)
    os.unlink(zdict_file)
