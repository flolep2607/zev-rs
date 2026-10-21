# zev-router

One load-aware router for the Kev (zev-fast) and vLLM pools, on [Pingora](https://github.com/cloudflare/pingora).
A port of learning-helm's `florian-nginx` (OpenResty + Lua): Pingora does the proxying, `src/fleet.rs` decides where
each request goes.

    cargo build --release
    zev-router router.json

## What it does per request

1. The path picks a **pool** (`pool_paths`; anything unlisted is `default`). A request, and every retry of it, only
   ever sees pods of its pool: Kev's `/v1/systemone` never lands on a vLLM chat pod.
2. With the `affinity_header` (default `x-session-id`) it tries the session's **home** pod first: weighted rendezvous
   hash, so every router replica agrees with no shared state.
3. Otherwise a **weighted-random pod with a free slot**, scored `gpu_multiplier / (1 + running + waiting)` from each
   pod's own `/metrics` (x0.1 over 95% KV cache, x0.5 on a sample older than 15 s, x0.05 for 30 s after a failure).
4. Every pod full: the request **waits** for a slot to be released (a `Notify`, not a poll), up to `queue_timeout_s`,
   then 503. No pod in the pool at all: 502 at once.
5. Connect error, broken upstream, or **429/500/502/503 before the first byte**: the pod is scored down and the
   request is replayed on an untried pod, 3 attempts in all. Bodies up to 64 KiB are buffered for the replay.

In the background, every `metrics_interval_s`: resolve each upstream (`dns` = a headless Service, or fixed `addrs`),
scrape `/metrics`. Both dialects are read: vLLM's `num_requests_running/waiting`, `kv_cache_usage_perc`,
`iteration_tokens_total_count`, and zev's `zev_in_flight` + `zev_passes_total` (summed over a pod's models). A step
counter frozen for `stall_s` while work is queued is a **wedged engine**: skipped outright, not scored down.

`/ready`, `/balancer-status` (JSON per pod) and `/metrics` (`zev_router_pod_*`, `zev_router_responses_total`) are
answered by the router itself.

## Config

```json
{
  "listen": "0.0.0.0:8000",
  "replicas": 2,
  "threads": 4,
  "pool_paths": { "/v1/systemone": "kev" },
  "upstreams": [
    { "name": "kev-backfill-4090", "pool": "kev", "port": 8008, "gpu_multiplier": 10, "max_inflight": 64,
      "dns": "argo-workflow-florian-nginx-kev-backfill-4090-pods.argo.svc.cluster.local" },
    { "name": "florian-agent-5090", "port": 8000, "gpu_multiplier": 17, "max_inflight": 64, "text_only": true,
      "dns": "argo-workflow-florian-nginx-florian-agent-5090-pods.argo.svc.cluster.local" }
  ]
}
```

`max_inflight` is per card for the whole set of routers; each of `replicas` takes its share (rounded up), as in
florian-nginx. Optional: `metrics_interval_s` (5), `stall_s` (30), `queue_timeout_s` (180), `affinity_header`.

## Against florian-nginx

Better: one process-wide state (no per-worker shared dicts, no JSON decode per pick), queued requests woken on
release instead of polling every 50 ms, no slot leaks (a pod's slots live in the `Pod` an in-flight request holds,
so nothing needs reclaiming), retry on 429, typed and unit-tested.

Not ported yet: `/drain`, replay capture, request-body deflate for rented boxes, the rented-box Secret
(`addrs` covers fixed backends), the image scan that keeps image requests off text-only pods (Pingora picks the
peer before the body is read: needs a body read in `request_filter`), the affinity wait/spill (a full home spills at
once), and a pool chosen by `model` (kev-combo answers every model name with its own adapter).

## Checks

    cargo test            # parsing, stall detection, pools, caps, weights, homing
    python3 smoke.py      # the real binary between a healthy and a wedged fake zev pod: 429s retried, body intact,
                          # the wedged pod skipped once stalled

Live, 2026-10-21: 100 real `/v1/systemone` calls through the router to the Kev NodePorts in 1.6 s, all 2xx.
