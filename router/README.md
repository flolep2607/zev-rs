# zev-router

One load-aware router for the Kev (zev-fast) and vLLM pools, a port of learning-helm's `florian-nginx`
(OpenResty + Lua) to Rust on [Pingora](https://github.com/cloudflare/pingora).

    cargo build --release
    zev-router router.json

Pingora is the server (HTTP/1 and h2 downstream, worker threads, graceful restart). Forwarding is the router's own,
through reqwest (hyper underneath): the image check, the `model` pool and per-pod compression all need the whole body
before a pod is chosen, and Pingora's proxy can only replay a body it buffered itself (64 KiB, smaller than a
screenshot prompt).

## Per request

1. **Drained?** 503 (`/drain` below).
2. **Read the body**, bounded by `max_body_bytes` (413 past it). Every retry replays it; it is dropped as soon as the
   response headers arrive, so a long stream does not hold its prompt.
3. **Pool and model**: `pool_paths[path]`, else `default`; a static `pool_models[body.model]` overrides it. Then
   discovery: if the body's `model` is a name some pod lists on `/v1/models` (a base model, a LoRA adapter, a zev
   model name), the request goes **only to pods listing it**, whatever their pool. That is the LoRA affinity, and it
   finds kev-combo by its model name with no `pool_models`. A name listed within the last hour whose pods are all gone
   is a **503 naming the model, at once**: zev answers any name with its own adapter, so falling back would return
   another model's answer. A name never listed (e.g. `kev-4b`, served by zev under its own names) keeps the path's
   pool. A request and its retries stay among the pods chosen here.
4. **Image check**, only for a pool with a `text_only` pod: a `"type": "image_url"` content part keeps the request off
   text-only pods (a bare `image_url` in a prompt or schema does not).
5. **Admission**. With the `affinity_header` (default `x-session-id`), the session's **home** pod (weighted rendezvous
   hash: every replica agrees, losing a pod rehomes only its sessions) if it has a free slot. A full home is waited for
   up to `affinity_wait_s`, unless another pod with room scores `spill_ratio` times the home. Otherwise a
   **weighted-random pod with a free slot**, scored `gpu_multiplier / (1 + running + waiting)` from the pod's own
   `/metrics` (x0.1 over 95% KV cache, x0.5 on a sample older than 15 s, x0.05 for 30 s after a failure; a stalled pod
   scores 0). Every pod full: wait for a release (woken by a `Notify`), up to `queue_timeout_s`, then 503. No pod in
   the pool: 502 at once.
6. **Attempts**, up to 3: a connect error, no headers within 600 s, or a **429/500/502/503** goes to an untried pod,
   and the pod is scored down for 30 s. With no untried pod left, the last answer is passed through.
   Per attempt, for a **rented box only**: the body compressed when it is at least `compress_min` (zstd for a box
   marked `zstd`, with `zstd_dict` once the box reports holding it and uncompressed until then; zlib for `deflate`),
   and its `Authorization: Bearer <api_key>`.
   An in-cluster pod is never sent a compressed body or that key: `compress` cannot even be set on a configured
   upstream, because on the LAN compression is CPU spent for nothing.
7. **Stream back**, cut at once by `/drain` (dropping the upstream connection is what makes vLLM abort the
   generation). The slot is released after the last byte.

## Connections to pods

`http2: true` on an upstream: **h2c** (HTTP/2 over plain TCP, prior knowledge), one long-lived connection per pod that
multiplexes every request, no idle timeout, pinged every 10 s (5 s to answer) so a dead peer is noticed before a
request goes into it. zev serves h2c once axum's `http2` feature is on (zev-rs Cargo.toml); vLLM's uvicorn speaks
HTTP/1.1 only. Everything else: HTTP/1.1 keep-alive, up to 64 idle connections per pod. Both with TCP keep-alive.
HTTP/3 (QUIC) is not offered: neither server speaks it, and on a LAN it buys nothing over h2c.

## Compression for rented boxes

They are metered per GB and 96% of their traffic is inbound prompt (84 KB in, 3.2 KB out). On 1,333 captured agent
prompts, dictionary trained on the older three quarters and measured on the newest: zlib 1 2.86x (1.2 ms), zstd 3
3.58x (0.7 ms), zstd 3 + a 110 KB trained dictionary 16.1x (0.3 ms). That last figure is one capture: its prompts
share whole pages. **Across days it is ~6.8x** (dictionary from the 19 and 24 Aug captures, measured on the newest
quarter of 24 Aug; a 19 Aug dictionary alone gets 8.0x on all of 24 Aug), ~2.5x fewer bytes than zlib. Retrain the
dictionary as prompts drift.
`vast/train_zstd_dict.py` trains it from captures (~80 s for 4,454 prompts); it is committed as
`vast/prompts.zdict` (the prompts are product names and public page text).

**Rolling it out needs no image and no restart.** The router re-reads `zstd_dict` (mount it from a plain ConfigMap)
whenever the file changes; a file that is not a trained dictionary is refused and the old one stays. On every scrape
of a `zstd` box it GETs the box's `/zstd-dict` (inflate_mw), and on a mismatch PUTs its dictionary with the box's API
key, then reads the id again. It compresses for that box with zstd only once the box reports our id; until then the
body goes uncompressed, so a mismatch, a refused key or an old image costs bandwidth, never a request. The box keeps
its previous two dictionaries: a frame names its dictionary id, so a body compressed just before a swap still
decodes, and one made with an unknown dictionary is a 400, never wrong text. vast-holo marks such a box `+zstd`.

### Retraining it, in the router (`autotrain`)

Each replica keeps the most recent bodies it sends to `zstd` boxes, up to `max_samples` (2,000) and `max_bytes`
(130 MiB). They are refcounts on request buffers, not copies, and bodies over 1 MiB (screenshots) are skipped. It is a
ring, not a reservoir, because the dictionary should fit the newest traffic and the held-out quarter is the newest. It
also keeps a byte-weighted **live ratio** over roughly the last 500 bodies it sent, per dictionary id.

A training is due when there are at least `min_samples` (500) bodies, at least `min_interval_s` (1 h) has passed since
the last one, and one of these holds:

- **NoBaseline**: there is no measured ratio for the dictionary in use, because the router just started or just
  became leader.
- **Age**: `every_s` (24 h) has passed since the last training.
- **Drift**: the live ratio fell below `drift` (0.8) × the baseline.

It runs on a thread of its own at nice 19, with the zstd crate's trainer (`dict_bytes` 110,000, as in
`train_zstd_dict.py`). It trains on the older three quarters, then measures the candidate and the dictionary in use on
the newest quarter. With no dictionary loaded, it measures against plain zstd. The candidate is published only if it
is `margin` (3%) better. Either way the measured ratio becomes the new baseline. Both ratios and the verdict are
logged and shown.

**One replica trains.** A Kubernetes Lease (`lease`, default `zev-router-zdict`, `lease_s` 30) elects it, renewed every
tick. Expiry is judged as client-go does: a Lease whose record has not changed for `lease_s` by our own clock, so no
two clocks are compared. The leader writes the dictionary as `binaryData[key]` of the ConfigMap `configmap`. Every
replica mounts that ConfigMap as the directory holding `zstd_dict`, so the kubelet refreshes the file (within about a
minute) and the existing reload and box sync take it from there. If the API cannot be reached, nobody leads and the
dictionary in use stays. With neither `api` nor `KUBERNETES_SERVICE_HOST`, the router leads alone and writes
`zstd_dict` itself.

Mount the whole ConfigMap as a directory, **not with `subPath`**: a subPath mount is never refreshed. The router needs:

```yaml
kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata: { name: zev-router-zdict }
rules:
  - { apiGroups: [coordination.k8s.io], resources: [leases], resourceNames: [zev-router-zdict], verbs: [get, update] }
  - { apiGroups: [""], resources: [configmaps], resourceNames: [zev-router-zdict], verbs: [get, update] }
  # create cannot be limited by name; drop these two once the Lease and ConfigMap exist
  - { apiGroups: [coordination.k8s.io], resources: [leases], verbs: [create] }
  - { apiGroups: [""], resources: [configmaps], verbs: [create] }
```

Bound to the router's ServiceAccount with a RoleBinding. The identity is `HOSTNAME`, i.e. the pod name.

## In the background, every `metrics_interval_s`

Resolve each upstream (`dns` = a headless Service, `addrs` = fixed backends) and the rented boxes in `external_file`
(vast-holo.py's Secret, re-read every tick: renting a box never restarts the router; an entry without
`gpu_multiplier` is refused). Scrape every pod's `/metrics` in both dialects: vLLM's
`num_requests_running/waiting`, `kv_cache_usage_perc`, `iteration_tokens_total_count`, and zev's `zev_in_flight` +
`zev_passes_total` (summed over a pod's models). A step counter frozen for `stall_s` while work is queued is a
**wedged engine**, skipped outright. A failed scrape scores the pod down and keeps its last sample.
Then `/v1/models` (vLLM: base model, loaded LoRA adapters with their `parent`, `max_model_len`; zev: the names it
serves), once per pod `/openapi.json` to see whether vLLM allows runtime LoRA loading (`/v1/load_lora_adapter`;
noted, not used), and for a `zstd` box the dictionary sync above. Image capability, compression and http2 are not
discoverable: they come from config (`text_only`, the Secret's marks, `http2`) and are shown next to the
discovered facts.

## Endpoints of the router itself

- `/ready`
- `/balancer-status`: JSON per pod (pool, held/cap, score, load, stalled, failing; discovered `models`, `lora_runtime`,
  `box_dict`; configured `text_only`, `compress`, `http2`, `external`), the router's `zstd_dict` id, `drained_s`, and
  `autotrain` (leader, samples, live and baseline ratios, trainings, last reason/result/ratios/time).
- `/metrics`: `zev_router_pod_{held,cap,score,load,stalled,failing}`, `zev_router_pod_model{model,parent,max_len}`
  (one series per model or adapter a pod lists), `zev_router_pod_box_dict`, `zev_router_zstd_dict_id`,
  `zev_router_responses_total{pool,status}`, `zev_router_drained`, `zev_router_capture_dropped_total`, and with
  `autotrain` `zev_router_zdict_{leader,samples,live_ratio,baseline_ratio,trainings,last_training_timestamp_seconds}`.
- `/drain[?seconds=N]`: close admission for N s (default `drain_seconds`, 0 reopens) and cut every request in flight,
  on every replica: with `peers_dns` it fans out to each one with `&peer=1`. Refused (403) with an
  `X-Forwarded-For`, i.e. through an ingress. A TTL: a drain nobody lifts expires.

## Replay capture

With `capture.endpoint`, default-pool requests produce a `request` event when admitted and a `response` event after
the full response (status, content type, backend, duration, body up to `max_response_bytes`, `truncated`), joined by
`capture_id`, the same schema as florian-nginx. The receiver is HEAD-checked every `check_interval_s`; while it is down,
or past `max_pending` events in flight, nothing is captured. Delivery is one shot and never delays inference.

## Config

```json
{
  "listen": "0.0.0.0:8000", "replicas": 2, "threads": 4,
  "pool_paths": { "/v1/systemone": "kev" },
  "pool_models": {},
  "zstd_dict": "/etc/zev-router/zdict/prompts.zdict",
  "autotrain": {},
  "peers_dns": "argo-workflow-zev-router-peers.argo.svc.cluster.local",
  "external_file": "/etc/zev-router/external/endpoints.json",
  "capture": { "endpoint": "http://172.17.16.214:9009/events" },
  "upstreams": [
    { "name": "kev-backfill-4090", "pool": "kev", "port": 8008, "gpu_multiplier": 10, "max_inflight": 64,
      "http2": true, "dns": "argo-workflow-florian-nginx-kev-backfill-4090-pods.argo.svc.cluster.local" },
    { "name": "florian-agent-4090-fp8", "port": 8000, "gpu_multiplier": 12, "max_inflight": 50,
      "dns": "argo-workflow-florian-nginx-florian-agent-4090-fp8-pods.argo.svc.cluster.local" },
    { "name": "florian-agent-5090", "port": 8000, "gpu_multiplier": 17, "max_inflight": 64, "text_only": true,
      "dns": "argo-workflow-florian-nginx-florian-agent-5090-pods.argo.svc.cluster.local" }
  ]
}
```

`max_inflight` is per card for the whole set of routers; each of `replicas` takes its share, rounded up (losing a
replica under-admits, the safe direction). Defaults: `metrics_interval_s` 5, `stall_s` 30, `queue_timeout_s` 180,
`affinity_wait_s` 20, `spill_ratio` 3, `drain_seconds` 300, `peers_port` 8000, `external_max_inflight` 40,
`compress_min` 4096, `compress_level` 1 (zlib), `zstd_level` 3, `max_body_bytes` 100 MiB; capture: `check_interval_s` 30,
`max_request_bytes` 25 MiB, `max_response_bytes` 2 MiB, `max_pending` 2, `timeout_ms` 500; autotrain: `every_s` 86400,
`min_interval_s` 3600, `drift` 0.8, `margin` 0.03, `min_samples` 500, `max_samples` 2000, `max_bytes` 130 MiB,
`dict_bytes` 110000, `lease`/`configmap` `zev-router-zdict`, `key` `prompts.zdict`, `lease_s` 30, `namespace` the pod's,
`api` in-cluster.

## Against florian-nginx

Same behaviour, plus: one process-wide state (no per-worker shared dicts, no JSON decode per pick); queued requests
woken on release instead of polling every 50 ms; no slot leaks to reclaim (a pod's slots live in the `Pod` an
in-flight request holds); 429 retried; compression and the rented-box key decided per attempt, so a retry can go to
any pod, not only one that inflates; `/drain` cuts in-flight requests itself, no nginx reload; prompts released at
the first response byte; typed and tested.

Limits: replicas still keep separate slot counts (shared state, e.g. Redis, past ~3 replicas); each in-flight request
holds its body until its response headers arrive (bounded by `max_body_bytes`; nginx spilled bodies over 1 MiB to a
temp file instead); vLLM pods stay on HTTP/1.1 keep-alive (uvicorn has no HTTP/2).

## Checks

    cargo test        # parsing, stall detection, pools, caps, weights, text-only, homing + spill, the rented-box
                      # registry (zstd/deflate marks, never on a configured upstream), drain, image scan, model,
                      # zlib and zstd + dictionary round trips
    python3 smoke.py  # the real binary against fake pods, a fake capture receiver and a rented-box file: 429 retry
                      # with a 200 KB body, stall skip, model pool, image routing, rented key + deflate, capture,
                      # affinity, drain cutting a live stream and reopening, a zstd + dictionary box beating zlib,
                      # in-cluster pods never compressed, h2c: 200 requests at 50 concurrent over one TCP
                      # connection to a zev-like pod. Needs `cargo build --example fake_h2c_pod` and the
                      # zstandard Python package.
    python3 smoke_train.py  # two replicas, a stub Kubernetes API, a real inflate_mw box: only the Lease holder
                            # trains, publishes to the ConfigMap, both replicas and the box take the dictionary, a
                            # retrain on the same traffic is rejected, the other replica takes over the Lease when
                            # the leader is killed.

To exercise the Lease for real, run two replicas with `autotrain` in a namespace that has the Role above.
`kubectl get lease zev-router-zdict -o yaml` shows the holder. Delete the holder pod: the other replica takes over
within `lease_s`. For a quick end-to-end run, set `min_samples: 50, every_s: 600`.

Live, 2026-10-21: 200 real `/v1/systemone` calls through the router to the Kev NodePorts in 2.8 s, all 2xx.
