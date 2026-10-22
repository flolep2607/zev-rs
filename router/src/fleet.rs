//! The fleet: pods discovered per upstream, their latest /metrics sample, the slots this router replica holds on
//! them, and the pick. A port of learning-helm's florian-nginx (pick.lua, init_worker.lua) with one process-wide state
//! instead of nginx shared dicts, and waiting on a Notify instead of a 50 ms poll.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::sync::{watch, Notify};

#[derive(Deserialize, Clone, Debug)]
pub struct Upstream {
    pub name: String,
    #[serde(default = "default_pool")]
    pub pool: String,
    /// Headless Service name: DNS gives one A record per ready pod.
    #[serde(default)]
    pub dns: Option<String>,
    /// Fixed "ip:port" backends.
    #[serde(default)]
    pub addrs: Vec<String>,
    #[serde(default)]
    pub port: u16,
    /// Routing weight: a 4090 is 10. No default, as in florian-nginx: a guessed weight wrecks the split.
    pub gpu_multiplier: f64,
    /// Admission cap per card for the whole set of routers; each replica takes its share.
    pub max_inflight: usize,
    /// Started with --language-model-only: an image request 400s there, and 4xx is not retried.
    #[serde(default)]
    pub text_only: bool,
    /// Request-body compression, for rented boxes only: they are metered per GB and 96% of their traffic is
    /// inbound prompt. Never configurable for an in-cluster upstream: on the LAN it is CPU spent for nothing.
    #[serde(skip)]
    pub compress: Compress,
    /// Speaks h2c (HTTP/2 over plain TCP, prior knowledge): one long-lived connection per pod carries every request,
    /// pinged so a dead one is noticed while idle. zev with axum's http2 feature does; vLLM's uvicorn does not.
    #[serde(default)]
    pub http2: bool,
    /// Rented boxes run vLLM with --api-key; sent to them only, never to an in-cluster pod.
    #[serde(skip)]
    pub api_key: Option<String>,
}

fn default_pool() -> String {
    "default".into()
}

/// What a rented box's inflate_mw decodes. On 1,333 captured agent prompts (mean 79 KB, newest quarter, dictionary
/// trained on the older three): zlib 1 2.86x, zstd 3 3.58x, zstd 3 + a 110 KB dictionary 16.1x at 0.3 ms a body.
/// Across days it is ~6.8x (a 19-24 Aug dictionary on the newest 24 Aug prompts): same-day prompts share whole
/// pages, later ones only the system prompt and tool schemas. Retrain the dictionary as prompts drift.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Compress {
    #[default]
    None,
    Deflate,
    Zstd,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CaptureCfg {
    /// POST target for the replay events, e.g. http://172.17.16.214:9009/events.
    pub endpoint: String,
    /// Probed with HEAD every check_interval_s; empty = the endpoint itself. Any HTTP status means up.
    #[serde(default)]
    pub health_path: String,
    #[serde(default = "thirty")]
    pub check_interval_s: u64,
    #[serde(default = "mib25")]
    pub max_request_bytes: usize,
    #[serde(default = "mib2")]
    pub max_response_bytes: usize,
    /// Events awaiting delivery; past this they are dropped, so a slow receiver cannot eat the router's memory.
    #[serde(default = "two")]
    pub max_pending: usize,
    #[serde(default = "ms500")]
    pub timeout_ms: u64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub listen: String,
    /// Router replicas behind one Service. Each keeps its own slot counts, so caps are split between them;
    /// losing a replica under-admits, the safe direction. ponytail: no shared state; Redis if this passes ~3.
    #[serde(default = "one")]
    pub replicas: usize,
    /// Proxy worker threads. One process, so one shared state across all of them.
    #[serde(default = "four")]
    pub threads: usize,
    /// Exact request path -> pool. Anything unlisted is "default".
    #[serde(default)]
    pub pool_paths: HashMap<String, String>,
    /// The body's `model` -> pool, applied after pool_paths: kev-combo answers every model name with its own adapter,
    /// so its models need a pool of their own on the same /v1/systemone path.
    #[serde(default)]
    pub pool_models: HashMap<String, String>,
    pub upstreams: Vec<Upstream>,
    #[serde(default = "five")]
    pub metrics_interval_s: u64,
    #[serde(default = "thirty")]
    pub stall_s: u64,
    #[serde(default = "one_eighty")]
    pub queue_timeout_s: u64,
    #[serde(default = "session_header")]
    pub affinity_header: String,
    /// How long a session waits for its full home pod (its prefix cache) before taking any free slot.
    #[serde(default = "twenty")]
    pub affinity_wait_s: u64,
    /// ...cut short once another pod scores this many times the home; 0 = always wait the full affinity_wait_s.
    #[serde(default = "three")]
    pub spill_ratio: f64,
    /// Default stop length of /drain. A TTL: a drain nobody lifts expires.
    #[serde(default = "three_hundred")]
    pub drain_seconds: u64,
    /// Headless Service of the router replicas, so /drain reaches every one, not the one the Service picked.
    #[serde(default)]
    pub peers_dns: Option<String>,
    #[serde(default = "port8000")]
    pub peers_port: u16,
    #[serde(default)]
    pub capture: Option<CaptureCfg>,
    /// Rented GPUs (vast-holo.py's Secret), re-read every scrape: renting a box never restarts the router.
    #[serde(default)]
    pub external_file: Option<String>,
    #[serde(default = "forty")]
    pub external_max_inflight: usize,
    /// Compress request bodies at least this big for rented boxes that decode them; 0 = never.
    #[serde(default = "kib4")]
    pub compress_min: usize,
    /// zlib level for `deflate` boxes. 1: 2.7x for 0.5 ms on these prompts, against 3.2x for 6 ms at level 6.
    #[serde(default = "one32")]
    pub compress_level: u32,
    /// zstd level for `zstd` boxes. 3: 3.6x alone, ~6.8x with the dictionary, faster than zlib 1 either way.
    #[serde(default = "three32")]
    pub zstd_level: i32,
    /// A zstd dictionary trained on captured prompts (vast/train_zstd_dict.py). The box must hold the same file:
    /// the frame names its dictionary id, so a box without it answers 400 rather than guessing.
    #[serde(default)]
    pub zstd_dict: Option<String>,
    #[serde(default = "mib100")]
    pub max_body_bytes: usize,
    /// Retrain `zstd_dict` from the bodies sent to zstd boxes (train.rs). Needs `zstd_dict`: that file is where every
    /// replica picks the published dictionary up.
    #[serde(default)]
    pub autotrain: Option<crate::train::AutoTrainCfg>,
}

fn one() -> usize {
    1
}
fn two() -> usize {
    2
}
fn four() -> usize {
    4
}
fn forty() -> usize {
    40
}
fn five() -> u64 {
    5
}
fn twenty() -> u64 {
    20
}
fn thirty() -> u64 {
    30
}
fn one_eighty() -> u64 {
    180
}
fn three_hundred() -> u64 {
    300
}
fn three() -> f64 {
    3.0
}
fn one32() -> u32 {
    1
}
fn three32() -> i32 {
    3
}
fn port8000() -> u16 {
    8000
}
fn ms500() -> u64 {
    500
}
fn kib4() -> usize {
    4096
}
fn mib2() -> usize {
    2 << 20
}
fn mib25() -> usize {
    25 << 20
}
fn mib100() -> usize {
    100 << 20
}
fn session_header() -> String {
    "x-session-id".into()
}

/// What a scrape says about a pod. A pod speaks vLLM's names or zev's, never both.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    pub running: f64,
    pub waiting: f64,
    pub kv_cache: f64,
    /// The engine's step counter; None when absent, so stall detection fails open.
    pub iterations: Option<f64>,
}

pub fn parse_metrics(body: &str) -> Sample {
    let mut s = Sample::default();
    let (mut zev_load, mut zev_passes) = (0.0, None::<f64>);
    for line in body.lines().filter(|l| !l.starts_with('#')) {
        let Some((name, value)) = line.rsplit_once(char::is_whitespace) else {
            continue;
        };
        let Ok(v) = value.parse::<f64>() else {
            continue;
        };
        // The name up to its labels: `vllm:x{a="b"}` and `vllm:x` are the same series name.
        match name.split('{').next().unwrap_or("").trim() {
            "vllm:num_requests_running" => s.running = v,
            "vllm:num_requests_waiting" => s.waiting = v,
            "vllm:kv_cache_usage_perc" => s.kv_cache = v,
            "vllm:iteration_tokens_total_count" => s.iterations = Some(v),
            // zev serves several models per pod, one series each: summed. in_flight is queued + running, the unit
            // score() divides by; passes_total is the step counter. A Kev pod has no KV cache to fill.
            "zev_in_flight" => zev_load += v,
            "zev_passes_total" => zev_passes = Some(zev_passes.unwrap_or(0.0) + v),
            _ => {}
        }
    }
    if zev_passes.is_some() {
        s.running = zev_load;
        s.iterations = zev_passes;
    }
    s
}

/// Rented boxes from the mounted Secret. An entry without ip, port or gpu_multiplier is refused, not defaulted:
/// the weight decides the traffic split. A missing or malformed file means nothing rented.
pub fn external_upstreams(json: &str, default_cap: usize) -> Vec<Upstream> {
    #[derive(Deserialize)]
    struct Entry {
        name: Option<String>,
        ip: Option<String>,
        port: Option<u16>,
        gpu_multiplier: Option<f64>,
        max_inflight: Option<usize>,
        #[serde(default)]
        deflate: bool,
        /// Its image has zstandard and the dictionary (vast-holo's +zstd mark). Wins over deflate.
        #[serde(default)]
        zstd: bool,
        #[serde(default)]
        text_only: bool,
        #[serde(default)]
        http2: bool,
    }
    #[derive(Deserialize)]
    struct Doc {
        api_key: Option<String>,
        #[serde(default)]
        endpoints: Vec<serde_json::Value>,
    }
    let Ok(doc) = serde_json::from_str::<Doc>(json) else {
        log::error!("external endpoints file is not valid JSON, ignoring");
        return vec![];
    };
    let key = doc.api_key.filter(|k| !k.is_empty());
    doc.endpoints
        .into_iter()
        .filter_map(|v| {
            let e: Entry = serde_json::from_value(v.clone()).ok()?;
            let (Some(ip), Some(port), Some(mult)) = (e.ip, e.port, e.gpu_multiplier) else {
                log::error!("external endpoint ignored, needs ip, port and gpu_multiplier: {v}");
                return None;
            };
            Some(Upstream {
                name: e.name.unwrap_or_else(|| format!("external-{ip}")),
                pool: default_pool(), // rented boxes are vLLM
                dns: None,
                addrs: vec![format!("{ip}:{port}")],
                port,
                gpu_multiplier: mult,
                max_inflight: e.max_inflight.unwrap_or(default_cap),
                text_only: e.text_only,
                compress: match (e.zstd, e.deflate) {
                    (true, _) => Compress::Zstd,
                    (false, true) => Compress::Deflate,
                    _ => Compress::None,
                },
                http2: e.http2,
                api_key: key.clone(),
            })
        })
        .collect()
}

/// One entry of a pod's /v1/models. vLLM lists its base model and every loaded LoRA adapter (with `parent`, the
/// base it applies to) and `max_model_len`; zev lists the names it serves.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub parent: Option<String>,
    pub max_len: Option<u64>,
}

/// Either dialect of /v1/models: OpenAI's `{"data":[{"id","parent","max_model_len"}]}` (vLLM) or zev's
/// `{"models":[{"name"}]}`. None when it is neither, so a pod that does not answer keeps its last list.
pub fn parse_models(json: &str) -> Option<Vec<ModelInfo>> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let s = |e: &serde_json::Value, k: &str| e.get(k).and_then(|x| x.as_str()).map(str::to_string);
    if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
        return Some(
            data.iter()
                .filter_map(|e| {
                    Some(ModelInfo {
                        id: s(e, "id")?,
                        parent: s(e, "parent"),
                        max_len: e.get("max_model_len").and_then(|x| x.as_u64()),
                    })
                })
                .collect(),
        );
    }
    let models = v.get("models")?.as_array()?;
    Some(
        models
            .iter()
            .filter_map(|e| {
                Some(ModelInfo {
                    id: s(e, "name")?,
                    parent: None,
                    max_len: None,
                })
            })
            .collect(),
    )
}

#[derive(Debug)]
pub struct State {
    pub sample: Sample,
    pub last_seen: Option<Instant>,
    /// When the step counter froze with work queued; a pod frozen past stall_s is wedged and skipped.
    pub stalled_since: Option<Instant>,
    pub failing_until: Option<Instant>,
    /// Requests this replica has dispatched to the pod and not yet released.
    pub held: usize,
    /// From the pod's /v1/models; kept across a failed fetch.
    pub models: Vec<ModelInfo>,
    /// vLLM with VLLM_ALLOW_RUNTIME_LORA_UPDATING exposes /v1/load_lora_adapter; checked once per pod.
    pub lora_runtime: Option<bool>,
    /// The zstd dictionary id a rented box reports holding (its /zstd-dict); zstd is sent only when it is ours.
    pub box_dict: Option<u32>,
}

#[derive(Debug)]
pub struct Pod {
    pub addr: SocketAddr,
    pub up: Arc<Upstream>,
    pub cap: usize,
    pub state: Mutex<State>,
}

const STALE: Duration = Duration::from_secs(30);
const STALE_PENALTY_AFTER: Duration = Duration::from_secs(15);
const FAIL_FOR: Duration = Duration::from_secs(30);
const FAIL_PENALTY: f64 = 0.05; // soft, not a skip: a fleet-wide blip must not empty the list
const KV_CLIFF: f64 = 0.95;
/// How long a model name stays "known" after the last pod listing it went away: a request for it then fails fast
/// instead of falling back to a pod that would answer with another adapter (zev answers any name with its own).
const MODEL_MEMORY: Duration = Duration::from_secs(3600);

impl Pod {
    pub fn new(addr: SocketAddr, up: Arc<Upstream>, replicas: usize) -> Self {
        let cap = up.max_inflight.div_ceil(replicas.max(1));
        let state = State {
            sample: Sample::default(),
            last_seen: None,
            stalled_since: None,
            failing_until: None,
            held: 0,
            models: Vec::new(),
            lora_runtime: None,
            box_dict: None,
        };
        Pod {
            addr,
            up,
            cap,
            state: Mutex::new(state),
        }
    }

    pub fn stalled(st: &State, now: Instant, stall: Duration) -> bool {
        st.stalled_since
            .is_some_and(|t| now.duration_since(t) >= stall)
    }

    /// Spare capacity weighted by card speed. `running` counts with `waiting`: vLLM reports waiting > 0 only once
    /// running hits --max-num-seqs, so on waiting alone a pod at 31/32 scores the same as one at 2/32.
    pub fn score(&self, st: &State, now: Instant, stall: Duration) -> f64 {
        let Some(seen) = st.last_seen else { return 0.0 };
        if Self::stalled(st, now, stall) || now.duration_since(seen) > STALE {
            return 0.0;
        }
        let mut score = self.up.gpu_multiplier / (1.0 + st.sample.running + st.sample.waiting);
        if st.sample.kv_cache > KV_CLIFF {
            score *= 0.1;
        }
        if now.duration_since(seen) > STALE_PENALTY_AFTER {
            score *= 0.5;
        }
        if st.failing_until.is_some_and(|t| now < t) {
            score *= FAIL_PENALTY;
        }
        score
    }

    pub fn fail(&self) {
        self.state.lock().unwrap().failing_until = Some(Instant::now() + FAIL_FOR);
    }

    /// Record a scrape. The stall clock starts when the step counter stops moving while work is queued and is
    /// carried while it stays frozen; an idle pod legitimately stops stepping.
    pub fn record(&self, sample: Sample, now: Instant) {
        let mut st = self.state.lock().unwrap();
        let load = sample.running + sample.waiting;
        let frozen = sample.iterations.is_some()
            && st.last_seen.is_some()
            && sample.iterations == st.sample.iterations;
        st.stalled_since = if frozen && load > 0.0 {
            Some(st.stalled_since.unwrap_or(now))
        } else {
            None
        };
        st.sample = sample;
        st.last_seen = Some(now);
    }
}

/// What a request can go to. With `model` set (a name some pod lists, or listed within the last hour), only pods
/// listing it, whatever their pool: that is how a LoRA adapter or kev-combo's model finds its pods. Without it,
/// the path's pool as before.
#[derive(Clone, Copy, Debug)]
pub struct Want<'a> {
    pub pool: &'a str,
    pub model: Option<&'a str>,
    pub needs_image: bool,
}

/// The router's zstd dictionary, as read from `zstd_dict`.
pub struct ZDict {
    pub id: u32,
    pub raw: bytes::Bytes,
    pub enc: zstd::dict::EncoderDictionary<'static>,
}

pub struct Fleet {
    pub cfg: Config,
    pub pods: RwLock<Vec<Arc<Pod>>>,
    /// Woken when a slot is released or a scrape lands, so queued requests re-pick at once instead of polling.
    pub freed: Notify,
    /// Responses by (pool, status class), for /metrics.
    responses: Mutex<HashMap<(String, &'static str), u64>>,
    /// /drain: admission closed until then.
    drained_until: Mutex<Option<Instant>>,
    /// Bumped by /drain: every in-flight request watching it drops its upstream connection, which aborts the
    /// generation. nginx needed a reload for that; here it is a channel.
    pub cut: watch::Sender<u64>,
    /// HTTP/1.1 keep-alive pool: up to 64 idle connections per pod, reused across requests.
    h1: reqwest::Client,
    /// h2c: one connection per pod, multiplexing every request, kept for good and pinged every 10 s so a peer that
    /// vanished (node NotReady, SYN blackholed) is noticed before a request is sent into it.
    h2: reqwest::Client,
    /// Every model name a pod has listed, and when last: see MODEL_MEMORY.
    seen_models: Mutex<HashMap<String, Instant>>,
    /// zstd_dict, re-read when its file changes: retraining is a ConfigMap update, nothing restarts.
    zdict: RwLock<Option<Arc<ZDict>>>,
    zdict_stamp: Mutex<Option<(std::time::SystemTime, u64)>>,
    pub train: Option<Arc<crate::train::Trainer>>,
}

/// A dead node blackholes SYN rather than refusing it; the default would be a long wait before failover starts.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

impl Fleet {
    pub fn new(cfg: Config) -> Self {
        let tcp = || {
            reqwest::Client::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .tcp_keepalive(Duration::from_secs(30))
        };
        let h1 = tcp()
            .pool_max_idle_per_host(64)
            .build()
            .expect("http/1.1 client");
        let h2 = tcp()
            .http2_prior_knowledge()
            .http2_keep_alive_interval(Duration::from_secs(10))
            .http2_keep_alive_timeout(Duration::from_secs(5))
            .http2_keep_alive_while_idle(true)
            .http2_adaptive_window(true)
            .pool_idle_timeout(None)
            .build()
            .expect("h2c client");
        let train = cfg.autotrain.clone().map(|t| {
            let path = cfg
                .zstd_dict
                .clone()
                .expect("autotrain needs zstd_dict: where the dictionary is published");
            Arc::new(crate::train::Trainer::new(t, cfg.zstd_level, path))
        });
        let f = Fleet {
            cfg,
            pods: RwLock::new(Vec::new()),
            freed: Notify::new(),
            responses: Mutex::new(HashMap::new()),
            drained_until: Mutex::new(None),
            cut: watch::channel(0).0,
            h1,
            h2,
            seen_models: Mutex::new(HashMap::new()),
            zdict: RwLock::new(None),
            zdict_stamp: Mutex::new(None),
            train,
        };
        f.reload_zdict();
        f
    }

    /// For plain HTTP/1.1 calls of the router's own (the /drain fan-out to its peers).
    pub fn http1(&self) -> &reqwest::Client {
        &self.h1
    }

    /// The client for this upstream's protocol.
    pub fn client(&self, up: &Upstream) -> &reqwest::Client {
        if up.http2 {
            &self.h2
        } else {
            &self.h1
        }
    }

    pub fn stall(&self) -> Duration {
        Duration::from_secs(self.cfg.stall_s)
    }

    pub fn pool_for(&self, path: &str) -> &str {
        self.cfg
            .pool_paths
            .get(path)
            .map_or("default", String::as_str)
    }

    pub fn zdict(&self) -> Option<Arc<ZDict>> {
        self.zdict.read().unwrap().clone()
    }

    /// (Re)load zstd_dict when its file changed. A missing or unreadable file keeps the dictionary in use; one that
    /// is not a trained dictionary is refused with an error, also keeping the old one.
    pub fn reload_zdict(&self) {
        let Some(path) = &self.cfg.zstd_dict else {
            return;
        };
        let stamp = std::fs::metadata(path)
            .ok()
            .map(|m| (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len()));
        if stamp.is_none() || *self.zdict_stamp.lock().unwrap() == stamp {
            return;
        }
        *self.zdict_stamp.lock().unwrap() = stamp;
        let Ok(raw) = std::fs::read(path) else { return };
        let Some(id) = zstd::zstd_safe::get_dict_id_from_dict(&raw) else {
            log::error!("{path} is not a trained zstd dictionary (no dictionary id); keeping the one in use");
            return;
        };
        let enc = zstd::dict::EncoderDictionary::copy(&raw, self.cfg.zstd_level);
        log::warn!(
            "zstd dictionary {} loaded from {path} ({} bytes)",
            id.get(),
            raw.len()
        );
        *self.zdict.write().unwrap() = Some(Arc::new(ZDict {
            id: id.get(),
            raw: raw.into(),
            enc,
        }));
    }

    /// Does `model` pin the request to the pods that list it? Yes when a pod lists it now or did within the last
    /// hour. A name never listed (e.g. "kev-4b", which zev serves under its own names) keeps the path's pool.
    pub fn model_known(&self, model: &str) -> bool {
        self.seen_models
            .lock()
            .unwrap()
            .get(model)
            .is_some_and(|t| t.elapsed() < MODEL_MEMORY)
    }

    fn eligible(p: &Pod, st: &State, w: &Want) -> bool {
        let fits = match w.model {
            Some(m) => st.models.iter().any(|x| x.id == m),
            None => p.up.pool == w.pool,
        };
        fits && !(w.needs_image && p.up.text_only)
    }

    /// Only a request that could reach a text-only pod needs its body scanned for images.
    pub fn has_text_only(&self, w: &Want) -> bool {
        self.pods
            .read()
            .unwrap()
            .iter()
            .any(|p| p.up.text_only && Self::eligible(p, &p.state.lock().unwrap(), w))
    }

    pub fn drained(&self) -> bool {
        self.drained_until
            .lock()
            .unwrap()
            .is_some_and(|t| Instant::now() < t)
    }

    /// Close admission for `seconds` (0 = reopen) and cut everything in flight.
    pub fn drain(&self, seconds: u64) {
        *self.drained_until.lock().unwrap() =
            (seconds > 0).then(|| Instant::now() + Duration::from_secs(seconds));
        if seconds > 0 {
            self.cut.send_modify(|n| *n += 1);
        }
        self.freed.notify_waiters(); // queued requests re-check and give up
    }

    pub fn drain_remaining_s(&self) -> f64 {
        self.drained_until.lock().unwrap().map_or(0.0, |t| {
            t.saturating_duration_since(Instant::now()).as_secs_f64()
        })
    }

    pub fn count(&self, pool: &str, code: u16) {
        let class = match code {
            0 => "none",
            429 => "429",
            c if c >= 500 => "5xx",
            c if c >= 400 => "4xx",
            _ => "2xx",
        };
        *self
            .responses
            .lock()
            .unwrap()
            .entry((pool.to_string(), class))
            .or_default() += 1;
    }

    /// Weighted-random pick among the eligible pods with a free slot, taking the slot. `known` counts the eligible
    /// pods at all, so the caller can tell "all full" (wait) from "none" (fail now).
    pub fn reserve(&self, w: &Want, tried: &HashSet<SocketAddr>) -> (Option<Arc<Pod>>, usize) {
        let now = Instant::now();
        let stall = self.stall();
        let pods = self.pods.read().unwrap();
        let mut known = 0;
        let mut cands: Vec<(f64, &Arc<Pod>)> = Vec::new();
        for p in pods.iter() {
            let st = p.state.lock().unwrap();
            if !Self::eligible(p, &st, w) {
                continue;
            }
            known += 1;
            if tried.contains(&p.addr) {
                continue;
            }
            let score = p.score(&st, now, stall);
            if score > 0.0 && st.held < p.cap {
                cands.push((score, p));
            }
        }
        while !cands.is_empty() {
            let total: f64 = cands.iter().map(|c| c.0).sum();
            let mut r = rand::random::<f64>() * total;
            let i = cands.iter().position(|c| {
                r -= c.0;
                r <= 0.0
            });
            let (_, p) = cands.swap_remove(i.unwrap_or(cands.len() - 1));
            if self.take(p) {
                return (Some(p.clone()), known);
            }
        }
        (None, known)
    }

    /// Weighted rendezvous hash: every replica agrees on a session's home with no shared state, and losing a pod
    /// rehomes only its own sessions. Failing or stalled pods are skipped.
    pub fn home(&self, w: &Want, session: &str) -> Option<Arc<Pod>> {
        let now = Instant::now();
        let stall = self.stall();
        let pods = self.pods.read().unwrap();
        pods.iter()
            .filter(|p| {
                let st = p.state.lock().unwrap();
                Self::eligible(p, &st, w)
                    && st.last_seen.is_some()
                    && !Pod::stalled(&st, now, stall)
                    && !st.failing_until.is_some_and(|t| now < t)
            })
            .map(|p| {
                let u = (fnv(&format!("{session}|{}", p.addr)) as f64 / u64::MAX as f64).max(1e-12);
                (p.up.gpu_multiplier / -u.ln(), p)
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, p)| p.clone())
    }

    /// Stop waiting for a full home now? Yes once another pod with a free slot scores `ratio` times the home:
    /// rendezvous hashing balances session count, not weight, and a few heavy sessions pinned pods at cap while a
    /// third of the slots sat idle (florian-nginx, 2026-08-11).
    pub fn crowded(&self, home: &Pod, w: &Want) -> bool {
        let ratio = self.cfg.spill_ratio;
        if ratio <= 0.0 {
            return false;
        }
        let now = Instant::now();
        let stall = self.stall();
        let home_score = home.score(&home.state.lock().unwrap(), now, stall);
        if home_score <= 0.0 {
            return true;
        }
        self.pods
            .read()
            .unwrap()
            .iter()
            .filter(|p| p.addr != home.addr)
            .any(|p| {
                let st = p.state.lock().unwrap();
                Self::eligible(p, &st, w)
                    && st.held < p.cap
                    && p.score(&st, now, stall) >= ratio * home_score
            })
    }

    pub fn take(&self, p: &Arc<Pod>) -> bool {
        let mut st = p.state.lock().unwrap();
        if st.held < p.cap {
            st.held += 1;
            true
        } else {
            false
        }
    }

    pub fn release(&self, p: &Pod) {
        let mut st = p.state.lock().unwrap();
        st.held = st.held.saturating_sub(1);
        drop(st);
        self.freed.notify_waiters();
    }

    /// Resolve every upstream, keep the Pod (and its held count) for addresses still present, drop the rest. A
    /// request in flight on a dropped pod holds its own Arc and releases into it: nothing leaks, nothing to reclaim.
    pub async fn discover(&self) {
        let mut ups: Vec<Upstream> = self.cfg.upstreams.clone();
        if let Some(path) = &self.cfg.external_file {
            // Absent = nothing rented, the normal state.
            if let Ok(json) = std::fs::read_to_string(path) {
                ups.extend(external_upstreams(&json, self.cfg.external_max_inflight));
            }
        }
        let mut found: Vec<(SocketAddr, Arc<Upstream>)> = Vec::new();
        for up in ups {
            let up = Arc::new(up);
            if let Some(dns) = &up.dns {
                match tokio::net::lookup_host((dns.as_str(), up.port)).await {
                    Ok(addrs) => found.extend(addrs.map(|a| (a, up.clone()))),
                    Err(e) => log::warn!("resolve {dns}: {e}"),
                }
            }
            for a in &up.addrs {
                match a.parse() {
                    Ok(a) => found.push((a, up.clone())),
                    Err(e) => log::error!("upstream {}: bad addr {a}: {e}", up.name),
                }
            }
        }
        let mut pods = self.pods.write().unwrap();
        let old: HashMap<SocketAddr, Arc<Pod>> = pods.drain(..).map(|p| (p.addr, p)).collect();
        *pods = found
            .into_iter()
            .map(|(a, up)| match old.get(&a) {
                Some(p)
                    if p.up.name == up.name
                        && p.up.api_key == up.api_key
                        && p.up.compress == up.compress =>
                {
                    p.clone()
                }
                _ => Arc::new(Pod::new(a, up, self.cfg.replicas)),
            })
            .collect();
    }

    /// One scrape of every pod: /metrics (load, stall), /v1/models (what it serves), once its runtime-LoRA
    /// capability, and for a rented zstd box its dictionary, pushed when it is not ours. A failed /metrics marks the
    /// pod failing and keeps its last sample: zeros would score a hung pod as idle, the most attractive backend.
    pub async fn scrape(self: &Arc<Self>) {
        let pods: Vec<Arc<Pod>> = self.pods.read().unwrap().clone();
        let stall = self.stall();
        let mut set = tokio::task::JoinSet::new();
        for p in pods {
            let f = self.clone();
            set.spawn(async move {
                // Over the pod's own connection: on h2 the scrape rides the request stream's connection, and a
                // scrape that fails is also news about that connection.
                let http = f.client(&p.up).clone();
                let get = |path: &str| {
                    let mut req = http
                        .get(format!("http://{}{path}", p.addr))
                        .timeout(Duration::from_secs(2));
                    if let Some(key) = &p.up.api_key {
                        req = req.bearer_auth(key);
                    }
                    req.send()
                };
                let text = |r: reqwest::Result<reqwest::Response>| async move {
                    match r {
                        Ok(r) if r.status().is_success() => r.text().await.ok(),
                        _ => None,
                    }
                };
                match text(get("/metrics").await).await {
                    Some(b) => {
                        p.record(parse_metrics(&b), Instant::now());
                        let st = p.state.lock().unwrap();
                        if st.stalled_since.is_some_and(|t| t.elapsed() >= stall) {
                            log::error!(
                                "stalled engine on {} ({}): step counter frozen with work queued",
                                p.addr,
                                p.up.name
                            );
                        }
                    }
                    None => {
                        log::warn!("metrics fetch failed for {} ({})", p.addr, p.up.name);
                        p.fail();
                        return;
                    }
                }
                if let Some(models) = text(get("/v1/models").await)
                    .await
                    .as_deref()
                    .and_then(parse_models)
                {
                    {
                        let now = Instant::now();
                        let mut seen = f.seen_models.lock().unwrap();
                        for m in &models {
                            seen.insert(m.id.clone(), now);
                        }
                    }
                    let vllm = models.iter().any(|m| m.max_len.is_some());
                    let ask_lora = vllm && p.state.lock().unwrap().lora_runtime.is_none();
                    p.state.lock().unwrap().models = models;
                    if ask_lora {
                        // ponytail: openapi.json is read once per pod to see whether runtime LoRA loading is on;
                        // on-demand loading itself is not implemented.
                        let api = text(get("/openapi.json").await).await;
                        p.state.lock().unwrap().lora_runtime =
                            api.map(|a| a.contains("/v1/load_lora_adapter"));
                    }
                }
                if p.up.compress == Compress::Zstd {
                    f.sync_dict(&p).await;
                }
            });
        }
        while set.join_next().await.is_some() {}
        self.freed.notify_waiters();
    }

    /// Make a rented box hold our dictionary: read its id, PUT ours on a mismatch, read again. Only the id the box
    /// reports is recorded, so a failed or refused push (wrong key, old image) leaves it uncompressed, never broken.
    async fn sync_dict(&self, p: &Pod) {
        let http = self.client(&p.up);
        let url = format!("http://{}/zstd-dict", p.addr);
        let held = || async {
            let mut req = http.get(&url).timeout(Duration::from_secs(2));
            if let Some(key) = &p.up.api_key {
                req = req.bearer_auth(key);
            }
            let r = req.send().await.ok().filter(|r| r.status().is_success())?;
            r.json::<serde_json::Value>()
                .await
                .ok()?
                .get("dict_id")?
                .as_u64()
                .map(|i| i as u32)
        };
        let mut id = held().await;
        if let Some(d) = self.zdict().filter(|d| id != Some(d.id)) {
            let mut put = http
                .put(&url)
                .body(d.raw.clone())
                .timeout(Duration::from_secs(10));
            if let Some(key) = &p.up.api_key {
                put = put.bearer_auth(key);
            }
            match put.send().await {
                Ok(r) if r.status().is_success() => {
                    id = held().await;
                    log::warn!(
                        "pushed zstd dictionary {} to {} ({}): box now holds {id:?}",
                        d.id,
                        p.addr,
                        p.up.name
                    );
                }
                Ok(r) => log::warn!(
                    "{} ({}) refused zstd dictionary {}: {}",
                    p.addr,
                    p.up.name,
                    d.id,
                    r.status()
                ),
                Err(e) => log::warn!("pushing zstd dictionary to {} failed: {e}", p.addr),
            }
        }
        p.state.lock().unwrap().box_dict = id;
    }

    /// The dictionary to compress for this box with, if it holds ours. None with a dictionary configured means: send
    /// it uncompressed until the box confirms; with none configured, plain zstd.
    pub fn box_ready(&self, p: &Pod) -> Result<Option<Arc<ZDict>>, ()> {
        match self.zdict() {
            None => Ok(None),
            Some(d) if p.state.lock().unwrap().box_dict == Some(d.id) => Ok(Some(d)),
            Some(_) => Err(()),
        }
    }

    pub fn status_json(&self) -> String {
        let now = Instant::now();
        let stall = self.stall();
        let pods: Vec<_> = self
            .pods
            .read()
            .unwrap()
            .iter()
            .map(|p| {
                let st = p.state.lock().unwrap();
                serde_json::json!({
                    "name": p.up.name, "pool": p.up.pool, "addr": p.addr.to_string(),
                    "held": st.held, "cap": p.cap, "score": p.score(&st, now, stall),
                    "running": st.sample.running, "waiting": st.sample.waiting, "kv_cache": st.sample.kv_cache,
                    "iterations": st.sample.iterations, "stalled": Pod::stalled(&st, now, stall),
                    "failing": st.failing_until.is_some_and(|t| now < t),
                    // discovered
                    "models": st.models, "lora_runtime": st.lora_runtime, "box_dict": st.box_dict,
                    // configured: not discoverable from vLLM
                    "text_only": p.up.text_only, "compress": format!("{:?}", p.up.compress).to_lowercase(),
                    "http2": p.up.http2, "external": p.up.api_key.is_some(),
                    "seconds_since_scrape": st.last_seen.map(|t| now.duration_since(t).as_secs_f64()),
                })
            })
            .collect();
        serde_json::json!({
            "drained_s": self.drain_remaining_s(),
            "zstd_dict": self.zdict().map(|d| d.id),
            "autotrain": self.train.as_ref().map(|t| t.status()),
            "pods": pods,
        })
        .to_string()
    }

    /// Per-pod gauges labelled by pool and upstream, the models each pod serves, plus responses by pool and status.
    pub fn prometheus(&self) -> String {
        use std::fmt::Write;
        let now = Instant::now();
        let stall = self.stall();
        let mut out = String::new();
        let gauges = [
            (
                "zev_router_pod_held",
                "requests this replica dispatched to the pod and not yet released",
            ),
            (
                "zev_router_pod_cap",
                "admission cap for the pod on this replica",
            ),
            (
                "zev_router_pod_score",
                "spare capacity x card weight; 0 = not selectable",
            ),
            (
                "zev_router_pod_load",
                "running + waiting from the pod's own /metrics",
            ),
            (
                "zev_router_pod_stalled",
                "1 if the pod's step counter froze with work queued; skipped",
            ),
            (
                "zev_router_pod_failing",
                "1 if the pod failed a request or a scrape in the last 30 s; scored down",
            ),
        ];
        let pods = self.pods.read().unwrap();
        for (i, (name, help)) in gauges.iter().enumerate() {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge");
            for p in pods.iter() {
                let st = p.state.lock().unwrap();
                let v = match i {
                    0 => st.held as f64,
                    1 => p.cap as f64,
                    2 => p.score(&st, now, stall),
                    3 => st.sample.running + st.sample.waiting,
                    4 => Pod::stalled(&st, now, stall) as u8 as f64,
                    _ => st.failing_until.is_some_and(|t| now < t) as u8 as f64,
                };
                let _ = writeln!(
                    out,
                    "{name}{{pool=\"{}\",upstream=\"{}\",pod=\"{}\"}} {v}",
                    p.up.pool, p.up.name, p.addr
                );
            }
        }
        let _ = writeln!(out, "# HELP zev_router_pod_model 1 per model or LoRA adapter the pod lists on /v1/models (parent = base model of an adapter)\n# TYPE zev_router_pod_model gauge");
        for p in pods.iter() {
            let st = p.state.lock().unwrap();
            for m in &st.models {
                let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
                let _ = writeln!(
                    out,
                    "zev_router_pod_model{{pool=\"{}\",upstream=\"{}\",pod=\"{}\",model=\"{}\",parent=\"{}\",max_len=\"{}\"}} 1",
                    p.up.pool, p.up.name, p.addr, esc(&m.id), esc(m.parent.as_deref().unwrap_or("")),
                    m.max_len.map_or(String::new(), |l| l.to_string())
                );
            }
        }
        let _ = writeln!(out, "# HELP zev_router_pod_box_dict zstd dictionary id a rented box reports holding (0 = none)\n# TYPE zev_router_pod_box_dict gauge");
        for p in pods.iter().filter(|p| p.up.compress == Compress::Zstd) {
            let id = p.state.lock().unwrap().box_dict.unwrap_or(0);
            let _ = writeln!(
                out,
                "zev_router_pod_box_dict{{pool=\"{}\",upstream=\"{}\",pod=\"{}\"}} {id}",
                p.up.pool, p.up.name, p.addr
            );
        }
        let _ = writeln!(out, "# HELP zev_router_zstd_dict_id the dictionary id this router compresses with (0 = none)\n# TYPE zev_router_zstd_dict_id gauge");
        let _ = writeln!(
            out,
            "zev_router_zstd_dict_id {}",
            self.zdict().map_or(0, |d| d.id)
        );
        if let Some(t) = &self.train {
            t.prometheus(&mut out);
        }
        let _ = writeln!(out, "# HELP zev_router_drained 1 while /drain holds admission closed\n# TYPE zev_router_drained gauge");
        let _ = writeln!(out, "zev_router_drained {}", self.drained() as u8);
        let _ = writeln!(out, "# HELP zev_router_responses_total responses by pool and status class\n# TYPE zev_router_responses_total counter");
        for ((pool, class), n) in self.responses.lock().unwrap().iter() {
            let _ = writeln!(
                out,
                "zev_router_responses_total{{pool=\"{pool}\",status=\"{class}\"}} {n}"
            );
        }
        out
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn up(name: &str, pool: &str, mult: f64, cap: usize) -> Arc<Upstream> {
        Arc::new(Upstream {
            name: name.into(),
            pool: pool.into(),
            dns: None,
            addrs: vec![],
            port: 8000,
            gpu_multiplier: mult,
            max_inflight: cap,
            text_only: false,
            compress: Compress::None,
            http2: false,
            api_key: None,
        })
    }

    fn fleet(pods: Vec<(&str, Arc<Upstream>, f64)>) -> Fleet {
        let cfg: Config = serde_json::from_str(r#"{"listen":"0:0","upstreams":[]}"#).unwrap();
        let f = Fleet::new(cfg);
        let now = Instant::now();
        *f.pods.write().unwrap() = pods
            .into_iter()
            .map(|(a, u, running)| {
                let p = Arc::new(Pod::new(a.parse().unwrap(), u, 1));
                p.record(
                    Sample {
                        running,
                        ..Default::default()
                    },
                    now,
                );
                p
            })
            .collect();
        f
    }

    #[test]
    fn parses_zev_and_vllm() {
        let zev =
            "# HELP x\nzev_in_flight{model=\"kev-latest\"} 7\nzev_in_flight{model=\"jev\"} 2\n\
                   zev_passes_total{model=\"kev-latest\"} 100\nzev_passes_total{model=\"jev\"} 5\n\
                   zev_pass_duration_seconds_count{model=\"kev-latest\"} 100\n";
        assert_eq!(
            parse_metrics(zev),
            Sample {
                running: 9.0,
                waiting: 0.0,
                kv_cache: 0.0,
                iterations: Some(105.0)
            }
        );
        let vllm = "vllm:num_requests_running{m=\"x\"} 3\nvllm:num_requests_waiting{m=\"x\"} 2\n\
                    vllm:kv_cache_usage_perc{m=\"x\"} 0.5\nvllm:iteration_tokens_total_count{m=\"x\"} 42\n\
                    vllm:iteration_tokens_total_sum{m=\"x\"} 9999\n";
        assert_eq!(
            parse_metrics(vllm),
            Sample {
                running: 3.0,
                waiting: 2.0,
                kv_cache: 0.5,
                iterations: Some(42.0)
            }
        );
        assert_eq!(
            parse_metrics("up 1\n").iterations,
            None,
            "neither: stall detection fails open"
        );
    }

    #[test]
    fn stall_needs_frozen_counter_and_load() {
        let p = Pod::new("1.1.1.1:1".parse().unwrap(), up("a", "default", 10.0, 4), 1);
        let t0 = Instant::now();
        let busy = |i| Sample {
            running: 5.0,
            iterations: Some(i),
            ..Default::default()
        };
        p.record(busy(10.0), t0);
        p.record(busy(10.0), t0 + Duration::from_secs(10));
        p.record(busy(10.0), t0 + Duration::from_secs(45));
        let st = p.state.lock().unwrap();
        assert_eq!(
            st.stalled_since,
            Some(t0 + Duration::from_secs(10)),
            "the clock is carried, not restarted"
        );
        let now = t0 + Duration::from_secs(45);
        assert!(Pod::stalled(&st, now, Duration::from_secs(30)));
        assert_eq!(
            p.score(&st, now, Duration::from_secs(30)),
            0.0,
            "a wedged pod is skipped, not scored down"
        );
        drop(st);
        p.record(
            Sample {
                iterations: Some(10.0),
                ..Default::default()
            },
            now,
        );
        assert!(
            p.state.lock().unwrap().stalled_since.is_none(),
            "frozen but idle is not a stall"
        );
    }

    #[test]
    fn pools_caps_and_weights() {
        let f = fleet(vec![
            ("10.0.0.1:8000", up("chat", "default", 10.0, 2), 0.0),
            ("10.0.0.2:8008", up("kev", "kev", 10.0, 2), 0.0),
        ]);
        let none = HashSet::new();
        for _ in 0..2 {
            let (p, known) = f.reserve(
                &Want {
                    pool: "kev",
                    model: None,
                    needs_image: false,
                },
                &none,
            );
            assert_eq!(
                (p.unwrap().addr.port(), known),
                (8008, 1),
                "a kev request sees only kev pods"
            );
        }
        let (p, known) = f.reserve(
            &Want {
                pool: "kev",
                model: None,
                needs_image: false,
            },
            &none,
        );
        assert!(
            p.is_none() && known == 1,
            "full is not empty: wait, don't 502"
        );
        let (_, known) = f.reserve(
            &Want {
                pool: "nope",
                model: None,
                needs_image: false,
            },
            &none,
        );
        assert_eq!(known, 0, "no pod of the pool: fail fast");
        let kev = f.pods.read().unwrap()[1].clone();
        f.release(&kev);
        assert!(
            f.reserve(
                &Want {
                    pool: "kev",
                    model: None,
                    needs_image: false
                },
                &none
            )
            .0
            .is_some(),
            "a released slot is reusable"
        );
        let tried: HashSet<_> = [kev.addr].into();
        assert!(
            f.reserve(
                &Want {
                    pool: "kev",
                    model: None,
                    needs_image: false
                },
                &tried
            )
            .0
            .is_none(),
            "a retry never goes back to a tried pod"
        );

        // Weights: an idle 3090 (5) beats a two-thirds-full 4090 (10), at equal load the 4090 wins 2:1.
        let now = Instant::now();
        let s = |mult, running| {
            let p = Pod::new(
                "1.1.1.1:1".parse().unwrap(),
                up("x", "default", mult, 64),
                1,
            );
            p.record(
                Sample {
                    running,
                    ..Default::default()
                },
                now,
            );
            let st = p.state.lock().unwrap();
            p.score(&st, now, Duration::from_secs(30))
        };
        assert!(s(5.0, 0.0) > s(10.0, 21.0));
        assert!((s(10.0, 4.0) / s(5.0, 4.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn text_only_pods_never_get_images() {
        let mut text = (*up("text", "default", 10.0, 4)).clone();
        text.text_only = true;
        let f = fleet(vec![
            ("10.0.0.1:8000", up("mm", "default", 10.0, 4), 0.0),
            ("10.0.0.2:8000", Arc::new(text), 0.0),
        ]);
        assert!(
            f.has_text_only(&Want {
                pool: "default",
                model: None,
                needs_image: false
            }) && !f.has_text_only(&Want {
                pool: "kev",
                model: None,
                needs_image: false
            })
        );
        for _ in 0..4 {
            assert_eq!(
                f.reserve(
                    &Want {
                        pool: "default",
                        model: None,
                        needs_image: true
                    },
                    &HashSet::new()
                )
                .0
                .unwrap()
                .up
                .name,
                "mm"
            );
        }
        let (p, known) = f.reserve(
            &Want {
                pool: "default",
                model: None,
                needs_image: true,
            },
            &HashSet::new(),
        );
        assert!(
            p.is_none() && known == 1,
            "text-only pods are not even known to an image request"
        );
        for i in 0..20 {
            assert_eq!(
                f.home(
                    &Want {
                        pool: "default",
                        model: None,
                        needs_image: true
                    },
                    &format!("s{i}")
                )
                .unwrap()
                .up
                .name,
                "mm"
            );
        }
    }

    #[test]
    fn home_is_stable_spills_and_skips_failing() {
        let f = fleet(vec![
            ("10.0.0.1:8000", up("a", "default", 10.0, 8), 0.0),
            ("10.0.0.2:8000", up("a", "default", 10.0, 8), 0.0),
        ]);
        let h1 = f
            .home(
                &Want {
                    pool: "default",
                    model: None,
                    needs_image: false,
                },
                "sess-1",
            )
            .unwrap();
        assert_eq!(
            f.home(
                &Want {
                    pool: "default",
                    model: None,
                    needs_image: false
                },
                "sess-1"
            )
            .unwrap()
            .addr,
            h1.addr,
            "same session, same pod"
        );
        assert!(
            !f.crowded(
                &h1,
                &Want {
                    pool: "default",
                    model: None,
                    needs_image: false
                }
            ),
            "equal pods: hold the prefix"
        );
        h1.record(
            Sample {
                running: 40.0,
                ..Default::default()
            },
            Instant::now(),
        );
        assert!(
            f.crowded(
                &h1,
                &Want {
                    pool: "default",
                    model: None,
                    needs_image: false
                }
            ),
            "a jammed home spills to a clearly idler sibling"
        );
        h1.fail();
        assert_ne!(
            f.home(
                &Want {
                    pool: "default",
                    model: None,
                    needs_image: false
                },
                "sess-1"
            )
            .unwrap()
            .addr,
            h1.addr,
            "a failing home rehomes"
        );
    }

    #[test]
    fn external_registry_refuses_weightless_entries() {
        let json = r#"{"api_key":"s3cret","endpoints":[
            {"name":"vast-1","ip":"1.2.3.4","port":41234,"gpu_multiplier":12.7,"max_inflight":41,"deflate":true},
            {"name":"no-weight","ip":"5.6.7.8","port":41235},
            {"ip":"9.9.9.9","port":8000,"gpu_multiplier":10},
            {"ip":"9.9.9.8","port":8000,"gpu_multiplier":10,"deflate":true,"zstd":true}]}"#;
        let ups = external_upstreams(json, 40);
        assert_eq!(ups.len(), 3, "an entry without gpu_multiplier is refused");
        assert_eq!(
            (ups[0].gpu_multiplier, ups[0].max_inflight, ups[0].compress),
            (12.7, 41, Compress::Deflate)
        );
        assert_eq!(ups[1].compress, Compress::None, "no mark, no compression");
        assert_eq!(ups[2].compress, Compress::Zstd, "zstd wins over deflate");
        let cfg: Config = serde_json::from_str(
            r#"{"listen":"0:0","upstreams":[{"name":"lan","gpu_multiplier":1,"max_inflight":1,"deflate":true,"compress":"zstd"}]}"#,
        )
        .unwrap();
        assert_eq!(
            cfg.upstreams[0].compress,
            Compress::None,
            "an in-cluster upstream can never compress"
        );
        assert_eq!(
            ups[1].max_inflight, 40,
            "cap falls back to external_max_inflight"
        );
        assert_eq!(ups[1].api_key.as_deref(), Some("s3cret"));
        assert!(external_upstreams("{ not json", 40).is_empty());
    }

    #[test]
    fn drain_closes_and_reopens() {
        let f = fleet(vec![]);
        assert!(!f.drained());
        let mut cut = f.cut.subscribe();
        f.drain(60);
        assert!(
            f.drained() && cut.has_changed().unwrap(),
            "drain closes admission and signals in-flight requests"
        );
        cut.mark_unchanged();
        f.drain(0);
        assert!(
            !f.drained() && !cut.has_changed().unwrap(),
            "resume reopens without cutting anything"
        );
    }

    #[test]
    fn parses_both_models_dialects() {
        let vllm = r#"{"object":"list","data":[
            {"id":"holo","max_model_len":32768,"parent":null},
            {"id":"lora-a","max_model_len":32768,"parent":"holo"}]}"#;
        let m = parse_models(vllm).unwrap();
        assert_eq!(
            m[1],
            ModelInfo {
                id: "lora-a".into(),
                parent: Some("holo".into()),
                max_len: Some(32768)
            }
        );
        let zev = r#"{"models":[{"name":"kev-latest","kind":"kev"},{"name":"jev-latest"}]}"#;
        let ids: Vec<_> = parse_models(zev)
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(ids, ["kev-latest", "jev-latest"]);
        assert!(
            parse_models("zev_in_flight 3").is_none(),
            "not /v1/models: keep the last list"
        );
    }

    #[test]
    fn a_listed_model_pins_the_request_to_its_pods() {
        let f = fleet(vec![
            ("10.0.0.1:8000", up("base", "default", 10.0, 8), 0.0),
            ("10.0.0.2:8000", up("lora", "default", 10.0, 8), 0.0),
            ("10.0.0.3:8008", up("combo", "kev-combo", 10.0, 8), 0.0),
        ]);
        let set = |i: usize, ids: &[&str]| {
            let p = f.pods.read().unwrap()[i].clone();
            p.state.lock().unwrap().models = ids
                .iter()
                .map(|id| ModelInfo {
                    id: id.to_string(),
                    parent: None,
                    max_len: None,
                })
                .collect();
            let now = Instant::now();
            for id in ids {
                f.seen_models.lock().unwrap().insert(id.to_string(), now);
            }
        };
        set(0, &["holo"]);
        set(1, &["holo", "lora-a"]);
        set(2, &["combo-r3"]);
        let none = HashSet::new();
        for _ in 0..8 {
            let w = Want {
                pool: "default",
                model: Some("lora-a"),
                needs_image: false,
            };
            assert_eq!(
                f.reserve(&w, &none).0.unwrap().up.name,
                "lora",
                "an adapter goes only where it is loaded"
            );
        }
        // Across pools: kev-combo's model found from /v1/systemone's pool, no pool_models needed.
        let w = Want {
            pool: "kev",
            model: Some("combo-r3"),
            needs_image: false,
        };
        assert_eq!(f.reserve(&w, &none).0.unwrap().up.name, "combo");
        assert!(
            f.model_known("combo-r3") && !f.model_known("kev-4b"),
            "a name never listed keeps the path's pool"
        );
        // Its pod goes away: the name is still known for an hour, so the request finds no pod (fail fast)
        // instead of falling back to one that would answer with another adapter.
        f.pods.write().unwrap().pop();
        let (p, known) = f.reserve(&w, &none);
        assert!(p.is_none() && known == 0 && f.model_known("combo-r3"));
        f.seen_models
            .lock()
            .unwrap()
            .insert("combo-r3".into(), Instant::now() - MODEL_MEMORY);
        assert!(!f.model_known("combo-r3"), "forgotten after an hour");
    }

    #[test]
    fn zstd_waits_for_the_box_to_hold_our_dictionary() {
        let dir = std::env::temp_dir().join(format!("zev-router-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("d.zdict");
        let train = |w: &str| {
            let samples: Vec<String> = (0..300)
                .map(|i| format!("{{\"page\":\"{w} {i}\",\"tools\":[\"click\",\"type\"]}}"))
                .collect();
            zstd::dict::from_samples(&samples, 4096).unwrap()
        };
        let mut cfg: Config = serde_json::from_str(r#"{"listen":"0:0","upstreams":[]}"#).unwrap();
        let p = Pod::new(
            "1.1.1.1:1".parse().unwrap(),
            up("box", "default", 1.0, 4),
            1,
        );
        let f = Fleet::new(cfg.clone());
        assert!(
            matches!(f.box_ready(&p), Ok(None)),
            "no dictionary configured: plain zstd"
        );

        std::fs::write(&path, train("a")).unwrap();
        cfg.zstd_dict = Some(path.to_string_lossy().into());
        let f = Fleet::new(cfg);
        let a = f.zdict().unwrap().id;
        assert!(
            f.box_ready(&p).is_err(),
            "the box has not confirmed: send uncompressed"
        );
        p.state.lock().unwrap().box_dict = Some(a);
        assert_eq!(f.box_ready(&p).unwrap().unwrap().id, a);

        // Retrained: the file changes, the router reloads it, and the box (still on a) goes back to uncompressed.
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&path, train("b")).unwrap();
        f.reload_zdict();
        let b = f.zdict().unwrap().id;
        assert_ne!(a, b);
        assert!(
            f.box_ready(&p).is_err(),
            "a box on the old dictionary is sent nothing compressed with the new"
        );
        // Not a dictionary: refused, the one in use stays.
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&path, b"just some bytes, no magic").unwrap();
        f.reload_zdict();
        assert_eq!(f.zdict().unwrap().id, b);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
