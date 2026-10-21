//! The fleet: pods discovered per upstream, their latest /metrics sample, the slots this router replica holds on
//! them, and the pick. A port of learning-helm's florian-nginx (pick.lua, init_worker.lua) with one process-wide state
//! instead of nginx shared dicts, and waiting on a Notify instead of a 50 ms poll.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::sync::Notify;

#[derive(Deserialize, Clone, Debug)]
pub struct Upstream {
    pub name: String,
    #[serde(default = "default_pool")]
    pub pool: String,
    /// Headless Service name: DNS gives one A record per ready pod.
    #[serde(default)]
    pub dns: Option<String>,
    /// Fixed "ip:port" backends (rented boxes, NodePorts for a smoke test).
    #[serde(default)]
    pub addrs: Vec<String>,
    pub port: u16,
    /// Routing weight: a 4090 is 10. No default, as in florian-nginx: a guessed weight wrecks the split.
    pub gpu_multiplier: f64,
    /// Admission cap per card for the whole fleet of routers; each replica takes its share.
    pub max_inflight: usize,
    #[serde(default)]
    pub text_only: bool,
}

fn default_pool() -> String {
    "default".into()
}

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub listen: String,
    /// Router replicas behind one Service. Each keeps its own slot counts, so caps are split between them;
    /// losing a replica under-admits, the safe direction. ponytail: no shared state; Redis if this passes ~3.
    #[serde(default = "one")]
    pub replicas: usize,
    #[serde(default)]
    pub pool_paths: HashMap<String, String>,
    pub upstreams: Vec<Upstream>,
    #[serde(default = "five")]
    pub metrics_interval_s: u64,
    #[serde(default = "thirty")]
    pub stall_s: u64,
    #[serde(default = "one_eighty")]
    pub queue_timeout_s: u64,
    #[serde(default = "session_header")]
    pub affinity_header: String,
    /// Proxy worker threads. One process, so one shared state across all of them.
    #[serde(default = "four")]
    pub threads: usize,
}

fn four() -> usize {
    4
}

fn one() -> usize {
    1
}
fn five() -> u64 {
    5
}
fn thirty() -> u64 {
    30
}
fn one_eighty() -> u64 {
    180
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

#[derive(Debug)]
pub struct State {
    pub sample: Sample,
    pub last_seen: Option<Instant>,
    /// When the step counter froze with work queued; a pod frozen past stall_s is wedged and skipped.
    pub stalled_since: Option<Instant>,
    pub failing_until: Option<Instant>,
    /// Requests this replica has dispatched to the pod and not yet released.
    pub held: usize,
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

impl Pod {
    pub fn new(addr: SocketAddr, up: Arc<Upstream>, replicas: usize) -> Self {
        let cap = up.max_inflight.div_ceil(replicas.max(1));
        let state = State {
            sample: Sample::default(),
            last_seen: None,
            stalled_since: None,
            failing_until: None,
            held: 0,
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

pub struct Fleet {
    pub cfg: Config,
    pub pods: RwLock<Vec<Arc<Pod>>>,
    /// Woken when a slot is released or a scrape lands, so queued requests re-pick at once instead of polling.
    pub freed: Notify,
    /// Responses by (pool, status class), for /metrics.
    responses: Mutex<HashMap<(String, &'static str), u64>>,
}

impl Fleet {
    pub fn new(cfg: Config) -> Self {
        Fleet {
            cfg,
            pods: RwLock::new(Vec::new()),
            freed: Notify::new(),
            responses: Mutex::new(HashMap::new()),
        }
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
                    "seconds_since_scrape": st.last_seen.map(|t| now.duration_since(t).as_secs_f64()),
                })
            })
            .collect();
        serde_json::json!({ "pods": pods }).to_string()
    }

    /// Per-pod gauges labelled by pool and upstream, plus responses by pool and status class.
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
        let _ = writeln!(out, "# HELP zev_router_responses_total responses by pool and status class\n# TYPE zev_router_responses_total counter");
        for ((pool, class), n) in self.responses.lock().unwrap().iter() {
            let _ = writeln!(
                out,
                "zev_router_responses_total{{pool=\"{pool}\",status=\"{class}\"}} {n}"
            );
        }
        out
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

    /// Weighted-random pick among the pool's pods with a free slot, taking the slot. `known` counts the pool's pods
    /// at all, so the caller can tell "all full" (wait) from "none discovered" (fail now).
    pub fn reserve(
        &self,
        pool: &str,
        tried: &HashSet<SocketAddr>,
        needs_image: bool,
    ) -> (Option<Arc<Pod>>, usize) {
        let now = Instant::now();
        let stall = self.stall();
        let pods = self.pods.read().unwrap();
        let mut known = 0;
        let mut cands: Vec<(f64, &Arc<Pod>)> = Vec::new();
        for p in pods
            .iter()
            .filter(|p| p.up.pool == pool && !(needs_image && p.up.text_only))
        {
            known += 1;
            if tried.contains(&p.addr) {
                continue;
            }
            let st = p.state.lock().unwrap();
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
            let mut st = p.state.lock().unwrap();
            if st.held < p.cap {
                st.held += 1;
                return (Some(p.clone()), known);
            }
        }
        (None, known)
    }

    /// Weighted rendezvous hash: every replica agrees on a session's home with no shared state, and losing a pod
    /// rehomes only its own sessions. Failing or stalled pods are skipped.
    pub fn home(&self, pool: &str, session: &str, needs_image: bool) -> Option<Arc<Pod>> {
        let now = Instant::now();
        let stall = self.stall();
        let pods = self.pods.read().unwrap();
        pods.iter()
            .filter(|p| p.up.pool == pool && !(needs_image && p.up.text_only))
            .filter(|p| {
                let st = p.state.lock().unwrap();
                st.last_seen.is_some()
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
        let mut found: Vec<(SocketAddr, Arc<Upstream>)> = Vec::new();
        for up in &self.cfg.upstreams {
            let up = Arc::new(up.clone());
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
                Some(p) if p.up.name == up.name => p.clone(),
                _ => Arc::new(Pod::new(a, up, self.cfg.replicas)),
            })
            .collect();
    }

    /// One scrape of every pod's /metrics. A failed scrape marks the pod failing and keeps its last sample: zeros
    /// would score a hung pod as idle, the most attractive backend there is.
    pub async fn scrape(&self, http: &reqwest::Client) {
        let pods: Vec<Arc<Pod>> = self.pods.read().unwrap().clone();
        let stall = self.stall();
        let mut set = tokio::task::JoinSet::new();
        for p in pods {
            let http = http.clone();
            set.spawn(async move {
                let url = format!("http://{}/metrics", p.addr);
                let body = match http.get(&url).send().await {
                    Ok(r) if r.status().is_success() => r.text().await.ok(),
                    _ => None,
                };
                match body {
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
                    }
                }
            });
        }
        while set.join_next().await.is_some() {}
        self.freed.notify_waiters();
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
            let (p, known) = f.reserve("kev", &none, false);
            assert_eq!(
                (p.unwrap().addr.port(), known),
                (8008, 1),
                "a kev request sees only kev pods"
            );
        }
        let (p, known) = f.reserve("kev", &none, false);
        assert!(
            p.is_none() && known == 1,
            "full is not empty: wait, don't 502"
        );
        let (_, known) = f.reserve("nope", &none, false);
        assert_eq!(known, 0, "no pod of the pool: fail fast");
        let kev = f.pods.read().unwrap()[1].clone();
        f.release(&kev);
        assert!(
            f.reserve("kev", &none, false).0.is_some(),
            "a released slot is reusable"
        );
        let tried: HashSet<_> = [kev.addr].into();
        assert!(
            f.reserve("kev", &tried, false).0.is_none(),
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
    fn home_is_stable_and_skips_failing() {
        let f = fleet(vec![
            ("10.0.0.1:8000", up("a", "default", 10.0, 8), 0.0),
            ("10.0.0.2:8000", up("a", "default", 10.0, 8), 0.0),
        ]);
        let h1 = f.home("default", "sess-1", false).unwrap().addr;
        assert_eq!(
            f.home("default", "sess-1", false).unwrap().addr,
            h1,
            "same session, same pod"
        );
        f.pods
            .read()
            .unwrap()
            .iter()
            .find(|p| p.addr == h1)
            .unwrap()
            .fail();
        assert_ne!(
            f.home("default", "sess-1", false).unwrap().addr,
            h1,
            "a failing home rehomes"
        );
    }
}
