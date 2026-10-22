//! The router retrains its zstd dictionary itself. Prompts drift (a dictionary from one day gets 16x on that day and
//! ~6.8x a week later), so it keeps the most recent bodies it compresses for rented boxes, and on its schedule (every
//! `every_s`, or sooner when the live ratio falls below `drift` x the ratio the dictionary was accepted at) trains a
//! candidate on the older three quarters and measures it against the dictionary in use on the newest quarter. Only a
//! candidate `margin` better is published.
//!
//! With replicas, one trains: a Kubernetes Lease elects it, and it publishes to a ConfigMap every replica mounts as
//! `zstd_dict`, so they all pick it up with the existing reload, and the existing sync pushes it to the boxes. If
//! the API is unreachable no one is leader and the dictionary in use stays. Outside Kubernetes the router is its own
//! leader and writes the `zstd_dict` file.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use bytes::Bytes;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::kube::Kube;

#[derive(Deserialize, Clone, Debug)]
pub struct AutoTrainCfg {
    #[serde(default = "day")]
    pub every_s: u64,
    /// Never two trainings closer than this, whatever triggers them.
    #[serde(default = "hour")]
    pub min_interval_s: u64,
    /// Retrain early once the live ratio falls below this fraction of the ratio the dictionary was accepted at.
    #[serde(default = "drift")]
    pub drift: f64,
    /// A candidate replaces the dictionary in use only if it compresses the held-out bodies this much better.
    #[serde(default = "margin")]
    pub margin: f64,
    #[serde(default = "n500")]
    pub min_samples: usize,
    #[serde(default = "n2000")]
    pub max_samples: usize,
    #[serde(default = "mib130")]
    pub max_bytes: usize,
    /// The size vast/train_zstd_dict.py trains: past ~110 KB the ratio stops improving on these prompts.
    #[serde(default = "dict_size")]
    pub dict_bytes: usize,
    #[serde(default = "lease_name")]
    pub lease: String,
    #[serde(default = "lease_name")]
    pub configmap: String,
    /// The ConfigMap key, i.e. the file name `zstd_dict` points at in the mounted directory.
    #[serde(default = "dict_key")]
    pub key: String,
    #[serde(default = "thirty")]
    pub lease_s: u64,
}

fn day() -> u64 {
    86_400
}
fn hour() -> u64 {
    3_600
}
fn drift() -> f64 {
    0.8
}
fn margin() -> f64 {
    0.03
}
fn n500() -> usize {
    500
}
fn n2000() -> usize {
    2_000
}
fn mib130() -> usize {
    130 << 20
}
fn dict_size() -> usize {
    110_000
}
fn lease_name() -> String {
    "inference-router-rs-zdict".into()
}
fn dict_key() -> String {
    "prompts.zdict".into()
}
fn thirty() -> u64 {
    30
}

/// A body past this is a screenshot prompt: base64 image bytes teach a dictionary nothing.
const SAMPLE_MAX: usize = 1 << 20;
/// The live ratio is a byte-weighted average over roughly the last this-many bodies.
const LIVE_SPAN: f64 = 500.0;
/// ...and is not reported before this many.
const LIVE_MIN: u64 = 50;

/// Why a training starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Why {
    /// No measured ratio for the dictionary in use: just started, or just became leader.
    NoBaseline,
    Age,
    Drift,
}

/// Is a training due? Pure, for the tests.
pub fn due(
    cfg: &AutoTrainCfg,
    samples: usize,
    since_last: Option<Duration>,
    baseline: Option<f64>,
    live: Option<f64>,
) -> Option<Why> {
    if samples < cfg.min_samples || since_last.is_some_and(|d| d.as_secs() < cfg.min_interval_s) {
        return None;
    }
    let Some(base) = baseline else {
        return Some(Why::NoBaseline);
    };
    if since_last.is_none_or(|d| d.as_secs() >= cfg.every_s) {
        return Some(Why::Age);
    }
    live.filter(|l| *l < cfg.drift * base).map(|_| Why::Drift)
}

/// Publish the candidate? Only if it beats the dictionary in use (plain zstd when there is none) by the margin.
pub fn better(candidate: f64, current: f64, margin: f64) -> bool {
    candidate >= current * (1.0 + margin)
}

/// Original bytes over compressed bytes for these bodies.
pub fn ratio(bodies: &[Bytes], level: i32, dict: Option<&[u8]>) -> f64 {
    let mut c = match dict {
        Some(d) => zstd::bulk::Compressor::with_dictionary(level, d),
        None => zstd::bulk::Compressor::new(level),
    }
    .expect("zstd compressor");
    let (mut raw, mut z) = (0usize, 0usize);
    for b in bodies {
        raw += b.len();
        z += c.compress(b).map_or(b.len(), |v| v.len());
    }
    raw as f64 / z.max(1) as f64
}

pub struct Outcome {
    pub dict: Vec<u8>,
    pub id: u32,
    pub candidate: f64,
    pub current: f64,
}

/// Train on the older three quarters (oldest first in `bodies`), measure the candidate and the dictionary in use on
/// the newest quarter: what the next hours of traffic look like most.
pub fn train(
    bodies: &[Bytes],
    current: Option<&[u8]>,
    level: i32,
    dict_bytes: usize,
) -> Result<Outcome, String> {
    let cut = bodies.len() - bodies.len() / 4;
    let (fit, held) = bodies.split_at(cut);
    let dict =
        zstd::dict::from_samples(fit, dict_bytes).map_err(|e| format!("training failed: {e}"))?;
    let id = zstd::zstd_safe::get_dict_id_from_dict(&dict)
        .ok_or("the trainer returned no dictionary id")?
        .get();
    Ok(Outcome {
        candidate: ratio(held, level, Some(&dict)),
        current: ratio(held, level, current),
        dict,
        id,
    })
}

#[derive(Default)]
struct Ring {
    q: VecDeque<Bytes>,
    bytes: usize,
}

/// Byte-weighted moving ratio of what was actually sent, under one dictionary id.
#[derive(Default)]
struct Live {
    dict: u32,
    raw: f64,
    z: f64,
    n: u64,
}

#[derive(Default)]
struct Status {
    baseline: Option<f64>,
    last_run: Option<Instant>,
    last_run_unix: Option<u64>,
    last_reason: Option<Why>,
    last_result: String,
    last_candidate: Option<f64>,
    last_current: Option<f64>,
    trainings: u64,
}

pub struct Trainer {
    cfg: AutoTrainCfg,
    level: i32,
    /// zstd_dict, written directly when there is no Kubernetes API.
    dict_path: String,
    ring: Mutex<Ring>,
    live: Mutex<Live>,
    st: Mutex<Status>,
    running: AtomicBool,
    leader: AtomicBool,
    kube: Option<Arc<Kube>>,
}

impl Trainer {
    pub fn new(cfg: AutoTrainCfg, level: i32, dict_path: String, kube: Option<Arc<Kube>>) -> Self {
        log::warn!(
            "zstd dictionary auto-training on: {}",
            match &kube {
                Some(k) => format!(
                    "Lease {}/{} as {}, publishing to ConfigMap {}",
                    k.ns, cfg.lease, k.id, cfg.configmap
                ),
                None => format!("no Kubernetes API, leader by default, writing {dict_path}"),
            }
        );
        Trainer {
            cfg,
            level,
            dict_path,
            ring: Mutex::default(),
            live: Mutex::default(),
            st: Mutex::default(),
            running: AtomicBool::new(false),
            leader: AtomicBool::new(false),
            kube,
        }
    }

    /// Keep a body that went (or would have gone) to a zstd box. A refcount, not a copy; the oldest go first.
    /// ponytail: a ring, not a reservoir: the point is the newest traffic, and the held-out quarter is the newest.
    pub fn sample(&self, body: &Bytes) {
        if body.len() > SAMPLE_MAX {
            return;
        }
        let mut r = self.ring.lock().unwrap();
        r.bytes += body.len();
        r.q.push_back(body.clone());
        while r.q.len() > self.cfg.max_samples || r.bytes > self.cfg.max_bytes {
            let old = r.q.pop_front().unwrap();
            r.bytes -= old.len();
        }
    }

    /// A body sent with zstd under dictionary `dict` (0 = none): `raw` bytes became `z`.
    pub fn sent(&self, dict: u32, raw: usize, z: usize) {
        let mut l = self.live.lock().unwrap();
        if l.dict != dict {
            *l = Live {
                dict,
                ..Live::default()
            };
        }
        let keep = 1.0 - 1.0 / LIVE_SPAN;
        l.raw = l.raw * keep + raw as f64;
        l.z = l.z * keep + z as f64;
        l.n += 1;
    }

    fn live(&self) -> Option<f64> {
        let l = self.live.lock().unwrap();
        (l.n >= LIVE_MIN).then(|| l.raw / l.z.max(1.0))
    }

    /// Every background tick: renew the Lease, and if leader and a training is due, start one on its own thread.
    pub async fn tick(self: &Arc<Self>, current: Option<Bytes>) {
        let leader = match &self.kube {
            Some(k) => k.lead(&self.cfg.lease, self.cfg.lease_s).await,
            None => true,
        };
        if leader != self.leader.swap(leader, Ordering::Relaxed) {
            log::warn!(
                "zstd dictionary training: {}",
                if leader {
                    "now the leader"
                } else {
                    "no longer the leader"
                }
            );
            // A new leader measures the dictionary in use before trusting a ratio it did not see.
            self.st.lock().unwrap().baseline = None;
        }
        if !leader || self.running.load(Ordering::Relaxed) {
            return;
        }
        let why = {
            let st = self.st.lock().unwrap();
            let n = self.ring.lock().unwrap().q.len();
            due(
                &self.cfg,
                n,
                st.last_run.map(|t| t.elapsed()),
                st.baseline,
                self.live(),
            )
        };
        let Some(why) = why else { return };
        self.running.store(true, Ordering::Relaxed);
        let bodies: Vec<Bytes> = self.ring.lock().unwrap().q.iter().cloned().collect();
        log::warn!(
            "zstd dictionary training ({why:?}) on {} bodies",
            bodies.len()
        );
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (level, size) = (self.level, self.cfg.dict_bytes);
        std::thread::spawn(move || {
            // Lowest priority: on Linux nice() applies to the calling thread, so the proxy threads keep the CPU.
            unsafe { libc::nice(19) };
            let _ = tx.send(train(&bodies, current.as_deref(), level, size));
        });
        let me = self.clone();
        tokio::spawn(async move {
            let res = rx
                .await
                .unwrap_or_else(|_| Err("training thread died".into()));
            me.finish(why, res).await;
            me.running.store(false, Ordering::Relaxed);
        });
    }

    async fn finish(&self, why: Why, res: Result<Outcome, String>) {
        let mut baseline = None;
        let (result, cand, cur) = match res {
            Err(e) => (e, None, None),
            Ok(o) if !better(o.candidate, o.current, self.cfg.margin) => {
                baseline = Some(o.current);
                (
                    format!(
                        "rejected: {:.2}x is not {:.0}% over {:.2}x",
                        o.candidate,
                        self.cfg.margin * 100.0,
                        o.current
                    ),
                    Some(o.candidate),
                    Some(o.current),
                )
            }
            Ok(o) => match self.publish(&o.dict).await {
                Ok(()) => {
                    baseline = Some(o.candidate);
                    (
                        format!("published {}", o.id),
                        Some(o.candidate),
                        Some(o.current),
                    )
                }
                Err(e) => (
                    format!("not published: {e}"),
                    Some(o.candidate),
                    Some(o.current),
                ),
            },
        };
        log::warn!(
            "zstd dictionary training ({why:?}): candidate {} vs in use {} on the newest quarter: {result}",
            cand.map_or("-".into(), |r| format!("{r:.2}x")),
            cur.map_or("-".into(), |r| format!("{r:.2}x"))
        );
        let mut st = self.st.lock().unwrap();
        if baseline.is_some() {
            st.baseline = baseline;
        }
        st.last_run = Some(Instant::now());
        st.last_run_unix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());
        st.last_reason = Some(why);
        st.last_result = result;
        st.last_candidate = cand;
        st.last_current = cur;
        st.trainings += 1;
    }

    async fn publish(&self, dict: &[u8]) -> Result<(), String> {
        match &self.kube {
            Some(k) => {
                k.put_configmap(&self.cfg.configmap, &self.cfg.key, dict)
                    .await
            }
            None => {
                let tmp = format!("{}.tmp", self.dict_path);
                std::fs::write(&tmp, dict)
                    .and_then(|_| std::fs::rename(&tmp, &self.dict_path))
                    .map_err(|e| format!("{}: {e}", self.dict_path))
            }
        }
    }

    pub fn status(&self) -> Value {
        let st = self.st.lock().unwrap();
        let r = self.ring.lock().unwrap();
        json!({
            "leader": self.leader.load(Ordering::Relaxed), "training": self.running.load(Ordering::Relaxed),
            "samples": r.q.len(), "sample_bytes": r.bytes, "live_ratio": self.live(), "baseline_ratio": st.baseline,
            "trainings": st.trainings, "last_run_unix": st.last_run_unix,
            "last_reason": st.last_reason.map(|w| format!("{w:?}")), "last_result": st.last_result,
            "last_candidate_ratio": st.last_candidate, "last_current_ratio": st.last_current,
        })
    }

    pub fn prometheus(&self, out: &mut String) {
        use std::fmt::Write;
        let st = self.st.lock().unwrap();
        let n = self.ring.lock().unwrap().q.len();
        let g = |out: &mut String, name: &str, help: &str, v: f64| {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
        };
        g(
            out,
            "inference_router_zdict_leader",
            "1 if this replica trains the zstd dictionary",
            self.leader.load(Ordering::Relaxed) as u8 as f64,
        );
        g(
            out,
            "inference_router_zdict_samples",
            "bodies kept for the next training",
            n as f64,
        );
        g(
            out,
            "inference_router_zdict_live_ratio",
            "raw/compressed bytes sent to zstd boxes, last ~500 bodies (0 = too few)",
            self.live().unwrap_or(0.0),
        );
        g(
            out,
            "inference_router_zdict_baseline_ratio",
            "held-out ratio of the dictionary in use when last measured (0 = not yet)",
            st.baseline.unwrap_or(0.0),
        );
        g(
            out,
            "inference_router_zdict_last_training_timestamp_seconds",
            "end of the last training (0 = none)",
            st.last_run_unix.unwrap_or(0) as f64,
        );
        g(
            out,
            "inference_router_zdict_trainings",
            "trainings run by this replica",
            st.trainings as f64,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AutoTrainCfg {
        serde_json::from_str(r#"{"min_samples": 10}"#).unwrap()
    }

    #[test]
    fn trigger() {
        let c = cfg();
        let h = |n: u64| Some(Duration::from_secs(n * 3600));
        assert_eq!(due(&c, 9, None, None, None), None, "too few samples");
        assert_eq!(
            due(&c, 10, None, None, None),
            Some(Why::NoBaseline),
            "first run measures the dictionary in use"
        );
        assert_eq!(
            due(&c, 10, Some(Duration::from_secs(600)), None, Some(1.0)),
            None,
            "min interval holds even for drift"
        );
        assert_eq!(
            due(&c, 10, h(2), Some(6.8), Some(6.0)),
            None,
            "6.0 is above 0.8 x 6.8"
        );
        assert_eq!(due(&c, 10, h(2), Some(6.8), Some(5.0)), Some(Why::Drift));
        assert_eq!(
            due(&c, 10, h(2), Some(6.8), None),
            None,
            "no live ratio yet: wait"
        );
        assert_eq!(due(&c, 10, h(24), Some(6.8), Some(6.8)), Some(Why::Age));
        assert_eq!(
            due(&c, 10, h(2), None, Some(6.8)),
            Some(Why::NoBaseline),
            "a new leader measures first"
        );
    }

    #[test]
    fn gate() {
        assert!(better(7.0, 6.8, 0.0));
        assert!(!better(6.9, 6.8, 0.03), "1.5% better is noise");
        assert!(better(7.01, 6.8, 0.03));
        assert!(better(6.0, 3.3, 0.03), "anything trained beats plain zstd");
    }

    #[test]
    fn ring_and_live() {
        let mut c = cfg();
        c.max_samples = 3;
        c.max_bytes = 25;
        let t = Trainer {
            kube: None,
            ..Trainer::new(c, 3, "/nonexistent".into(), None)
        };
        for i in 0..5u8 {
            t.sample(&Bytes::from(vec![i; 10]));
        }
        let r = t.ring.lock().unwrap();
        assert_eq!(r.q.len(), 2, "25 bytes hold two 10-byte bodies");
        assert_eq!(
            (r.q[0][0], r.q[1][0], r.bytes),
            (3, 4, 20),
            "oldest out first"
        );
        drop(r);
        for _ in 0..LIVE_MIN {
            t.sent(7, 1000, 100);
        }
        assert!((t.live().unwrap() - 10.0).abs() < 1e-9);
        t.sent(8, 1000, 500);
        assert_eq!(t.live(), None, "a new dictionary starts its own average");
    }

    #[test]
    fn trains_and_measures_on_the_newest_quarter() {
        let bodies: Vec<Bytes> = (0..400)
            .map(|i| Bytes::from(format!("{{\"system\":\"You are a product classifier. Answer with the node id.\",\"item\":\"widget {i} model {}\"}}", i * 13)))
            .collect();
        let o = train(&bodies, None, 3, 4096).unwrap();
        assert!(o.id != 0);
        assert!(
            better(o.candidate, o.current, 0.03),
            "{} vs plain {}",
            o.candidate,
            o.current
        );
        let again = train(&bodies, Some(&o.dict), 3, 4096).unwrap();
        assert!(
            !better(again.candidate, again.current, 0.03),
            "the same traffic again is not worth a swap"
        );
    }
}
