//! /v1/systemone on a Candle backbone. Request handlers encode and queue; each model worker (one per `--devices`
//! entry, all taking from one queue) runs continuous batching: it takes the queued requests that fit one pass's token
//! budget, runs one packed pass over their new states and all their rows (a row continues its state within the same
//! pass), replies, and takes the next pass's requests. States are kept in an LRU prefix cache per worker.

use super::entrant::Entrant;
use super::kernels::{Pack, StateCache};
use super::model::Model;
use super::readout::Readout;
use super::RowSpec;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Result;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

const MAX_QUEUE_DEPTH: usize = 1024;

/// Per row, per question group of the row: the readout's values (probabilities or label logits); the pass time in ms.
pub type Outputs = Vec<Vec<Vec<f64>>>;
type JobReply = tokio::sync::oneshot::Sender<std::result::Result<(Outputs, f64), String>>;

pub struct Job {
    state: Vec<u32>,
    rows: Vec<RowSpec>,
    reply: JobReply,
}

impl Job {
    fn tokens(&self) -> usize {
        self.state.len() + self.rows.iter().map(|r| r.ids.len()).sum::<usize>()
    }
}

/// LRU of state prefixes keyed by their token ids (kev.serve.PrefixCache).
struct Prefixes {
    size: usize,
    order: VecDeque<Vec<u32>>,
    map: HashMap<Vec<u32>, Arc<StateCache>>,
    hits: usize,
    misses: usize,
}

impl Prefixes {
    fn new(size: usize) -> Self {
        Self {
            size,
            order: VecDeque::new(),
            map: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }
    fn get(&mut self, k: &[u32]) -> Option<Arc<StateCache>> {
        let v = self.map.get(k).cloned()?;
        if self.order.back().is_none_or(|last| last.as_slice() != k) {
            if let Some(pos) = self.order.iter().position(|x| x.as_slice() == k) {
                if let Some(item) = self.order.remove(pos) {
                    self.order.push_back(item);
                }
            }
        }
        Some(v)
    }
    fn put(&mut self, k: Vec<u32>, v: Arc<StateCache>) {
        if self.size == 0 {
            return;
        }
        if let Some(pos) = self.order.iter().position(|x| *x == k) {
            self.order.remove(pos);
        }
        self.order.push_back(k.clone());
        self.map.insert(k, v);
        while self.order.len() > self.size {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

/// Serving options.
#[derive(Clone, Debug)]
pub struct Opts {
    pub prefix_cache: usize,
    pub pass_tokens: usize, // token budget of one pass (a larger single request still runs, alone)
}

/// What /health reads. A worker counts as alive from the end of its warm-up until its thread exits (a drop guard, so a
/// panic counts too), and each finished pass stamps `last_pass_ms`. Requests in flight with no pass for STALL_SECS is
/// a wedged GPU: a card that falls off the bus leaves its thread blocked inside a CUDA call, with the process up and a
/// plain "ok" route still answering, so nothing restarts it.
#[derive(Default)]
pub struct Health {
    alive: AtomicUsize,
    in_flight: AtomicUsize,
    last_pass_ms: AtomicU64,
}

/// A 32k-token pass takes seconds; two minutes without one while requests wait is not slow, it is stuck.
const STALL_SECS: u64 = 120;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

struct Counted<'a>(&'a AtomicUsize);

impl Drop for Counted<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Worker {
    m: Model,
    readout: Readout,
    cache: Prefixes,
    health: Arc<Health>,
}

impl Worker {
    /// One packed pass for `jobs`: their new states (each distinct state once) and all their rows.
    fn pass(&mut self, jobs: &[Job]) -> Result<Vec<Outputs>> {
        let dev = self.m.dev.clone();
        let mut seqs = Vec::new();
        let mut ids: Vec<u32> = Vec::new();
        let mut made: Vec<(Vec<u32>, Arc<StateCache>)> = Vec::new();
        let mut states: Vec<Option<Arc<StateCache>>> = Vec::with_capacity(jobs.len());
        let mut hits = Vec::with_capacity(jobs.len());
        for j in jobs {
            if j.state.is_empty() {
                states.push(None);
                hits.push(false);
                continue;
            }
            let c = if let Some(c) = self.cache.get(&j.state) {
                hits.push(true);
                c
            } else {
                hits.push(false);
                match made.iter().find(|(k, _)| *k == j.state) {
                    Some((_, c)) => c.clone(),
                    None => {
                        let c = Arc::new(StateCache::alloc(
                            j.state.len(),
                            &self.m.cache,
                            self.m.dt,
                            &dev,
                        )?);
                        seqs.push((j.state.len(), None, Some(c.clone())));
                        ids.extend_from_slice(&j.state);
                        made.push((j.state.clone(), c.clone()));
                        c
                    }
                }
            };
            states.push(Some(c));
        }
        let mut picks: Vec<u32> = Vec::new();
        let mut rows: Vec<&RowSpec> = Vec::new();
        for (j, st) in jobs.iter().zip(&states) {
            for r in &j.rows {
                let start = ids.len();
                picks.extend(r.picks.iter().map(|&o| (start + o) as u32));
                seqs.push((r.ids.len(), st.clone(), None));
                ids.extend_from_slice(&r.ids);
                rows.push(r);
            }
        }
        let pack = Pack::new(seqs, &dev)?;
        let h = self.m.forward(&pack, &ids, &picks)?;
        let mut per_row = self.readout.run(&self.m, &h, &rows)?.into_iter();
        for (k, c) in made {
            self.cache.put(k, c);
        }
        self.cache.hits += hits.iter().filter(|h| **h).count();
        self.cache.misses += hits.iter().filter(|h| !**h).count();
        let mut out = Vec::with_capacity(jobs.len());
        for j in jobs {
            let o: Outputs = per_row.by_ref().take(j.rows.len()).collect();
            if o.len() != j.rows.len() {
                candle_core::bail!("mismatched row count from the readout");
            }
            out.push(o);
        }
        Ok(out)
    }

    /// Continuous batching: take the queued requests that fit one pass, run it, reply, repeat.
    fn serve(mut self, rx: Arc<Mutex<mpsc::Receiver<Job>>>, opts: Opts) {
        // warm-up: NVRTC compiles the kernels and cuBLAS picks its algorithms on the first pass (seconds), before any
        // request can wait on it
        let warm = Pack::new(vec![(4, None, None)], &self.m.dev)
            .and_then(|p| self.m.forward(&p, &[1, 2, 3, 4], &[3]));
        if let Err(e) = warm.and_then(|_| self.m.dev.synchronize()) {
            eprintln!("kev: warm-up pass failed: {e}");
        }
        let health = self.health.clone();
        health.alive.fetch_add(1, Ordering::SeqCst);
        let _alive = Counted(&health.alive);
        health.last_pass_ms.store(now_ms(), Ordering::SeqCst);
        let mut carry: Option<Job> = None;
        loop {
            let mut batch = Vec::new();
            let mut tokens = 0;
            {
                let rx = rx.lock().unwrap();
                let first = match carry.take() {
                    Some(j) => j,
                    None => match rx.recv() {
                        Ok(j) => j,
                        Err(_) => return,
                    },
                };
                tokens += first.tokens();
                batch.push(first);
                while tokens < opts.pass_tokens {
                    match rx.try_recv() {
                        Ok(j) => {
                            if tokens + j.tokens() > opts.pass_tokens {
                                carry = Some(j);
                                break;
                            }
                            tokens += j.tokens();
                            batch.push(j);
                        }
                        Err(_) => break,
                    }
                }
            }
            let t0 = Instant::now();
            let res = self.pass(&batch);
            let _ = self.m.dev.synchronize();
            health.last_pass_ms.store(now_ms(), Ordering::SeqCst);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            match res {
                Ok(rs) => {
                    for (j, o) in batch.into_iter().zip(rs) {
                        let _ = j.reply.send(Ok((o, ms)));
                    }
                }
                Err(e) => {
                    #[cfg(feature = "cuda")]
                    let mem = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                        .map(|(f, t)| format!("{} of {} MiB free", f >> 20, t >> 20))
                        .unwrap_or_default();
                    #[cfg(not(feature = "cuda"))]
                    let mem = "";
                    eprintln!(
                        "kev: pass of {} requests ({tokens} tokens, longest state {}) failed: {e} {mem}",
                        batch.len(),
                        batch.iter().map(|j| j.state.len()).max().unwrap_or(0)
                    );
                    for j in batch {
                        let _ = j.reply.send(Err(e.to_string()));
                    }
                }
            }
        }
    }
}

/// One served model: the names requests select it by, its entrant, its queue.
pub struct Served {
    pub names: Vec<String>,
    pub entrant: Entrant,
    pub card: Value,
    tx: mpsc::SyncSender<Job>,
    health: Arc<Health>,
}

impl Served {
    /// Queue a state and its rows; -> the readout outputs per row and the pass time.
    pub async fn run(
        &self,
        state: Vec<u32>,
        rows: Vec<RowSpec>,
    ) -> std::result::Result<(Outputs, f64), (StatusCode, String)> {
        self.health.in_flight.fetch_add(1, Ordering::SeqCst);
        let _in_flight = Counted(&self.health.in_flight);
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tx
            .try_send(Job {
                state,
                rows,
                reply: tx,
            })
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => (
                    StatusCode::TOO_MANY_REQUESTS,
                    "inference queue saturated; please retry shortly".to_string(),
                ),
                mpsc::TrySendError::Disconnected(_) => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "model thread stopped".to_string(),
                ),
            })?;
        rx.await
            .map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "model thread dropped the request".to_string(),
                )
            })?
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))
    }
}

/// Start one model worker per (model, readout) (each on its own device, or several on one), all serving one queue.
pub fn spawn(
    workers: Vec<(Model, Readout)>,
    entrant: Entrant,
    names: Vec<String>,
    opts: Opts,
    card: Value,
) -> Served {
    let (tx, rx) = mpsc::sync_channel::<Job>(MAX_QUEUE_DEPTH);
    let rx = Arc::new(Mutex::new(rx));
    let health = Arc::new(Health::default());
    for (i, (m, readout)) in workers.into_iter().enumerate() {
        let (rx, opts, health) = (rx.clone(), opts.clone(), health.clone());
        std::thread::Builder::new()
            .name(format!(
                "{}-{i}",
                names.first().map_or("model", |n| n.as_str())
            ))
            .spawn(move || {
                Worker {
                    m,
                    readout,
                    cache: Prefixes::new(opts.prefix_cache),
                    health,
                }
                .serve(rx, opts)
            })
            .expect("spawn model thread");
    }
    Served {
        names,
        entrant,
        card,
        tx,
        health,
    }
}

/// Why a model is unhealthy, if it is: no live worker, or requests waiting with no pass for STALL_SECS. An idle model
/// (nothing in flight) is healthy however long since its last pass.
fn health_problem(workers: usize, in_flight: usize, idle_s: u64) -> Option<&'static str> {
    if workers == 0 {
        Some("model thread stopped")
    } else if in_flight > 0 && idle_s > STALL_SECS {
        Some("stalled: requests waiting and no pass finished")
    } else {
        None
    }
}

/// /health: 200 while every served model has a live worker and is not stalled, else 503 with the reason. Both
/// readiness (out of the Service) and liveness (restart the pod) can probe it.
async fn health(State(s): State<AppState>) -> (StatusCode, Json<Value>) {
    let now = now_ms();
    let mut ok = true;
    let models: Vec<Value> = s
        .models
        .iter()
        .map(|m| {
            let h = &m.health;
            let (alive, in_flight) = (
                h.alive.load(Ordering::SeqCst),
                h.in_flight.load(Ordering::SeqCst),
            );
            let idle_s = now.saturating_sub(h.last_pass_ms.load(Ordering::SeqCst)) / 1000;
            let problem = health_problem(alive, in_flight, idle_s);
            ok &= problem.is_none();
            json!({
                "name": m.names.first(), "workers": alive, "in_flight": in_flight,
                "seconds_since_pass": idle_s, "problem": problem,
            })
        })
        .collect();
    let code = if ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        code,
        Json(
            json!({"status": if ok { "ok" } else { "unhealthy" }, "engine": "zev-candle", "models": models}),
        ),
    )
}

#[derive(Clone)]
pub struct AppState {
    models: Arc<Vec<Served>>,
    api_key: Option<String>,
}

type Reply = std::result::Result<Json<Value>, (StatusCode, Json<Value>)>;

fn err(code: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"detail": msg.into()})))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn systemone(State(s): State<AppState>, headers: HeaderMap, body: String) -> Reply {
    if let Some(key) = &s.api_key {
        let got = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let expected = format!("Bearer {key}");
        if !constant_time_eq(got.as_bytes(), expected.as_bytes()) {
            return Err(err(
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>",
            ));
        }
    }
    let req: Value = serde_json::from_str(&body)
        .map_err(|e| err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
    // `model` selects a served model by name; any other name (e.g. the harness's "jev-latest") gets the first one
    let name = req.get("model").and_then(Value::as_str).unwrap_or("");
    let m = s
        .models
        .iter()
        .find(|m| m.names.iter().any(|n| n == name))
        .unwrap_or(&s.models[0]);
    m.entrant
        .answer(m, &req)
        .await
        .map(Json)
        .map_err(|(c, e)| err(c, e))
}

async fn list_models(State(s): State<AppState>) -> Json<Value> {
    let cards: Vec<Value> = s
        .models
        .iter()
        .flat_map(|m| {
            m.names.iter().map(|n| {
                let mut c = m.card.clone();
                c["name"] = json!(n);
                c
            })
        })
        .collect();
    Json(json!({ "models": cards }))
}

pub fn router(models: Vec<Served>) -> Router {
    let s = AppState {
        models: Arc::new(models),
        api_key: std::env::var("KEV_API_KEY").ok().filter(|k| !k.is_empty()),
    };
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/systemone", post(systemone))
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_problem() {
        assert_eq!(health_problem(1, 0, 10_000), None); // idle for hours: healthy
        assert_eq!(health_problem(1, 5, 3), None); // busy, passes finishing
        assert_eq!(health_problem(0, 0, 0), Some("model thread stopped"));
        assert!(health_problem(1, 5, STALL_SECS + 1)
            .unwrap()
            .starts_with("stalled")); // wedged GPU
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"Bearer secret123", b"Bearer secret123"));
        assert!(!constant_time_eq(b"Bearer secret123", b"Bearer secret124"));
        assert!(!constant_time_eq(b"Bearer secret123", b"Bearer secret12"));
        assert!(!constant_time_eq(b"Bearer secret12", b"Bearer secret123"));
        assert!(!constant_time_eq(b"", b"Bearer secret123"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_prefixes_lru() {
        let mut cache = Prefixes {
            size: 2,
            order: VecDeque::new(),
            map: HashMap::new(),
            hits: 0,
            misses: 0,
        };

        let dummy_cache1 = Arc::new(StateCache {
            len: 1,
            conv: None,
            rec: None,
            k: None,
            v: None,
        });
        let dummy_cache2 = Arc::new(StateCache {
            len: 2,
            conv: None,
            rec: None,
            k: None,
            v: None,
        });
        let dummy_cache3 = Arc::new(StateCache {
            len: 3,
            conv: None,
            rec: None,
            k: None,
            v: None,
        });

        cache.put(vec![1, 2], dummy_cache1.clone());
        cache.put(vec![3, 4], dummy_cache2.clone());

        assert_eq!(cache.get(&[1, 2]).unwrap().len, 1);

        // Putting 3rd item should evict [3, 4] because [1, 2] was recently accessed
        cache.put(vec![5, 6], dummy_cache3.clone());

        assert!(cache.get(&[3, 4]).is_none());
        assert_eq!(cache.get(&[1, 2]).unwrap().len, 1);
        assert_eq!(cache.get(&[5, 6]).unwrap().len, 3);
    }
}
