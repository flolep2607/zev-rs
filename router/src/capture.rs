//! Best-effort replay capture (florian-nginx capture.lua): a `request` event when the request is admitted and a
//! `response` event after the full response, joined by capture_id, POSTed to a receiver. Never blocks or fails
//! inference: no retries, a bounded number of events in flight, and nothing at all while the receiver is down.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::fleet::CaptureCfg;

pub struct Capture {
    cfg: CaptureCfg,
    http: reqwest::Client,
    available: Arc<AtomicBool>,
    pending: Arc<AtomicUsize>,
    next_id: AtomicU64,
    pub dropped: AtomicU64,
}

/// One captured exchange: the response side accumulates here while it streams.
pub struct Record {
    pub id: String,
    started: SystemTime,
    pub body: Vec<u8>,
    pub truncated: bool,
}

fn now_s(t: SystemTime) -> f64 {
    t.duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

impl Capture {
    pub fn new(cfg: CaptureCfg) -> Arc<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.timeout_ms))
            .build()
            .expect("capture http client");
        Arc::new(Capture {
            cfg,
            http,
            available: Arc::new(AtomicBool::new(false)),
            pending: Arc::new(AtomicUsize::new(0)),
            next_id: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        })
    }

    /// HEAD the receiver; any HTTP status means the process is up. Run every check_interval_s.
    pub async fn check(&self) {
        let url = if self.cfg.health_path.is_empty() {
            self.cfg.endpoint.clone()
        } else {
            let base = self
                .cfg
                .endpoint
                .splitn(4, '/')
                .take(3)
                .collect::<Vec<_>>()
                .join("/");
            format!("{base}{}", self.cfg.health_path)
        };
        let up = self.http.head(&url).send().await.is_ok();
        if up != self.available.swap(up, Ordering::Relaxed) {
            log::warn!(
                "capture receiver {}: {}",
                self.cfg.endpoint,
                if up {
                    "available"
                } else {
                    "unavailable, capture paused"
                }
            );
        }
    }

    /// Start a capture for this request, or None (receiver down, body too big). The request event goes out now.
    pub fn begin(
        &self,
        method: &str,
        uri: &str,
        content_type: Option<&str>,
        session: Option<&str>,
        body: &[u8],
    ) -> Option<Record> {
        if !self.available.load(Ordering::Relaxed) || body.len() > self.cfg.max_request_bytes {
            return None;
        }
        let started = SystemTime::now();
        let id = format!(
            "{:x}-{:x}",
            now_s(started) as u64,
            self.next_id.fetch_add(1, Ordering::Relaxed)
        );
        self.send(serde_json::json!({
            "schema_version": 1, "event": "request", "capture_id": id, "timestamp": now_s(started),
            "request": { "method": method, "uri": uri, "content_type": content_type, "session_id": session,
                         "body": String::from_utf8_lossy(body) },
        }));
        Some(Record {
            id,
            started,
            body: Vec::new(),
            truncated: false,
        })
    }

    pub fn tee(&self, rec: &mut Record, chunk: &[u8]) {
        let room = self.cfg.max_response_bytes.saturating_sub(rec.body.len());
        if chunk.len() > room {
            rec.truncated = true;
        }
        rec.body.extend_from_slice(&chunk[..chunk.len().min(room)]);
    }

    pub fn finish(&self, rec: Record, status: u16, content_type: Option<&str>, backend: &str) {
        let now = SystemTime::now();
        self.send(serde_json::json!({
            "schema_version": 1, "event": "response", "capture_id": rec.id, "timestamp": now_s(now),
            "response": { "status": status, "content_type": content_type, "backend": backend,
                          "duration_ms": now.duration_since(rec.started).map_or(0, |d| d.as_millis() as u64),
                          "truncated": rec.truncated, "body": String::from_utf8_lossy(&rec.body) },
        }));
    }

    /// Detached, one shot. A failed delivery pauses capture until the next health check succeeds.
    fn send(&self, event: serde_json::Value) {
        if self.pending.fetch_add(1, Ordering::Relaxed) >= self.cfg.max_pending {
            self.pending.fetch_sub(1, Ordering::Relaxed);
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let (http, url) = (self.http.clone(), self.cfg.endpoint.clone());
        let (pending, available) = (self.pending.clone(), self.available.clone());
        tokio::spawn(async move {
            let res = http.post(&url).json(&event).send().await;
            if !matches!(res, Ok(r) if r.status().is_success()) {
                available.store(false, Ordering::Relaxed);
            }
            pending.fetch_sub(1, Ordering::Relaxed);
        });
    }
}
