//! zev-router: one load-aware router for the Kev (zev-fast) and vLLM pools, a port of learning-helm's florian-nginx.
//!
//!   zev-router router.json
//!
//! Pingora is the server (HTTP/1 + h2 downstream, threads, graceful restart). Forwarding is ours, through a
//! keep-alive reqwest client, because three decisions need the whole body before a pod is chosen -- the image check,
//! the `model` pool and per-pod compression -- and Pingora can only replay a body it buffered itself (64 KiB).
//!
//! Per request: drained? 503. Read the body. Pool from the path, then from the body's `model`. Image scan if the pool
//! has a text-only pod. Session home with a free slot, waiting up to affinity_wait_s for it unless a sibling is
//! clearly idler; else a weighted-random pod with a free slot, waiting for a release up to queue_timeout_s. Up to 3
//! attempts: a connect error or a 429/500/502/503 goes to an untried pod. The body is dropped once the response
//! headers arrive, then the response streams back, cut at once by /drain.

mod capture;
mod fleet;

use std::collections::HashSet;
use std::io::Write as _;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use capture::Capture;
use fleet::{Config, Fleet, Pod};
use pingora::http::ResponseHeader;
use pingora::prelude::*;
use pingora::proxy::{http_proxy_service, ProxyHttp, Session};
use pingora::server::ShutdownWatch;
use pingora::services::background::{background_service, BackgroundService};

/// Attempts per request, the first included (florian-nginx: proxy_next_upstream_tries 3).
const MAX_ATTEMPTS: usize = 3;
/// A dead node blackholes SYN rather than refusing it; the default would be a long wait before failover starts.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Non-streaming replies send nothing until done, so this covers queue + generation, per read.
const READ_TIMEOUT: Duration = Duration::from_secs(600);
/// Headers that describe one hop, never forwarded either way.
const HOP: [&str; 9] = [
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "host",
    "content-length",
];

struct Router {
    fleet: Arc<Fleet>,
    capture: Option<Arc<Capture>>,
    http: reqwest::Client,
}

/// Why a request ended without an upstream response: the status to answer with.
struct Reject(u16, &'static str);

/// Does this body carry an image? The content part's type tag, not the bare word: prompts and JSON schemas say
/// `image_url` as plain text, and each of those 502'd once only the text-only 5090 was left (2026-10-20).
fn carries_image(body: &[u8]) -> bool {
    let mut rest = body;
    while let Some(i) = find(rest, b"\"type\"") {
        let after = &rest[i + 6..];
        let s = after
            .iter()
            .position(|c| !c.is_ascii_whitespace())
            .unwrap_or(after.len());
        if after.get(s) == Some(&b':') {
            let v = &after[s + 1..];
            let t = v
                .iter()
                .position(|c| !c.is_ascii_whitespace())
                .unwrap_or(v.len());
            if v[t..].starts_with(b"\"image_url\"") {
                return true;
            }
        }
        rest = after;
    }
    false
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The top-level `model` of a JSON body, without decoding the rest of it (a screenshot body is megabytes).
fn model_of(body: &[u8]) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct M {
        model: Option<String>,
    }
    serde_json::from_slice::<M>(body).ok()?.model
}

/// zlib stream (RFC 1950): what `Content-Encoding: deflate` means, and what inflate_mw expects.
fn deflate(body: &[u8], level: u32) -> Option<Vec<u8>> {
    let mut e = flate2::write::ZlibEncoder::new(
        Vec::with_capacity(body.len() / 2),
        flate2::Compression::new(level),
    );
    e.write_all(body).ok()?;
    e.finish().ok().filter(|z| z.len() < body.len())
}

async fn respond(session: &mut Session, code: u16, ctype: &str, body: String) -> Result<()> {
    let mut h = ResponseHeader::build(code, Some(2))?;
    h.insert_header("content-type", ctype)?;
    h.insert_header("content-length", body.len().to_string())?;
    session.write_response_header(Box::new(h), false).await?;
    session.write_response_body(Some(body.into()), true).await
}

impl Router {
    /// A slot on the session's home, else on any pod; waits for releases while every pod is full.
    async fn admit(
        &self,
        pool: &str,
        sid: Option<&str>,
        needs_image: bool,
    ) -> std::result::Result<Arc<Pod>, Reject> {
        let f = &self.fleet;
        let start = Instant::now();
        let deadline = start + Duration::from_secs(f.cfg.queue_timeout_s);
        let affinity_until = start + Duration::from_secs(f.cfg.affinity_wait_s);
        let none = HashSet::new();
        loop {
            // Armed before looking, so a release between the look and the wait is not missed.
            let freed = f.freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            if f.drained() {
                return Err(Reject(503, "drained"));
            }
            let home = sid.and_then(|s| f.home(pool, s, needs_image));
            if let Some(h) = &home {
                if f.take(h) {
                    return Ok(h.clone());
                }
            }
            // Holding out for the home's prefix: only while affinity_wait_s lasts and no sibling is clearly idler.
            let hold = home.as_ref().is_some_and(|h| {
                Instant::now() < affinity_until && !f.crowded(h, pool, needs_image)
            });
            if !hold {
                let (pod, known) = f.reserve(pool, &none, needs_image);
                if let Some(p) = pod {
                    return Ok(p);
                }
                if known == 0 {
                    return Err(Reject(502, "no pods discovered in this pool"));
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(Reject(503, "queue timeout: every pod at cap"));
            }
            // Releases and scrapes wake this; the 1 s bound re-checks pods going stale or coming back.
            let _ = tokio::time::timeout((deadline - now).min(Duration::from_secs(1)), freed).await;
        }
    }

    /// Hand the slot back; on failure also score the pod down for 30 s so concurrent requests stop choosing it.
    fn done(&self, pod: &Pod, failed: bool) {
        if failed {
            pod.fail();
        }
        self.fleet.release(pod);
    }

    /// One attempt: the request to `pod`, the body compressed if it inflates bodies, its key if it is a rented box.
    async fn attempt(
        &self,
        session: &Session,
        pod: &Pod,
        body: &Bytes,
    ) -> std::result::Result<reqwest::Response, String> {
        let req = session.req_header();
        let url = format!(
            "http://{}{}",
            pod.addr,
            req.uri.path_and_query().map_or("/", |p| p.as_str())
        );
        let method = reqwest::Method::from_bytes(req.method.as_str().as_bytes())
            .unwrap_or(reqwest::Method::POST);
        let mut out = self.http.request(method, url);
        for (k, v) in req.headers.iter() {
            if !HOP.contains(&k.as_str())
                && !(pod.up.api_key.is_some() && k.as_str() == "authorization")
            {
                out = out.header(k.as_str(), v.as_bytes());
            }
        }
        if let Some(key) = &pod.up.api_key {
            out = out.bearer_auth(key);
        }
        let cfg = &self.fleet.cfg;
        let squeezed = (pod.up.deflate
            && cfg.compress_min > 0
            && body.len() >= cfg.compress_min
            && !req.headers.contains_key("content-encoding"))
        .then(|| deflate(body, cfg.compress_level))
        .flatten();
        out = match squeezed {
            Some(z) => out.header("content-encoding", "deflate").body(z),
            None => out.body(body.clone()), // Bytes: a refcount, not a copy
        };
        // Per wait, like nginx's proxy_read_timeout; a whole-request reqwest timeout would cut long streams.
        match tokio::time::timeout(READ_TIMEOUT, out.send()).await {
            Ok(r) => r.map_err(|e| e.to_string()),
            Err(_) => Err(format!(
                "no response headers in {}s",
                READ_TIMEOUT.as_secs()
            )),
        }
    }

    async fn serve(&self, session: &mut Session) -> Result<()> {
        let f = &self.fleet;
        if f.drained() {
            f.count("drained", 503);
            return respond(session, 503, "text/plain", "drained\n".into()).await;
        }

        // The whole body, bounded: routing needs it and every retry replays it.
        let mut buf = Vec::new();
        while let Some(chunk) = session.read_request_body().await? {
            if buf.len() + chunk.len() > f.cfg.max_body_bytes {
                return respond(
                    session,
                    413,
                    "text/plain",
                    "request body too large\n".into(),
                )
                .await;
            }
            buf.extend_from_slice(&chunk);
        }
        let mut body = Some(Bytes::from(buf));
        let b = body.as_ref().unwrap();

        let path = session.req_header().uri.path().to_string();
        let mut pool = f.pool_for(&path).to_string();
        if !f.cfg.pool_models.is_empty() {
            if let Some(p) = model_of(b).and_then(|m| f.cfg.pool_models.get(&m).cloned()) {
                pool = p;
            }
        }
        let needs_image = f.has_text_only(&pool) && carries_image(b);
        let headers = &session.req_header().headers;
        let sid = headers
            .get(f.cfg.affinity_header.as_str())
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);

        // Capture is for the vLLM agent traffic: Kev's ~450 calls/s would only crowd it out of max_pending.
        let mut rec = self
            .capture
            .as_ref()
            .filter(|_| pool == "default")
            .and_then(|c| {
                let ctype = headers.get("content-type").and_then(|v| v.to_str().ok());
                let uri = session.req_header().uri.to_string();
                c.begin(
                    session.req_header().method.as_str(),
                    &uri,
                    ctype,
                    sid.as_deref(),
                    b,
                )
            });

        let mut pod = match self.admit(&pool, sid.as_deref(), needs_image).await {
            Ok(p) => p,
            Err(Reject(code, why)) => {
                f.count(&pool, code);
                return respond(session, code, "text/plain", format!("{why}\n")).await;
            }
        };

        let mut tried = HashSet::new();
        let resp = loop {
            tried.insert(pod.addr);
            let res = self.attempt(session, &pod, body.as_ref().unwrap()).await;
            let retryable = match &res {
                Ok(r) => matches!(r.status().as_u16(), 429 | 500 | 502 | 503),
                Err(_) => true,
            };
            if !retryable {
                break res;
            }
            // A zev pod answers 429 when its queue is full, and a wedged one answers nothing else: 1.04M of them in
            // 16 h on 2026-10-21 while it looked like the idlest pod. Fast to get, so the next pod costs ~nothing.
            let next = (tried.len() < MAX_ATTEMPTS)
                .then(|| f.reserve(&pool, &tried, needs_image).0)
                .flatten();
            match next {
                Some(n) => {
                    log::warn!(
                        "{} failed ({}), retrying on {}",
                        pod.addr,
                        res.as_ref()
                            .map_or_else(|e| e.clone(), |r| r.status().to_string()),
                        n.addr
                    );
                    self.done(&pod, true);
                    pod = n;
                }
                // Nowhere left to go: the last answer, even a 429, beats a made-up 502.
                None => break res,
            }
        };

        let mut resp = match resp {
            Ok(r) => r,
            Err(e) => {
                log::error!("{} failed for good: {e}", pod.addr);
                self.done(&pod, true);
                f.count(&pool, 502);
                return respond(session, 502, "text/plain", "upstream failed\n".into()).await;
            }
        };
        // Retries are over: the prompt is dead weight from here to the last token (what held 3.5 GB in
        // mariojcr's Pingora gateway at 10k streams).
        body.take();

        let status = resp.status().as_u16();
        let mut h = ResponseHeader::build(status, Some(resp.headers().len() + 1))?;
        for (k, v) in resp.headers() {
            if !HOP.contains(&k.as_str()) {
                h.append_header(k.as_str().to_string(), v.as_bytes())?;
            }
        }
        match resp.content_length() {
            Some(n) => h.insert_header("content-length", n.to_string())?,
            None => h.insert_header("transfer-encoding", "chunked")?, // SSE; else close-delimited, no keep-alive
        }
        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        session.write_response_header(Box::new(h), false).await?;

        // Stream it back. /drain cuts here: dropping `resp` closes the upstream connection, which is what makes
        // vLLM abort the generation.
        let mut cut = f.cut.subscribe();
        let mut failed = status == 429 || status >= 500;
        loop {
            let chunk = tokio::select! {
                c = tokio::time::timeout(READ_TIMEOUT, resp.chunk()) => c,
                _ = cut.changed() => { failed = false; log::warn!("drain cut a request on {}", pod.addr); break }
            };
            match chunk {
                Ok(Ok(Some(c))) => {
                    if let (Some(cap), Some(r)) = (self.capture.as_ref(), rec.as_mut()) {
                        cap.tee(r, &c);
                    }
                    if session.write_response_body(Some(c), false).await.is_err() {
                        break; // the client left; nothing to tell it
                    }
                }
                Ok(Ok(None)) => {
                    let _ = session.write_response_body(None, true).await;
                    break;
                }
                _ => {
                    failed = true; // read error or 600 s of silence mid-response
                    break;
                }
            }
        }
        self.done(&pod, failed);
        f.count(&pool, status);
        if let (Some(cap), Some(r)) = (self.capture.as_ref(), rec.take()) {
            cap.finish(r, status, ctype.as_deref(), &pod.addr.to_string());
        }
        Ok(())
    }

    /// /drain[?seconds=N][&peer=1]: close admission for N seconds (default drain_seconds, 0 = reopen) and cut every
    /// request in flight, on every router replica. Cluster-internal: anything that came through an ingress (which
    /// always adds X-Forwarded-For) is refused.
    async fn drain(&self, session: &mut Session) -> Result<()> {
        let req = session.req_header();
        if req.headers.contains_key("x-forwarded-for") {
            return respond(session, 403, "text/plain", "cluster-internal only\n".into()).await;
        }
        let query = req.uri.query().unwrap_or("");
        let param = |k: &str| {
            query
                .split('&')
                .find_map(|kv| kv.strip_prefix(k).and_then(|v| v.strip_prefix('=')))
        };
        let seconds = param("seconds")
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.fleet.cfg.drain_seconds);
        let mut peers = Vec::new();
        match (&self.fleet.cfg.peers_dns, param("peer")) {
            // Fan out to every replica, this one included through the Service; ?peer=1 stops the recursion.
            (Some(dns), None) => {
                let addrs: Vec<SocketAddr> =
                    tokio::net::lookup_host((dns.as_str(), self.fleet.cfg.peers_port))
                        .await
                        .map(|a| a.collect())
                        .unwrap_or_default();
                for a in addrs {
                    let ok = self
                        .http
                        .post(format!("http://{a}/drain?seconds={seconds}&peer=1"))
                        .send()
                        .await
                        .is_ok_and(|r| r.status().is_success());
                    peers.push(serde_json::json!({ "addr": a.to_string(), "ok": ok }));
                }
                if peers.is_empty() {
                    self.fleet.drain(seconds); // no peer resolved: at least this one
                }
            }
            _ => self.fleet.drain(seconds),
        }
        let reply =
            serde_json::json!({ "drained_s": self.fleet.drain_remaining_s(), "peers": peers });
        respond(session, 200, "application/json", reply.to_string()).await
    }
}

#[async_trait]
impl ProxyHttp for Router {
    type CTX = ();

    fn new_ctx(&self) {}

    /// Everything happens here and the request never reaches Pingora's own proxy: see the module docs.
    async fn request_filter(&self, session: &mut Session, _: &mut ()) -> Result<bool> {
        let path = session.req_header().uri.path().to_string();
        match path.as_str() {
            "/ready" => respond(session, 200, "text/plain", "ok\n".into()).await?,
            "/balancer-status" => {
                respond(session, 200, "application/json", self.fleet.status_json()).await?
            }
            "/metrics" => {
                let mut m = self.fleet.prometheus();
                if let Some(c) = &self.capture {
                    let n = c.dropped.load(std::sync::atomic::Ordering::Relaxed);
                    m.push_str(&format!("# TYPE zev_router_capture_dropped_total counter\nzev_router_capture_dropped_total {n}\n"));
                }
                respond(session, 200, "text/plain; version=0.0.4", m).await?
            }
            "/drain" => self.drain(session).await?,
            _ => self.serve(session).await?,
        }
        Ok(true)
    }

    async fn upstream_peer(&self, _: &mut Session, _: &mut ()) -> Result<Box<HttpPeer>> {
        Error::e_explain(
            InternalError,
            "unreachable: request_filter answers every request",
        )
    }
}

struct Background {
    fleet: Arc<Fleet>,
    capture: Option<Arc<Capture>>,
}

#[async_trait]
impl BackgroundService for Background {
    async fn start(&self, mut shutdown: ShutdownWatch) {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("http client");
        let mut tick =
            tokio::time::interval(Duration::from_secs(self.fleet.cfg.metrics_interval_s));
        let check_every = self
            .fleet
            .cfg
            .capture
            .as_ref()
            .map_or(30, |c| c.check_interval_s.max(1));
        let mut check = tokio::time::interval(Duration::from_secs(check_every));
        loop {
            tokio::select! {
                _ = shutdown.changed() => return,
                _ = tick.tick() => {
                    self.fleet.discover().await;
                    self.fleet.scrape(&http).await;
                }
                _ = check.tick(), if self.capture.is_some() => self.capture.as_ref().unwrap().check().await,
            }
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let path = std::env::args()
        .nth(1)
        .expect("usage: zev-router <config.json>");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(&path).expect("read config"))
        .expect("parse config");
    let fleet = Arc::new(Fleet::new(cfg.clone()));
    let capture = cfg
        .capture
        .clone()
        .filter(|c| !c.endpoint.is_empty())
        .map(Capture::new);
    let http = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .pool_max_idle_per_host(64)
        .build()
        .expect("http client");

    let mut server = Server::new(None).expect("server");
    server.bootstrap();
    let router = Router {
        fleet: fleet.clone(),
        capture: capture.clone(),
        http,
    };
    let mut proxy = http_proxy_service(&server.configuration, router);
    proxy.add_tcp(&cfg.listen);
    proxy.threads = Some(cfg.threads);
    server.add_service(proxy);
    server.add_service(background_service(
        "background",
        Background { fleet, capture },
    ));
    server.run_forever();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_scan_wants_the_type_tag() {
        assert!(carries_image(
            br#"{"content":[{"type": "image_url","image_url":{"url":"data:..."}}]}"#
        ));
        assert!(carries_image(br#"{"type":"image_url"}"#));
        assert!(!carries_image(
            br#"{"schema":{"image_url":{"type":"string"}},"prompt":"return an 'image_url' field"}"#
        ));
        assert!(!carries_image(b"no json at all"));
    }

    #[test]
    fn model_and_deflate() {
        assert_eq!(
            model_of(br#"{"model":"combo-r3","state":"x","questions":{}}"#).as_deref(),
            Some("combo-r3")
        );
        assert_eq!(model_of(br#"{"state":"x"}"#), None);
        let body = "the same invoice line again and again ".repeat(200);
        let z = deflate(body.as_bytes(), 1).expect("compressible");
        let mut back = String::new();
        std::io::Read::read_to_string(&mut flate2::read::ZlibDecoder::new(&z[..]), &mut back)
            .unwrap();
        assert_eq!(back, body, "zlib round trip");
        assert!(deflate(b"x", 1).is_none(), "not worth it: send it as is");
    }
}
