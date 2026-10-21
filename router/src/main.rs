//! zev-router: one load-aware router for the Kev (zev-fast) and vLLM pools. Pingora does the proxying (keep-alive
//! pools, streaming, timeouts, graceful restart); fleet.rs decides where each request goes.
//!
//!   zev-router router.json
//!
//! Per request: the path picks a pool, a session header picks a home pod (rendezvous hash) when it has room, else a
//! weighted-random pod with a free slot; with none free the request waits for a release, up to queue_timeout_s.
//! A connect error, a broken upstream, or a 429/500/502/503 before the first byte is retried on an untried pod.

mod fleet;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use fleet::{Config, Fleet, Pod};
use pingora::http::ResponseHeader;
use pingora::prelude::*;
use pingora::proxy::{http_proxy_service, ProxyHttp, Session};
use pingora::server::ShutdownWatch;
use pingora::services::background::{background_service, BackgroundService};

/// Attempts per request, the first included (florian-nginx: proxy_next_upstream_tries 3).
const MAX_ATTEMPTS: u8 = 3;
/// A dead node blackholes SYN rather than refusing it; the default would be a long wait before failover starts.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Non-streaming replies send nothing until done, so this covers queue + generation.
const READ_TIMEOUT: Duration = Duration::from_secs(600);

struct Router {
    fleet: Arc<Fleet>,
}

#[derive(Default)]
struct Ctx {
    pool: String,
    slot: Option<Arc<Pod>>,
    tried: HashSet<SocketAddr>,
    attempts: u8,
}

impl Router {
    /// Hand the slot back; on failure also score the pod down for 30 s so concurrent requests stop choosing it.
    fn drop_slot(&self, ctx: &mut Ctx, failed: bool) {
        if let Some(p) = ctx.slot.take() {
            if failed {
                p.fail();
            }
            self.fleet.release(&p);
        }
    }

    async fn pick(&self, session: &Session, ctx: &Ctx) -> Result<Arc<Pod>> {
        let f = &self.fleet;
        let deadline = Instant::now() + Duration::from_secs(f.cfg.queue_timeout_s);
        let sid = (ctx.attempts == 1)
            .then(|| {
                session
                    .req_header()
                    .headers
                    .get(f.cfg.affinity_header.as_str())
            })
            .flatten()
            .and_then(|v| v.to_str().ok());
        loop {
            // Armed before looking, so a release between the look and the wait is not missed.
            let freed = f.freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            // ponytail: a full home spills at once; florian-nginx waits affinity.waitSeconds for it unless a
            // sibling is clearly idler (spillRatio). Port that when sessions route here.
            if let Some(home) = sid.and_then(|s| f.home(&ctx.pool, s, false)) {
                if f.take(&home) {
                    return Ok(home);
                }
            }
            let (pod, known) = f.reserve(&ctx.pool, &ctx.tried, false);
            if let Some(p) = pod {
                return Ok(p);
            }
            if known == 0 {
                return Error::e_explain(
                    HTTPStatus(502),
                    format!("no pods discovered in pool {}", ctx.pool),
                );
            }
            // A retry does not queue: the request already waited once, and nginx's balancer phase could not either.
            if ctx.attempts > 1 {
                return Error::e_explain(HTTPStatus(502), "no untried pod with room for the retry");
            }
            let now = Instant::now();
            if now >= deadline {
                return Error::e_explain(HTTPStatus(503), "queue timeout: every pod at cap");
            }
            // Releases and scrapes wake this; the 1 s bound re-checks pods going stale or coming back.
            let _ = tokio::time::timeout((deadline - now).min(Duration::from_secs(1)), freed).await;
        }
    }
}

async fn respond(session: &mut Session, code: u16, ctype: &str, body: String) -> Result<()> {
    let mut h = ResponseHeader::build(code, Some(2))?;
    h.insert_header("content-type", ctype)?;
    h.insert_header("content-length", body.len().to_string())?;
    session.write_response_header(Box::new(h), false).await?;
    session.write_response_body(Some(body.into()), true).await
}

#[async_trait]
impl ProxyHttp for Router {
    type CTX = Ctx;

    fn new_ctx(&self) -> Ctx {
        Ctx::default()
    }

    async fn request_filter(&self, session: &mut Session, ctx: &mut Ctx) -> Result<bool> {
        let path = session.req_header().uri.path().to_string();
        match path.as_str() {
            "/ready" => respond(session, 200, "text/plain", "ok\n".into()).await?,
            "/balancer-status" => {
                respond(session, 200, "application/json", self.fleet.status_json()).await?
            }
            "/metrics" => {
                respond(
                    session,
                    200,
                    "text/plain; version=0.0.4",
                    self.fleet.prometheus(),
                )
                .await?
            }
            _ => {
                ctx.pool = self.fleet.pool_for(&path).to_string();
                // Keep the body so an attempt that fails before the first byte can be replayed elsewhere. Pingora
                // keeps up to 64 KiB: a Kev body fits, a large vLLM prompt is not replayed once it is sent.
                session.enable_retry_buffering();
                return Ok(false);
            }
        }
        Ok(true)
    }

    async fn upstream_peer(&self, session: &mut Session, ctx: &mut Ctx) -> Result<Box<HttpPeer>> {
        ctx.attempts += 1;
        let pod = self.pick(session, ctx).await?;
        ctx.tried.insert(pod.addr);
        let mut peer = HttpPeer::new(pod.addr, false, String::new());
        peer.options.connection_timeout = Some(CONNECT_TIMEOUT);
        peer.options.read_timeout = Some(READ_TIMEOUT);
        ctx.slot = Some(pod);
        Ok(Box::new(peer))
    }

    async fn upstream_response_filter(
        &self,
        _session: &mut Session,
        resp: &mut ResponseHeader,
        ctx: &mut Ctx,
    ) -> Result<()> {
        let code = resp.status.as_u16();
        // A zev pod answers 429 when its queue is full, and a wedged one answers nothing else: 1.04M of them in
        // 16 h on 2026-10-21 while it looked like the idlest pod. Turned into an error here, before the client has
        // a byte, error_while_proxy can still send it elsewhere. The last attempt passes the status through.
        if matches!(code, 429 | 500 | 502 | 503) && ctx.attempts < MAX_ATTEMPTS {
            return Error::e_explain(HTTPStatus(code), "upstream status, retrying on another pod");
        }
        Ok(())
    }

    fn fail_to_connect(
        &self,
        _session: &mut Session,
        _peer: &HttpPeer,
        ctx: &mut Ctx,
        mut e: Box<Error>,
    ) -> Box<Error> {
        self.drop_slot(ctx, true);
        e.set_retry(ctx.attempts < MAX_ATTEMPTS); // nothing was sent: always safe
        e
    }

    fn error_while_proxy(
        &self,
        _peer: &HttpPeer,
        session: &mut Session,
        mut e: Box<Error>,
        ctx: &mut Ctx,
        _client_reused: bool,
    ) -> Box<Error> {
        self.drop_slot(ctx, true);
        // POST is retried on purpose, as florian-nginx's non_idempotent: a Kev call is a pure function and a vLLM
        // one at worst repeats work. Never once the client has a byte, never without the whole body buffered.
        let replayable = session.response_written().is_none() && !session.retry_buffer_truncated();
        e.set_retry(replayable && ctx.attempts < MAX_ATTEMPTS);
        e
    }

    async fn logging(&self, session: &mut Session, e: Option<&Error>, ctx: &mut Ctx) {
        let code = session.response_written().map_or(0, |r| r.status.as_u16());
        let failed = e.is_some() || code == 429 || code >= 500;
        self.drop_slot(ctx, failed);
        if !ctx.pool.is_empty() {
            self.fleet.count(&ctx.pool, code);
        }
    }
}

struct Scraper {
    fleet: Arc<Fleet>,
}

#[async_trait]
impl BackgroundService for Scraper {
    async fn start(&self, mut shutdown: ShutdownWatch) {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("http client");
        let mut tick =
            tokio::time::interval(Duration::from_secs(self.fleet.cfg.metrics_interval_s));
        loop {
            tokio::select! {
                _ = shutdown.changed() => return,
                _ = tick.tick() => {
                    self.fleet.discover().await;
                    self.fleet.scrape(&http).await;
                }
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

    let mut server = Server::new(None).expect("server");
    server.bootstrap();
    let mut proxy = http_proxy_service(
        &server.configuration,
        Router {
            fleet: fleet.clone(),
        },
    );
    proxy.add_tcp(&cfg.listen);
    proxy.threads = Some(cfg.threads);
    server.add_service(proxy);
    server.add_service(background_service("scraper", Scraper { fleet }));
    server.run_forever();
}
