//! A stand-in zev pod for smoke.py: HTTP/1.1 and h2c on one port (axum's serve, as zev's), zev-style /metrics, and
//! /v1/systemone answering the protocol it was reached over, so the test can tell h2c from HTTP/1.1.
//!
//!   cargo run --example fake_h2c_pod -- 18190

use std::sync::atomic::{AtomicU64, Ordering};

use axum::{extract::Request, routing::get, routing::post, Router};

static PASSES: AtomicU64 = AtomicU64::new(0);

#[tokio::main]
async fn main() {
    let port = std::env::args().nth(1).unwrap_or_else(|| "18190".into());
    let app = Router::new()
        .route(
            "/metrics",
            get(|| async {
                let n = PASSES.fetch_add(1, Ordering::Relaxed) + 1;
                format!(
                    "zev_in_flight{{model=\"fake\"}} 0\nzev_passes_total{{model=\"fake\"}} {n}\n"
                )
            }),
        )
        .route(
            "/v1/systemone",
            post(|req: Request| async move {
                format!("{{\"answers\":{{}},\"version\":\"{:?}\"}}", req.version())
            }),
        );
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
