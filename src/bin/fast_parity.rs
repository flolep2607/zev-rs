//! How far does a re-merged backbone drift from a freshly merged one?
//!
//! For the Kev adapters A and B over one base: FRESH_B = `Model::load(base, B)` against REMERGED_B =
//! `Model::load(base, A).remerge(base, B)` (rebuilt from the base; `--delta`: `remerge_delta`) and the reverse, FRESH_A against REMERGED_A, each with its own
//! adapter's pointer head. Every request goes through the prompt encoder and readout /v1/systemone uses (one request per
//! pass, so both models see the same batches), and the option probabilities are compared. `KEV_W8=int8` is honoured
//! as `zev serve` honours it: both models are quantized the way serving quantizes them.
//!
//!   KEV_W8=int8 cargo run --release --features cuda --bin fast_parity -- \
//!     --adapter-a jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 --adapter-b /root/.cache/kev-combo-r8b1-step-400 \
//!     --requests-a kev-4b.jsonl --requests-b agent-4b.jsonl

use clap::Parser;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zev::kev::entrant::Entrant;
use zev::kev::load::{device, kev_base, kev_served, resolve, LoadOpts};
use zev::kev::model::Model;
use zev::kev::serve::Opts;
use zev::kev::{to_record, Question};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Parity of a re-merged Kev backbone against a freshly merged one")]
struct Args {
    /// The base checkpoint (dir or hub id), or a Kev adapter whose head names its base
    #[arg(long, default_value = "jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101")]
    base: String,
    /// Adapter A (dir or hub id[@revision]: adapter_model.safetensors, adapter_config.json, head)
    #[arg(long)]
    adapter_a: String,
    /// Adapter B
    #[arg(long)]
    adapter_b: String,
    /// JSONL of request bodies {model, state, questions} (a {"request": ...} wrapper is unwrapped) used for both adapters
    #[arg(long)]
    requests: Option<PathBuf>,
    /// Requests of adapter A's traffic: run through FRESH_A and REMERGED_A (B -> A)
    #[arg(long)]
    requests_a: Option<PathBuf>,
    /// Requests of adapter B's traffic: run through FRESH_B and REMERGED_B (A -> B)
    #[arg(long)]
    requests_b: Option<PathBuf>,
    /// Use only the first N requests of each file
    #[arg(long)]
    limit: Option<usize>,
    /// Re-merge by delta from the other adapter's weights (fast, reads no base weights, drifts) instead of rebuilding
    /// the projections from the base checkpoint (the default, equal to a fresh load)
    #[arg(long)]
    delta: bool,
    /// Also compare the remerged and fresh weights projection by projection (dequantized). With no request files,
    /// only that and the remerge cost run.
    #[arg(long)]
    per_proj: bool,
    /// bf16 (needed by KEV_W8) or f32
    #[arg(long, default_value = "bf16")]
    dtype: String,
    /// CUDA device
    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn read_requests(path: &Path, limit: Option<usize>) -> Res<Vec<Value>> {
    let mut out = Vec::new();
    for (i, line) in std::fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        out.push(v.get("request").cloned().unwrap_or(v));
        if limit.is_some_and(|n| out.len() >= n) {
            break;
        }
    }
    Ok(out)
}

/// Free device memory in bytes (0 without CUDA).
fn free_bytes() -> u64 {
    #[cfg(feature = "cuda")]
    if let Ok((free, _)) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info() {
        return free as u64;
    }
    0
}

fn argmax(p: &[f64]) -> usize {
    (0..p.len()).fold(0, |m, i| if p[i] > p[m] { i } else { m }) // as kev::to_answers
}

/// The probabilities of every question of every request, as `/v1/systemone` computes them; None for a request it refuses.
fn run_all(rt: &tokio::runtime::Runtime, served: &zev::kev::serve::Served, reqs: &[Value]) -> Vec<Option<Vec<Vec<f64>>>> {
    let Entrant::Kev(enc) = &served.entrant else { unreachable!("kev_served builds a Kev entrant") };
    reqs.iter()
        .enumerate()
        .map(|(i, req)| {
            let answer = || -> Res<Vec<Vec<f64>>> {
                let (state, qs): (String, Vec<Question>) = to_record(req)?;
                let (ids, rows) = enc.encode(&state, &qs)?;
                let (out, _ms) = rt.block_on(served.run(ids, rows)).map_err(|(c, e)| format!("{c}: {e}"))?;
                Ok(out.into_iter().flatten().collect())
            };
            answer().map_err(|e| eprintln!("fast_parity: request {i} skipped: {e}")).ok()
        })
        .collect()
}

fn load_served(spec: &str, dir: &Path, base_spec: &str, base: &Path, model: Model, o: &LoadOpts) -> Res<zev::kev::serve::Served> {
    kev_served(spec, dir, base_spec, base, vec![model], "merged", o, Instant::now())
}

#[derive(Clone, Copy)]
struct Mode {
    /// remerge_delta (dequantized weights - delta_from + delta_to) instead of remerge (rebuilt from the base)
    delta: bool,
    per_proj: bool,
}

/// |remerged - fresh| of every projection, dequantized, straight from the weights: the worst ones, then each kind.
/// Steps are in units of the fresh row's int8 step (a delta remerge may differ by about 1.5; a rebuild by 0).
fn per_proj(label: &str, fresh: &Model, re: &Model) -> Res<()> {
    use candle_core::DType::F32;
    let mut rows = Vec::new(); // (name, max |dw|, max steps, rms dw / rms w, fraction of elements differing)
    for ((name, pf), (_, pr)) in fresh.projections().into_iter().zip(re.projections()) {
        let ((wf, sf), (wr, _)) = (pf.dequantized()?, pr.dequantized()?);
        let d = (&wr - &wf)?.abs()?;
        let scalar = |t: candle_core::Tensor| t.to_scalar::<f32>();
        let max = scalar(d.max_all()?)?;
        let steps = match &sf {
            Some(s) => scalar(d.broadcast_div(&s.unsqueeze(1)?)?.max_all()?)?,
            None => f32::NAN,
        };
        let rel = scalar(d.sqr()?.mean_all()?)?.sqrt() / scalar(wf.sqr()?.mean_all()?)?.sqrt().max(f32::MIN_POSITIVE);
        let differing = scalar(d.gt(0f64)?.to_dtype(F32)?.mean_all()?)?;
        rows.push((name, max, steps, rel, differing));
    }
    let key = |r: &(String, f32, f32, f32, f32)| if r.2.is_nan() { r.1 } else { r.2 };
    println!("== per projection, {label}: |remerged - fresh| of the dequantized weights");
    println!("   projections differing   {} of {}", rows.iter().filter(|r| r.1 > 0.0).count(), rows.len());
    println!("   {:<26} {:>10} {:>8} {:>10} {:>9}", "kind (worst layer)", "max|dw|", "steps", "rms dw/w", "differing");
    for kind in ["mixer_in", "mixer_out", "gate_up", "down"] {
        let worst = rows.iter().filter(|r| r.0.ends_with(kind)).max_by(|a, b| key(a).total_cmp(&key(b)));
        if let Some(r) = worst {
            println!("   {:<26} {:>10.3e} {:>8} {:>10.3e} {:>8.2}%", r.0, r.1, if r.2.is_nan() { "-".into() } else { format!("{:.3}", r.2) }, r.3, 100.0 * r.4);
        }
    }
    rows.sort_by(|a, b| key(b).total_cmp(&key(a)));
    println!("   worst 8 projections:");
    for r in rows.iter().take(8) {
        println!("   {:<26} {:>10.3e} {:>8} {:>10.3e} {:>8.2}%", r.0, r.1, if r.2.is_nan() { "-".into() } else { format!("{:.3}", r.2) }, r.3, 100.0 * r.4);
    }
    Ok(())
}

/// FRESH_to against Model::load(from) re-merged to `to` on `reqs`, printed under `label`.
#[allow(clippy::too_many_arguments)]
fn direction(
    label: &str,
    base_spec: &str,
    base: &Path,
    (from_spec, from): (&str, &Path),
    (to_spec, to): (&str, &Path),
    reqs: &[Value],
    mode: Mode,
    o: &LoadOpts,
) -> Res<()> {
    let dev = device(o.devices[0])?;
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let fresh = Model::load(base, Some(to), o.dtype, &dev)?;
    let src = Model::load(base, Some(from), o.dtype, &dev)?;
    dev.synchronize()?;
    let (free0, t0) = (free_bytes(), Instant::now());
    let re = if mode.delta { src.remerge_delta(base, from, to)? } else { src.remerge(base, to)? };
    dev.synchronize()?;
    let remerge_ms = t0.elapsed().as_secs_f64() * 1e3;
    let extra = free0.saturating_sub(free_bytes());
    drop(src);
    if mode.per_proj {
        per_proj(label, &fresh, &re)?;
    }
    if reqs.is_empty() {
        println!("== {label}: remerge {remerge_ms:.0} ms, extra VRAM {:.0} MiB (no requests given)", extra as f64 / 1048576.0);
        return Ok(());
    }

    let fresh = load_served(to_spec, to, base_spec, base, fresh, o)?;
    let re = load_served(to_spec, to, base_spec, base, re, o)?;
    let (pf, pr) = (run_all(&rt, &fresh, reqs), run_all(&rt, &re, reqs));

    let (mut n_req, mut n_ans, mut flips, mut decisive) = (0, 0, 0, 0);
    let mut deltas: Vec<f64> = Vec::new();
    for (f, r) in pf.iter().zip(&pr) {
        let (Some(f), Some(r)) = (f, r) else { continue };
        n_req += 1;
        for (f, r) in f.iter().zip(r).filter(|(f, r)| !f.is_empty() && f.len() == r.len()) {
            n_ans += 1;
            deltas.extend(f.iter().zip(r.iter()).map(|(a, b)| (a - b).abs()));
            if argmax(f) != argmax(r) {
                flips += 1;
                // a flip across a near-tie of the fresh answer is noise; a decisive one has a 0.1 lead there
                let mut s = f.clone();
                s.sort_by(|a, b| b.total_cmp(a));
                if s[0] - s.get(1).copied().unwrap_or(0.0) > 0.1 {
                    decisive += 1;
                }
            }
        }
    }
    deltas.sort_by(f64::total_cmp);
    let q = |p: f64| deltas.get(((deltas.len() as f64 * p) as usize).min(deltas.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let pct = |n: usize| if n_ans == 0 { 0.0 } else { 100.0 * n as f64 / n_ans as f64 };
    println!("== {label}: remerged {from_spec} -> {to_spec} vs fresh {to_spec}");
    println!("   requests            {n_req} answered of {} ({} skipped)", reqs.len(), reqs.len() - n_req);
    println!("   answers             {n_ans} ({} option probabilities)", deltas.len());
    println!("   max |dp|            {:.6}", q(1.0));
    println!("   p99 |dp|            {:.6}", q(0.99));
    println!("   mean |dp|           {:.6}", deltas.iter().sum::<f64>() / deltas.len().max(1) as f64);
    println!("   choice flips        {flips} ({:.2}%), {decisive} decisive (fresh lead > 0.1)", pct(flips));
    println!("   remerge wall        {remerge_ms:.0} ms");
    println!("   extra VRAM          {:.0} MiB (pool growth: new weights plus the transient f32 buffers)", extra as f64 / 1048576.0);
    Ok(())
}

fn main() -> Res<()> {
    let a = Args::parse();
    let dtype = match a.dtype.as_str() {
        "f32" | "fp32" => candle_core::DType::F32,
        _ => candle_core::DType::BF16,
    };
    let o = LoadOpts {
        dtype,
        devices: vec![a.device],
        serve: Opts { prefix_cache: 8, pass_tokens: 8192 },
        temperature: None,
    };
    // `--base` may be the Kev adapter that names the base, or the base itself
    let base_dir = resolve(&a.base)?;
    let (base_spec, base) = kev_base(&base_dir)?.unwrap_or((a.base.clone(), base_dir));
    let (da, db) = (resolve(&a.adapter_a)?, resolve(&a.adapter_b)?);
    let reqs = |specific: &Option<PathBuf>| -> Res<Vec<Value>> {
        match specific.as_ref().or(a.requests.as_ref()) {
            Some(p) => read_requests(p, a.limit),
            None if a.per_proj => Ok(Vec::new()), // weights only
            None => Err("give --requests, or --requests-a and --requests-b".into()),
        }
    };
    let mode = Mode { delta: a.delta, per_proj: a.per_proj };
    let (req_a, req_b) = (reqs(&a.requests_a)?, reqs(&a.requests_b)?);
    eprintln!(
        "fast_parity: base {base_spec}, KEV_W8={}, {}, {} A requests, {} B requests",
        std::env::var("KEV_W8").unwrap_or_default(),
        if a.delta { "remerge_delta" } else { "remerge from the base" },
        req_a.len(),
        req_b.len()
    );
    direction("A->B", &base_spec, &base, (&a.adapter_a, &da), (&a.adapter_b, &db), &req_b, mode, &o)?;
    direction("B->A", &base_spec, &base, (&a.adapter_b, &db), (&a.adapter_a, &da), &req_a, mode, &o)?;
    Ok(())
}
