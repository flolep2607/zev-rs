//! `zev serve --model <dir | hub id[@revision]>`: find the checkpoint (the local Hugging Face cache, downloading what
//! is missing), tell its family from the files it ships, and load it on each device.
//!
//! - head.pt / head.safetensors next to an adapter: Kev (the base named by the head's metadata, LoRA merged; several
//!   Kev adapters on the same base share one copy of it, each LoRA unmerged: `load_all`);
//! - pointer_head.safetensors next to backbone/: a pre-merged Kev (NeoHorse);
//! - decider_config.json: decider (and its fine-tunes);
//! - jevk5_config.json: JevK5.

use super::entrant::{Decider, Entrant, Intern, JevK5, Winnow};
use super::model::{read_json, Model};
use super::readout::{LabelHead, PointerHead, Readout};
use super::serve::{spawn, Opts, Served};
use super::Encoder;
use candle_core::{DType, Device};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct LoadOpts {
    pub dtype: DType,
    pub devices: Vec<usize>,
    pub serve: Opts,
    pub temperature: Option<f64>,
}

fn hub_dir() -> PathBuf {
    if let Ok(d) = std::env::var("HF_HUB_CACHE") {
        return d.into();
    }
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface")
        });
    home.join("hub")
}

/// A directory, or `org/name[@revision]` in the local Hugging Face cache (downloaded when missing).
pub fn resolve(spec: &str) -> Res<PathBuf> {
    if Path::new(spec).is_dir() {
        return Ok(spec.into());
    }
    let (repo, rev) = spec.split_once('@').unwrap_or((spec, "main"));
    if repo.contains("..") || rev.contains('/') || rev.contains('\\') || rev.contains("..") {
        return Err(format!("invalid model spec: {spec}").into());
    }
    let root = hub_dir().join(format!("models--{}", repo.replace('/', "--")));
    let sha = std::fs::read_to_string(root.join("refs").join(rev))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| rev.to_string());
    if let Ok(rd) = std::fs::read_dir(root.join("snapshots")) {
        let mut hits: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(&sha))
            .map(|e| e.path())
            .collect();
        hits.sort();
        if let Some(h) = hits.pop() {
            return Ok(h);
        }
    }
    download(repo, rev, &root)
}

/// The files a checkpoint needs to serve (weights, configs, tokenizer, heads), from huggingface.co.
fn download(repo: &str, rev: &str, root: &Path) -> Res<PathBuf> {
    let token = std::env::var("HF_TOKEN").ok();
    let get = |url: &str| {
        let mut r = ureq::get(url);
        if let Some(t) = &token {
            r = r.header("Authorization", &format!("Bearer {t}"));
        }
        r.call()
    };
    let info: Value = get(&format!(
        "https://huggingface.co/api/models/{repo}/revision/{rev}"
    ))?
    .body_mut()
    .read_json()?;
    let sha = info["sha"].as_str().ok_or("hub: no sha")?.to_string();
    let snap = root.join("snapshots").join(&sha);
    let wanted = |f: &str| {
        let top = !f.contains('/') || f.starts_with("backbone/");
        (top && [
            ".json",
            ".safetensors",
            ".txt",
            ".jinja",
            ".model",
            ".pt",
            ".py",
        ]
        .iter()
        .any(|x| f.ends_with(x)))
            || (f.starts_with("gguf/") && f.ends_with("Q8_0.gguf"))
    };
    for s in info["siblings"].as_array().into_iter().flatten() {
        let f = s["rfilename"].as_str().unwrap_or("");
        // Validate against path traversal or absolute destinations
        if f.is_empty() || f.contains("..") || Path::new(f).is_absolute() {
            continue;
        }
        let dst = snap.join(f);
        if !wanted(f) || dst.exists() {
            continue;
        }
        eprintln!("zev: downloading {repo}/{f}");
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = dst.with_extension("part");
        let mut resp = get(&format!("https://huggingface.co/{repo}/resolve/{sha}/{f}"))?;
        let mut out = std::fs::File::create(&tmp)?;
        std::io::copy(&mut resp.body_mut().as_reader(), &mut out)?;
        std::fs::rename(&tmp, &dst)?;
    }
    std::fs::create_dir_all(root.join("refs"))?;
    if !rev.chars().all(|c| c.is_ascii_hexdigit()) {
        std::fs::write(root.join("refs").join(rev), &sha)?;
    }
    Ok(snap)
}

fn tokenizer(dir: &Path) -> Res<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| format!("{}: {e}", dir.display()).into())
}

fn device(i: usize) -> Res<Device> {
    let dev = if cfg!(feature = "cuda") {
        Device::new_cuda(i)?
    } else {
        Device::Cpu
    };
    #[cfg(feature = "cuda")]
    super::kernels::tune(&dev)?;
    Ok(dev)
}

/// Load `spec` once per device and serve it under its own names (plus `spec` itself).
pub fn load(spec: &str, o: &LoadOpts) -> Res<Served> {
    let dir = resolve(spec)?;
    let t0 = std::time::Instant::now();
    let dtype = format!("{:?}", o.dtype).to_lowercase();
    let has = |f: &str| dir.join(f).exists();
    let mut workers = Vec::new();
    let (entrant, mut names, card) = if let Some((base_spec, base)) = kev_base(&dir)? {
        // Kev: the base the head was trained on, with the adapter merged
        let mut models = Vec::new();
        for &d in &o.devices {
            models.push(Model::load(&base, Some(&dir), o.dtype, &device(d)?)?);
        }
        return kev_served(spec, &dir, &base_spec, &base, models, "merged", o, t0);
    } else if has("pointer_head.safetensors") && has("backbone/config.json") {
        let meta = PointerHead::load(&dir, o.temperature, &Device::Cpu)?;
        for &d in &o.devices {
            let dev = device(d)?;
            let m = Model::load(&dir.join("backbone"), None, o.dtype, &dev)?;
            workers.push((
                m,
                Readout::Pointer(PointerHead::load(&dir, o.temperature, &dev)?),
            ));
        }
        let card = json!({"description": "Kev pointer head on a merged backbone, Candle backend", "backend": "candle", "dtype": dtype,
                          "temperature": meta.temperature});
        let tok = if has("tokenizer.json") {
            tokenizer(&dir)?
        } else {
            tokenizer(&dir.join("backbone"))?
        };
        (
            Entrant::Kev(Encoder::new(tok)?),
            vec!["kev-latest".to_string()],
            card,
        )
    } else if has("gguf")
        && read_json(&dir.join("config.json")).is_ok_and(|c| c.to_string().contains("gemma4"))
    {
        // winnow: a Gemma 4 GGUF (Q8_0) with the repo's config.json and tokenizer.json. Its published form is this GGUF
        // under llama.cpp, so it runs in llama.cpp's precision whatever --dtype says: f32 activations, q8_1 x Q8_0
        // products, f16 attention.
        for &d in &o.devices {
            let dev = device(d)?;
            let m = Model::load(&dir, None, DType::F32, &dev)?;
            let softcap = m.final_softcap;
            workers.push((m, Readout::Labels(LabelHead { softcap, f32: true })));
        }
        let cfg = read_json(&dir.join("config.json"))?;
        let bos = cfg
            .pointer("/text_config/bos_token_id")
            .or(cfg.get("bos_token_id"))
            .and_then(Value::as_u64)
            .unwrap_or(2) as u32;
        let cap = std::env::var("ZEV_WINNOW_LABELS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let name = dir
            .file_name()
            .map_or("winnow".into(), |n| n.to_string_lossy().to_string());
        let card = json!({"description": format!("Winnow letter readout ({cap} labels) on Gemma 4 Q8_0, Candle backend"), "backend": "candle", "dtype": "q8_0 x q8_1, f32"});
        (
            Entrant::Winnow(Winnow::new(tokenizer(&dir)?, bos, cap, spec.to_string())?),
            vec![name],
            card,
        )
    } else if has("decider_config.json") || has("jevk5_config.json") || intern_meta(&dir).is_some()
    {
        for &d in &o.devices {
            let dev = device(d)?;
            let m = Model::load(&dir, None, o.dtype, &dev)?;
            let softcap = m.final_softcap;
            workers.push((
                m,
                Readout::Labels(LabelHead {
                    softcap,
                    f32: false,
                }),
            ));
        }
        if let Some((name, t)) = intern_meta(&dir) {
            let t = o.temperature.unwrap_or(t);
            let card = json!({"description": "Intern-Decision marker readout, Candle backend", "backend": "candle", "dtype": dtype, "temperature": t});
            (
                Entrant::Intern(Intern::new(tokenizer(&dir)?, name.clone(), t)?),
                vec![name],
                card,
            )
        } else if has("decider_config.json") {
            let cfg = read_json(&dir.join("decider_config.json"))?;
            let name = format!("decider-{}", cfg["version"].as_str().unwrap_or("dev"));
            let card = json!({"description": format!("decider label readout, Candle backend ({})", cfg["version"].as_str().unwrap_or("")),
                              "backend": "candle", "dtype": dtype, "temperature": cfg["temperature"], "temperature_by_type": cfg["temperature_by_type"]});
            (
                Entrant::Decider(Decider::new(tokenizer(&dir)?, &cfg, name.clone())?),
                vec![name],
                card,
            )
        } else {
            let cfg = read_json(&dir.join("jevk5_config.json"))?;
            let card = json!({"description": "JevK5 letter readout with knockout, Candle backend", "backend": "candle", "dtype": dtype,
                              "temperature": cfg["temperature"], "knockout_temperature": cfg["knockout_temperature"]});
            (
                Entrant::JevK5(JevK5::new(tokenizer(&dir)?, &cfg, spec.to_string())?),
                vec!["jevk5".to_string()],
                card,
            )
        }
    } else {
        return Err(format!("{}: cannot tell the model family (no head.pt, decider_config.json or jevk5_config.json)", dir.display()).into());
    };
    names.push(spec.to_string());
    eprintln!(
        "zev: loaded {spec} ({}) x{} in {:.1}s as {names:?}",
        dir.display(),
        workers.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(spawn(workers, entrant, names, o.serve.clone(), card))
}

/// A Kev checkpoint (a pointer head next to an adapter): the base the head was trained on, as a spec and a local dir.
fn kev_base(dir: &Path) -> Res<Option<(String, PathBuf)>> {
    let has = |f: &str| dir.join(f).exists();
    if !((has("head.pt") || has("head.safetensors")) && has("adapter_config.json")) {
        return Ok(None);
    }
    let head_meta = super::convert::head_pt(&dir.join("head.pt"))
        .map(|(_, m)| m)
        .unwrap_or_default();
    let acfg = read_json(&dir.join("adapter_config.json"))?;
    let base_name = head_meta["base"]
        .as_str()
        .or(acfg["base_model_name_or_path"].as_str())
        .ok_or("kev: no base model named")?
        .to_string();
    let base_spec = match head_meta["base_revision"].as_str() {
        Some(r) => format!("{base_name}@{r}"),
        None => base_name,
    };
    let base = resolve(&base_spec)?;
    Ok(Some((base_spec, base)))
}

/// Serve Kev backbones (one per device, `--devices` order) with `dir`'s pointer head.
#[allow(clippy::too_many_arguments)]
fn kev_served(
    spec: &str,
    dir: &Path,
    base_spec: &str,
    base: &Path,
    models: Vec<Model>,
    how: &str,
    o: &LoadOpts,
    t0: std::time::Instant,
) -> Res<Served> {
    let meta = PointerHead::load(dir, o.temperature, &Device::Cpu)?;
    let mut workers = Vec::new();
    for m in models {
        let head = PointerHead::load(dir, o.temperature, &m.dev)?;
        workers.push((m, Readout::Pointer(head)));
    }
    let dtype = format!("{:?}", o.dtype).to_lowercase();
    let card = json!({"description": format!("Kev pointer head on {base_spec} (LoRA {how}), Candle backend, temperature {:.2}", meta.temperature),
                      "backend": "candle", "dtype": dtype, "temperature": meta.temperature});
    let names = vec!["kev-latest".to_string(), "jev-latest".to_string(), spec.to_string()];
    eprintln!(
        "zev: loaded {spec} ({}) x{} in {:.1}s as {names:?}",
        dir.display(),
        workers.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(spawn(workers, Entrant::Kev(Encoder::new(tokenizer(base)?)?), names, o.serve.clone(), card))
}

/// Every `--model`, in order (the first answers unknown names). Kev adapters that share a base load it once per device:
/// the first of them is merged into it (fused int8 path, full speed), each other one attaches unmerged as its own LoRA
/// minus the merged one's (`Model::with_adapter`), exact. N adapters cost one base plus N-1 small (A, B) sets against N
/// merged backbones. List the busiest adapter first.
pub fn load_all(specs: &[String], o: &LoadOpts) -> Res<Vec<Served>> {
    let dirs: Vec<PathBuf> = specs.iter().map(|s| resolve(s)).collect::<Res<_>>()?;
    let bases: Vec<Option<(String, PathBuf)>> = dirs.iter().map(|d| kev_base(d)).collect::<Res<_>>()?;
    let shared = |b: &Path| bases.iter().flatten().filter(|(_, x)| x == b).count() > 1;
    let mut loaded: Vec<(PathBuf, PathBuf, Vec<Model>)> = Vec::new(); // per base dir: the merged adapter, its model per device
    let mut out = Vec::new();
    for ((spec, dir), base) in specs.iter().zip(&dirs).zip(&bases) {
        let t0 = std::time::Instant::now();
        match base {
            Some((base_spec, b)) if shared(b) => {
                // the first adapter on a base is merged into it (its requests take the fused int8 path); the others
                // attach as exact deltas against it
                match loaded.iter().find(|(x, _, _)| x == b) {
                    None => {
                        let mut ms = Vec::new();
                        for &d in &o.devices {
                            ms.push(Model::load(b, Some(dir), o.dtype, &device(d)?)?);
                        }
                        out.push(kev_served(spec, dir, base_spec, b, ms.clone(), "merged, base shared", o, t0)?);
                        loaded.push((b.clone(), dir.clone(), ms));
                    }
                    Some((_, hot, ms)) => {
                        let models = ms
                            .iter()
                            .map(|m| m.with_adapter(b, dir, Some(hot)))
                            .collect::<candle_core::Result<Vec<_>>>()?;
                        out.push(kev_served(spec, dir, base_spec, b, models, "unmerged delta on a shared base", o, t0)?);
                    }
                }
            }
            _ => out.push(load(spec, o)?),
        }
    }
    Ok(out)
}

/// Intern-Decision ships its runtime as inference.py: its MODEL_NAME and DEFAULT_TEMPERATURE.
fn intern_meta(dir: &Path) -> Option<(String, f64)> {
    let src = std::fs::read_to_string(dir.join("inference.py")).ok()?;
    let get = |key: &str| {
        src.lines().find_map(|l| {
            l.strip_prefix(key)?
                .trim()
                .strip_prefix('=')
                .map(|v| v.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
        })
    };
    let name = get("MODEL_NAME")?;
    name.starts_with("Intern-Decision").then(|| {
        (
            name,
            get("DEFAULT_TEMPERATURE")
                .and_then(|t| t.parse().ok())
                .unwrap_or(1.0),
        )
    })
}

/// The pre-`--model` form: `--kev <adapter dir> --base <base dir> [--head <dir>]`.
pub fn load_kev(kev: &Path, base: &Path, head: &Path, o: &LoadOpts) -> Res<Served> {
    let t0 = std::time::Instant::now();
    let mut workers = Vec::new();
    let mut temp = 1.0;
    for &d in &o.devices {
        let dev = device(d)?;
        let m = Model::load(base, Some(kev), o.dtype, &dev)?;
        let h = PointerHead::load(head, o.temperature, &dev)?;
        temp = h.temperature;
        workers.push((m, Readout::Pointer(h)));
    }
    eprintln!(
        "kev: loaded {} x{} in {:.1}s, temperature {temp:.4}",
        kev.display(),
        workers.len(),
        t0.elapsed().as_secs_f64()
    );
    let card = json!({
        "description": format!("Kev pointer head on {}, Candle backend, temperature {:.2}", base.display(), temp),
        "release_date": "2026-09-25", "backend": "candle", "dtype": format!("{:?}", o.dtype).to_lowercase(), "temperature": temp,
    });
    Ok(spawn(
        workers,
        Entrant::Kev(Encoder::new(tokenizer(base)?)?),
        vec!["kev-latest".into(), "jev-latest".into()],
        o.serve.clone(),
        card,
    ))
}
