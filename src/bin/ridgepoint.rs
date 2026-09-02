//! ridgepoint CLI — fit / scan / devices (v0).

use ridgepoint::core::*;
use ridgepoint::{detect, registry, render};

fn fail(msg: &str) -> ! {
    eprintln!("ridgepoint: {msg}");
    std::process::exit(2);
}

const USAGE: &str = "usage:
  ridgepoint fit  <model> [--gpu id:N|auto[:N]] [--engine vllm|llamacpp] [--dtype fp16|fp8|q4_k_m|...] [--kv-cache-dtype fp16|fp8] [--ctx N] [--prompt N] [--json]
  ridgepoint scan <model> [same flags; sweeps ctx, so --ctx is ignored]
  ridgepoint devices        detect local GPUs";

struct Opts {
    model: Option<String>,
    gpu: String,
    engine: String,
    dtype: String,
    kv_bytes: u8,
    ctx: u32,
    prompt: u32,
    json: bool,
}

fn req(args: &[String], i: usize, flag: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| fail(&format!("missing value for {flag}")))
}

fn parse(args: &[String]) -> Opts {
    let mut o = Opts {
        model: None,
        gpu: "a100-80gb:1".into(),
        engine: "vllm".into(),
        dtype: "fp16".into(),
        kv_bytes: 2,
        ctx: 4096,
        prompt: 2048,
        json: false,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => { i += 1; o.gpu = req(args, i, "--gpu"); }
            "--engine" => { i += 1; o.engine = req(args, i, "--engine"); }
            "--dtype" => { i += 1; o.dtype = req(args, i, "--dtype"); }
            "--kv-cache-dtype" => {
                i += 1;
                o.kv_bytes = match req(args, i, "--kv-cache-dtype").as_str() {
                    "fp16" | "bf16" => 2,
                    "fp8" => 1,
                    x => fail(&format!("unknown kv-cache-dtype {x}")),
                };
            }
            "--ctx" => { i += 1; o.ctx = req(args, i, "--ctx").parse().unwrap_or_else(|_| fail("--ctx must be an integer")); }
            "--prompt" => { i += 1; o.prompt = req(args, i, "--prompt").parse().unwrap_or_else(|_| fail("--prompt must be an integer")); }
            "--json" => o.json = true,
            s if s.starts_with("--") => fail(&format!("unknown flag {s}")),
            _ => {
                if o.model.is_none() {
                    o.model = Some(args[i].clone());
                } else {
                    fail(&format!("unexpected argument {}", args[i]));
                }
            }
        }
        i += 1;
    }
    o
}

/// Resolve `--gpu` into a DeviceSet. Supports "id:N" and "auto[:N]" (system detection).
fn resolve_hw(gpu: &str) -> DeviceSet {
    if let Some(rest) = gpu.strip_prefix("auto") {
        let want: Option<u32> = match rest.strip_prefix(':') {
            Some(n) => Some(n.parse().unwrap_or_else(|_| fail("--gpu auto:N count must be an integer"))),
            None if rest.is_empty() => None,
            None => fail("bad --gpu value (expected auto or auto:N)"),
        };
        let gpus = detect::detect_gpus()
            .unwrap_or_else(|e| fail(&format!("--gpu auto: {e}. Pass --gpu <id>:N instead (see `ridgepoint devices`).")));
        let available = gpus.len() as u32;
        let count = want.unwrap_or(available).clamp(1, available);
        let device = detect::match_device(&gpus[0].name, gpus[0].vram_bytes);
        DeviceSet { device, count }
    } else {
        let (id, n) = match gpu.split_once(':') {
            Some((d, n)) => (d.to_string(), n.parse::<u32>().unwrap_or_else(|_| fail("--gpu count must be an integer"))),
            None => (gpu.to_string(), 1),
        };
        let device = registry::device(&id)
            .unwrap_or_else(|| fail(&format!("unknown gpu: {id} (try `ridgepoint devices`, or --gpu auto)")));
        DeviceSet { device, count: n }
    }
}

fn build(o: &Opts) -> (ModelShape, DeviceSet, Box<dyn QuantScheme>, Box<dyn Allocator>) {
    let model_id = o.model.clone().unwrap_or_else(|| fail("missing <model>"));
    let model = registry::model(&model_id).unwrap_or_else(|| fail(&format!("unknown model: {model_id}")));
    let hw = resolve_hw(&o.gpu);
    let quant = registry::quant(&o.dtype).unwrap_or_else(|| fail(&format!("unknown dtype: {}", o.dtype)));
    let alloc: Box<dyn Allocator> = match o.engine.as_str() {
        "vllm" => Box::new(Vllm { util: 0.90 }),
        "llamacpp" | "llama.cpp" => Box::new(LlamaCpp),
        x => fail(&format!("unknown engine: {x}")),
    };
    (model, hw, quant, alloc)
}

fn cmd_fit(args: &[String]) {
    let o = parse(args);
    let (m, hw, q, e) = build(&o);
    let w = Workload { ctx: o.ctx, prompt_tokens: o.prompt, concurrency: None, kv_bytes: o.kv_bytes };
    let r = fit(&m, &hw, q.as_ref(), e.as_ref(), &w, &Calibration::uncalibrated());
    if o.json { println!("{}", render::json(&r)); } else { print!("{}", render::human(&r)); }
}

fn cmd_scan(args: &[String]) {
    let o = parse(args);
    let (m, hw, q, e) = build(&o);
    let ctxs = [512u32, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
    let r = scan(&m, &hw, q.as_ref(), e.as_ref(), o.prompt, o.kv_bytes, &ctxs, &Calibration::uncalibrated());
    if o.json { println!("{}", render::scan_json(&r)); } else { print!("{}", render::scan_human(&r)); }
}

fn cmd_devices() {
    match detect::detect_gpus() {
        Ok(gpus) => {
            println!("detected {} GPU(s):", gpus.len());
            for g in &gpus {
                let d = detect::match_device(&g.name, g.vram_bytes);
                println!(
                    "  [{}] {}  ({:.0} GB)  → {} · {} · {}",
                    g.index, g.name, g.vram_bytes as f64 / 1e9, d.name, d.arch, d.interconnect
                );
            }
            println!("\nuse:  --gpu auto   (all)   ·   --gpu auto:N   (N of them)");
        }
        Err(e) => {
            eprintln!("no GPUs detected: {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("fit") => cmd_fit(&args[1..]),
        Some("scan") => cmd_scan(&args[1..]),
        Some("devices") => cmd_devices(),
        _ => fail(USAGE),
    }
}
