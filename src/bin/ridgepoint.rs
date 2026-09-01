//! ridgepoint CLI — the fit-first surface (v0).
//!
//! usage: ridgepoint fit <model> [--gpu id:N] [--engine vllm|llamacpp]
//!                                [--dtype fp16|fp8|q4_k_m] [--ctx N] [--prompt N] [--json]

use ridgepoint::core::*;
use ridgepoint::{registry, render};

fn fail(msg: &str) -> ! {
    eprintln!("ridgepoint: {msg}");
    std::process::exit(2);
}

const USAGE: &str =
    "usage: ridgepoint fit <model> [--gpu id:N] [--engine vllm|llamacpp] [--dtype fp16|fp8|q4_k_m] [--kv-cache-dtype fp16|fp8] [--ctx N] [--prompt N] [--json]";

/// Value that must follow a flag at position `i`.
fn req(args: &[String], i: usize, flag: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| fail(&format!("missing value for {flag}")))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] != "fit" {
        fail(USAGE);
    }

    let mut model_id: Option<String> = None;
    let mut gpu = "a100-80gb:1".to_string();
    let mut engine = "vllm".to_string();
    let mut dtype = "fp16".to_string();
    let mut ctx: u32 = 4096;
    let mut prompt: u32 = 2048;
    let mut kv_bytes: u8 = 2; // KV-cache dtype, default fp16; independent of weight dtype
    let mut json = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => { i += 1; gpu = req(&args, i, "--gpu"); }
            "--engine" => { i += 1; engine = req(&args, i, "--engine"); }
            "--dtype" => { i += 1; dtype = req(&args, i, "--dtype"); }
            "--ctx" => {
                i += 1;
                ctx = req(&args, i, "--ctx").parse().unwrap_or_else(|_| fail("--ctx must be an integer"));
            }
            "--prompt" => {
                i += 1;
                prompt = req(&args, i, "--prompt").parse().unwrap_or_else(|_| fail("--prompt must be an integer"));
            }
            "--kv-cache-dtype" => {
                i += 1;
                kv_bytes = match req(&args, i, "--kv-cache-dtype").as_str() {
                    "fp16" | "bf16" => 2,
                    "fp8" => 1,
                    o => fail(&format!("unknown kv-cache-dtype {o}")),
                };
            }
            "--json" => json = true,
            s if s.starts_with("--") => fail(&format!("unknown flag {s}")),
            _ => {
                if model_id.is_none() {
                    model_id = Some(args[i].clone());
                } else {
                    fail(&format!("unexpected argument {}", args[i]));
                }
            }
        }
        i += 1;
    }

    let model_id = model_id.unwrap_or_else(|| fail("missing <model>"));
    let model = registry::model(&model_id).unwrap_or_else(|| fail(&format!("unknown model: {model_id}")));

    let (dev_id, count) = match gpu.split_once(':') {
        Some((d, n)) => (d.to_string(), n.parse::<u32>().unwrap_or_else(|_| fail("--gpu count must be an integer"))),
        None => (gpu.clone(), 1),
    };
    let device = registry::device(&dev_id).unwrap_or_else(|| fail(&format!("unknown gpu: {dev_id}")));
    let hw = DeviceSet { device, count };

    let quant = registry::quant(&dtype).unwrap_or_else(|| fail(&format!("unknown dtype: {dtype}")));
    let alloc: Box<dyn Allocator> = match engine.as_str() {
        "vllm" => Box::new(Vllm { util: 0.90 }),
        "llamacpp" | "llama.cpp" => Box::new(LlamaCpp),
        other => fail(&format!("unknown engine: {other}")),
    };

    let w = Workload { ctx, prompt_tokens: prompt, concurrency: None, kv_bytes };
    let report = fit(&model, &hw, quant.as_ref(), alloc.as_ref(), &w, &Calibration::uncalibrated());

    if json {
        println!("{}", render::json(&report));
    } else {
        print!("{}", render::human(&report));
    }
}
