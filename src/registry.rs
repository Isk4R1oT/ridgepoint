//! Built-in registry — popular models / devices / quants shipped as data, so
//! `ridgepoint fit llama-3-70b ...` works offline.
//!
//! VRAM is USABLE bytes (binary GiB minus a ~0.75 GiB driver reserve) — calibrated against what
//! vLLM actually reports (A100-80GB → 79.25 GiB usable, NOT 80e9 decimal). Quant bpw are REAL
//! (AWQ ≈ 4.5, Q4_K_M ≈ 4.8), some measured on hardware.

use crate::core::*;

/// Usable VRAM in bytes: nominal binary GiB minus a ~0.75 GiB driver/context reserve (measured).
fn usable(nominal_gib: f64) -> u64 {
    (nominal_gib * 1_073_741_824.0 - 0.8e9) as u64
}

pub fn model(id: &str) -> Option<ModelShape> {
    // (layers, d_model, n_params, active_params, state, kv_kind)
    let (layers, d_model, n_params, active, state, kv_kind): (u32, u32, u64, u64, StateGeometry, &'static str) = match id {
        // MHA (older dense)
        "llama-2-7b" => (32, 4096, 6_738_415_616, 6_738_415_616, StateGeometry::Mha { head_dim: 128, n_heads: 32 }, "MHA"),
        // GQA (modern dense)
        "llama-3-8b" => (32, 4096, 8_030_261_248, 8_030_261_248, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),
        "llama-3-70b" => (80, 8192, 70_553_706_496, 70_553_706_496, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),
        "mistral-7b" => (32, 4096, 7_241_732_096, 7_241_732_096, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),
        "qwen2.5-7b" => (28, 3584, 7_615_616_512, 7_615_616_512, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 4 }, "GQA"),
        "deepseek-r1-distill-14b" => (48, 5120, 14_770_033_664, 14_770_033_664, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),
        // GQA + MoE
        "mixtral-8x7b" => (32, 4096, 46_702_792_704, 12_900_000_000, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA·MoE"),
        // MLA + MoE (DeepSeek)
        "deepseek-v2-lite" => (27, 2048, 15_706_000_000, 2_400_000_000, StateGeometry::Mla { d_c: 512, d_rope: 64 }, "MLA·MoE"),
        "deepseek-r1" => (61, 7168, 671_000_000_000, 37_000_000_000, StateGeometry::Mla { d_c: 512, d_rope: 64 }, "MLA·MoE"),
        _ => return None,
    };
    Some(ModelShape { id: id.into(), layers, kv_layers: layers, d_model, n_params, active_params: active, state, kv_kind })
}

pub fn device(id: &str) -> Option<Device> {
    // (usable VRAM bytes, mem_bw B/s, peak fp16 FLOP/s, arch, interconnect)
    let (vram, bw, flops, arch, ic): (u64, u64, f64, &'static str, &'static str) = match id {
        "a100-40gb" => (usable(40.0), 1_555_000_000_000, 312e12, "Ampere", "NVLink3 600GB/s"),
        "a100-80gb" => (usable(80.0), 2_039_000_000_000, 312e12, "Ampere", "NVLink3 600GB/s"),
        "rtx3090-24gb" => (usable(24.0), 936_000_000_000, 71e12, "Ampere", "NVLink3 bridge"),
        "h100-80gb" => (usable(80.0), 3_350_000_000_000, 989e12, "Hopper", "NVLink4 900GB/s"),
        "h200-141gb" => (usable(141.0), 4_800_000_000_000, 989e12, "Hopper", "NVLink4 900GB/s"),
        "l40s-48gb" => (usable(48.0), 864_000_000_000, 362e12, "Ada", "PCIe4"),
        "rtx4090-24gb" => (usable(24.0), 1_008_000_000_000, 165e12, "Ada", "PCIe4"),
        "b200-192gb" => (usable(192.0), 8_000_000_000_000, 2250e12, "Blackwell", "NVLink5 1.8TB/s"),
        "gb200-192gb" => (usable(192.0), 8_000_000_000_000, 2500e12, "Blackwell", "NVLink5 1.8TB/s"),
        "rtx5090-32gb" => (usable(32.0), 1_792_000_000_000, 210e12, "Blackwell", "PCIe5"),
        "mi300x-192gb" => (usable(192.0), 5_300_000_000_000, 1300e12, "CDNA3", "Infinity Fabric"),
        _ => return None,
    };
    Some(Device { name: id.into(), vram_bytes: vram, mem_bw_bytes_s: bw, peak_flops: flops, arch, interconnect: ic })
}

/// Weight-quant table with REAL effective bytes-per-weight (some measured on hardware).
pub fn quant(name: &str) -> Option<Box<dyn QuantScheme>> {
    let (canon, bpw): (&'static str, f64) = match name {
        "fp16" => ("fp16", 2.0),
        "bf16" => ("bf16", 2.0),
        "fp8" => ("fp8", 1.0),
        "int8" => ("int8", 1.0),
        "q8_0" => ("q8_0", 8.50 / 8.0),
        "q6_k" => ("q6_k", 6.56 / 8.0),
        "q5_k_m" => ("q5_k_m", 5.67 / 8.0),
        "q4_k_m" | "q4km" => ("q4_k_m", 4.80 / 8.0),
        "q4_0" => ("q4_0", 4.55 / 8.0),
        "q3_k_m" => ("q3_k_m", 3.91 / 8.0),
        "q2_k" => ("q2_k", 3.35 / 8.0),
        "awq" | "awq-int4" => ("awq-int4", 4.51 / 8.0), // measured: 70.6B AWQ = 37.06 GiB
        "gptq" | "gptq-int4" => ("gptq-int4", 4.25 / 8.0),
        _ => return None,
    };
    Some(Box::new(Quant::new(canon, bpw)))
}
