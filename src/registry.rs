//! Built-in registry — popular models / devices / quants shipped as data, so
//! `ridgepoint fit llama-3-70b ...` works offline. Custom shapes via `--dims` later.
//!
//! NOTE (correctness item, §9): VRAM here is the marketing label taken as decimal
//! bytes. Real usable VRAM (GiB vs GB + driver reserve) is a calibration target.

use crate::core::*;

pub fn model(id: &str) -> Option<ModelShape> {
    // (id, layers, n_params, active_params, state, kv_kind)
    let (layers, n_params, active, state, kv_kind): (u32, u64, u64, StateGeometry, &'static str) = match id {
        // --- MHA (older dense) ---
        "llama-2-7b" => (32, 6_738_415_616, 6_738_415_616, StateGeometry::Mha { head_dim: 128, n_heads: 32 }, "MHA"),

        // --- GQA (modern dense) ---
        "llama-3-8b" => (32, 8_030_261_248, 8_030_261_248, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),
        "llama-3-70b" => (80, 70_553_706_496, 70_553_706_496, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),
        "mistral-7b" => (32, 7_241_732_096, 7_241_732_096, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),
        "qwen2.5-7b" => (28, 7_615_616_512, 7_615_616_512, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 4 }, "GQA"),
        "deepseek-r1-distill-14b" => (48, 14_770_033_664, 14_770_033_664, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA"),

        // --- GQA + MoE (memory holds all experts; decode reads active only) ---
        "mixtral-8x7b" => (32, 46_702_792_704, 12_900_000_000, StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 }, "GQA·MoE"),

        // --- MLA + MoE (DeepSeek): tiny KV per token despite 671B ---
        "deepseek-r1" => (61, 671_000_000_000, 37_000_000_000, StateGeometry::Mla { d_c: 512, d_rope: 64 }, "MLA·MoE"),

        _ => return None,
    };
    Some(ModelShape { id: id.into(), layers, n_params, active_params: active, state, kv_kind })
}

pub fn device(id: &str) -> Option<Device> {
    // (vram bytes, mem_bw B/s, peak fp16 FLOP/s, arch, interconnect)
    let (vram, bw, flops, arch, ic): (u64, u64, f64, &'static str, &'static str) = match id {
        // Ampere
        "a100-40gb" => (40_000_000_000, 1_555_000_000_000, 312e12, "Ampere", "NVLink3 600GB/s"),
        "a100-80gb" => (80_000_000_000, 2_039_000_000_000, 312e12, "Ampere", "NVLink3 600GB/s"),
        "rtx3090-24gb" => (24_000_000_000, 936_000_000_000, 71e12, "Ampere", "NVLink3 bridge"),
        // Hopper
        "h100-80gb" => (80_000_000_000, 3_350_000_000_000, 989e12, "Hopper", "NVLink4 900GB/s"),
        "h200-141gb" => (141_000_000_000, 4_800_000_000_000, 989e12, "Hopper", "NVLink4 900GB/s"),
        // Ada (no NVLink — PCIe only)
        "l40s-48gb" => (48_000_000_000, 864_000_000_000, 362e12, "Ada", "PCIe4"),
        "rtx4090-24gb" => (24_000_000_000, 1_008_000_000_000, 165e12, "Ada", "PCIe4"),
        // Blackwell (native FP8/FP4, huge HBM3e bandwidth)
        "b200-192gb" => (192_000_000_000, 8_000_000_000_000, 2250e12, "Blackwell", "NVLink5 1.8TB/s"),
        "gb200-192gb" => (192_000_000_000, 8_000_000_000_000, 2500e12, "Blackwell", "NVLink5 1.8TB/s"),
        "rtx5090-32gb" => (32_000_000_000, 1_792_000_000_000, 210e12, "Blackwell", "PCIe5"),
        // AMD
        "mi300x-192gb" => (192_000_000_000, 5_300_000_000_000, 1300e12, "CDNA3", "Infinity Fabric"),
        _ => return None,
    };
    Some(Device { name: id.into(), vram_bytes: vram, mem_bw_bytes_s: bw, peak_flops: flops, arch, interconnect: ic })
}

/// Full weight-quant table with REAL effective bytes-per-weight. GGUF k-quants carry
/// per-block scales/mins, so effective bpw exceeds the nominal bit count.
pub fn quant(name: &str) -> Option<Box<dyn QuantScheme>> {
    let (canon, bpw): (&'static str, f64) = match name {
        "fp16" => ("fp16", 2.0),
        "bf16" => ("bf16", 2.0),
        "fp8" => ("fp8", 1.0),
        "int8" => ("int8", 1.0),
        "q8_0" => ("q8_0", 8.50 / 8.0),
        "q6_k" => ("q6_k", 6.56 / 8.0),
        "q5_k_m" => ("q5_k_m", 5.67 / 8.0),
        "q4_k_m" | "q4km" => ("q4_k_m", 4.80 / 8.0), // real ≈ 4.8, not the "4.5" label
        "q4_0" => ("q4_0", 4.55 / 8.0),
        "q3_k_m" => ("q3_k_m", 3.91 / 8.0),
        "q2_k" => ("q2_k", 3.35 / 8.0),
        "awq" | "awq-int4" => ("awq-int4", 4.25 / 8.0),
        "gptq" | "gptq-int4" => ("gptq-int4", 4.16 / 8.0),
        _ => return None,
    };
    Some(Box::new(Quant::new(canon, bpw)))
}
