//! Built-in registry — popular models / devices / quants shipped as data, so
//! `ridgepoint fit llama-3-70b ...` works offline. Custom shapes via `--dims` later.
//!
//! NOTE (correctness item, §9): VRAM here is the marketing label taken as decimal
//! bytes. Real usable VRAM (GiB vs GB + driver reserve) is a calibration target.

use crate::core::*;

pub fn model(id: &str) -> Option<ModelShape> {
    match id {
        "llama-3-70b" => Some(ModelShape {
            id: id.into(),
            layers: 80,
            n_params: 70_553_706_496,
            state: StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 },
            kv_kind: "GQA",
        }),
        "deepseek-r1-distill-14b" => Some(ModelShape {
            id: id.into(),
            layers: 48,
            n_params: 14_770_033_664,
            state: StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 },
            kv_kind: "GQA",
        }),
        _ => None,
    }
}

pub fn device(id: &str) -> Option<Device> {
    match id {
        "a100-80gb" => Some(Device {
            name: id.into(),
            vram_bytes: 80_000_000_000,
            mem_bw_bytes_s: 2_039_000_000_000,
            peak_flops: 312e12,
        }),
        "rtx4090-24gb" => Some(Device {
            name: id.into(),
            vram_bytes: 24_000_000_000,
            mem_bw_bytes_s: 1_008_000_000_000,
            peak_flops: 165e12,
        }),
        _ => None,
    }
}

pub fn quant(name: &str) -> Option<Box<dyn QuantScheme>> {
    match name {
        "fp16" | "bf16" => Some(Box::new(Fp16)),
        "fp8" => Some(Box::new(Fp8)),
        "q4_k_m" | "q4km" => Some(Box::new(Q4KM)),
        _ => None,
    }
}
