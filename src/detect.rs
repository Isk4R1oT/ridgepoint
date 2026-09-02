//! System GPU auto-detection. Queries the local machine so the user need not type
//! hardware by hand. NVIDIA via `nvidia-smi`; graceful fallback when absent (e.g. macOS).
//!
//! nvidia-smi reports name + total memory but NOT bandwidth/FLOPs, so we map the
//! detected name onto a registry Device (authoritative specs) and only fall back to a
//! specs-unknown Device when the card is not in the registry.

use crate::core::Device;
use crate::registry;
use std::process::Command;

pub struct DetectedGpu {
    pub index: u32,
    pub name: String,
    pub vram_bytes: u64,
}

/// Enumerate NVIDIA GPUs on this system. Err (not panic) when none / no driver.
pub fn detect_gpus() -> Result<Vec<DetectedGpu>, String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let mut gpus = Vec::new();
            for line in text.lines() {
                let p: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if p.len() >= 3 {
                    let mib: u64 = p[2].parse().unwrap_or(0); // memory.total, MiB
                    gpus.push(DetectedGpu {
                        index: p[0].parse().unwrap_or(0),
                        name: p[1].to_string(),
                        vram_bytes: mib * 1024 * 1024,
                    });
                }
            }
            if gpus.is_empty() {
                Err("nvidia-smi returned no GPUs".into())
            } else {
                Ok(gpus)
            }
        }
        Ok(_) => Err("nvidia-smi exited with an error".into()),
        Err(_) => Err("nvidia-smi not found (no NVIDIA GPU or driver on this machine)".into()),
    }
}

/// Map a detected (name, VRAM) onto a registry Device for authoritative bw/FLOPs,
/// or a specs-unknown Device carrying only the reported VRAM.
pub fn match_device(name: &str, vram_bytes: u64) -> Device {
    let n = name.to_lowercase();
    let id = if n.contains("h200") {
        "h200-141gb"
    } else if n.contains("h100") {
        "h100-80gb"
    } else if n.contains("gb200") || n.contains("b200") {
        "b200-192gb"
    } else if n.contains("a100") && (n.contains("40") || vram_bytes < 60_000_000_000) {
        "a100-40gb"
    } else if n.contains("a100") {
        "a100-80gb"
    } else if n.contains("l40s") {
        "l40s-48gb"
    } else if n.contains("5090") {
        "rtx5090-32gb"
    } else if n.contains("4090") {
        "rtx4090-24gb"
    } else if n.contains("3090") {
        "rtx3090-24gb"
    } else if n.contains("mi300") {
        "mi300x-192gb"
    } else {
        ""
    };

    if let Some(mut d) = registry::device(id) {
        d.name = format!("{} (detected)", d.name);
        return d;
    }
    Device {
        name: format!("{} (detected, specs unknown)", name),
        vram_bytes,
        mem_bw_bytes_s: 0,
        peak_flops: 0.0,
        arch: "unknown",
        interconnect: "unknown",
    }
}
