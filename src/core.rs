//! ridgepoint core — the invariant memory / roofline model.
//!
//! The core encodes ONLY what is true for any model and any engine:
//!   I1 conservation, I2 additive per-layer state, I3 roofline.
//! Everything model/engine/quant-specific is an extension trait
//! (`StateGeometry`, `QuantScheme`, `Allocator`). `fit` never names a concrete
//! engine — it takes `&dyn Allocator`. That signature is the P1/P2-share guarantee.

/// An answer that is not a single number.
///
/// Deterministic quantities are `exact` (low == best == high). Quantities that
/// depend on a measured coefficient carry a low..high band; `calibrated` says
/// whether that band was fit to real hardware yet (G1). This is the
/// "interval, not point" principle made structural.
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub low: f64,
    pub best: f64,
    pub high: f64,
    pub exact: bool,
    pub calibrated: bool,
}

impl Interval {
    pub fn exact(v: f64) -> Self {
        Interval { low: v, best: v, high: v, exact: true, calibrated: true }
    }
    pub fn band(low: f64, best: f64, high: f64, calibrated: bool) -> Self {
        Interval { low, best, high, exact: false, calibrated }
    }
}

// ---------------------------------------------------------------------------
// Invariant data
// ---------------------------------------------------------------------------

/// Minimal model description needed for v0 memory + speed. Fuller geometry
/// (d_model, n_heads, vocab) can be added without touching the core math.
#[derive(Clone, Debug)]
pub struct ModelShape {
    pub id: String,
    pub layers: u32,
    pub n_params: u64,
    pub state: StateGeometry,
    pub kv_kind: &'static str, // display label: "GQA" / "MLA" / ...
}

#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub vram_bytes: u64,
    pub mem_bw_bytes_s: u64,
    pub peak_flops: f64,
}

#[derive(Clone, Debug)]
pub struct DeviceSet {
    pub device: Device,
    pub count: u32,
}

impl DeviceSet {
    pub fn total_vram(&self) -> u64 { self.device.vram_bytes * self.count as u64 }
    pub fn total_bw(&self) -> f64 { self.device.mem_bw_bytes_s as f64 * self.count as f64 }
    pub fn total_flops(&self) -> f64 { self.device.peak_flops * self.count as f64 }
}

/// The operating point. `concurrency = None` means "solve for the maximum".
/// `kv_bytes` = KV-cache element size (2 = fp16, 1 = fp8) — a serving decision
/// independent of the weight quant.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    pub ctx: u32,
    pub prompt_tokens: u32,
    pub concurrency: Option<u32>,
    pub kv_bytes: u8,
}

/// The ONLY place measured inputs enter the model. Uncalibrated defaults are wide
/// bands with `calibrated = false`; RunPod (G1) replaces them with fit values.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    pub overhead: Interval, // non-KV runtime overhead (activations, CUDA graphs), bytes
    pub mfu: Interval,      // prefill compute efficiency (0..1)
    pub mbu: Interval,      // decode memory-bandwidth efficiency (0..1)
}

impl Calibration {
    pub fn uncalibrated() -> Self {
        Calibration {
            overhead: Interval::band(2.5e9, 3.0e9, 4.0e9, false),
            mfu: Interval::band(0.30, 0.40, 0.50, false),
            mbu: Interval::band(0.60, 0.70, 0.85, false),
        }
    }
}

// ---------------------------------------------------------------------------
// E1 — StateGeometry: per-layer state as a FUNCTION of sequence length n.
// Linear for attention, constant for SSM. Invariant I2: never hardcode "linear".
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum StateGeometry {
    Mha { head_dim: u32, n_heads: u32 },
    Gqa { head_dim: u32, n_kv_heads: u32 },
    Mla { d_c: u32, d_rope: u32 },
    Ssm { state_bytes_per_layer: u64 },
}

impl StateGeometry {
    /// Bytes of attention/recurrent state for ONE layer at sequence length `n`.
    ///
    /// `state_bytes()` multiplies this by the layer count — do NOT multiply by
    /// layers here. `kv_bytes` is the KV element size (2 = fp16, 1 = fp8).
    ///
    /// Branches to implement:
    ///   Mha { head_dim, n_heads }    → K and V, per head, per token:
    ///                                   2 * n_heads * head_dim * n * kv_bytes
    ///   Gqa { head_dim, n_kv_heads } → same, but only n_kv_heads KV heads:
    ///                                   2 * n_kv_heads * head_dim * n * kv_bytes
    ///   Mla { d_c, d_rope }          → one compressed latent + rope key per token,
    ///                                   no per-head blow-up: (d_c + d_rope) * n * kv_bytes
    ///   Ssm { state_bytes_per_layer } → recurrent state is CONSTANT in n:
    ///                                   state_bytes_per_layer
    ///
    /// The leading `2` in MHA/GQA is "K and V". GQA swaps n_heads → n_kv_heads;
    /// MLA drops the per-head factor entirely (that is the ~50–100× win).
    pub fn layer_state_bytes(&self, n: u64, kv_bytes: u8) -> u64 {
        let kb = kv_bytes as u64;
        match *self {
            // K and V, one entry per attention head per token.
            StateGeometry::Mha { head_dim, n_heads } => {
                2 * n_heads as u64 * head_dim as u64 * n * kb
            }
            // GQA: KV is shared across query heads → only n_kv_heads store it.
            StateGeometry::Gqa { head_dim, n_kv_heads } => {
                2 * n_kv_heads as u64 * head_dim as u64 * n * kb
            }
            // MLA: one compressed latent + one rope key per token, no per-head factor.
            StateGeometry::Mla { d_c, d_rope } => {
                (d_c as u64 + d_rope as u64) * n * kb
            }
            // SSM: recurrent state is constant in n (invariant I2: fn of n, not linear).
            StateGeometry::Ssm { state_bytes_per_layer } => state_bytes_per_layer,
        }
    }
}

// ---------------------------------------------------------------------------
// E2 — QuantScheme: REAL bytes per weight (not the marketing label).
// ---------------------------------------------------------------------------

/// Weight quantization ONLY. The KV-cache dtype is a SEPARATE axis (`Workload::kv_bytes`),
/// exactly as serving engines expose it: vLLM's `--kv-cache-dtype` is independent of the
/// weight quant, and configs like "fp4 weights + fp16 KV" are common in practice.
pub trait QuantScheme {
    fn bytes_per_weight(&self) -> f64;
    fn name(&self) -> &str;
}

pub struct Fp16;
impl QuantScheme for Fp16 {
    fn bytes_per_weight(&self) -> f64 { 2.0 }
    fn name(&self) -> &str { "fp16" }
}

pub struct Fp8;
impl QuantScheme for Fp8 {
    fn bytes_per_weight(&self) -> f64 { 1.0 }
    fn name(&self) -> &str { "fp8" }
}

/// GGUF Q4_K_M — REAL ≈ 4.8 bits/weight, not the "4.5" label.
pub struct Q4KM;
impl QuantScheme for Q4KM {
    fn bytes_per_weight(&self) -> f64 { 4.8 / 8.0 }
    fn name(&self) -> &str { "q4_k_m" }
}

// ---------------------------------------------------------------------------
// E3 — Allocator: the VRAM partition policy. The ONLY thing that differs P1↔P2.
// ---------------------------------------------------------------------------

pub trait Allocator {
    /// KV pool available after this engine's allocation, as an interval (OOM honesty).
    fn kv_pool_bytes(&self, total_vram: u64, weights: u64, overhead: Interval) -> Interval;
    fn name(&self) -> &str;
    /// vLLM-style pre-grab utilisation, if any (for display).
    fn util(&self) -> Option<f64> { None }
    fn managed_pool(&self, total_vram: u64) -> Option<u64> {
        self.util().map(|u| (total_vram as f64 * u) as u64)
    }
}

/// vLLM pre-reserves `util × VRAM` at startup and pages KV into that pool.
pub struct Vllm {
    pub util: f64,
}
impl Allocator for Vllm {
    fn kv_pool_bytes(&self, total_vram: u64, weights: u64, overhead: Interval) -> Interval {
        let grab = total_vram as f64 * self.util;
        let w = weights as f64;
        // low pool ⇐ high overhead ; high pool ⇐ low overhead
        Interval::band(
            grab - w - overhead.high,
            grab - w - overhead.best,
            grab - w - overhead.low,
            overhead.calibrated,
        )
    }
    fn name(&self) -> &str { "vLLM" }
    fn util(&self) -> Option<f64> { Some(self.util) }
}

/// llama.cpp is incremental — no big pre-grab; usable ≈ VRAM − weights − overhead.
pub struct LlamaCpp;
impl Allocator for LlamaCpp {
    fn kv_pool_bytes(&self, total_vram: u64, weights: u64, overhead: Interval) -> Interval {
        let v = total_vram as f64;
        let w = weights as f64;
        Interval::band(v - w - overhead.high, v - w - overhead.best, v - w - overhead.low, overhead.calibrated)
    }
    fn name(&self) -> &str { "llama.cpp" }
}

// ---------------------------------------------------------------------------
// Invariant core functions — depend only on the traits above.
// ---------------------------------------------------------------------------

pub fn weights_bytes(m: &ModelShape, q: &dyn QuantScheme) -> u64 {
    (m.n_params as f64 * q.bytes_per_weight()).round() as u64
}

/// Σ over layers of per-layer state at sequence length `n` (invariant I2).
/// `kv_bytes` is the KV-cache element size — independent of the weight quant.
pub fn state_bytes(m: &ModelShape, n: u64, kv_bytes: u8) -> u64 {
    m.state.layer_state_bytes(n, kv_bytes) * m.layers as u64
}

/// TTFT — prefill is compute-bound (right of the ridge point): 2·N·P / (flops·MFU).
pub fn ttft_s(m: &ModelShape, hw: &DeviceSet, w: &Workload, c: &Calibration) -> Interval {
    let work = 2.0 * m.n_params as f64 * w.prompt_tokens as f64; // fwd FLOPs ≈ 2·params·tokens
    let flops = hw.total_flops();
    // higher MFU ⇒ lower time
    Interval::band(
        work / (flops * c.mfu.high),
        work / (flops * c.mfu.best),
        work / (flops * c.mfu.low),
        c.mfu.calibrated,
    )
}

/// Decode tok/s per request — memory-bound (left of ridge): (bw·MBU)/(weights + active_kv).
pub fn decode_tok_s(m: &ModelShape, hw: &DeviceSet, q: &dyn QuantScheme, w: &Workload, c: &Calibration) -> Interval {
    let read = (weights_bytes(m, q) + state_bytes(m, w.ctx as u64, w.kv_bytes)) as f64; // bytes read per token
    let bw = hw.total_bw();
    Interval::band(bw * c.mbu.low / read, bw * c.mbu.best / read, bw * c.mbu.high / read, c.mbu.calibrated)
}

/// Batch at which decode flips memory-bound → compute-bound (the ridge point):
/// B* = (peak_flops / bandwidth) · weights_bytes / (2·params).
pub fn ridge_batch(m: &ModelShape, hw: &DeviceSet, q: &dyn QuantScheme) -> f64 {
    (hw.total_flops() / hw.total_bw()) * (weights_bytes(m, q) as f64) / (2.0 * m.n_params as f64)
}

pub struct FitReport {
    pub model: ModelShape,
    pub hw: DeviceSet,
    pub quant_name: String,
    pub engine_name: String,
    pub util: Option<f64>,
    pub total_vram: u64,
    pub managed_pool: Option<u64>,
    pub weights: u64,
    pub overhead: Interval,
    pub kv_pool: Interval,
    pub kv_per_token: u64,
    pub bytes_per_seq: u64,
    pub kv_bytes: u8,
    pub ctx: u32,
    pub max_seqs: (i64, i64, i64),
    pub serves: bool,
    pub naive_would_say: bool,
    pub ttft: Interval,
    pub decode: Interval,
    pub ridge_batch: f64,
    pub recommendations: Vec<(String, String)>,
    pub not_modeled: Vec<&'static str>,
    pub calibrated: bool,
}

/// The flagship path: engine-aware fit + capacity + speed. Never names an engine.
pub fn fit(
    m: &ModelShape,
    hw: &DeviceSet,
    q: &dyn QuantScheme,
    e: &dyn Allocator,
    w: &Workload,
    c: &Calibration,
) -> FitReport {
    let total_vram = hw.total_vram();
    let weights = weights_bytes(m, q);
    let kv_pool = e.kv_pool_bytes(total_vram, weights, c.overhead);

    let kv_per_token = state_bytes(m, 1, w.kv_bytes);
    let bytes_per_seq = state_bytes(m, w.ctx as u64, w.kv_bytes);
    let seqs = |pool: f64| -> i64 {
        if bytes_per_seq == 0 { 0 } else { (pool / bytes_per_seq as f64).floor() as i64 }
    };
    let max_seqs = (seqs(kv_pool.low).max(0), seqs(kv_pool.best).max(0), seqs(kv_pool.high).max(0));
    let serves = max_seqs.1 >= 1;
    let naive_would_say = (weights as f64) < (total_vram as f64);

    let ttft = ttft_s(m, hw, w, c);
    let decode = decode_tok_s(m, hw, q, w, c);
    let rb = ridge_batch(m, hw, q);

    // v0 recommendations: only when it won't serve.
    let mut recs: Vec<(String, String)> = Vec::new();
    if !serves {
        // Only the WEIGHTS shrink under fp8; the KV cache keeps the user's kv_bytes.
        let fp8 = Fp8;
        let w8 = weights_bytes(m, &fp8);
        let pool8 = e.kv_pool_bytes(total_vram, w8, c.overhead);
        let seq8 = state_bytes(m, w.ctx as u64, w.kv_bytes);
        let n8 = if seq8 == 0 { 0 } else { (pool8.best / seq8 as f64).floor() as i64 };
        if n8 >= 1 {
            recs.push(("--dtype fp8".to_string(), format!("~{} seqs @{}k", n8, w.ctx / 1024)));
        }
        for extra in [2u32, 4u32, 8u32] {
            if extra <= hw.count { continue; }
            let hw2 = DeviceSet { device: hw.device.clone(), count: extra };
            let pool2 = e.kv_pool_bytes(hw2.total_vram(), weights, c.overhead);
            let n2 = if bytes_per_seq == 0 { 0 } else { (pool2.best / bytes_per_seq as f64).floor() as i64 };
            if n2 >= 1 {
                recs.push((format!("--gpu {}:{}", hw.device.name, extra), "fits".to_string()));
                break;
            }
        }
    }

    FitReport {
        model: m.clone(),
        hw: hw.clone(),
        quant_name: q.name().to_string(),
        engine_name: e.name().to_string(),
        util: e.util(),
        total_vram,
        managed_pool: e.managed_pool(total_vram),
        weights,
        overhead: c.overhead,
        kv_pool,
        kv_per_token,
        bytes_per_seq,
        kv_bytes: w.kv_bytes,
        ctx: w.ctx,
        max_seqs,
        serves,
        naive_would_say,
        ttft,
        decode,
        ridge_batch: rb,
        recommendations: recs,
        not_modeled: vec!["prefix-cache", "chunked-prefill"],
        calibrated: c.overhead.calibrated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry;

    // Frozen hand-computed C1 predictions (predict-then-measure: the "predict" leg).
    #[test]
    fn c1_llama_70b_deterministic_memory() {
        let m = registry::model("llama-3-70b").unwrap();
        // weights: 70,553,706,496 params × 2 bytes
        assert_eq!(weights_bytes(&m, &Fp16), 141_107_412_992);
        // kv per token (GQA, fp16): 2·8·128·2 = 4096 B/layer × 80 layers
        assert_eq!(state_bytes(&m, 1, 2), 327_680);
        // per sequence @ 4096 ctx
        assert_eq!(state_bytes(&m, 4096, 2), 1_342_177_280);
    }

    // KV dtype is independent of weight dtype (the bug Igor caught): fp8 KV halves it.
    #[test]
    fn kv_dtype_is_independent() {
        let m = registry::model("llama-3-70b").unwrap();
        assert_eq!(state_bytes(&m, 4096, 1), state_bytes(&m, 4096, 2) / 2);
    }

    // C1 verdict: naive says "fits", engine-aware says it won't serve.
    #[test]
    fn c1_wont_serve_but_naive_would() {
        let m = registry::model("llama-3-70b").unwrap();
        let dev = registry::device("a100-80gb").unwrap();
        let hw = DeviceSet { device: dev, count: 2 };
        let w = Workload { ctx: 4096, prompt_tokens: 2048, concurrency: None, kv_bytes: 2 };
        let r = fit(&m, &hw, &Fp16, &Vllm { util: 0.90 }, &w, &Calibration::uncalibrated());
        assert!(!r.serves, "should not serve at fp16 on 2×A100");
        assert!(r.naive_would_say, "naive weights<VRAM check would wrongly say fits");
        assert_eq!(r.max_seqs.1, 0);
    }

    // MLA geometry (Igor #3: don't trust it without a test). Latent+rope, no per-head factor.
    #[test]
    fn mla_drops_the_per_head_factor() {
        let mla = StateGeometry::Mla { d_c: 512, d_rope: 64 };
        assert_eq!(mla.layer_state_bytes(100, 2), (512 + 64) * 100 * 2);
        // and it is far smaller than a GQA layer at the same length
        let gqa = StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 };
        assert!(mla.layer_state_bytes(4096, 2) < gqa.layer_state_bytes(4096, 2));
    }

    // MHA→GQA is a LINEAR memory shrink of n_heads/n_kv_heads (not quadratic): here 8×.
    #[test]
    fn gqa_is_linear_shrink_over_mha() {
        let mha = StateGeometry::Mha { head_dim: 128, n_heads: 64 };
        let gqa = StateGeometry::Gqa { head_dim: 128, n_kv_heads: 8 };
        assert_eq!(mha.layer_state_bytes(4096, 2), gqa.layer_state_bytes(4096, 2) * 8);
    }
}
