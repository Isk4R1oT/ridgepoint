//! Rendering — Style A boxed dashboard (human) + `--json`.
//! Color is deliberately omitted in v0 (added later); layout carries the structure.

use crate::core::{FitReport, Interval};

fn gb(bytes: f64) -> String {
    format!("{:.1}", bytes / 1e9)
}

/// low·best·high for a memory interval, in GB.
fn mem_band(i: &Interval) -> String {
    if i.exact {
        format!("{:>6} GB  exact", gb(i.best))
    } else {
        let flag = if i.calibrated { "" } else { "  ~est" };
        format!("{:>6} ·{:>6} ·{:>6} GB{}", gb(i.low), gb(i.best), gb(i.high), flag)
    }
}

fn boxed(lines: &[String]) -> String {
    let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let bar: String = "─".repeat(w + 2);
    let mut out = format!("╭{}╮\n", bar);
    for l in lines {
        let pad = w - l.chars().count();
        out.push_str(&format!("│ {}{} │\n", l, " ".repeat(pad)));
    }
    out.push_str(&format!("╰{}╯\n", bar));
    out
}

pub fn human(r: &FitReport) -> String {
    let params_b = format!("{:.1}B", r.model.n_params as f64 / 1e9);
    let kv = if r.kv_bytes == 1 { "fp8" } else { "fp16" };
    let engine_line = match r.util {
        Some(u) => format!("{}× {} · {} {:.2} · KV {}", r.hw.count, r.hw.device.name, r.engine_name, u, kv),
        None => format!("{}× {} · {} · KV {}", r.hw.count, r.hw.device.name, r.engine_name, kv),
    };
    let header = boxed(&[
        "ridgepoint fit".to_string(),
        format!("{} · {} · {} · {}", r.model.id, r.model.kv_kind, params_b, r.quant_name),
        engine_line,
    ]);

    let mut s = String::new();
    s.push_str(&header);
    s.push('\n');

    // VERDICT
    if r.serves {
        s.push_str(&format!("  ✓  SERVES        ~{} seqs @ {} ctx (best)\n", r.max_seqs.1, r.ctx));
    } else {
        s.push_str("  ✗  WON'T SERVE\n");
        s.push_str(&format!("     KV pool ≈ {} GB after the engine's grab\n", gb(r.kv_pool.best)));
        if r.naive_would_say {
            s.push_str(&format!(
                "     a naive weights<VRAM check says \"fits\" ({}<{}) — wrong\n",
                gb(r.weights as f64), gb(r.total_vram as f64)
            ));
        }
    }
    s.push('\n');

    // MEMORY
    s.push_str("  MEMORY                low     best    high\n");
    s.push_str(&format!("    weights  {:<7}{}\n", r.quant_name, mem_band(&Interval::exact(r.weights as f64))));
    s.push_str(&format!("    overhead        {}\n", mem_band(&r.overhead)));
    s.push_str(&format!("    KV pool         {}   ←\n", mem_band(&r.kv_pool)));
    s.push('\n');

    // SPEED
    let cal = if r.calibrated { "calibrated" } else { "roofline · uncalibrated" };
    s.push_str(&format!("  SPEED  {}\n", cal));
    s.push_str(&format!(
        "    TTFT @prompt      {:>5.2} ·{:>5.2} ·{:>5.2} s\n",
        r.ttft.low, r.ttft.best, r.ttft.high
    ));
    s.push_str(&format!(
        "    decode/req        {:>5.0} ·{:>5.0} ·{:>5.0} tok/s\n",
        r.decode.low, r.decode.best, r.decode.high
    ));
    s.push_str(&format!("    regime   memory-bound · ridge ≈ batch {:.0}\n", r.ridge_batch));
    s.push('\n');

    // CAPACITY
    s.push_str(&format!(
        "  CAPACITY @{}       {} seqs · {} GB/seq\n",
        r.ctx, r.max_seqs.1, gb(r.bytes_per_seq as f64)
    ));

    // FIX
    if !r.recommendations.is_empty() {
        s.push('\n');
        s.push_str("  FIX");
        for (i, (change, effect)) in r.recommendations.iter().enumerate() {
            let sep = if i == 0 { "  " } else { "       " };
            s.push_str(&format!("{}{} → {}\n", sep, change, effect));
        }
    }

    s.push('\n');
    s.push_str(&format!(
        "  not modeled: {}   ·   calibrated: {}\n",
        r.not_modeled.join(", "),
        if r.calibrated { "yes" } else { "no" }
    ));
    s
}

fn iv_json(i: &Interval) -> String {
    if i.exact {
        format!("{{\"best\": {:.4e}, \"exact\": true}}", i.best)
    } else {
        format!(
            "{{\"low\": {:.4e}, \"best\": {:.4e}, \"high\": {:.4e}, \"calibrated\": {}}}",
            i.low, i.best, i.high, i.calibrated
        )
    }
}

pub fn json(r: &FitReport) -> String {
    let recs: Vec<String> = r
        .recommendations
        .iter()
        .map(|(c, e)| format!("{{\"change\": \"{}\", \"effect\": \"{}\"}}", c, e))
        .collect();
    let nm: Vec<String> = r.not_modeled.iter().map(|s| format!("\"{}\"", s)).collect();
    format!(
        "{{\n  \"model\": {{\"id\": \"{}\", \"kv_kind\": \"{}\", \"params\": {}, \"layers\": {}}},\n  \
         \"engine\": {{\"name\": \"{}\", \"util\": {}}},\n  \
         \"verdict\": {{\"serves\": {}, \"naive_would_say\": {}}},\n  \
         \"memory\": {{\"weights_bytes\": {}, \"overhead_bytes\": {}, \"kv_pool_bytes\": {}}},\n  \
         \"speed\": {{\"ttft_s\": {}, \"decode_tok_s\": {}, \"regime\": \"memory_bound\", \"ridge_batch\": {:.0}}},\n  \
         \"capacity\": {{\"ctx\": {}, \"max_sequences\": {}, \"bytes_per_seq\": {}}},\n  \
         \"recommendations\": [{}],\n  \"not_modeled\": [{}],\n  \"calibrated\": {}\n}}",
        r.model.id, r.model.kv_kind, r.model.n_params, r.model.layers,
        r.engine_name, r.util.map(|u| format!("{:.2}", u)).unwrap_or_else(|| "null".into()),
        r.serves, r.naive_would_say,
        iv_json(&Interval::exact(r.weights as f64)), iv_json(&r.overhead), iv_json(&r.kv_pool),
        iv_json(&r.ttft), iv_json(&r.decode), r.ridge_batch,
        r.ctx,
        format!("{{\"low\": {}, \"best\": {}, \"high\": {}}}", r.max_seqs.0, r.max_seqs.1, r.max_seqs.2),
        r.bytes_per_seq,
        recs.join(", "), nm.join(", "), r.calibrated
    )
}
