# ridgepoint

**LLM inference sizing that models the engine, not just the weights.**

`ridgepoint` answers the question every VRAM calculator gets wrong: *after your serving
engine (vLLM, SGLang, …) grabs its memory pool, how many concurrent tokens of KV cache
actually fit — and where does throughput land on the latency/throughput frontier?*

The name is the **ridge point** of the roofline model: the arithmetic intensity where
decode stops being memory-bandwidth-bound. It's the single number this tool reasons about.

> 🚧 **Early development.** This `0.0.0` release reserves the name. v0 (Rust core + Python
> bindings via PyO3/maturin, one code path for both a CLI and `pip install ridgepoint`) is
> in progress.

## Why another calculator?

Most "will it fit?" tools compute `weights + KV < VRAM`. That is the wrong model for paged
engines: vLLM pre-reserves `gpu_memory_utilization × VRAM` (default 90%) at startup and
pages KV into that pool. ridgepoint models what actually decides the answer:

- **Engine-aware capacity** — usable KV pool *after* the pre-grab → max concurrent tokens.
- **Intervals, not points** — throughput sits on a Pareto frontier set by your latency budget.
- **MLA vs GQA KV geometry** — DeepSeek-style MLA is ~50–100× smaller than the GQA formula predicts.
- **Real quant bpw** — Q4_K_M is ~4.8 bits/weight, not the label's 4.5.
- **OOM honesty** — never silently under-predict memory.

## License

MIT
