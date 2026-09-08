"""HuggingFace input adapter — turns any 'org/repo' into ridgepoint dims.

This is the ONE changeable edge (HF API, auth, gated repos, formats). It is isolated here via
`huggingface_hub` (the maintained canonical client) so the Rust core never touches HF and stays
pure math. `fetch_shape` returns a dict consumed by `_core.fit_shape` / `_core.scan_shape`.

n_params / active_params are ESTIMATES from HF metadata (documented) — they feed interval-valued
overhead/decode, so approximation is honest, not silent.
"""
import json
import urllib.request

from huggingface_hub import hf_hub_download

_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float": 4, "half": 2, "int8": 1}


def _download_json(repo: str, filename: str, revision: str):
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    with open(path) as f:
        return json.load(f)


def fetch_shape(repo: str, revision: str = "main") -> dict:
    """Fetch a HuggingFace model's config and derive ridgepoint dims. Raises on missing fields."""
    cfg = _download_json(repo, "config.json", revision)
    tc = cfg.get("text_config") or cfg  # multimodal configs nest attention under text_config

    def field(name, default=None):
        return tc.get(name, cfg.get(name, default))

    layers = field("num_hidden_layers")
    d_model = field("hidden_size")
    n_heads = field("num_attention_heads")
    if layers is None or d_model is None or n_heads is None:
        raise ValueError(f"{repo}: config.json missing num_hidden_layers/hidden_size/num_attention_heads")
    layers, d_model, n_heads = int(layers), int(d_model), int(n_heads)
    n_kv = int(field("num_key_value_heads", n_heads))
    head_dim = int(field("head_dim") or (d_model // n_heads))

    # HYBRID attention: only full-attention layers hold a growing KV cache. Detect from
    # `layer_types` (array of "full_attention"/"linear_attention") or `full_attention_interval`
    # (every Nth layer is full — Qwen3-Next). Dense models keep kv_layers == layers; getting
    # this wrong over-estimates KV ~N× and flips "won't serve" verdicts on hybrid models.
    kv_layers = layers
    layer_types = field("layer_types")
    interval = field("full_attention_interval")
    if isinstance(layer_types, list) and layer_types:
        kv_layers = sum(1 for t in layer_types if "full" in str(t).lower())
    elif interval:
        kv_layers = layers // int(interval)
    kv_layers = max(1, min(kv_layers, layers))

    out = {"id": repo, "layers": layers, "kv_layers": kv_layers, "d_model": d_model}

    # geometry: MLA (DeepSeek) if a compressed-KV latent is present; else GQA; MHA when kv==heads
    if field("kv_lora_rank"):
        out["geom"] = "mla"
        out["d_c"] = int(field("kv_lora_rank"))
        out["d_rope"] = int(field("qk_rope_head_dim") or 64)
    elif n_kv == n_heads:
        out["geom"] = "mha"
        out["head_dim"] = head_dim
        out["n_heads"] = n_heads
    else:
        out["geom"] = "gqa"
        out["head_dim"] = head_dim
        out["n_kv_heads"] = n_kv

    n_params = _params_from_index(repo, revision, cfg, tc)
    out["n_params"] = n_params
    out["active_params"] = _active_params(cfg, tc, n_params)
    return out


def _params_from_index(repo: str, revision: str, cfg: dict, tc: dict) -> int:
    """Exact parameter count from safetensors tensor SHAPES (dtype-agnostic).

    Dividing `total_size` by a config dtype breaks when the checkpoint is stored at a
    different width than `torch_dtype` claims — e.g. GLM-4.7-Flash ships 1-byte (fp8)
    weights with `torch_dtype` absent, so the old code assumed bf16 and under-counted 2×.
    Shapes don't lie: sum numel over every tensor header. Falls back to the dtype division,
    then to a config estimate, if the headers can't be read (gated repo / offline)."""
    try:
        n = sum(_safetensors_numel(repo, revision, fn) for fn in _safetensors_shards(repo, revision))
        if n > 0:
            return n
    except Exception:
        pass
    try:
        idx = _download_json(repo, "model.safetensors.index.json", revision)
        total_bytes = int(idx["metadata"]["total_size"])
        dt = str(cfg.get("torch_dtype") or tc.get("torch_dtype") or "bfloat16").lower()
        return int(total_bytes / _DTYPE_BYTES.get(dt, 2))
    except Exception:
        # Last resort: rough dense transformer estimate (~12·L·h² + 2·V·h). Approximate.
        L, h = int(tc.get("num_hidden_layers")), int(tc.get("hidden_size"))
        v = int(cfg.get("vocab_size", 32000))
        return int(12 * L * h * h + 2 * v * h)


def _safetensors_shards(repo: str, revision: str) -> list:
    """Safetensors shard filenames — the sharded index's weight_map, or the single file."""
    try:
        idx = _download_json(repo, "model.safetensors.index.json", revision)
        return sorted(set(idx["weight_map"].values()))
    except Exception:
        return ["model.safetensors"]


def _safetensors_numel(repo: str, revision: str, filename: str) -> int:
    """Sum of tensor element counts in one safetensors file, read from its header only:
    8-byte little-endian header length, then a JSON header {name: {dtype, shape, ...}}."""
    import struct

    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"

    def _range(a: int, b: int) -> bytes:
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "ridgepoint"})
        return urllib.request.urlopen(req, timeout=30).read()

    header_len = struct.unpack("<Q", _range(0, 7))[0]
    header = json.loads(_range(8, 7 + header_len))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        numel = 1
        for dim in meta.get("shape") or []:
            numel *= int(dim)
        total += numel
    return total


def _active_params(cfg: dict, tc: dict, n_params: int) -> int:
    """Active (per-token) params for MoE; equals n_params for dense. ESTIMATE.

    Covers the common MoE field names across families: Mixtral (num_local_experts),
    Qwen (num_experts), DeepSeek (n_routed_experts) + experts-per-token.
    """
    def pick(*names):
        for n in names:
            v = tc.get(n)
            v = v if v is not None else cfg.get(n)
            if v:
                return v
        return None

    ne = pick("num_local_experts", "num_experts", "n_routed_experts")
    k = pick("num_experts_per_tok", "num_experts_per_token", "moe_topk")
    if not ne or not k:
        return n_params  # dense
    # Crude estimate: only k/ne of the (dominant) expert params are active; floor keeps shared
    # attention/embeds/shared-experts in play. Feeds interval-valued decode → honest approximation.
    frac = float(k) / float(ne)
    return max(int(n_params * frac), int(n_params * 0.12))
