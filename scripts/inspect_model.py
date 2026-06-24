"""
inspect_model.py — Human-readable architecture and weight inspection for FastVLM.

Inspects the raw weight dictionaries (PyTorch HF or MLX) for any of the six
FastVLM model checkpoints:

    PyTorch (HF bf16):  weights/fastvlm-{variant}/   (llava_qwen.safetensors etc.)
    MLX (fp16/quantized): weights/fastvlm-{variant}-mlx/  (*.npz or *.safetensors)

WHAT THIS SHOWS
---------------
This script answers: "what exists in this model and what does it look like?"

It does NOT show:
  - Operations or control flow (use inspect_aimodel.py for the compiled graph)
  - KV cache management (use verify_runtime.py or read fastvlm_decoder.py)
  - Numerical quality (use compare_weights.py or verify_*.py)

OUTPUT MODES
------------
--mode summary   (default)
    One line per logical module group showing parameter count and dtype.
    Fast overview of the whole model.

--mode layers
    One line per weight tensor: name, shape, dtype, size in MB.
    Designed so that diffing PyTorch vs MLX output is clean and useful:

        python scripts/inspect_model.py --variant 1.5b --source pytorch > /tmp/pt.txt
        python scripts/inspect_model.py --variant 1.5b --source mlx    > /tmp/mlx.txt
        diff /tmp/pt.txt /tmp/mlx.txt

    The diff will show exactly which tensors changed dtype, which gained
    quantization siblings (.scales / .biases), and nothing else.

--mode flow
    Shows the data flow through the full model with tensor shapes at each
    stage boundary. Derived from config values — no forward pass needed.
    This is the ground truth for the architecture doc tables.

--mode config
    Prints the raw config JSON for the variant's LLM and vision tower.

USAGE
-----
    python scripts/inspect_model.py --variant 0.5b --source pytorch
    python scripts/inspect_model.py --variant 1.5b --source mlx --mode layers
    python scripts/inspect_model.py --variant 7b   --source pytorch --mode flow
    python scripts/inspect_model.py --variant 1.5b --source pytorch --mode config

    # Diff PyTorch vs MLX for 1.5B:
    python scripts/inspect_model.py --variant 1.5b --source pytorch --mode layers > /tmp/pt.txt
    python scripts/inspect_model.py --variant 1.5b --source mlx    --mode layers > /tmp/mlx.txt
    diff /tmp/pt.txt /tmp/mlx.txt

    # Compare all three variants side by side (flow mode):
    for v in 0.5b 1.5b 7b; do
        echo "=== $v ===" && python scripts/inspect_model.py --variant $v --source pytorch --mode flow
    done
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def _load_pytorch_weights(weights_dir: Path) -> dict:
    """Load HF safetensors or pytorch_model.bin weights as {name: tensor}."""
    try:
        from safetensors import safe_open
        # Try single-file safetensors first
        for pattern in ["*.safetensors"]:
            files = sorted(weights_dir.glob(pattern))
            if files:
                tensors = {}
                for f in files:
                    with safe_open(f, framework="pt", device="cpu") as sf:
                        for key in sf.keys():
                            tensors[key] = sf.get_tensor(key)
                return tensors
    except ImportError:
        pass

    # Fallback: torch.load
    import torch
    for pattern in ["pytorch_model.bin", "pytorch_model-*.bin"]:
        files = sorted(weights_dir.glob(pattern))
        if files:
            tensors = {}
            for f in files:
                tensors.update(torch.load(f, map_location="cpu", weights_only=True))
            return tensors

    raise FileNotFoundError(f"No PyTorch weights found in {weights_dir}")


def _load_mlx_weights(weights_dir: Path) -> dict:
    """Load MLX weights (npz or safetensors) as {name: numpy array}."""
    import numpy as np

    # Try safetensors first (newer MLX format)
    try:
        from safetensors.numpy import load_file as np_load_file
        files = sorted(weights_dir.glob("*.safetensors"))
        if files:
            tensors = {}
            for f in files:
                tensors.update(np_load_file(str(f)))
            return tensors
    except (ImportError, Exception):
        pass

    # Try npz
    files = sorted(weights_dir.glob("*.npz"))
    if files:
        tensors = {}
        for f in files:
            d = np.load(f, allow_pickle=False)
            tensors.update({k: d[k] for k in d.files})
        return tensors

    raise FileNotFoundError(f"No MLX weights found in {weights_dir}")


def _load_config(weights_dir: Path) -> dict:
    """Load config.json from the weights directory."""
    cfg_path = weights_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json in {weights_dir}")
    return json.loads(cfg_path.read_text())


def _tensor_shape(t) -> list:
    """Return shape as list regardless of torch.Tensor or numpy array."""
    try:
        return list(t.shape)
    except Exception:
        return []


def _tensor_dtype(t) -> str:
    """Return dtype string regardless of torch.Tensor or numpy array."""
    try:
        return str(t.dtype).replace("torch.", "")
    except Exception:
        return "?"


def _tensor_bytes(t) -> int:
    """Return total bytes of a tensor."""
    import math
    shape = _tensor_shape(t)
    if not shape:
        return 0
    n = math.prod(shape)
    dtype = _tensor_dtype(t)
    # uint32 (packed int4/int8) is 4 bytes per element
    bytes_per = {"float32": 4, "float16": 2, "bfloat16": 2,
                 "uint32": 4, "int32": 4, "uint8": 1, "int8": 1}.get(dtype, 4)
    return n * bytes_per


def _mb(n_bytes: int) -> str:
    if n_bytes >= 1_073_741_824:
        return f"{n_bytes / 1_073_741_824:.2f} GB"
    if n_bytes >= 1_048_576:
        return f"{n_bytes / 1_048_576:.1f} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes} B"


# ---------------------------------------------------------------------------
# Weight directory resolution
# ---------------------------------------------------------------------------

def _resolve_dirs(variant: str, source: str, repo_root: Path):
    """Return (weights_dir, config_dir) for the given variant and source."""
    if source == "pytorch":
        d = repo_root / "weights" / f"fastvlm-{variant}"
    else:
        d = repo_root / "weights" / f"fastvlm-{variant}-mlx"

    if not d.exists():
        raise FileNotFoundError(
            f"Weights directory not found: {d}\n"
            f"Expected layout: weights/fastvlm-{{variant}}/ and weights/fastvlm-{{variant}}-mlx/"
        )

    # Config lives in the pytorch directory regardless of source
    cfg_dir = repo_root / "weights" / f"fastvlm-{variant}"
    return d, cfg_dir


# ---------------------------------------------------------------------------
# Component classification
# ---------------------------------------------------------------------------

def _classify_key(key: str) -> str:
    """Classify a weight key into component: vision_encoder | projector | decoder."""
    # HF key prefixes
    if key.startswith("model.vision_tower") or key.startswith("vision_tower"):
        return "vision_encoder"
    if key.startswith("model.mm_projector") or key.startswith("mm_projector"):
        return "projector"
    if (key.startswith("model.embed_tokens") or key.startswith("model.layers")
            or key.startswith("model.norm") or key.startswith("lm_head")):
        return "decoder"
    # MLX flat key prefixes (may differ)
    if "vision" in key.split(".")[0]:
        return "vision_encoder"
    if "projector" in key.split(".")[0] or "mm_proj" in key:
        return "projector"
    return "decoder"


# ---------------------------------------------------------------------------
# --mode layers
# ---------------------------------------------------------------------------

def mode_layers(tensors: dict, source: str, variant: str):
    """Print one deterministic line per tensor for diffing."""
    header = f"# FastVLM {variant.upper()} — {source}"
    print(header)
    print("#")
    print(f"# {'Key':<70} {'Shape':<25} {'Dtype':<12} {'Size':>10}")
    print(f"# {'-'*70} {'-'*25} {'-'*12} {'-'*10}")

    total_bytes = 0
    for key in sorted(tensors.keys()):
        t = tensors[key]
        shape = _tensor_shape(t)
        dtype = _tensor_dtype(t)
        nb = _tensor_bytes(t)
        total_bytes += nb
        shape_str = str(shape)
        print(f"  {key:<70} {shape_str:<25} {dtype:<12} {_mb(nb):>10}")

    print(f"#")
    print(f"# Total: {len(tensors)} tensors, {_mb(total_bytes)}")


# ---------------------------------------------------------------------------
# --mode summary
# ---------------------------------------------------------------------------

def mode_summary(tensors: dict, source: str, variant: str):
    """Print grouped summary by component and module kind."""
    import re

    print(f"\n{'='*70}")
    print(f"  FastVLM {variant.upper()} — {source}")
    print(f"{'='*70}")

    components = {"vision_encoder": {}, "projector": {}, "decoder": {}}
    for key, t in tensors.items():
        comp = _classify_key(key)
        components[comp][key] = t

    comp_labels = {
        "vision_encoder": "Vision Encoder (FastViTHD)",
        "projector": "Projector (mlp2x_gelu)",
        "decoder": "Decoder (Qwen2)",
    }

    for comp, label in comp_labels.items():
        comp_tensors = components[comp]
        if not comp_tensors:
            continue

        total_bytes = sum(_tensor_bytes(t) for t in comp_tensors.values())
        total_params = sum(
            (lambda s: __import__('math').prod(s) if s else 0)(_tensor_shape(t))
            for t in comp_tensors.values()
        )

        print(f"\n  {label}")
        print(f"  {'─'*60}")
        print(f"  Tensors : {len(comp_tensors)}")
        print(f"  Params  : {total_params:,}")
        print(f"  Size    : {_mb(total_bytes)}")

        # Group by module kind
        groups = {}
        for key in sorted(comp_tensors.keys()):
            # Extract module kind: strip variant-specific layer numbers
            parts = key.split(".")
            # Collapse layer indices to show module types not per-layer detail
            condensed = []
            for p in parts:
                if re.match(r"^\d+$", p):
                    condensed.append("N")
                else:
                    condensed.append(p)
            kind = ".".join(condensed)
            groups.setdefault(kind, []).append(key)

        # Show unique module kinds with dtype and count
        seen_kinds = {}
        for key in sorted(comp_tensors.keys()):
            t = comp_tensors[key]
            dtype = _tensor_dtype(t)
            shape = _tensor_shape(t)
            # Get kind without layer number
            parts = key.split(".")
            condensed = []
            for p in parts:
                if __import__("re").match(r"^\d+$", p):
                    condensed.append("N")
                else:
                    condensed.append(p)
            kind = ".".join(condensed)
            if kind not in seen_kinds:
                seen_kinds[kind] = {"dtype": dtype, "shape": shape,
                                    "count": 1, "bytes": _tensor_bytes(t)}
            else:
                seen_kinds[kind]["count"] += 1
                seen_kinds[kind]["bytes"] += _tensor_bytes(t)

        print(f"\n  {'Module kind':<55} {'Dtype':<12} {'Count':>6} {'Size':>10}")
        print(f"  {'─'*55} {'─'*12} {'─'*6} {'─'*10}")
        for kind, info in sorted(seen_kinds.items()):
            print(f"  {kind:<55} {info['dtype']:<12} {info['count']:>6} "
                  f"{_mb(info['bytes']):>10}")

    # Grand total
    all_bytes = sum(_tensor_bytes(t) for t in tensors.values())
    all_params = sum(
        (lambda s: __import__('math').prod(s) if s else 0)(_tensor_shape(t))
        for t in tensors.values()
    )
    print(f"\n{'─'*70}")
    print(f"  TOTAL   {all_params:,} parameters   {_mb(all_bytes)}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# --mode flow
# ---------------------------------------------------------------------------

def mode_flow(cfg: dict, variant: str, source: str):
    """Print the data flow with tensor shapes at each stage boundary."""

    # Extract LLM config — may be nested under 'text_config' or 'llm_config'
    llm = cfg.get("text_config") or cfg.get("llm_config") or cfg
    vis = cfg.get("vision_config") or {}

    hidden    = llm.get("hidden_size", "?")
    n_layers  = llm.get("num_hidden_layers", "?")
    n_heads   = llm.get("num_attention_heads", "?")
    n_kv      = llm.get("num_key_value_heads", "?")
    head_dim  = llm.get("head_dim") or (
        hidden // n_heads if isinstance(hidden, int) and isinstance(n_heads, int) else "?")
    inter     = llm.get("intermediate_size", "?")
    vocab     = llm.get("vocab_size", "?")
    rope_t    = llm.get("rope_theta", "?")
    mm_hidden = cfg.get("mm_hidden_size", 3072)

    # KV dim
    kv_dim = (n_kv * head_dim) if isinstance(n_kv, int) and isinstance(head_dim, int) else "?"

    # Quantization label
    quant = {"0.5b": "fp16", "1.5b": "int8", "7b": "int4"}.get(variant, "?")
    src_dtype = "bf16" if source == "pytorch" else f"fp16/{quant}"

    print(f"\n{'='*70}")
    print(f"  FastVLM {variant.upper()} — {source} ({src_dtype}) — Full Data Flow")
    print(f"{'='*70}")
    print(f"""
  INPUT
  ─────────────────────────────────────────────────
  pixel_values          [1, 3, 1024, 1024]  float32
    │  (fp32 → fp16 cast, first op in vision encoder)
    ▼

  VISION ENCODER (FastViTHD)              shared across all variants
  ─────────────────────────────────────────────────
  patch_embed           Conv2d(3→64, 4×4, stride=4)
    │  [1, 64, 256, 256]
    ▼
  network.0–8           RepMixer / MobileOneBlock / MHSA stages
    │  [1, 768, 16, 16]    (intermediate — stage-dependent)
    ▼
  conv_exp              MobileOneBlock Conv2d(→3072, 1×1)
    │  [1, 3072, 16, 16]
    ▼
  reshape + transpose   [B, C, H*W] → [B, H*W, C]
    │
  image_features        [1, 256, 3072]     float16
    │
    ▼

  PROJECTOR (mlp2x_gelu)                  hidden_size = {hidden}
  ─────────────────────────────────────────────────
  layers.0              Linear({mm_hidden} → {hidden})
    │  [1, 256, {hidden}]
    ▼
  layers.1              GELU (tanh approximation)
    │  [1, 256, {hidden}]
    ▼
  layers.2              Linear({hidden} → {hidden})
    │
  projected_features    [1, 256, {hidden}]    float16
    │
    │   text prompt → tokenizer → input_ids [1, N]
    │                           → embed_tokens → [1, N, {hidden}]
    ▼
  concat([projected_features, text_embeds], dim=1)
    │
  prefill_embeds        [1, 256+N, {hidden}]  float16
    │
    ▼

  DECODER (Qwen2 — {variant.upper()})
  ─────────────────────────────────────────────────
  Config:
    num_hidden_layers  = {n_layers}
    num_attention_heads = {n_heads}
    num_key_value_heads = {n_kv}   (GQA ratio {f"{n_heads//n_kv}:1" if isinstance(n_heads,int) and isinstance(n_kv,int) and n_kv else "?"})
    head_dim           = {head_dim}
    intermediate_size  = {inter}
    vocab_size         = {vocab}
    rope_theta         = {rope_t}

  ┌─ × {n_layers} transformer layers ──────────────────────────┐
  │                                                          │
  │  input_layernorm     RMSNorm({hidden})                   │
  │    ▼                                                     │
  │  q_proj              Linear({hidden} → {n_heads if isinstance(n_heads,int) and isinstance(head_dim,int) else "?"}{f"×{head_dim}" if isinstance(head_dim,int) else ""} = {n_heads*head_dim if isinstance(n_heads,int) and isinstance(head_dim,int) else "?"})    │
  │  k_proj              Linear({hidden} → {n_kv if isinstance(n_kv,int) and isinstance(head_dim,int) else "?"}{f"×{head_dim}" if isinstance(head_dim,int) else ""} = {n_kv*head_dim if isinstance(n_kv,int) and isinstance(head_dim,int) else "?"})     │
  │  v_proj              Linear({hidden} → {n_kv*head_dim if isinstance(n_kv,int) and isinstance(head_dim,int) else "?"})              │
  │    ▼                                                     │
  │  RoPE                applied to Q, K                     │
  │    ▼                                                     │
  │  GQA attention       SDPA, is_causal=True                │
  │    KV cache read:    k_cache[layer, :, :pos, :]          │
  │    KV cache write:   slice_scatter new K,V at pos        │
  │    ▼                                                     │
  │  o_proj              Linear({n_heads*head_dim if isinstance(n_heads,int) and isinstance(head_dim,int) else "?"} → {hidden})              │
  │    ▼                                                     │
  │  post_attn_layernorm RMSNorm({hidden})                   │
  │    ▼                                                     │
  │  gate_proj           Linear({hidden} → {inter})          │
  │  up_proj             Linear({hidden} → {inter})          │
  │  SwiGLU              gate * silu(up)                     │
  │  down_proj           Linear({inter} → {hidden})          │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
    │
    ▼
  norm                  RMSNorm({hidden})
    ▼
  lm_head               Linear({hidden} → {vocab})
    │
  logits                [1, L, {vocab}]   float16

  ─────────────────────────────────────────────────
  DECODE LOOP (one token at a time after prefill)
  ─────────────────────────────────────────────────
  argmax(logits[:, -1, :])  →  next_token_id  int32
    │
    └─ embed_tokens(next_token_id)  →  [1, 1, {hidden}]
         │  (feed back as input_embeds for next decode step)
         ▼
       repeat until EOS or max_tokens

  KV CACHE (state, persisted across decode steps)
  ─────────────────────────────────────────────────
  k_cache               [{n_layers}, 1, MAX_SEQ_LEN, {kv_dim}]  float16
  v_cache               [{n_layers}, 1, MAX_SEQ_LEN, {kv_dim}]  float16
  kv_dim = n_kv_heads × head_dim = {n_kv} × {head_dim} = {kv_dim}
""")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# --mode config
# ---------------------------------------------------------------------------

def mode_config(cfg: dict, variant: str):
    """Pretty-print the raw config."""
    print(f"\n=== FastVLM {variant.upper()} config.json ===\n")
    print(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect FastVLM model weights and architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--variant", choices=["0.5b", "1.5b", "7b"], required=True)
    parser.add_argument("--source", choices=["pytorch", "mlx"], default="pytorch")
    parser.add_argument(
        "--mode",
        choices=["summary", "layers", "flow", "config"],
        default="summary",
        help="Output mode (default: summary)"
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Override weights root directory (default: auto-detected from repo root)"
    )
    args = parser.parse_args()

    # Find repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if not (repo_root / "weights").exists():
        # Try current directory
        repo_root = Path.cwd()
    if args.weights_dir:
        weights_dir = args.weights_dir
        cfg_dir = weights_dir
    else:
        try:
            weights_dir, cfg_dir = _resolve_dirs(args.variant, args.source, repo_root)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    # Config mode doesn't need weights
    if args.mode in ("config", "flow"):
        try:
            cfg = _load_config(cfg_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        if args.mode == "config":
            mode_config(cfg, args.variant)
        else:
            mode_flow(cfg, args.variant, args.source)
        return

    # Load weights
    print(f"Loading {args.source} weights for {args.variant}...", file=sys.stderr)
    try:
        if args.source == "pytorch":
            tensors = _load_pytorch_weights(weights_dir)
        else:
            tensors = _load_mlx_weights(weights_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading weights: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(tensors)} tensors.", file=sys.stderr)

    if args.mode == "summary":
        mode_summary(tensors, args.source, args.variant)
    elif args.mode == "layers":
        mode_layers(tensors, args.source, args.variant)


if __name__ == "__main__":
    main()
