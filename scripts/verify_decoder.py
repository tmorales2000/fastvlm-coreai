"""
verify_decoder.py — Three-stage decoder verification gate.

Written against the stable FastVLMDecoder API:
  forward(inputs_embeds, position_ids, k_cache, v_cache) -> logits

Stages run in order and stop at the first failure:

  Stage 1 — CORRECTNESS (fp32, port vs HF Qwen2)
      Does our re-authored decoder compute the same thing as the original?
      Both models receive identical inputs_embeds (from safetensors directly,
      bypassing embed_tokens which lives in embed.aimodel). Run in fp32 to
      isolate structural bugs from fp16 rounding noise.
      Target: > 80 dB. We achieve ~113 dB.

  Stage 2 — KV CACHE + fp16 HEALTH (fp16, prefill + decode vs full pass)
      Runs the real fp16 artifact across a cached multi-step decode.
      Checks cache correctness and fp16 health (NaN/Inf/saturation).
      Target: > 40 dB decode PSNR. We achieve ~72 dB.

  Stage 3 — COMPRESSION QUALITY (optional)
      Measures PSNR loss from compression vs fp16 baseline.
      Prepare-only (no finalize) — model stays runnable as plain PyTorch.
      Enable with --compression 4bit / 8bit or --compression-config YAML.
      Target: > 35 dB for int8, > 25 dB for int4.

Usage:
    python scripts/verify_decoder.py --variant 0.5b
    python scripts/verify_decoder.py --variant 1.5b --stage correctness
    python scripts/verify_decoder.py --variant 1.5b --prefill 6 --decode 4
    python scripts/verify_decoder.py --variant 1.5b --compression 8bit
    python scripts/verify_decoder.py --variant 7b   --compression 4bit
    python scripts/verify_decoder.py --variant 7b   --compression-config recipes/7b_mixed.yaml
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, "scripts")
from fastvlm_decoder import (  # noqa: E402
    FastVLMDecoder,
    FastVLMDecoderStateful,
    KEY_CACHE_NAME,
    VALUE_CACHE_NAME,
    _load_decoder_weights,
)
from quantization import (  # noqa: E402
    MACOS_NAMED_PRESETS,
    load_compression_config,
    apply_quantization_from_config,
)

# ── Pass thresholds ───────────────────────────────────────────────────────────
CORRECTNESS_PASS      = 80.0
CORRECTNESS_MARGINAL  = 50.0
CACHE_PASS            = 40.0
COMPRESSION_PASS_INT8 = 35.0
COMPRESSION_PASS_INT4 = 25.0
FP16_OVERFLOW         = 60000.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float(), b.float()
    mse = ((a_f - b_f) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    peak = b_f.abs().max().item() ** 2
    if peak == 0:
        return float("inf")
    return 10 * np.log10(peak / mse)


def _load_embed_weights(weights_dir: str) -> torch.Tensor:
    """Load embed_tokens weight from safetensors (excluded from decoder weights)."""
    for path in sorted(glob.glob(f"{weights_dir}/*.safetensors")):
        d = st.load_file(path)
        for k, v in d.items():
            if "embed_tokens.weight" in k:
                return v  # (vocab, hidden) bf16
    raise FileNotFoundError(f"embed_tokens.weight not found in {weights_dir}")


def _make_kv_cache(text_cfg, max_ctx: int, dtype: torch.dtype):
    """Build zero-initialised k/v cache tensors matching the decoder's expected shape."""
    n_layers   = text_cfg.num_hidden_layers
    n_kv_heads = text_cfg.num_key_value_heads
    head_dim   = text_cfg.hidden_size // text_cfg.num_attention_heads
    shape = (n_layers, 1, n_kv_heads, max_ctx, head_dim)
    k = torch.zeros(shape, dtype=dtype)
    v = torch.zeros(shape, dtype=dtype)
    return k, v


def _random_embeds(hidden: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(1, seq_len, hidden, dtype=dtype)


# ── Model builders ────────────────────────────────────────────────────────────

def _build_port(text_cfg, weights_dir: str, dtype: torch.dtype) -> FastVLMDecoder:
    """Load decoder at requested dtype. from_weights() is fp16-only so we load manually."""
    weights = _load_decoder_weights(weights_dir, dtype=dtype)
    weights = {k.removeprefix("model."): v for k, v in weights.items()}
    model   = FastVLMDecoder(text_cfg).to(dtype=dtype)
    missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
    if set(missing):
        raise RuntimeError(f"Port missing keys: {set(missing)}")
    if unexpected:
        raise RuntimeError(f"Port unexpected keys: {unexpected}")
    return model.eval()


def _build_hf_reference(text_cfg, weights_dir: str) -> Qwen2ForCausalLM:
    """HF Qwen2 in fp32. embed_tokens omitted (not in decoder weights) — tolerated."""
    weights = _load_decoder_weights(weights_dir, dtype=torch.float32)
    try:
        model = Qwen2ForCausalLM(text_cfg).to(torch.float32)
    except Exception:
        model = Qwen2ForCausalLM(Qwen2Config(**text_cfg.to_dict())).to(torch.float32)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    # embed_tokens lives in embed.aimodel — always missing here, always tolerated
    tolerated    = {"model.embed_tokens.weight", "lm_head.weight"}
    real_missing = set(missing) - tolerated
    if real_missing:
        raise RuntimeError(f"HF reference missing keys: {sorted(real_missing)[:8]}")
    if unexpected:
        raise RuntimeError(f"HF reference unexpected keys: {sorted(unexpected)[:8]}")
    return model.eval()


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def stage_correctness(text_cfg, weights_dir: str) -> bool:
    print("\n" + "=" * 56)
    print("STAGE 1 — CORRECTNESS (fp32, port vs HF Qwen2)")
    print("=" * 56)

    hf   = _build_hf_reference(text_cfg, weights_dir)
    port = _build_port(text_cfg, weights_dir, torch.float32)
    n_layers = text_cfg.num_hidden_layers

    hf_acts, port_acts = {}, {}

    def hf_hook(i):
        def hook(_m, _inp, out):
            hf_acts[i] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return hook

    def port_hook(i):
        def hook(_m, _inp, out):
            port_acts[i] = out.detach().float()
        return hook

    handles = []
    for i in range(n_layers):
        handles.append(hf.model.layers[i].register_forward_hook(hf_hook(i)))
        handles.append(port.layers[i].register_forward_hook(port_hook(i)))

    torch.manual_seed(0)
    L = 8

    # Get embeddings from safetensors directly (embed_tokens not in HF reference)
    embed_w = _load_embed_weights(weights_dir).to(torch.float32)  # (vocab, hidden)
    input_ids = torch.randint(1, text_cfg.vocab_size, (1, L), dtype=torch.long)
    embeds    = embed_w[input_ids]  # (1, L, hidden) fp32
    pos_ids   = torch.arange(L, dtype=torch.long).unsqueeze(0)

    # KV cache for our port (fp32 for Stage 1)
    k_fp32, v_fp32 = _make_kv_cache(text_cfg, max_ctx=256, dtype=torch.float32)

    with torch.no_grad():
        hf_out   = hf(inputs_embeds=embeds, position_ids=pos_ids).logits
        # Port is loaded in fp32 for Stage 1 — pass fp32 inputs throughout
        port_out = port(
            embeds,                  # fp32 embeds
            pos_ids.to(torch.int32),
            k_fp32,                  # fp32 cache
            v_fp32,
        )

    for h in handles:
        h.remove()

    print(f"\n{'layer':>6} | {'PSNR (dB)':>10} | note")
    print("-" * 44)
    worst, worst_layer, prev = float("inf"), -1, float("inf")
    for i in range(n_layers):
        score = psnr(port_acts.get(i, torch.zeros(1)), hf_acts.get(i, torch.zeros(1)))
        drop  = prev - score if np.isfinite(prev) and np.isfinite(score) else 0.0
        note  = f"<-- drops {drop:.0f} dB" if drop > 15 else ""
        print(f"{i:>6} | {score:>10.1f} | {note}")
        if score < worst:
            worst, worst_layer = score, i
        prev = score

    logits_score = psnr(port_out.float(), hf_out.float())
    print("-" * 44)
    print(f"final logits PSNR : {logits_score:.1f} dB")
    print(f"worst layer       : {worst_layer} ({worst:.1f} dB)")

    if logits_score > CORRECTNESS_PASS:
        print(f"\n[PASS] {logits_score:.1f} dB — port matches HF Qwen2.")
        return True
    if logits_score > CORRECTNESS_MARGINAL:
        print(
            f"\n[MARGINAL] {logits_score:.1f} dB. Likely rope_theta (expect 1e6) "
            f"or SDPA scale mismatch. Uniform floor → rope; cliff at layer "
            f"{worst_layer} → block bug."
        )
        return False
    print(
        f"\n[FAIL] {logits_score:.1f} dB — architecture mismatch at layer "
        f"{worst_layer}. Check qkv split, head_dim, rope_theta."
    )
    return False


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def stage_cache(text_cfg, weights_dir: str, n_prefill: int, n_decode: int) -> bool:
    print("\n" + "=" * 56)
    print(f"STAGE 2 — KV CACHE + fp16 HEALTH (prefill={n_prefill}, decode={n_decode})")
    print("=" * 56)

    port     = _build_port(text_cfg, weights_dir, torch.float16)
    hidden   = text_cfg.hidden_size
    total    = n_prefill + n_decode
    max_ctx  = total + 4  # small ceiling sufficient for verification

    # Use fixed random embeds so both paths see identical inputs
    torch.manual_seed(0)
    all_embeds = torch.randn(1, total, hidden, dtype=torch.float16)

    # Reference: single full pass, empty cache, all tokens at once.
    # pos_ids length = total (offset=0, L=total, seq_len=total).
    pos_full  = torch.arange(total, dtype=torch.int32).unsqueeze(0)
    k_ref, v_ref = _make_kv_cache(text_cfg, max_ctx, torch.float16)
    with torch.no_grad():
        ref_logits = port(all_embeds, pos_full, k_ref, v_ref)

    # Cached path: prefill then per-token decode steps.
    # The decoder derives offset = seq_len - L where seq_len = pos_ids.shape[-1].
    # Prefill:  L=n_prefill, offset=0 → pos_ids = [0..n_prefill]  (len n_prefill)
    # Decode t: L=1, offset=n_prefill+t → pos_ids = [0..n_prefill+t+1) (len n_prefill+t+1)
    k_cache, v_cache = _make_kv_cache(text_cfg, max_ctx, torch.float16)
    cached = torch.zeros_like(ref_logits)

    with torch.no_grad():
        # Prefill
        pos_pre = torch.arange(n_prefill, dtype=torch.int32).unsqueeze(0)
        pre_out = port(all_embeds[:, :n_prefill], pos_pre, k_cache, v_cache)
        cached[:, :n_prefill] = pre_out

        # Decode: one token at a time, position_ids covers full past+current
        for t in range(n_decode):
            p       = n_prefill + t
            pos_dec = torch.arange(p + 1, dtype=torch.int32).unsqueeze(0)
            step_out = port(
                all_embeds[:, p:p + 1],  # single token embed
                pos_dec,                  # [0..p] — offset = p, L = 1
                k_cache, v_cache,
            )
            cached[:, p] = step_out[:, 0]

    has_nan  = torch.isnan(cached).any().item()
    has_inf  = torch.isinf(cached).any().item()
    max_abs  = cached.abs().max().item()
    overflow = max_abs > FP16_OVERFLOW

    print(f"\nfp16 NaN / Inf    : {has_nan} / {has_inf}")
    print(f"fp16 max |logit|  : {max_abs:.0f}  (fp16 ceiling 65504)")

    first = psnr(
        cached[:, n_prefill:n_prefill + 1],
        ref_logits[:, n_prefill:n_prefill + 1],
    )
    span = psnr(cached[:, n_prefill:], ref_logits[:, n_prefill:])
    print(f"first decode PSNR : {first:.1f} dB")
    print(f"decode span PSNR  : {span:.1f} dB")

    if has_nan or has_inf:
        print("\n[FAIL] NaN/Inf in fp16 output.")
        return False
    if overflow:
        print(f"\n[FAIL] fp16 logits near saturation ({max_abs:.0f}).")
        return False
    if span > CACHE_PASS and first > CACHE_PASS:
        print(f"\n[PASS] fp16 cached decode clean ({span:.1f} dB).")
        return True
    print(
        "\n[FAIL] Cached decode diverges. Check offset, head reshape, "
        "and that mutable_slice_update resolves to coreai_models."
    )
    return False


# ── Stage 3 ───────────────────────────────────────────────────────────────────

def stage_compression(
    text_cfg,
    weights_dir: str,
    compression_config: dict,
    compression_label: str,
) -> bool:
    print("\n" + "=" * 56)
    print(f"STAGE 3 — COMPRESSION QUALITY ({compression_label})")
    print("=" * 56)
    print("Comparing compressed vs fp16 baseline (prepare only, not finalized).")

    hidden   = text_cfg.hidden_size
    n_layers = text_cfg.num_hidden_layers
    n_kv     = text_cfg.num_key_value_heads
    head_dim = hidden // text_cfg.num_attention_heads
    max_ctx  = 64

    torch.manual_seed(0)
    L       = 8
    embeds  = _random_embeds(hidden, L, torch.float16)
    pos_ids = torch.arange(L, dtype=torch.int32).unsqueeze(0)
    k_ex    = torch.zeros(n_layers, 1, n_kv, max_ctx, head_dim, dtype=torch.float16)
    v_ex    = torch.zeros_like(k_ex)

    example_inputs = (embeds, pos_ids, k_ex.clone(), v_ex.clone())

    # fp16 baseline
    port_fp16 = _build_port(text_cfg, weights_dir, torch.float16)
    with torch.no_grad():
        fp16_out = port_fp16(embeds, pos_ids, k_ex.clone(), v_ex.clone())

    # Compressed (prepare only — stays runnable)
    print(f"Applying {compression_label}...")
    port_comp = _build_port(text_cfg, weights_dir, torch.float16)
    port_comp = apply_quantization_from_config(
        port_comp, compression_config, example_inputs, finalize=False
    )
    port_comp.eval()

    with torch.no_grad():
        comp_out = port_comp(embeds, pos_ids, k_ex.clone(), v_ex.clone())

    score = psnr(fp16_out, comp_out)

    # Determine threshold from dtype in config.
    # Presets use module_type_configs (global_config is None).
    # YAML recipes may use global_config. Check both.
    def _extract_dtype(cfg: dict) -> str:
        """Extract dtype string from config — handles both string and torch.dtype values."""
        def _normalise(d) -> str | None:
            if d is None:
                return None
            if isinstance(d, str):
                return d
            # torch.dtype object — e.g. torch.int8, torch.int4
            return str(d).replace("torch.", "")

        # Try global_config first
        gc = cfg.get("global_config") or {}
        if isinstance(gc, dict):
            d = _normalise(gc.get("op_state_spec", {}).get("weight", {}).get("dtype"))
            if d:
                return d
        # Fall back to first non-None module_type_configs entry
        for v in (cfg.get("module_type_configs") or {}).values():
            if isinstance(v, dict):
                d = _normalise(v.get("op_state_spec", {}).get("weight", {}).get("dtype"))
                if d:
                    return d
        return "int4"  # safe default

    dtype_str = _extract_dtype(compression_config)
    threshold = COMPRESSION_PASS_INT8 if dtype_str == "int8" else COMPRESSION_PASS_INT4

    print(f"\nCompressed vs fp16 PSNR : {score:.1f} dB")
    print(f"Pass threshold ({dtype_str:4s})   : {threshold:.0f} dB")

    if score > threshold:
        print(f"\n[PASS] {score:.1f} dB — compression viable for export.")
        return True
    print(
        f"\n[FAIL] {score:.1f} dB — too lossy. Consider 8bit instead of 4bit, "
        "or a mixed-precision YAML from scan_quantization_sensitivity.py."
    )
    return False


# ── Driver ────────────────────────────────────────────────────────────────────

def verify(
    variant: str,
    stage: str,
    n_prefill: int,
    n_decode: int,
    compression_config: dict | None,
    compression_label: str,
) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying decoder: {variant}")
    if compression_config is not None:
        print(f"Compression:       {compression_label}")

    config   = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    if stage == "correctness":
        sys.exit(0 if stage_correctness(text_cfg, weights_dir) else 1)
    if stage == "cache":
        sys.exit(0 if stage_cache(text_cfg, weights_dir, n_prefill, n_decode) else 1)
    if stage == "compression":
        if compression_config is None:
            print("ERROR: --stage compression requires --compression or --compression-config.")
            sys.exit(1)
        sys.exit(
            0 if stage_compression(text_cfg, weights_dir, compression_config, compression_label)
            else 1
        )

    # all
    if not stage_correctness(text_cfg, weights_dir):
        print("\n>>> Stopped at Stage 1.")
        sys.exit(1)
    if not stage_cache(text_cfg, weights_dir, n_prefill, n_decode):
        print("\n>>> Stopped at Stage 2.")
        sys.exit(1)
    if compression_config is not None:
        if not stage_compression(text_cfg, weights_dir, compression_config, compression_label):
            print("\n>>> Stopped at Stage 3.")
            sys.exit(1)

    print("\n" + "=" * 56)
    stages = "Stages 1–3" if compression_config is not None else "Stages 1–2"
    print(f"ALL STAGES PASS ({stages}) — safe to export.")
    print("=" * 56)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "correctness", "cache", "compression"],
    )
    ap.add_argument("--prefill", type=int, default=6)
    ap.add_argument("--decode",  type=int, default=4)

    cg = ap.add_mutually_exclusive_group()
    cg.add_argument(
        "--compression",
        choices=list(MACOS_NAMED_PRESETS.keys()),
        default=None, metavar="PRESET",
        help=f"Named preset: {', '.join(MACOS_NAMED_PRESETS.keys())}.",
    )
    cg.add_argument(
        "--compression-config",
        type=Path, default=None, metavar="YAML",
        help="Path to quantization_config YAML recipe.",
    )

    args = ap.parse_args()

    compression_config: dict | None = None
    compression_label: str = "none"

    if args.compression_config is not None:
        if not args.compression_config.is_file():
            ap.error(f"--compression-config: not found: {args.compression_config}")
        compression_config, compression_label = load_compression_config(
            args.compression_config, platform="macOS"
        )
    elif args.compression is not None:
        compression_config, compression_label = load_compression_config(
            args.compression, platform="macOS"
        )

    verify(
        variant=args.variant,
        stage=args.stage,
        n_prefill=args.prefill,
        n_decode=args.decode,
        compression_config=compression_config,
        compression_label=compression_label,
    )
