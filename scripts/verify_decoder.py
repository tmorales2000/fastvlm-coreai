"""
verify_decoder.py — FastVLM decoder verification.

STAGES
------
Stage 1 — CORRECTNESS (HF weights -> fp32 port)
    Does the re-authored port compute the same thing as the original model?
    Reference is stock HF Qwen2ForCausalLM from the same weights. Run in
    fp32 ONLY as a diagnostic isolation trick: divergence here is structural
    (fusion order, rope, head reshape), not a precision issue. Per-layer PSNR
    localises a bug to a specific transformer block. PSNR is measured against
    the HF reference output. This is the ONLY stage that runs by default.

Stage 2 — PRECISION (fp32 -> fp16)
    Cast the verified fp32 port to float16 — the mandatory ANE execution
    precision. NOT optional, NOT "compression": every decoder export goes
    through this stage regardless of whether quantization follows. PSNR is
    measured against the Stage 1 fp32 port (not the original HF weights) —
    this isolates the fp16 cast's own error from any residual Stage 1
    structural imprecision. Includes NaN/Inf/overflow checks and a cached
    prefill+decode health check, since fp16 failure modes often only surface
    in the cached multi-step decode path. Always runs after Stage 1 passes.

Stage 3 — QUANTIZATION (fp16 -> int8 | int4), optional
    Apply coreai-opt weight-only quantization on top of the Stage 2 fp16
    model. Mutually exclusive: int8 or int4, never both, no "all". PSNR is
    measured against the Stage 2 fp16 output — this isolates quantization-only
    error from the fp16 cast's error. Only runs if --quantize is passed.

    Results directly inform the --quantize flag in export_fastvlm.py.

    For KV cache correctness across separate runtime calls, see verify_runtime.py.

VALIDATED PRODUCTION TARGETS
-----------------------------
  0.5B : Stage 2 only, no quantization (matches Apple MLX, unquantized)
  1.5B : Stage 3 int8  (46.2 dB vs fp16, PASS)
  7B   : Stage 3 int8  (int4 fails at 22.4 dB vs fp16 — fp16 pre-cast
                        sensitivity; see psnr_results.md)

USAGE
-----
  python scripts/verify_decoder.py --variant 1.5b
  python scripts/verify_decoder.py --variant 1.5b --quantize int8
  python scripts/verify_decoder.py --variant 1.5b --quantize int4
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, "scripts")
from quantization import QUANTIZATION_LEVELS, apply_quantization, psnr  # noqa: E402
from fastvlm_decoder import (  # noqa: E402
    FastVLMDecoderStateful,
    _load_decoder_weights,
    _mutate_state_dict,
)

# Pass thresholds (dB). Engineering judgments, not Apple specifications.
CORRECTNESS_PASS     = 80.0   # Stage 1, fp32 cross-model; we achieve ~113 dB
CORRECTNESS_MARGINAL = 50.0   # below here is definitely wrong
PRECISION_PASS       = 40.0   # Stage 2, fp16 cached decode vs fp32 port
QUANTIZATION_PASS    = 40.0   # Stage 3, quantized vs fp16

# fp16 range is 0..65504. Flag logits above this as overflow risk.
FP16_OVERFLOW_THRESHOLD = 60000.0


# ─── Model builders ───────────────────────────────────────────────────────────


def _build_port(text_cfg, weights_dir: str, dtype: torch.dtype) -> FastVLMDecoderStateful:
    weights = _load_decoder_weights(weights_dir, dtype=dtype)
    _mutate_state_dict(weights)
    weights = {k.removeprefix("model."): v for k, v in weights.items()}
    model = FastVLMDecoderStateful(text_cfg).to(dtype=dtype)
    missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
    actual_missing = set(missing) - {"k_cache", "v_cache"}
    if actual_missing:
        raise RuntimeError(f"Port missing keys: {actual_missing}")
    if unexpected:
        raise RuntimeError(f"Port unexpected keys: {unexpected}")
    model.eval()
    return model


def _build_hf_reference(text_cfg, weights_dir: str) -> Qwen2ForCausalLM:
    """Stock HF Qwen2 in fp32, UNFUSED q/k/v (do not run _mutate_state_dict)."""
    weights = _load_decoder_weights(weights_dir, dtype=torch.float32)
    try:
        model = Qwen2ForCausalLM(text_cfg).to(torch.float32)
    except Exception:
        model = Qwen2ForCausalLM(Qwen2Config(**text_cfg.to_dict())).to(torch.float32)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    tolerated = {"lm_head.weight"} if getattr(text_cfg, "tie_word_embeddings", False) else set()
    real_missing = set(missing) - tolerated
    if real_missing:
        raise RuntimeError(f"HF reference missing keys: {sorted(real_missing)[:8]} ...")
    if unexpected:
        raise RuntimeError(f"HF reference unexpected keys: {sorted(unexpected)[:8]} ...")
    model.eval()
    return model


def _example_inputs(text_cfg) -> tuple[torch.Tensor, ...]:
    """Minimal example inputs for coreai-opt tracing during Stage 3 preparation."""
    torch.manual_seed(0)
    B, L = 1, 8
    input_ids = torch.randint(1, text_cfg.vocab_size, (B, L), dtype=torch.int32)
    pos_ids = torch.arange(L, dtype=torch.int32).unsqueeze(0)
    return (input_ids, pos_ids)


# ─── Stage 1: correctness vs HF ───────────────────────────────────────────────


def stage_correctness(text_cfg, weights_dir: str) -> tuple[bool, torch.Tensor]:
    """
    Stage 1 — CORRECTNESS. Returns (passed, fp32_port_logits).
    fp32_port_logits is the Stage 1 output, reused as the Stage 2 reference.
    """
    print("\n" + "=" * 56)
    print("STAGE 1 — CORRECTNESS (HF weights -> fp32 port)")
    print("=" * 56)

    n_layers = text_cfg.num_hidden_layers
    print(f"\nhidden_size           : {text_cfg.hidden_size}")
    print(f"num_hidden_layers     : {n_layers}")
    print(f"num_attention_heads   : {text_cfg.num_attention_heads}")
    print(f"num_key_value_heads   : {text_cfg.num_key_value_heads}")
    print(f"vocab_size            : {text_cfg.vocab_size}")
    print(f"tie_word_embeddings   : {getattr(text_cfg, 'tie_word_embeddings', False)}")

    hf = _build_hf_reference(text_cfg, weights_dir)
    port = _build_port(text_cfg, weights_dir, torch.float32)

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
    B, L = 1, 8
    input_ids = torch.randint(1, text_cfg.vocab_size, (B, L), dtype=torch.long)
    pos_ids = torch.arange(L, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, position_ids=pos_ids.long()).logits
        port.k_cache.zero_()
        port.v_cache.zero_()
        port_out = port(input_ids.to(torch.int32), pos_ids)

    for h in handles:
        h.remove()

    print(f"\n{'layer':>6} | {'PSNR (dB)':>10} | note")
    print("-" * 44)
    worst, worst_layer, prev = float("inf"), -1, float("inf")
    for i in range(n_layers):
        score = psnr(port_acts[i], hf_acts[i])
        drop = prev - score if np.isfinite(prev) and np.isfinite(score) else 0.0
        note = f"<-- drops {drop:.0f} dB" if drop > 15 else ""
        print(f"{i:>6} | {score:>10.1f} | {note}")
        if score < worst:
            worst, worst_layer = score, i
        prev = score

    logits_score = psnr(port_out, hf_out)
    print("-" * 44)
    print(f"final logits PSNR : {logits_score:.1f} dB")
    print(f"worst layer       : {worst_layer} ({worst:.1f} dB)")

    if logits_score > CORRECTNESS_PASS:
        print(f"\n[PASS] {logits_score:.1f} dB — port matches HF Qwen2.")
        return True, port_out
    if logits_score > CORRECTNESS_MARGINAL:
        print(
            f"\n[MARGINAL] {logits_score:.1f} dB. Likely rope_theta or SDPA scale "
            f"mismatch. Uniform mid-40s floor = rope signature; single-layer cliff = "
            f"block bug at layer {worst_layer}."
        )
        return False, port_out
    print(
        f"\n[FAIL] {logits_score:.1f} dB — architecture mismatch. Start at layer "
        f"{worst_layer}: check qkv fusion order (q,k,v), head_dim split, rope_theta."
    )
    return False, port_out


# ─── Stage 2: precision (fp32 -> fp16) ─────────────────────────────────────────


def stage_precision(
    text_cfg,
    weights_dir: str,
) -> tuple[bool, FastVLMDecoderStateful]:
    """
    Stage 2 — PRECISION. Cast to fp16 and run NaN/Inf/overflow + cached
    decode health checks. PSNR is measured against a freshly-built fp32
    port run over the SAME prefill+decode token sequence used here — NOT
    the Stage 1 8-token-only reference, which has a different sequence
    length and can't be sliced against the decode span. Building a
    dedicated fp32 reference (rather than reusing Stage 1's output)
    still isolates the fp16 cast's own error, since both fp32 and fp16
    runs use the identical port and the identical input sequence.

    Returns (passed, fp16_port) — fp16_port is reused as the Stage 3 base
    model.
    """
    print("\n" + "=" * 56)
    print("STAGE 2 — PRECISION (fp32 -> fp16)")
    print("=" * 56)

    port_fp32 = _build_port(text_cfg, weights_dir, torch.float32)
    port_fp16 = _build_port(text_cfg, weights_dir, torch.float16)

    n_prefill, n_decode = 6, 4
    total = n_prefill + n_decode
    torch.manual_seed(0)
    input_ids = torch.randint(1, text_cfg.vocab_size, (1, total), dtype=torch.int32)
    pos_full = torch.arange(total, dtype=torch.int32).unsqueeze(0)

    # fp32 reference over the identical prefill+decode sequence
    port_fp32.k_cache.zero_()
    port_fp32.v_cache.zero_()
    with torch.no_grad():
        ref_fp32 = port_fp32(input_ids, pos_full)

    # Cached: prefill then per-token decode, in fp16
    port_fp16.k_cache.zero_()
    port_fp16.v_cache.zero_()
    cached = torch.zeros(1, total, text_cfg.vocab_size, dtype=torch.float16)
    with torch.no_grad():
        pre_out = port_fp16(input_ids[:, :n_prefill], pos_full[:, :n_prefill])
        cached[:, :n_prefill] = pre_out
        for t in range(n_decode):
            p = n_prefill + t
            step_out = port_fp16(input_ids[:, p:p + 1], pos_full[:, :p + 1])
            cached[:, p] = step_out[:, 0]

    has_nan = torch.isnan(cached).any().item()
    has_inf = torch.isinf(cached).any().item()
    max_abs = cached.abs().max().item()
    overflow_risk = max_abs > FP16_OVERFLOW_THRESHOLD

    print(f"\nfp16 NaN / Inf     : {has_nan} / {has_inf}")
    print(f"fp16 max |logit|   : {max_abs:.0f}  (fp16 ceiling 65504)")

    # PSNR vs the matching fp32 port run — isolates fp16 cast error specifically.
    first_decode = psnr(
        cached[:, n_prefill:n_prefill + 1].float(),
        ref_fp32[:, n_prefill:n_prefill + 1],
    )
    decode_span = psnr(cached[:, n_prefill:].float(), ref_fp32[:, n_prefill:])
    print(f"first decode step PSNR vs fp32 port: {first_decode:.1f} dB")
    print(f"decode span PSNR vs fp32 port      : {decode_span:.1f} dB")

    if has_nan or has_inf:
        print("\n[FAIL] NaN/Inf in fp16 cached output.")
        return False, port_fp16
    if overflow_risk:
        print(f"\n[FAIL] fp16 logits near saturation ({max_abs:.0f}).")
        return False, port_fp16
    if decode_span > PRECISION_PASS and first_decode > PRECISION_PASS:
        print(f"\n[PASS] fp16 cached decode clean ({decode_span:.1f} dB).")
        return True, port_fp16
    print(f"\n[FAIL] fp16 cached decode diverges ({decode_span:.1f} dB).")
    return False, port_fp16


# ─── Stage 3: quantization (fp16 -> int8 | int4), optional ───────────────────


def stage_quantization(
    text_cfg,
    weights_dir: str,
    level: str,
) -> bool:
    """
    Stage 3 — QUANTIZATION. Applies int8 or int4 weight-only quantization
    to a fresh fp32 port (apply_quantization performs its own fp32->fp16
    cast internally, matching Apple's two-part scheme). PSNR is measured
    against a freshly-built Stage 2 fp16 reference — isolates
    quantization-only error.
    """
    print("\n" + "=" * 56)
    print(f"STAGE 3 — QUANTIZATION ({level.upper()}, fp16 -> {level})")
    print("=" * 56)

    port = _build_port(text_cfg, weights_dir, torch.float32)
    example_inputs = _example_inputs(text_cfg)

    print(f"Applying {level} quantization...")
    quantized, _ = apply_quantization(port, level, example_inputs)

    torch.manual_seed(0)
    B, L = 1, 8
    input_ids = torch.randint(1, text_cfg.vocab_size, (B, L), dtype=torch.int32)
    pos_ids = torch.arange(L, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        if hasattr(quantized, "k_cache"):
            quantized.k_cache.zero_()
            quantized.v_cache.zero_()
        quantized_out = quantized(input_ids, pos_ids)

    fp16_port = _build_port(text_cfg, weights_dir, torch.float16)
    torch.manual_seed(0)
    with torch.no_grad():
        if hasattr(fp16_port, "k_cache"):
            fp16_port.k_cache.zero_()
            fp16_port.v_cache.zero_()
        fp16_ref_out = fp16_port(input_ids, pos_ids)

    score = psnr(quantized_out, fp16_ref_out.float())
    print(f"\nPSNR vs fp16 (Stage 2) reference : {score:.1f} dB")

    if score > QUANTIZATION_PASS:
        print(f"[PASS] {score:.1f} dB — {level} quantization viable.")
        return True
    print(
        f"[FAIL] {score:.1f} dB — unacceptable quality loss at {level}. "
        f"Consider a less aggressive quantization level."
    )
    return False


# ─── Driver ───────────────────────────────────────────────────────────────────


def verify(variant: str, quantize: str | None) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying decoder: {variant} ({weights_dir})")
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    passed, fp32_ref = stage_correctness(text_cfg, weights_dir)
    if not passed:
        print("\n>>> Stage 1 FAILED. Fix correctness before Stage 2.")
        sys.exit(1)

    passed, _ = stage_precision(text_cfg, weights_dir)
    if not passed:
        print("\n>>> Stage 2 FAILED. Fix fp16 precision issues before Stage 3.")
        sys.exit(1)

    if not quantize:
        print("\n" + "=" * 56)
        print("STAGES 1-2 PASS — safe to export at fp16.")
        print("=" * 56)
        sys.exit(0)

    q_passed = stage_quantization(text_cfg, weights_dir, quantize)

    print("\n" + "=" * 56)
    if q_passed:
        print(f"STAGES 1-3 PASS — safe to export at {quantize}.")
    else:
        print(f"STAGE 3 ({quantize}) FAILED.")
    print("=" * 56)
    sys.exit(0 if q_passed else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Verify FastVLM decoder correctness, precision, and quantization quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/verify_decoder.py --variant 1.5b
  python scripts/verify_decoder.py --variant 1.5b --quantize int8
  python scripts/verify_decoder.py --variant 1.5b --quantize int4
""",
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument(
        "--quantize",
        choices=QUANTIZATION_LEVELS,
        default=None,
        help="Run Stage 3 at this quantization level after Stages 1-2 pass. "
             "Default: none (Stages 1-2 only, fp16 export target).",
    )
    args = ap.parse_args()
    verify(args.variant, args.quantize)
