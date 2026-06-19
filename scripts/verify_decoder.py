"""
verify_decoder.py — FastVLM decoder verification.

STAGES
------
Stage 1 — CORRECTNESS (fp32, port vs HF Qwen2)
    Does the re-authored port compute the same thing as the original model?
    Reference is stock HF Qwen2ForCausalLM from the same weights. Run in
    fp32 ONLY as a diagnostic isolation trick: divergence here is structural
    (fusion order, rope, head reshape), not a precision issue. Per-layer PSNR
    localises a bug to a specific transformer block. This is the ONLY stage
    that runs by default.

Compression stages (--compression flag)
    Apply coreai-opt compression to the verified fp32 port and compare
    against the Stage 1 fp32 reference. Reports PSNR at the specified
    precision level(s). The former Stage 2 (fp16 health check) is now
    --compression fp16.

    Supported levels:
      fp16            : Cast to float16 (former Stage 2)
      int8            : coreai-opt weight-only int8 quantization
      int8-palettized : coreai-opt 8-bit k-means palettization
      int4            : coreai-opt weight-only int4 quantization
      int4-palettized : coreai-opt 4-bit k-means palettization
      all             : Run all five levels in sequence

    These compare compressed output against the fp32 reference to measure
    quality degradation at each precision level. Results directly inform the
    --compression flag in export_fastvlm.py.

    For KV cache correctness across separate runtime calls, see verify_runtime.py.

USAGE
-----
  python scripts/verify_decoder.py --variant 1.5b
  python scripts/verify_decoder.py --variant 1.5b --compression fp16
  python scripts/verify_decoder.py --variant 1.5b --compression int4-palettized
  python scripts/verify_decoder.py --variant 1.5b --compression all
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, "scripts")
from compression import COMPRESSION_LEVELS, apply_compression, psnr  # noqa: E402
from fastvlm_decoder import (  # noqa: E402
    FastVLMDecoderStateful,
    _load_decoder_weights,
    _mutate_state_dict,
)

# Pass thresholds (dB). Engineering judgments, not Apple specifications.
CORRECTNESS_PASS     = 80.0   # fp32 cross-model; we achieve ~113 dB
CORRECTNESS_MARGINAL = 50.0   # below here is definitely wrong
COMPRESSION_PASS     = 40.0   # minimum acceptable PSNR after compression

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
    """Minimal example inputs for coreai-opt tracing during compression preparation."""
    torch.manual_seed(0)
    B, L = 1, 8
    input_ids = torch.randint(1, text_cfg.vocab_size, (B, L), dtype=torch.int32)
    pos_ids = torch.arange(L, dtype=torch.int32).unsqueeze(0)
    return (input_ids, pos_ids)


# ─── Stage 1: correctness vs HF ───────────────────────────────────────────────


def stage_correctness(text_cfg, weights_dir: str) -> tuple[bool, torch.Tensor]:
    """
    Run Stage 1 correctness check. Returns (passed, fp32_reference_logits).
    fp32_reference_logits is returned so stage_compression can reuse it
    without reloading weights.
    """
    print("\n" + "=" * 56)
    print("STAGE 1 — CORRECTNESS (fp32, port vs HF Qwen2)")
    print("=" * 56)

    hf = _build_hf_reference(text_cfg, weights_dir)
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


# ─── Compression stages ───────────────────────────────────────────────────────


def stage_compression(
    text_cfg,
    weights_dir: str,
    level: str,
    fp32_ref: torch.Tensor,
) -> bool:
    """
    Apply one compression level to the fp32 port and compare against fp32_ref.

    fp32_ref is the logits tensor from Stage 1 — reused here rather than
    recomputed so we're comparing against the same reference output.
    """
    print("\n" + "=" * 56)
    print(f"COMPRESSION — {level.upper()} vs fp32 reference")
    print("=" * 56)

    port = _build_port(text_cfg, weights_dir, torch.float32)
    example_inputs = _example_inputs(text_cfg)

    print(f"Applying {level} compression...")
    compressed, _ = apply_compression(port, level, example_inputs)

    # For fp16, run the exact same cache-health checks the old Stage 2 did,
    # since fp16 has unique overflow/NaN failure modes worth diagnosing
    # separately from just PSNR degradation.
    if level == "fp16":
        return _check_fp16_health(compressed, text_cfg, fp32_ref)

    # For quantization/palettization levels: run a forward pass and compare
    # against the fp32 reference.
    torch.manual_seed(0)
    B, L = 1, 8
    input_ids = torch.randint(1, text_cfg.vocab_size, (B, L), dtype=torch.int32)
    pos_ids = torch.arange(L, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        if hasattr(compressed, "k_cache"):
            compressed.k_cache.zero_()
            compressed.v_cache.zero_()
        compressed_out = compressed(input_ids, pos_ids)

    score = psnr(compressed_out, fp32_ref)
    print(f"\nPSNR vs fp32 reference: {score:.1f} dB")

    if score > COMPRESSION_PASS:
        print(f"[PASS] {score:.1f} dB — {level} compression viable.")
        return True
    print(
        f"[FAIL] {score:.1f} dB — unacceptable quality loss at {level}. "
        f"Consider a less aggressive compression level."
    )
    return False


def _check_fp16_health(
    port_fp16: FastVLMDecoderStateful,
    text_cfg,
    fp32_ref: torch.Tensor,
) -> bool:
    """
    fp16-specific health checks: NaN/Inf, overflow risk, and cache correctness
    across a prefill + decode sequence. More thorough than a single PSNR
    comparison because fp16 pathology often surfaces specifically in the
    cached multi-step decode path, not on a single forward pass.
    """
    n_prefill, n_decode = 6, 4
    total = n_prefill + n_decode
    torch.manual_seed(0)
    input_ids = torch.randint(1, text_cfg.vocab_size, (1, total), dtype=torch.int32)
    pos_full = torch.arange(total, dtype=torch.int32).unsqueeze(0)

    # Full pass reference in fp16
    port_fp16.k_cache.zero_()
    port_fp16.v_cache.zero_()
    with torch.no_grad():
        ref_fp16 = port_fp16(input_ids, pos_full)

    # Cached: prefill then per-token decode
    port_fp16.k_cache.zero_()
    port_fp16.v_cache.zero_()
    cached = torch.zeros_like(ref_fp16)
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

    first_decode = psnr(
        cached[:, n_prefill:n_prefill + 1],
        ref_fp16[:, n_prefill:n_prefill + 1],
    )
    decode_span = psnr(cached[:, n_prefill:], ref_fp16[:, n_prefill:])
    print(f"first decode step PSNR: {first_decode:.1f} dB")
    print(f"decode span PSNR      : {decode_span:.1f} dB")

    if has_nan or has_inf:
        print("\n[FAIL] NaN/Inf in fp16 cached output.")
        return False
    if overflow_risk:
        print(f"\n[FAIL] fp16 logits near saturation ({max_abs:.0f}).")
        return False
    if decode_span > COMPRESSION_PASS and first_decode > COMPRESSION_PASS:
        print(f"\n[PASS] fp16 cached decode clean ({decode_span:.1f} dB).")
        return True
    print(f"\n[FAIL] fp16 cached decode diverges ({decode_span:.1f} dB).")
    return False


# ─── Driver ───────────────────────────────────────────────────────────────────


def verify(variant: str, compression: list[str] | None) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying decoder: {variant} ({weights_dir})")
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    passed, fp32_ref = stage_correctness(text_cfg, weights_dir)
    if not passed:
        print("\n>>> Stage 1 FAILED. Fix correctness before testing compression.")
        sys.exit(1)

    if not compression:
        print("\n" + "=" * 56)
        print("ALL STAGES PASS — safe to export.")
        print("=" * 56)
        sys.exit(0)

    levels = COMPRESSION_LEVELS if "all" in compression else compression
    all_passed = True
    for level in levels:
        level_passed = stage_compression(text_cfg, weights_dir, level, fp32_ref)
        all_passed = all_passed and level_passed

    print("\n" + "=" * 56)
    if all_passed:
        print("ALL COMPRESSION CHECKS PASS.")
    else:
        print("ONE OR MORE COMPRESSION CHECKS FAILED.")
    print("=" * 56)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Verify FastVLM decoder correctness and compression quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/verify_decoder.py --variant 1.5b
  python scripts/verify_decoder.py --variant 1.5b --compression fp16
  python scripts/verify_decoder.py --variant 1.5b --compression int4-palettized
  python scripts/verify_decoder.py --variant 1.5b --compression all
""",
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument(
        "--compression",
        nargs="+",
        choices=COMPRESSION_LEVELS + ["all"],
        default=None,
        help="Compression level(s) to test after Stage 1. Default: none (Stage 1 only).",
    )
    args = ap.parse_args()
    verify(args.variant, args.compression)
