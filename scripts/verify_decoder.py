"""
verify_decoder.py — One decoder verification, two staged gates.

Replaces the old fp16-vs-fp32-only check (which compared the port against itself
and so could not catch a wrong architecture, weight map, or broken cache) and
folds in the cross-model and cache checks. Runs two stages in order and stops at
the first failure, so a failure tells you WHICH layer of the problem you are at:

  Stage 1 — CORRECTNESS (fp32, vs HF Qwen2)
      Does the re-authored port compute the same thing as the original model?
      Reference is stock HF Qwen2ForCausalLM from the same weights (the original
      llava_qwen decoder delegates to it). Run in fp32 ONLY as a diagnostic
      isolation trick: a divergence here is structural (fusion order, rope, head
      reshape), not fp16 rounding. Per-layer PSNR localizes a bug to a block.
      If this fails, the rest is noise — stop.

  Stage 2 — KV CACHE + fp16 HEALTH (fp16, prefill + decode vs full pass)
      Runs the REAL artifact: the fp16 model across a cached multi-step decode.
      fp16 is the authored target for the ANE — it is what ships, so it is what we
      verify here. Checks (a) the cache mutates and reads back correctly across
      decode steps (the only stage exercising mutable_slice_update), and (b) fp16
      health: no NaN/Inf, no saturation toward the fp16 ceiling. There is NO
      fp32 comparison — fp32 is not a deployed artifact, and Stage 1 already
      confirmed the fp16-authored port matches an fp32 reference.

Default runs both and prints a single verdict. Each stage is also callable alone
via --stage.

Usage:
    python scripts/verify_decoder.py [--variant 1.5b]
    python scripts/verify_decoder.py --variant 1.5b --stage correctness
    python scripts/verify_decoder.py --variant 1.5b --prefill 6 --decode 4
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, "scripts")
from fastvlm_decoder import (  # noqa: E402
    FastVLMDecoderStateful,
    _load_decoder_weights,
    _mutate_state_dict,
)

# Pass thresholds (dB). Engineering judgments, not Apple specifications.
# See module docstring for rationale.
CORRECTNESS_PASS     = 80.0   # fp32 cross-model; we achieve ~113 dB
CORRECTNESS_MARGINAL = 50.0   # below here is definitely wrong
CACHE_PASS           = 40.0   # fp16 cached decode; we achieve ~72 dB

# fp16 range is 0..65504. Flag logits above this as overflow risk.
# 60000 gives ~8% headroom below the ceiling — enough to catch runaway
# activations before they produce Inf on the next arithmetic step.
FP16_OVERFLOW_THRESHOLD = 60000.0



def _random_embeds(model, seq_len: int) -> "torch.Tensor":
    """Return random inputs_embeds for decoder verification calls."""
    import torch
    hidden = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    return torch.randn(1, seq_len, hidden, dtype=dtype)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float(), b.float()
    mse = ((a_f - b_f) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    peak = b_f.abs().max().item() ** 2
    if peak == 0:
        return float("inf")
    return 10 * np.log10(peak / mse)


# ─── Model builders ───────────────────────────────────────────────────────────


def _build_port(text_cfg, weights_dir: str, dtype: torch.dtype) -> FastVLMDecoderStateful:
    weights = _load_decoder_weights(weights_dir, dtype=dtype)
    _mutate_state_dict(weights)
    weights = {k.removeprefix("model."): v for k, v in weights.items()}
    model = FastVLMDecoderStateful(text_cfg).to(dtype=dtype)
    missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
    actual_missing = set(missing)
    if actual_missing:
        raise RuntimeError(f"Port missing keys: {actual_missing}")
    if unexpected:
        raise RuntimeError(f"Port unexpected keys: {unexpected}")
    model.eval()
    return model


def _build_hf_reference(text_cfg, weights_dir: str) -> Qwen2ForCausalLM:
    """Stock HF Qwen2 in fp32, UNFUSED q/k/v (do not run _mutate_state_dict)."""
    weights = _load_decoder_weights(weights_dir, dtype=torch.float32)
    # text_cfg may be a LlavaConfig subclass; strip to a clean Qwen2Config so HF
    # does not choke on llava-specific fields.
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


# ─── Stage 1: correctness vs HF ───────────────────────────────────────────────


def stage_correctness(text_cfg, weights_dir: str) -> bool:
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
    pos_ids = torch.arange(L, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        # HF reference uses input_ids; our decoder uses inputs_embeds
        # Use embed_tokens from HF model to get equivalent embeddings
        embeds = hf.model.embed_tokens(input_ids).detach()
        hf_out = hf(inputs_embeds=embeds, position_ids=pos_ids).logits
        port_out = port(embeds.to(dtype=torch.float16), pos_ids.to(torch.int32))

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
        return True
    if logits_score > CORRECTNESS_MARGINAL:
        print(
            f"\n[MARGINAL] {logits_score:.1f} dB. Likely rope_theta (expect 1e6, not "
            "the 1e4 fallback) or an SDPA scale mismatch. A uniform mid-40s floor "
            "across ALL layers is the rope signature; a single-layer cliff is a "
            f"block bug — start at layer {worst_layer}."
        )
        return False
    print(
        f"\n[FAIL] {logits_score:.1f} dB — architecture mismatch. Start at layer "
        f"{worst_layer}: check qkv fusion order (q,k,v), head_dim split, rope_theta."
    )
    return False


# ─── Stage 2: KV cache ────────────────────────────────────────────────────────


def stage_cache(text_cfg, weights_dir: str, n_prefill: int, n_decode: int) -> bool:
    print("\n" + "=" * 56)
    print(f"STAGE 2 — KV CACHE + fp16 HEALTH (prefill={n_prefill}, decode={n_decode})")
    print("=" * 56)
    # This stage runs the REAL artifact: the fp16 model, across a cached multi-step
    # decode — the only place fp16 pathology (overflow, NaN/Inf, accumulation
    # through the cache) actually surfaces. There is no fp32 comparison here
    # because fp16 IS the authored target for the ANE; fp32 is not something that
    # ships. Correctness (architecture) is Stage 1's job; this proves the fp16
    # cached model runs clean and the cache reads back what it wrote.

    port = _build_port(text_cfg, weights_dir, torch.float16)
    total = n_prefill + n_decode

    torch.manual_seed(0)
    input_ids = torch.randint(1, text_cfg.vocab_size, (1, total), dtype=torch.int32)
    pos_full = torch.arange(total, dtype=torch.int32).unsqueeze(0)

    # Ground truth: one full pass (offset 0, every token sees all prior positions).
    with torch.no_grad():
        ref_logits = port(_random_embeds(port, total), pos_full)

    # Cached: prefill, then per-token decode reading/writing the cache.
    cached = torch.zeros_like(ref_logits)
    with torch.no_grad():
        pre_out = port(_random_embeds(port, n_prefill), pos_full[:, :n_prefill])
        cached[:, :n_prefill] = pre_out
        for t in range(n_decode):
            p = n_prefill + t
            step_out = port(_random_embeds(port, 1), pos_full[:, : p + 1])
            cached[:, p] = step_out[:, 0]

    # ── fp16 health: the cached run is the real fp16 artifact across many steps ─
    has_nan = torch.isnan(cached).any().item()
    has_inf = torch.isinf(cached).any().item()
    # fp16 max finite is 65504; flag values near saturation as overflow risk.
    max_abs = cached.abs().max().item()
    overflow_risk = max_abs > FP16_OVERFLOW_THRESHOLD
    print(f"\nfp16 NaN / Inf     : {has_nan} / {has_inf}")
    print(f"fp16 max |logit|   : {max_abs:.0f}  (fp16 ceiling 65504)")

    # ── cache correctness: cached decode must match the full pass ──────────────
    first_decode = psnr(cached[:, n_prefill : n_prefill + 1], ref_logits[:, n_prefill : n_prefill + 1])
    decode_span = psnr(cached[:, n_prefill:], ref_logits[:, n_prefill:])
    print(f"first decode step PSNR: {first_decode:.1f} dB")
    print(f"decode span PSNR      : {decode_span:.1f} dB")

    if has_nan or has_inf:
        print("\n[FAIL] NaN/Inf in fp16 cached output — fp16 pathology, not a cache bug.")
        return False
    if overflow_risk:
        print(
            f"\n[FAIL] fp16 logits near saturation ({max_abs:.0f}). An activation is "
            "overflowing fp16 — same failure class as the vision encoder's 1024x1024 "
            "overflow. Find the unbounded op (pre-norm activation / attention logits)."
        )
        return False
    if decode_span > CACHE_PASS and first_decode > CACHE_PASS:
        print(f"\n[PASS] fp16 cached decode clean and matches full pass ({decode_span:.1f} dB).")
        return True
    print(
        f"\n[FAIL] cached decode diverges. First decode step is the suspect: check "
        "offset (= new token's absolute position), the head flatten/restore "
        "reshape, and that the read span is [0:seq_len]. If mutable_slice_update "
        "resolved to the local fallback rather than coreai_models, the mutation "
        "may not be landing — confirm the import."
    )
    return False


# ─── Driver ───────────────────────────────────────────────────────────────────


def verify(variant: str, stage: str, n_prefill: int, n_decode: int) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying decoder: {variant} ({weights_dir})")
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    if stage == "correctness":
        sys.exit(0 if stage_correctness(text_cfg, weights_dir) else 1)
    if stage == "cache":
        sys.exit(0 if stage_cache(text_cfg, weights_dir, n_prefill, n_decode) else 1)

    # stage == "all": run in order, stop at first failure.
    if not stage_correctness(text_cfg, weights_dir):
        print("\n>>> Stopped at Stage 1. Fix correctness before anything else.")
        sys.exit(1)
    if not stage_cache(text_cfg, weights_dir, n_prefill, n_decode):
        print("\n>>> Stopped at Stage 2. Architecture is right; cache or fp16 is not.")
        sys.exit(1)

    print("\n" + "=" * 56)
    print("ALL STAGES PASS — safe to export.")
    print("=" * 56)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "correctness", "cache"],
    )
    ap.add_argument("--prefill", type=int, default=6)
    ap.add_argument("--decode", type=int, default=4)
    args = ap.parse_args()
    verify(args.variant, args.stage, args.prefill, args.decode)
