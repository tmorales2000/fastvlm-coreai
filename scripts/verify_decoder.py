"""
verify_decoder.py — Four-phase decoder verification gate.

Phases run in order and stop at the first hard failure.

  Phase 1 — ARCHITECTURE CORRECTNESS (fp32, port vs HF Qwen2)
      Does our re-authored decoder compute the same thing as the original?
      Both models receive identical random inputs_embeds in fp32.
      Purpose: catch re-authoring bugs, not precision issues.
      Gate: >80 dB PASS, 50–80 dB MARGINAL (exits nonzero), <50 dB FAIL.
      Known-good implementations score ~100+ dB.

  Phase 2 — FP32 vs FP16 FIDELITY (realistic inputs)
      Runs the decoder in fp32 and fp16 on identical REALISTIC inputs
      from the full HF multimodal pipeline (fastvlm_fixtures.py).
      Reports the full metric suite (PSNR, NRMSE, cosine, KL, top-1, top-5,
      margin ratio). Establishes what the FP32→FP16 narrowing costs on
      the actual input distribution.
      Gate: informational (no hard PASS/FAIL) — baseline for Phase 4.

  Phase 3 — KV CACHE + FP16 HEALTH (fp16, prefill + decode vs full pass)
      Runs the fp16 decoder across a cached multi-step decode.
      Checks KV cache correctness and fp16 health (NaN/Inf/saturation).
      Gate: > 40 dB decode PSNR, no NaN/Inf, logits < 60000.

  Phase 4 — COMPRESSION QUALITY (realistic inputs, corpus aggregation)
      Measures quality loss from compression vs fp16 baseline using the full
      9-image corpus from fastvlm_fixtures.py. Evaluates the final generation
      position only (the first token decision). Reports mean and worst-case
      metrics across the corpus.
      Primary gate: top-5 overlap ≥ 80% mean AND ≥ 60% worst case.
      Top-1 agreement and KL divergence are context only — top-1 flips between
      candidates already favored by the FP16 baseline are expected under compression.
      INCONCLUSIVE if corpus fixtures are unavailable (no random fallback).
      Enable with --compression or --compression-config.

Usage:
    python scripts/verify_decoder.py --variant 0.5b
    python scripts/verify_decoder.py --variant 1.5b --stage correctness
    python scripts/verify_decoder.py --variant 1.5b --stage fidelity
    python scripts/verify_decoder.py --variant 1.5b --compression 8bit
    python scripts/verify_decoder.py --variant 7b   --compression 4bit
    python scripts/verify_decoder.py --variant 7b   --compression-config recipes/7b_mixed.yaml
"""

import argparse
import glob
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM


class PhaseResult(Enum):
    """Result enum for verification phases.

    Richer than bool — distinguishes INCONCLUSIVE (could not test)
    from CAUTION (tested, below threshold) from PASS/FAIL (hard gates).
    """
    PASS         = "PASS"          # Phase 1, 3: hard gate passed
    MARGINAL     = "MARGINAL"      # Phase 1: 50–80 dB, investigate
    FAIL         = "FAIL"          # Phase 1, 3: hard gate failed
    MEASURED     = "MEASURED"      # Phase 2: informational, fixture available
    SKIPPED      = "SKIPPED"       # Phase 2: fixture unavailable
    RECOMMEND    = "RECOMMEND"     # Phase 4: export for Core AI validation
    CAUTION      = "CAUTION"       # Phase 4: below threshold
    INCONCLUSIVE = "INCONCLUSIVE"  # Phase 4: corpus unavailable/incomplete

    def is_blocking(self) -> bool:
        """True if this result should stop the pipeline."""
        return self in (PhaseResult.FAIL, PhaseResult.MARGINAL, PhaseResult.CAUTION,
                        PhaseResult.INCONCLUSIVE)

    def is_ok_to_continue(self) -> bool:
        return not self.is_blocking()

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fastvlm_decoder import (  # noqa: E402
    FastVLMDecoder,
    FastVLMDecoderStateful,
    KEY_CACHE_NAME,
    VALUE_CACHE_NAME,
    KV_STATE_NAMES,
    _load_decoder_weights,
)
from fastvlm_fixtures import (  # noqa: E402
    build_decoder_fixture,
    DEFAULT_PROMPT,
)
from metrics import (  # noqa: E402
    full_report,
    print_report,
    psnr as metrics_psnr,
)
from quantization import (  # noqa: E402
    MACOS_NAMED_PRESETS,
    load_compression_config,
    apply_quantization_from_config,
)

# ── Phase 1 thresholds ────────────────────────────────────────────────────────
# Hard gates — MARGINAL exits nonzero (pipeline stops).
# Our known-good implementations score ~100+ dB; a sudden drop to 50–80 dB
# means the decoder cannot be trusted for compression evaluation.
CORRECTNESS_PASS     = 80.0   # dB — clean PASS, port matches HF Qwen2
CORRECTNESS_MARGINAL = 50.0   # dB — MARGINAL, exits nonzero, investigate

# ── Phase 3 thresholds ────────────────────────────────────────────────────────
CACHE_PASS     = 40.0    # dB — decode PSNR with KV cache
FP16_OVERFLOW  = 60000.0  # fp16 ceiling is 65504

# ── Phase 4 thresholds ────────────────────────────────────────────────────────
# Primary gate: top-5 overlap — "does the compressed model preserve the
# FP16 baseline's high-ranking candidate token set?"
# These are project heuristics intended to rank/reject obviously degraded
# recipes. They should be calibrated as Core AI A/B data is collected.
COMPRESSION_TOP5_PASS = 0.80   # top-5 overlap fraction — mean threshold
COMPRESSION_TOP5_FLOOR = 0.60  # top-5 overlap fraction — worst-case floor

# Default fixture image for Phase 2 and Phase 4
DEFAULT_FIXTURE_IMAGE = "test_assets/images/great_wave.jpg"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Legacy PSNR helper — delegates to canonical metrics.psnr."""
    return metrics_psnr(a, b)


def _load_embed_weights(weights_dir: str) -> torch.Tensor:
    """Load embed_tokens weight from safetensors (excluded from decoder weights)."""
    for path in sorted(glob.glob(f"{weights_dir}/*.safetensors")):
        d = st.load_file(path)
        for k, v in d.items():
            if "embed_tokens.weight" in k:
                return v
    raise FileNotFoundError(f"embed_tokens.weight not found in {weights_dir}")


def _make_kv_cache(text_cfg, max_ctx: int, dtype: torch.dtype,
                   device: torch.device | None = None):
    n_layers   = text_cfg.num_hidden_layers
    n_kv_heads = text_cfg.num_key_value_heads
    head_dim   = text_cfg.hidden_size // text_cfg.num_attention_heads
    shape = (n_layers, 1, n_kv_heads, max_ctx, head_dim)
    kw = dict(dtype=dtype) if device is None else dict(dtype=dtype, device=device)
    return torch.zeros(shape, **kw), torch.zeros(shape, **kw)


def _random_embeds(hidden: int, seq_len: int, dtype: torch.dtype,
                   device: torch.device | None = None) -> torch.Tensor:
    kwargs = dict(dtype=dtype) if device is None else dict(dtype=dtype, device=device)
    return torch.randn(1, seq_len, hidden, **kwargs)


# ── Model builders ────────────────────────────────────────────────────────────

def _build_port(text_cfg, weights_dir: str, dtype: torch.dtype,
                device: torch.device | None = None) -> FastVLMDecoder:
    weights = _load_decoder_weights(weights_dir, dtype=dtype)
    weights = {k.removeprefix("model."): v for k, v in weights.items()}
    model   = FastVLMDecoder(text_cfg)
    missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
    if set(missing):
        raise RuntimeError(f"Port missing keys: {set(missing)}")
    if unexpected:
        raise RuntimeError(f"Port unexpected keys: {unexpected}")
    # to(device) must come AFTER load_state_dict(assign=True) —
    # assign=True replaces parameters with loaded tensors (CPU from safetensors),
    # overwriting any prior device placement.
    model = model.to(dtype=dtype)
    if device is not None:
        model = model.to(device)
    return model.eval()


def _build_hf_reference(text_cfg, weights_dir: str,
                        device: torch.device | None = None) -> Qwen2ForCausalLM:
    weights = _load_decoder_weights(weights_dir, dtype=torch.float32)
    try:
        model = Qwen2ForCausalLM(text_cfg)
    except Exception:
        model = Qwen2ForCausalLM(Qwen2Config(**text_cfg.to_dict()))
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    tolerated    = {"model.embed_tokens.weight", "lm_head.weight"}
    real_missing = set(missing) - tolerated
    if real_missing:
        raise RuntimeError(f"HF reference missing keys: {sorted(real_missing)[:8]}")
    if unexpected:
        raise RuntimeError(f"HF reference unexpected keys: {sorted(unexpected)[:8]}")
    # to(device) after load_state_dict(assign=True) — same reason as _build_port
    model = model.to(torch.float32)
    if device is not None:
        model = model.to(device)
    return model.eval()


def _extract_dtype(cfg: dict) -> str:
    """Extract dtype string from compression config."""
    def _normalise(d) -> str | None:
        if d is None:
            return None
        if isinstance(d, str):
            return d
        return str(d).replace("torch.", "")

    gc = cfg.get("global_config") or {}
    if isinstance(gc, dict):
        d = _normalise(gc.get("op_state_spec", {}).get("weight", {}).get("dtype"))
        if d:
            return d
    for v in (cfg.get("module_type_configs") or {}).values():
        if isinstance(v, dict):
            d = _normalise(v.get("op_state_spec", {}).get("weight", {}).get("dtype"))
            if d:
                return d
    return "int4"


# ── Phase 1 — Architecture correctness ───────────────────────────────────────

def phase_correctness(text_cfg, weights_dir: str,
                      device: torch.device | None = None) -> PhaseResult:
    print("\n" + "=" * 60)
    print("PHASE 1 — ARCHITECTURE CORRECTNESS (fp32, port vs HF Qwen2)")
    print("=" * 60)
    print("Random token sequence using real embedding-table inputs.")
    print("Purpose: catch re-authoring bugs, not precision issues.")

    hf   = _build_hf_reference(text_cfg, weights_dir, device)
    port = _build_port(text_cfg, weights_dir, torch.float32, device)
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
    dev = device or torch.device("cpu")
    embed_w   = _load_embed_weights(weights_dir).to(device=dev, dtype=torch.float32)
    input_ids = torch.randint(1, text_cfg.vocab_size, (1, L), dtype=torch.long, device=dev)
    embeds    = embed_w[input_ids]
    pos_ids   = torch.arange(L, dtype=torch.long, device=dev).unsqueeze(0)
    k_fp32, v_fp32 = _make_kv_cache(text_cfg, max_ctx=256, dtype=torch.float32, device=device)

    with torch.no_grad():
        hf_out   = hf(inputs_embeds=embeds, position_ids=pos_ids).logits
        port_out = port(embeds, pos_ids.to(torch.int32), k_fp32, v_fp32)

    for h in handles:
        h.remove()

    print(f"\n{'layer':>6} | {'PSNR (dB)':>10} | note")
    print("-" * 44)
    worst, worst_layer, prev = float("inf"), -1, float("inf")
    for i in range(n_layers):
        score = _psnr(port_acts.get(i, torch.zeros(1)),
                      hf_acts.get(i, torch.zeros(1)))
        drop  = prev - score if np.isfinite(prev) and np.isfinite(score) else 0.0
        note  = f"<-- drops {drop:.0f} dB" if drop > 15 else ""
        print(f"{i:>6} | {score:>10.1f} | {note}")
        if score < worst:
            worst, worst_layer = score, i
        prev = score

    logits_score = _psnr(port_out.float(), hf_out.float())
    print("-" * 44)
    print(f"final logits PSNR : {logits_score:.1f} dB")
    print(f"worst layer       : {worst_layer} ({worst:.1f} dB)")

    if logits_score > CORRECTNESS_PASS:
        print(f"\n[PASS] {logits_score:.1f} dB — port matches HF Qwen2.")
        return PhaseResult.PASS
    if logits_score > CORRECTNESS_MARGINAL:
        print(
            f"\n[MARGINAL] {logits_score:.1f} dB — investigate layer {worst_layer}."
            f"\n  Known-good implementations score 100+ dB. This is worth investigating"
            f"\n  before trusting Phase 4 compression results."
        )
        return PhaseResult.MARGINAL
    print(f"\n[FAIL] {logits_score:.1f} dB — port diverges from HF Qwen2.")
    return PhaseResult.FAIL


# ── Phase 2 — FP32 vs FP16 fidelity ──────────────────────────────────────────

def phase_fidelity(
    text_cfg,
    weights_dir: str,
    variant: str,
    image_path: str = DEFAULT_FIXTURE_IMAGE,
    device: torch.device | None = None,
) -> PhaseResult:
    print("\n" + "=" * 60)
    print("PHASE 2 — FP32 vs FP16 FIDELITY (realistic inputs)")
    print("=" * 60)
    print("Measures what FP32→FP16 narrowing costs on real multimodal inputs.")
    print(f"Image: {Path(image_path).name}")

    # Load realistic fixture
    print("Loading fixture...", end=" ", flush=True)
    try:
        fixture = build_decoder_fixture(
            variant=variant,
            image_path=image_path,
            use_cache=True,
            verbose=False,
        )
        print(f"seq_len={fixture.seq_len} "
              f"({fixture.image_tokens} image + {fixture.text_tokens} text)")
    except FileNotFoundError as e:
        print(f"\n[SKIPPED] Fixture unavailable: {e}")
        print("          Run build_fixtures.py to pre-build corpus fixtures.")
        return PhaseResult.SKIPPED

    hidden   = text_cfg.hidden_size
    n_layers = text_cfg.num_hidden_layers
    n_kv     = text_cfg.num_key_value_heads
    head_dim = hidden // text_cfg.num_attention_heads
    seq_len  = fixture.seq_len
    max_ctx  = seq_len + 64  # must be >= seq_len for prefill to fit in cache

    # Use fixture inputs_embeds — on CPU, cast to appropriate dtype then move to device
    dev = device or torch.device("cpu")
    inputs_embeds_fp32 = fixture.inputs_embeds.float().to(dev)   # [1, seq, hidden]
    inputs_embeds_fp16 = fixture.inputs_embeds.to(dev)           # [1, seq, hidden] fp16
    pos_ids = fixture.position_ids.to(dev)                       # [1, seq] int32

    k32, v32 = _make_kv_cache(text_cfg, max_ctx, torch.float32, device)
    k16, v16 = _make_kv_cache(text_cfg, max_ctx, torch.float16, device)

    port_fp32 = _build_port(text_cfg, weights_dir, torch.float32, device)
    port_fp16 = _build_port(text_cfg, weights_dir, torch.float16, device)

    with torch.no_grad():
        out_fp32 = port_fp32(inputs_embeds_fp32, pos_ids.long(), k32, v32)
        out_fp16 = port_fp16(inputs_embeds_fp16, pos_ids, k16, v16)

    # Report full metric suite
    # Numerical fidelity: all positions
    # Behavioral: final position only (the generation decision)
    report_all   = full_report(out_fp32.float(), out_fp16.float())
    report_final = full_report(out_fp32[:, -1:].float(), out_fp16[:, -1:].float())

    print()
    print_report(report_all,
                 label="FP32 → FP16 (all positions — numerical fidelity):",
                 indent="  ")
    print()
    print_report(report_final,
                 label="FP32 → FP16 (final position — generation decision):",
                 indent="  ")

    # Phase 2 is informational — always MEASURED, never PASS/FAIL
    # The metrics establish the fp16 deployment baseline used as the Phase 4 reference.
    print(f"\n[MEASURED] FP16 deployment baseline established.")
    print(f"           Note: fixture inputs_embeds are stored as FP16, so Phase 2 measures")
    print(f"           decoder FP32 vs FP16 precision, not full pipeline FP32→FP16 loss.")
    return PhaseResult.MEASURED


# ── Phase 3 — KV cache + FP16 health ─────────────────────────────────────────

def phase_cache(
    text_cfg,
    weights_dir: str,
    n_prefill: int = 6,
    n_decode: int = 4,
    device: torch.device | None = None,
) -> PhaseResult:
    print("\n" + "=" * 60)
    print("PHASE 3 — KV CACHE + FP16 HEALTH")
    print("=" * 60)
    print(f"Prefill {n_prefill} tokens, decode {n_decode} steps with KV cache.")

    hidden   = text_cfg.hidden_size
    n_layers = text_cfg.num_hidden_layers
    n_kv     = text_cfg.num_key_value_heads
    head_dim = hidden // text_cfg.num_attention_heads
    total    = n_prefill + n_decode
    max_ctx  = min(total + 4, 256)

    torch.manual_seed(1)
    dev = device or torch.device("cpu")
    port_fp16 = _build_port(text_cfg, weights_dir, torch.float16, device)

    full_embeds = _random_embeds(hidden, total, torch.float16, device)

    # Cached pass: prefill n_prefill tokens, then decode n_decode tokens one-by-one.
    # Position IDs must be the full sequence seen so far — not just the current token.
    # For decode step at position p: position_ids = [0, 1, ..., p] so that
    # seq_len = p+1, L = 1, offset = p inside FastVLMAttention.
    k_cache, v_cache = _make_kv_cache(text_cfg, max_ctx, torch.float16, device)
    cached_logits = []
    with torch.no_grad():
        # Prefill: positions [0..n_prefill-1]
        prefill_pos = torch.arange(n_prefill, dtype=torch.int32, device=dev).unsqueeze(0)
        _ = port_fp16(
            full_embeds[:, :n_prefill],
            prefill_pos,
            k_cache, v_cache,
        )
        # Decode steps: each step passes ALL positions seen so far [0..pos]
        for step in range(n_decode):
            pos = n_prefill + step
            # position_ids = [0..pos] → offset = pos inside the decoder
            pos_dec = torch.arange(pos + 1, dtype=torch.int32, device=dev).unsqueeze(0)
            out = port_fp16(
                full_embeds[:, pos:pos+1],
                pos_dec,
                k_cache, v_cache,
            )
            cached_logits.append(out)
    cached = torch.cat(cached_logits, dim=1)  # [1, n_decode, vocab]

    has_nan  = torch.isnan(cached).any().item()
    has_inf  = torch.isinf(cached).any().item()
    max_abs  = cached.abs().max().item()
    overflow = max_abs > FP16_OVERFLOW
    print(f"fp16 max |logit|  : {max_abs:.0f} (ceiling 65504)")

    # Reference: one full FP16 pass over all tokens (no cache).
    # Then compare cached decode logits at positions [n_prefill..total-1]
    # against the full-pass logits at the same positions.
    # This directly answers: does the incremental cached decode agree with
    # the full-context reference? Any KV cache bug breaks this agreement.
    k_full, v_full = _make_kv_cache(text_cfg, max_ctx, torch.float16, device)
    full_pos = torch.arange(total, dtype=torch.int32, device=dev).unsqueeze(0)
    with torch.no_grad():
        ref_out = port_fp16(full_embeds, full_pos, k_full, v_full)
    ref = ref_out[:, n_prefill:, :]  # [1, n_decode, vocab]

    first = _psnr(cached[:, 0:1], ref[:, 0:1])
    span  = _psnr(cached, ref)
    print(f"first decode PSNR : {first:.1f} dB")
    print(f"decode span PSNR  : {span:.1f} dB")

    if has_nan or has_inf:
        print("\n[FAIL] NaN/Inf in fp16 output.")
        return PhaseResult.FAIL
    if overflow:
        print(f"\n[FAIL] fp16 logits near saturation ({max_abs:.0f}).")
        return PhaseResult.FAIL
    if span > CACHE_PASS and first > CACHE_PASS:
        print(f"\n[PASS] fp16 cached decode clean ({span:.1f} dB).")
        return PhaseResult.PASS
    print(
        "\n[FAIL] Cached decode diverges. Check offset, head reshape, "
        "and that mutable_slice_update resolves to coreai_models."
    )
    return PhaseResult.FAIL


# ── Phase 4 — Compression quality ────────────────────────────────────────────

def _run_one_fixture(
    fixture,
    port_fp16,
    port_comp,
    n_layers: int,
    n_kv: int,
    head_dim: int,
    device: torch.device | None = None,
) -> dict:
    """Run fp16 and compressed models on one fixture, return final-position metrics."""
    seq_len = fixture.seq_len
    max_ctx = seq_len + 64
    kw = dict(dtype=torch.float16) if device is None else dict(dtype=torch.float16, device=device)
    k_ex = torch.zeros(n_layers, 1, n_kv, max_ctx, head_dim, **kw)
    v_ex = torch.zeros_like(k_ex)

    dev = device or torch.device("cpu")
    with torch.no_grad():
        fp16_out = port_fp16(
            fixture.inputs_embeds.to(dev), fixture.position_ids.to(dev),
            k_ex.clone(), v_ex.clone(),
        )
        comp_out = port_comp(
            fixture.inputs_embeds.to(dev), fixture.position_ids.to(dev),
            k_ex.clone(), v_ex.clone(),
        )

    # Final position only — the actual generation decision
    return full_report(fp16_out[:, -1:].float(), comp_out[:, -1:].float())


def phase_compression(
    text_cfg,
    weights_dir: str,
    variant: str,
    compression_config: dict,
    compression_label: str,
    image_path: str = DEFAULT_FIXTURE_IMAGE,
    device: torch.device | None = None,
) -> PhaseResult:
    print("\n" + "=" * 60)
    print(f"PHASE 4 — COMPRESSION QUALITY ({compression_label})")
    print("=" * 60)
    print("Comparing compressed vs fp16 at the final generation position.")
    print("Evaluated over the full fixture corpus for representative coverage.")

    hidden   = text_cfg.hidden_size
    n_layers = text_cfg.num_hidden_layers
    n_kv     = text_cfg.num_key_value_heads
    head_dim = hidden // text_cfg.num_attention_heads

    # Load corpus fixtures — INCONCLUSIVE if unavailable
    from fastvlm_fixtures import build_corpus_fixtures, CORPUS_IMAGES
    print(f"\nLoading corpus fixtures ({len(CORPUS_IMAGES)} images)...")
    try:
        fixtures = build_corpus_fixtures(
            variant=variant,
            use_cache=True,
            verbose=True,
        )
    except Exception as e:
        print(f"\n[INCONCLUSIVE] Could not load corpus fixtures: {e}")
        print("  Phase 4 requires realistic multimodal fixtures.")
        print("  Ensure test_assets/images/ contains the corpus images.")
        return PhaseResult.INCONCLUSIVE

    if not fixtures:
        print("\n[INCONCLUSIVE] No fixtures available — corpus images missing.")
        print("  Phase 4 cannot evaluate recipe quality without realistic inputs.")
        return PhaseResult.INCONCLUSIVE

    from fastvlm_fixtures import CORPUS_IMAGES
    if len(fixtures) != len(CORPUS_IMAGES):
        print(f"\n[INCONCLUSIVE] Corpus incomplete: "
              f"{len(fixtures)}/{len(CORPUS_IMAGES)} fixtures available.")
        print(f"  Recipe recommendation requires the complete corpus.")
        print(f"  Run: python scripts/build_fixtures.py --variant {variant}")
        return PhaseResult.INCONCLUSIVE

    # Small calibration inputs for quantizer prepare — on CPU.
    # coreai_opt quantizer.prepare() does not support MPS — RoPEImpl's freqs
    # buffer is created on CPU at init time and causes device mismatch on MPS.
    # We prepare on CPU then move to device for inference.
    cpu = torch.device("cpu")
    cal_k = torch.zeros(n_layers, 1, n_kv, 64, head_dim, dtype=torch.float16)
    cal_v = torch.zeros_like(cal_k)
    example_inputs = (
        _random_embeds(hidden, 8, torch.float16),
        torch.arange(8, dtype=torch.int32).unsqueeze(0),
        cal_k, cal_v,
    )

    # Build fp16 baseline on device for fast inference
    port_fp16 = _build_port(text_cfg, weights_dir, torch.float16, device)

    # Build and prepare compressed model on CPU, then move to device
    print(f"\nApplying {compression_label}...")
    port_comp = _build_port(text_cfg, weights_dir, torch.float16)  # CPU
    port_comp = apply_quantization_from_config(
        port_comp, compression_config, example_inputs, finalize=False
    )
    port_comp.eval()
    if device is not None:
        port_comp = port_comp.to(device)

    # Evaluate over corpus
    print(f"\nEvaluating {len(fixtures)} fixtures...")
    all_reports = []
    worst_top5  = 1.0
    worst_image = ""

    for fix in fixtures:
        r = _run_one_fixture(fix, port_fp16, port_comp, n_layers, n_kv, head_dim, device)
        all_reports.append((Path(fix.image_path).name, r))
        if r["top5_overlap"] < worst_top5:
            worst_top5  = r["top5_overlap"]
            worst_image = Path(fix.image_path).name

    # Aggregate: mean and worst across corpus
    # Margin ratio: cap at ±100 before averaging to prevent pathological means
    # when fp16 reference margins are near zero (ratio blows up)
    def _mean_capped(key, cap=100.0):
        vals = [max(-cap, min(cap, r[key])) for _, r in all_reports]
        return sum(vals) / len(vals)

    def _mean(key):
        return sum(r[key] for _, r in all_reports) / len(all_reports)
    def _worst(key):
        return min(r[key] for _, r in all_reports)
    def _best(key):
        return max(r[key] for _, r in all_reports)

    n         = len(all_reports)


    print(f"\n  Corpus: {n} images")
    print(f"  Prompt: \"{DEFAULT_PROMPT}\"")
    print(f"  Evaluated at: final generation position (first token decision)")
    print()
    print(f"  {'Metric':<22} {'Mean':>10} {'Worst':>10}")
    print(f"  {'-'*44}")
    print(f"  {'PSNR (dB)':<22} {_mean('psnr_db'):>10.1f} {_worst('psnr_db'):>10.1f}")
    print(f"  {'NRMSE':<22} {_mean('nrmse'):>10.4f} {_best('nrmse'):>10.4f}")
    print(f"  {'Cosine similarity':<22} {_mean('cosine'):>10.4f} {_worst('cosine'):>10.4f}")
    print(f"  {'KL divergence':<22} {_mean('kl_divergence'):>10.4f} {_best('kl_divergence'):>10.4f}")
    print(f"  {'Top-5 overlap':<22} {_mean('top5_overlap'):>9.1%} {_worst('top5_overlap'):>9.1%}")
    print(f"  {'Top-1 agreement':<22} {_mean('top1_agreement'):>9.1%} {_worst('top1_agreement'):>9.1%}   (context only)")
    print(f"  {'Margin preservation':<22} {_mean_capped('margin_ratio'):>10.4f} {_worst('margin_ratio'):>10.4f}")

    top5_mean  = _mean("top5_overlap")
    top5_worst = _worst("top5_overlap")
    top5_pass_count = sum(1 for _, r in all_reports if r["top5_overlap"] >= COMPRESSION_TOP5_PASS)

    print(f"\n  Top-5 ≥{COMPRESSION_TOP5_PASS:.0%}: {top5_pass_count}/{n} images pass")
    if worst_image:
        print(f"  Worst case: {worst_image} (top-5={worst_top5:.1%})")

    # Primary gate: mean AND worst top-5 overlap
    mean_pass  = top5_mean  >= COMPRESSION_TOP5_PASS
    worst_pass = top5_worst >= COMPRESSION_TOP5_FLOOR

    print(f"\n  Primary gate:")
    print(f"    mean  top-5 overlap : {top5_mean:.1%}  (threshold: ≥{COMPRESSION_TOP5_PASS:.0%})")
    print(f"    worst top-5 overlap : {top5_worst:.1%}  (floor: ≥{COMPRESSION_TOP5_FLOOR:.0%})")

    if mean_pass and worst_pass:
        print(f"\n[RECOMMEND] Export {compression_label} recipe for Core AI runtime validation.")
        return PhaseResult.RECOMMEND

    if mean_pass and not worst_pass:
        print(f"\n[CAUTION] Mean quality acceptable but worst-case image ({worst_image}) "
              f"shows significant degradation ({worst_top5:.1%} top-5).")
        print(f"          Consider mixed-precision YAML to protect sensitive inputs.")
        return PhaseResult.CAUTION

    print(f"\n[CAUTION] Recipe quality below threshold (mean top-5={top5_mean:.1%}).")
    print(f"          Consider 8bit instead of 4bit, or a mixed-precision YAML recipe.")
    return PhaseResult.CAUTION


# ── Driver ────────────────────────────────────────────────────────────────────

def verify(
    variant: str,
    stage: str,
    n_prefill: int,
    n_decode: int,
    compression_config: dict | None,
    compression_label: str,
    image_path: str,
) -> None:
    weights_dir = str(REPO_ROOT / "weights" / f"fastvlm-{variant}")
    print(f"Verifying decoder: {variant}")
    if compression_config is not None:
        print(f"Compression:       {compression_label}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device:            {device} (Phase 1 always CPU for IEEE 754 precision)")

    # Phase 1 always runs on CPU regardless of MPS availability.
    # Architecture correctness requires IEEE 754 strict fp32 — MPS uses Metal
    # shaders with different rounding, producing lower PSNR scores that aren't
    # meaningful for bug detection. Phases 2-4 run on MPS for deployment accuracy.
    cpu = torch.device("cpu")

    config   = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    def _exit(result: PhaseResult) -> None:
        sys.exit(0 if result in (PhaseResult.PASS, PhaseResult.MEASURED,
                                  PhaseResult.SKIPPED, PhaseResult.RECOMMEND) else 1)

    if stage == "correctness":
        _exit(phase_correctness(text_cfg, weights_dir, cpu))
        return
    if stage == "fidelity":
        _exit(phase_fidelity(text_cfg, weights_dir, variant, image_path, device))
        return
    if stage == "cache":
        _exit(phase_cache(text_cfg, weights_dir, n_prefill, n_decode, device))
        return
    if stage == "compression":
        if compression_config is None:
            print("ERROR: --stage compression requires --compression or --compression-config.")
            sys.exit(1)
        _exit(phase_compression(
            text_cfg, weights_dir, variant,
            compression_config, compression_label, image_path, device,
        ))
        return

    # all phases
    r1 = phase_correctness(text_cfg, weights_dir, cpu)
    if r1.is_blocking():
        print(f"\n>>> Stopped at Phase 1 ({r1.value}).")
        sys.exit(1)
    r2 = phase_fidelity(text_cfg, weights_dir, variant, image_path, device)
    r3 = phase_cache(text_cfg, weights_dir, n_prefill, n_decode, device)
    if r3.is_blocking():
        print(f"\n>>> Stopped at Phase 3 ({r3.value}).")
        sys.exit(1)

    r4 = PhaseResult.SKIPPED
    if compression_config is not None:
        r4 = phase_compression(
            text_cfg, weights_dir, variant,
            compression_config, compression_label, image_path, device,
        )

    print("\n" + "=" * 60)
    print("DECODER VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"  Phase 1 — Architecture correctness : {r1.value}")
    print(f"  Phase 2 — FP16 fidelity            : {r2.value}")
    print(f"  Phase 3 — KV cache correctness     : {r3.value}")
    if compression_config is not None:
        print(f"  Phase 4 — Recipe quality           : {r4.value}")
    print()
    if r4 == PhaseResult.RECOMMEND:
        print(f"  Recommendation: Export {compression_label} recipe for Core AI runtime validation.")
        print(f"  Note: This does not validate the finalized Core AI artifact.")
        print(f"        Run verify_runtime.py after export for end-to-end validation.")
    elif r4 in (PhaseResult.CAUTION, PhaseResult.INCONCLUSIVE):
        print(f"  {r4.value}: Recipe not recommended for export. See Phase 4 output above.")
    elif r4 == PhaseResult.SKIPPED:
        print(f"  Decoder implementation ready for export testing.")
        print(f"  Add --compression to evaluate a recipe.")
    print("=" * 60)

    if r4.is_blocking():
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "correctness", "fidelity", "cache", "compression"],
        help="Which phase to run (default: all).",
    )
    ap.add_argument("--prefill", type=int, default=6,
                    help="Phase 3: number of prefill tokens.")
    ap.add_argument("--decode",  type=int, default=4,
                    help="Phase 3: number of cached decode steps.")
    ap.add_argument(
        "--image", default=DEFAULT_FIXTURE_IMAGE, metavar="PATH",
        help=f"Image for Phase 2 and Phase 4 fixtures (default: {DEFAULT_FIXTURE_IMAGE}).",
    )

    cg = ap.add_mutually_exclusive_group()
    cg.add_argument(
        "--compression",
        choices=list(MACOS_NAMED_PRESETS.keys()),
        default=None, metavar="PRESET",
        help=f"Named preset: {', '.join(MACOS_NAMED_PRESETS.keys())}.",
    )
    cg.add_argument(
        "--compression-config",
        default=None, metavar="YAML",
        help="Path to quantization_config YAML recipe.",
    )

    args = ap.parse_args()

    compression_config: dict | None = None
    compression_label: str = "none"

    if args.compression_config is not None:
        path = Path(args.compression_config)
        if not path.is_file():
            ap.error(f"--compression-config: not found: {path}")
        compression_config, compression_label = load_compression_config(
            str(path), platform="macOS"
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
        image_path=args.image,
    )
