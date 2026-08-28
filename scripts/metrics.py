"""
metrics.py — Canonical metric module for FastVLM-CoreAI verification.

Shared by verify_decoder.py, scan_quantization_sensitivity.py, and
verify_runtime.py. All metric functions operate on torch.Tensor inputs
and return Python floats.

PSNR is retained for continuity but is no longer the sole pass/fail
criterion. Behavioral metrics (top-k agreement, KL divergence, cosine
similarity, margin preservation) are the primary evidence for recipe
quality in Phase 4 verification.

All functions:
  - Accept fp16 or fp32 tensors, compute in float64 for stability
  - Accept any shape — flattened internally where needed
  - Return float (inf where appropriate, e.g. identical inputs)
  - Are stateless and side-effect-free

Reference convention: `ref` is always the baseline (fp16 or fp32
decoder output), `test` is the compressed/candidate output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── Numerical fidelity ────────────────────────────────────────────────────────

def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB).

    Peak is derived from the reference tensor's absolute maximum.
    Computed in float64 for stability.

    Returns inf when ref == test (zero error).
    Returns -inf when ref is all-zero (undefined peak).
    """
    ref_f  = ref.detach().double()
    test_f = test.detach().double()
    mse    = torch.mean((ref_f - test_f) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = ref_f.abs().max().item()
    if peak == 0.0:
        return float("-inf")
    return 20.0 * torch.log10(torch.tensor(peak)).item() \
         - 10.0 * torch.log10(torch.tensor(mse)).item()


def nrmse(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Normalized Root Mean Square Error.

    Normalized by the RMS of the reference, so the result is
    dimensionless and comparable across different output scales.
    Returns 0.0 when ref == test, inf when ref is all-zero.
    """
    ref_f  = ref.detach().double()
    test_f = test.detach().double()
    mse    = torch.mean((ref_f - test_f) ** 2).item()
    ref_rms = torch.sqrt(torch.mean(ref_f ** 2)).item()
    if ref_rms == 0.0:
        return float("inf") if mse > 0 else 0.0
    return (mse ** 0.5) / ref_rms


def cosine_similarity(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Cosine similarity between flattened ref and test tensors.

    Measures preservation of the overall logit direction.
    Returns values in [-1, 1]; 1.0 = identical direction.
    """
    ref_f  = ref.detach().double().flatten()
    test_f = test.detach().double().flatten()
    dot    = torch.dot(ref_f, test_f).item()
    norm_r = torch.norm(ref_f).item()
    norm_t = torch.norm(test_f).item()
    if norm_r == 0.0 or norm_t == 0.0:
        return 0.0
    return dot / (norm_r * norm_t)


def kl_divergence(ref: torch.Tensor, test: torch.Tensor) -> float:
    """KL divergence KL(ref_softmax || test_softmax) over last dimension.

    Measures distributional change in next-token probability distributions.
    Operates on raw logits — applies softmax internally.
    Averages over all positions if ref has multiple sequence positions.

    Returns 0.0 when ref == test.
    """
    ref_f  = ref.detach().float()
    test_f = test.detach().float()
    # Flatten to (N, vocab) where N = batch * seq positions
    if ref_f.dim() > 2:
        ref_f  = ref_f.reshape(-1, ref_f.shape[-1])
        test_f = test_f.reshape(-1, test_f.shape[-1])
    elif ref_f.dim() == 1:
        ref_f  = ref_f.unsqueeze(0)
        test_f = test_f.unsqueeze(0)

    log_p = F.log_softmax(ref_f,  dim=-1)
    log_q = F.log_softmax(test_f, dim=-1)
    p     = log_p.exp()
    # KL(p || q) = sum p * (log_p - log_q)
    kl_per_pos = (p * (log_p - log_q)).sum(dim=-1)
    return kl_per_pos.mean().item()


# ── Decision / behavioral fidelity ────────────────────────────────────────────

def top_k_agreement(
    ref: torch.Tensor,
    test: torch.Tensor,
    k: int = 1,
) -> float:
    """Fraction of positions where the top-k tokens agree.

    k=1: whether the candidate selects the same next token as ref.
    k>1: whether the candidate's top-k set equals ref's top-k set.

    Operates on last dimension (vocab). Averages over all positions.
    Returns values in [0, 1]; 1.0 = perfect agreement.
    """
    ref_f  = ref.detach().float()
    test_f = test.detach().float()

    if ref_f.dim() > 2:
        ref_f  = ref_f.reshape(-1, ref_f.shape[-1])
        test_f = test_f.reshape(-1, test_f.shape[-1])
    elif ref_f.dim() == 1:
        ref_f  = ref_f.unsqueeze(0)
        test_f = test_f.unsqueeze(0)

    if k == 1:
        ref_top  = ref_f.argmax(dim=-1)
        test_top = test_f.argmax(dim=-1)
        return (ref_top == test_top).float().mean().item()

    # k > 1: set intersection / k
    ref_topk  = ref_f.topk(k, dim=-1).indices.sort(dim=-1).values
    test_topk = test_f.topk(k, dim=-1).indices.sort(dim=-1).values
    match = (ref_topk == test_topk).all(dim=-1)
    return match.float().mean().item()


def top_k_overlap(
    ref: torch.Tensor,
    test: torch.Tensor,
    k: int = 5,
) -> float:
    """Mean fraction of overlap between ref and test top-k token sets.

    Unlike top_k_agreement (which requires exact set match), this measures
    partial overlap: |ref_topk ∩ test_topk| / k.
    Returns values in [0, 1]; 1.0 = complete overlap.
    """
    ref_f  = ref.detach().float()
    test_f = test.detach().float()

    if ref_f.dim() > 2:
        ref_f  = ref_f.reshape(-1, ref_f.shape[-1])
        test_f = test_f.reshape(-1, test_f.shape[-1])
    elif ref_f.dim() == 1:
        ref_f  = ref_f.unsqueeze(0)
        test_f = test_f.unsqueeze(0)

    n_pos = ref_f.shape[0]
    overlaps = []
    ref_topk  = ref_f.topk(k, dim=-1).indices
    test_topk = test_f.topk(k, dim=-1).indices

    for i in range(n_pos):
        ref_set  = set(ref_topk[i].tolist())
        test_set = set(test_topk[i].tolist())
        overlaps.append(len(ref_set & test_set) / k)

    return sum(overlaps) / len(overlaps)


def margin_preservation(
    ref: torch.Tensor,
    test: torch.Tensor,
) -> float:
    """Margin preservation of the reference decision.

    Computes the logit gap between the reference top-1 and top-2 tokens
    in BOTH distributions, then returns the ratio test/ref.

    ref_margin  = ref[ref_top1] - ref[ref_top2]
    test_margin = test[ref_top1] - test[ref_top2]
    result      = test_margin / ref_margin

    Interpretation:
      > 0 : test preserves the reference ordering (ref_top1 still leads ref_top2)
      < 0 : test reverses the reference ordering (ref_top2 now beats ref_top1)
      ~ 1 : test is as confident as ref about the reference decision
      > 1 : test is MORE confident than ref (can happen with quantization sharpening)

    Averages over all positions.
    """
    ref_f  = ref.detach().float()
    test_f = test.detach().float()

    if ref_f.dim() > 2:
        ref_f  = ref_f.reshape(-1, ref_f.shape[-1])
        test_f = test_f.reshape(-1, test_f.shape[-1])
    elif ref_f.dim() == 1:
        ref_f  = ref_f.unsqueeze(0)
        test_f = test_f.unsqueeze(0)

    # Reference top-1 and top-2 token indices
    ref_top2_vals, ref_top2_idx = ref_f.topk(2, dim=-1)
    ref_top1_idx = ref_top2_idx[:, 0]   # [N]
    ref_top2_idx = ref_top2_idx[:, 1]   # [N]

    # Reference margin: ref[ref_top1] - ref[ref_top2]
    ref_margin = ref_top2_vals[:, 0] - ref_top2_vals[:, 1]  # [N]

    # Test margin: test[ref_top1] - test[ref_top2] (same token positions)
    n = ref_f.shape[0]
    idx = torch.arange(n)
    test_at_ref_top1 = test_f[idx, ref_top1_idx]   # [N]
    test_at_ref_top2 = test_f[idx, ref_top2_idx]   # [N]
    test_margin = test_at_ref_top1 - test_at_ref_top2  # [N]

    # Avoid division by zero when ref margin is negligible
    safe_ref_margin = ref_margin.clamp(min=1e-4)
    ratio = test_margin / safe_ref_margin
    return ratio.mean().item()


# ── Composite report ──────────────────────────────────────────────────────────

def full_report(
    ref: torch.Tensor,
    test: torch.Tensor,
    label: str = "",
) -> dict[str, float]:
    """Compute the full metric suite and return as a dict.

    Suitable for Phase 4 recipe quality reporting.
    """
    return {
        "psnr_db":          psnr(ref, test),
        "nrmse":            nrmse(ref, test),
        "cosine":           cosine_similarity(ref, test),
        "kl_divergence":    kl_divergence(ref, test),
        "top1_agreement":   top_k_agreement(ref, test, k=1),
        "top5_overlap":     top_k_overlap(ref, test, k=5),
        "margin_ratio":     margin_preservation(ref, test),
    }


def print_report(
    metrics: dict[str, float],
    label: str = "",
    indent: str = "",
) -> None:
    """Print a formatted metric report."""
    if label:
        print(f"{indent}{label}")
    print(f"{indent}  PSNR              : {metrics['psnr_db']:>8.1f} dB")
    print(f"{indent}  NRMSE             : {metrics['nrmse']:>8.4f}")
    print(f"{indent}  Cosine similarity : {metrics['cosine']:>8.4f}")
    print(f"{indent}  KL divergence     : {metrics['kl_divergence']:>8.4f}")
    print(f"{indent}  Top-1 agreement   : {metrics['top1_agreement']:>7.1%}")
    print(f"{indent}  Top-5 overlap     : {metrics['top5_overlap']:>7.1%}")
    print(f"{indent}  Margin ratio      : {metrics['margin_ratio']:>8.4f}")
