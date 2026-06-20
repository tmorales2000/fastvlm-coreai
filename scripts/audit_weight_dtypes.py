"""
audit_weight_dtypes.py — Exhaustive dtype/shape audit of HF and Apple MLX
FastVLM checkpoints, across all three variants (0.5b, 1.5b, 7b).

PURPOSE
-------
We've been relying on spot-checks (a handful of grep'd keys) to infer:
  - HF checkpoint storage dtype is uniformly bfloat16
  - Apple's MLX quantizes ONLY nn.Linear weight matrices (not embeddings,
    not norms, not biases)
  - The set of quantized vs non-quantized module types is identical across
    0.5b/1.5b/7b (modulo 0.5b having no quantization at all)
  - Non-quantized tensors in MLX output are uniformly float16

This script checks ALL of these assumptions exhaustively, for every single
tensor in every checkpoint, rather than continuing to spot-check. Any
tensor that doesn't fit the expected pattern is flagged loudly.

WHAT IT REPORTS, PER VARIANT
------------------------------
1. HF checkpoint: dtype histogram across ALL tensors. Flags anything that
   isn't bfloat16.
2. MLX checkpoint: for every tensor, classifies it into one of:
     - quantized weight   (uint32, has matching .scales/.biases siblings)
     - quantization param (.scales or .biases, fp16)
     - unquantized float  (fp16, fp32, or bf16 — flags if not fp16)
     - unexpected         (anything not matching the above — always flagged)
   Then groups by "module kind" (embed_tokens, lm_head, layers.N.self_attn.*,
   layers.N.mlp.*, layers.N.*norm*) to show which module KINDS are quantized
   vs not, with a kind-level summary (not per-layer, since that's 28 rows
   of noise — but flags if behavior differs across layers within a variant).
3. Cross-variant comparison: are the same module kinds quantized in 1.5b
   and 7b? Any kind quantized in one but not the other is flagged.

USAGE
-----
  # HF checkpoint audit only (no MLX comparison, no external path needed):
  python scripts/audit_weight_dtypes.py

  # Full audit including Apple MLX checkpoints:
  python scripts/audit_weight_dtypes.py --mlx-checkpoints-dir /path/to/ml-fastvlm/checkpoints

  # Single variant, full audit:
  python scripts/audit_weight_dtypes.py --variant 7b --mlx-checkpoints-dir /path/to/ml-fastvlm/checkpoints

The MLX checkpoint subdirectory NAMES (e.g. llava-fastvithd_7b_stage3_llm.int4)
are fixed by Apple/mlx-vlm's own naming convention and are hardcoded in
MLX_DIR_NAMES below. The PARENT path to those directories is
environment-specific (depends on where ml-fastvlm was cloned on this
machine) and is intentionally NEVER hardcoded -- it must be supplied via
--mlx-checkpoints-dir, or MLX auditing is skipped entirely.
"""

import argparse
import collections
import os
import re

from safetensors import safe_open

HF_PATHS = {
    "0.5b": "weights/fastvlm-0.5b",
    "1.5b": "weights/fastvlm-1.5b",
    "7b":   "weights/fastvlm-7b",
}

# Directory NAMES are fixed (Apple/mlx-vlm's own naming convention), but the
# PARENT path to them is environment-specific (depends on where ml-fastvlm
# was cloned) and must never be hardcoded. Supplied via --mlx-checkpoints-dir;
# if omitted, MLX audit steps are skipped entirely and only the HF checkpoint
# audit runs.
MLX_DIR_NAMES = {
    "0.5b": "llava-fastvithd_0.5b_stage3_llm.fp16",
    "1.5b": "llava-fastvithd_1.5b_stage3_llm.int8",
    "7b":   "llava-fastvithd_7b_stage3_llm.int4",
}


def _mlx_paths(mlx_checkpoints_dir: str | None) -> dict[str, str]:
    """
    Build the variant -> full MLX checkpoint path mapping from a
    user-supplied parent directory. Returns an empty dict (all variants
    skipped) if mlx_checkpoints_dir is None.
    """
    if mlx_checkpoints_dir is None:
        return {}
    base = os.path.expanduser(mlx_checkpoints_dir)
    return {variant: os.path.join(base, name) for variant, name in MLX_DIR_NAMES.items()}

EXPECTED_NON_FP16_OK = set()  # nothing is expected to deviate; flag everything


def _safetensor_files(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))


def _module_kind(key: str) -> str:
    """
    Collapse a full tensor key down to a 'module kind' for grouping, e.g.:
      language_model.model.layers.13.self_attn.q_proj.weight -> self_attn.q_proj
      language_model.model.layers.13.input_layernorm.weight  -> input_layernorm
      language_model.lm_head.weight                          -> lm_head
      language_model.model.embed_tokens.weight                -> embed_tokens
    Strips the layer index so all 28 layers' q_proj collapse into one row.

    IMPORTANT: does NOT strip the .weight/.bias/.scales/.biases suffix.
    An earlier version stripped it, which collapsed e.g. q_proj.weight
    (quantized, uint32) and q_proj.bias (never quantized, fp16) into the
    same 'self_attn.q_proj' kind, producing false MIXED flags — a
    quantized weight with an unquantized bias is completely normal, not
    an inconsistency. Keeping the suffix means q_proj.weight and q_proj.bias
    are tracked as separate kinds, so MIXED only fires for genuine
    same-tensor-type inconsistency (e.g. layer 3's q_proj.weight quantized
    but layer 7's q_proj.weight not — which WOULD be a real problem).
    """
    k = key
    for prefix in ("language_model.", "model.mm_projector.", "model.vision_tower."):
        k = k.removeprefix(prefix)
    k = re.sub(r"^model\.", "", k)
    k = re.sub(r"layers\.\d+\.", "", k)
    return k


# ─── HF checkpoint audit ───────────────────────────────────────────────────────


def audit_hf(variant: str, path: str) -> None:
    print(f"\n{'─'*72}")
    print(f"HF CHECKPOINT — {variant}  ({path})")
    print("─" * 72)

    files = _safetensor_files(path)
    if not files:
        print(f"  [SKIP] no .safetensors files found at {path}")
        return

    dtype_counts: dict[str, int] = collections.Counter()
    dtype_examples: dict[str, tuple[str, list]] = {}
    total = 0
    non_float_keys = []  # int64 etc. -- bookkeeping tensors, not weights

    for fname in files:
        with safe_open(os.path.join(path, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                d = str(t.dtype)
                dtype_counts[d] += 1
                if d not in dtype_examples:
                    dtype_examples[d] = (k, list(t.shape))
                total += 1
                if not t.is_floating_point():
                    non_float_keys.append((k, d))

    print(f"  Total tensors: {total}")
    for d, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        ek, eshape = dtype_examples[d]
        flag = "" if d == "torch.bfloat16" else "  <-- non-bfloat16 (see classification below)"
        print(f"    {d:18s} : {count:4d} tensors  (e.g. {ek} {eshape}){flag}")

    if non_float_keys:
        print(f"  [INFO] {len(non_float_keys)} tensor(s) are non-floating-point "
              f"bookkeeping (e.g. BatchNorm num_batches_tracked) -- expected, "
              f"not weights, excluded from the bfloat16 check.")

    float_total = total - len(non_float_keys)
    float_non_bf16 = dtype_counts.get("torch.bfloat16", 0)
    float_non_bf16 = float_total - dtype_counts.get("torch.bfloat16", 0)
    if float_non_bf16 == 0:
        print(f"  [CONFIRMED] All {float_total} floating-point weight tensors are bfloat16.")
    else:
        print(f"  [FLAG] {float_non_bf16} floating-point tensor(s) are NOT bfloat16.")


# ─── MLX checkpoint audit ──────────────────────────────────────────────────────


def audit_mlx(variant: str, path: str) -> dict[str, str]:
    """
    Returns a dict: module_kind -> classification
      ("quantized" | "unquantized_fp16" | "unquantized_other" | "mixed")
    for cross-variant comparison.
    """
    print(f"\n{'─'*72}")
    print(f"MLX CHECKPOINT — {variant}  ({path})")
    print("─" * 72)

    files = _safetensor_files(path)
    if not files:
        print(f"  [SKIP] no .safetensors files found at {path}")
        return {}

    all_keys: dict[str, tuple[str, list]] = {}
    for fname in files:
        with safe_open(os.path.join(path, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                all_keys[k] = (str(t.dtype), list(t.shape))

    print(f"  Total tensors: {len(all_keys)}")

    # Identify quantized weight tensors: uint32 dtype, with .scales/.biases siblings
    quantized_bases = set()
    scale_bases = set()
    bias_bases = set()
    weight_keys = set()
    other_keys = set()

    for k, (dtype, shape) in all_keys.items():
        if k.endswith(".weight") and dtype == "torch.uint32":
            quantized_bases.add(k[: -len(".weight")])
            weight_keys.add(k)
        elif k.endswith(".scales"):
            scale_bases.add(k[: -len(".scales")])
        elif k.endswith(".biases"):
            bias_bases.add(k[: -len(".biases")])
        else:
            other_keys.add(k)

    # Sanity: every quantized weight should have matching scales + biases
    missing_scales = quantized_bases - scale_bases
    missing_biases = quantized_bases - bias_bases
    extra_scales = scale_bases - quantized_bases
    extra_biases = bias_bases - quantized_bases

    if missing_scales:
        print(f"  [FLAG] {len(missing_scales)} quantized weight(s) missing .scales: "
              f"{sorted(missing_scales)[:3]} ...")
    if missing_biases:
        print(f"  [FLAG] {len(missing_biases)} quantized weight(s) missing .biases: "
              f"{sorted(missing_biases)[:3]} ...")
    if extra_scales:
        print(f"  [FLAG] {len(extra_scales)} .scales tensor(s) with no uint32 weight: "
              f"{sorted(extra_scales)[:3]} ...")
    if extra_biases:
        print(f"  [FLAG] {len(extra_biases)} .biases tensor(s) with no uint32 weight: "
              f"{sorted(extra_biases)[:3]} ...")
    if not (missing_scales or missing_biases or extra_scales or extra_biases):
        print(f"  [OK] Every quantized weight has matching .scales + .biases "
              f"({len(quantized_bases)} quantized tensors).")

    # Classify every remaining tensor not part of a quantized triple
    quant_related_keys = set()
    for base in quantized_bases:
        quant_related_keys.add(base + ".weight")
        quant_related_keys.add(base + ".scales")
        quant_related_keys.add(base + ".biases")

    unquantized_keys = set(all_keys.keys()) - quant_related_keys
    unquantized_dtype_counts: dict[str, int] = collections.Counter()
    unquantized_non_fp16 = []
    for k in unquantized_keys:
        dtype, shape = all_keys[k]
        unquantized_dtype_counts[dtype] += 1
        if dtype != "torch.float16":
            unquantized_non_fp16.append((k, dtype, shape))

    print(f"\n  Unquantized tensors: {len(unquantized_keys)}")
    for d, count in sorted(unquantized_dtype_counts.items(), key=lambda x: -x[1]):
        flag = "" if d == "torch.float16" else "  <-- UNEXPECTED, not float16"
        print(f"    {d:18s} : {count:4d} tensors{flag}")
    if unquantized_non_fp16:
        print(f"  [FLAG] {len(unquantized_non_fp16)} unquantized tensor(s) NOT float16:")
        for k, d, shape in unquantized_non_fp16[:10]:
            print(f"      {k}  ({d}, {shape})")
    else:
        print(f"  [CONFIRMED] All unquantized tensors are float16.")

    # Module-kind level summary: which KINDS of modules are quantized?
    kind_status: dict[str, set] = collections.defaultdict(set)
    for k in weight_keys:
        kind_status[_module_kind(k)].add("quantized")
    for k in unquantized_keys:
        if k.endswith((".weight", ".bias")):
            kind_status[_module_kind(k)].add("unquantized")

    print(f"\n  Module-kind quantization summary:")
    print(f"  {'kind':35s} status")
    result: dict[str, str] = {}
    for kind in sorted(kind_status.keys()):
        statuses = kind_status[kind]
        if statuses == {"quantized"}:
            label = "QUANTIZED"
        elif statuses == {"unquantized"}:
            label = "fp16 (not quantized)"
        else:
            label = f"MIXED {statuses}  <-- FLAG: inconsistent across layers!"
        result[kind] = label
        print(f"    {kind:35s} {label}")

    return result


# ─── Cross-variant comparison ──────────────────────────────────────────────────


def compare_variants(results: dict[str, dict[str, str]]) -> None:
    print(f"\n{'─'*72}")
    print("CROSS-VARIANT MODULE-KIND COMPARISON")
    print("─" * 72)

    variants = [v for v in ("0.5b", "1.5b", "7b") if v in results and results[v]]
    if len(variants) < 2:
        print("  [SKIP] need at least 2 variants with data to compare.")
        return

    all_kinds = set()
    for v in variants:
        all_kinds.update(results[v].keys())

    header = f"  {'module kind':35s}" + "".join(f"{v:>22s}" for v in variants)
    print(header)
    print("  " + "-" * (35 + 22 * len(variants)))

    any_flag = False
    for kind in sorted(all_kinds):
        row_vals = [results[v].get(kind, "—") for v in variants]
        # Flag if the QUANTIZED/not-quantized status differs across variants
        # (0.5b is expected to differ from 1.5b/7b since it's unquantized
        # entirely — only flag if 1.5b and 7b disagree with each other)
        flag = ""
        if "1.5b" in variants and "7b" in variants:
            v15 = results["1.5b"].get(kind, "—")
            v7 = results["7b"].get(kind, "—")
            is_quant_15 = v15.startswith("QUANTIZED")
            is_quant_7 = v7.startswith("QUANTIZED")
            if v15 != "—" and v7 != "—" and is_quant_15 != is_quant_7:
                flag = "  <-- FLAG: 1.5b/7b disagree!"
                any_flag = True
        row = f"  {kind:35s}" + "".join(f"{v:>22s}" for v in row_vals) + flag
        print(row)

    if not any_flag:
        print("\n  [CONFIRMED] 1.5b and 7b quantize the identical set of module kinds.")
    else:
        print("\n  [FLAG] 1.5b and 7b quantize DIFFERENT module kinds — see rows above.")


# ─── Driver ───────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Exhaustive dtype/quantization audit of HF and Apple MLX "
                    "FastVLM checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # HF checkpoint audit only (no MLX comparison):
  python scripts/audit_weight_dtypes.py

  # Full audit including Apple MLX checkpoints:
  python scripts/audit_weight_dtypes.py --mlx-checkpoints-dir ~/git/tmorales2000/ml-fastvlm/checkpoints

  # Single variant:
  python scripts/audit_weight_dtypes.py --variant 7b --mlx-checkpoints-dir ~/git/tmorales2000/ml-fastvlm/checkpoints
""",
    )
    ap.add_argument("--variant", choices=["0.5b", "1.5b", "7b"], default=None,
                     help="Audit a single variant only. Default: all three.")
    ap.add_argument("--mlx-checkpoints-dir", default=None,
                     help="Parent directory containing Apple's MLX checkpoint "
                          "subdirectories (e.g. .../ml-fastvlm/checkpoints). "
                          "The subdirectory NAMES are fixed (Apple's own "
                          "naming convention, see MLX_DIR_NAMES) but this "
                          "parent path is environment-specific and is never "
                          "hardcoded. If omitted, only the HF checkpoint "
                          "audit runs and all MLX comparison is skipped.")
    args = ap.parse_args()

    variants = [args.variant] if args.variant else ["0.5b", "1.5b", "7b"]
    mlx_paths = _mlx_paths(args.mlx_checkpoints_dir)

    print("=" * 72)
    print("FastVLM weight dtype/quantization audit")
    print("=" * 72)

    if not mlx_paths:
        print("\n[INFO] --mlx-checkpoints-dir not supplied — running HF-only audit.")
        print("       Pass --mlx-checkpoints-dir <path> to also audit Apple's MLX weights.")

    mlx_results: dict[str, dict[str, str]] = {}

    for variant in variants:
        audit_hf(variant, HF_PATHS[variant])
        if variant in mlx_paths:
            mlx_results[variant] = audit_mlx(variant, mlx_paths[variant])

    if args.variant is None and mlx_paths:
        compare_variants(mlx_results)

    print(f"\n{'='*72}")
    print("AUDIT COMPLETE — review any [FLAG] lines above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
