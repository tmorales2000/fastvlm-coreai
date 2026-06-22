"""
compare_weights.py

Compares Apple MLX quantized weights (dequantized back to fp32) against the
original bf16 HuggingFace weights, layer by layer. Optionally also runs our
coreai-opt quantization from fp32 and compares side by side.

Usage:
    # Apple MLX only (fast):
    python scripts/compare_weights.py --variant 7b
    python scripts/compare_weights.py --variant 1.5b

    # Apple MLX + our coreai-opt scheme (slower, requires coreai_opt):
    python scripts/compare_weights.py --variant 7b --ours
    python scripts/compare_weights.py --variant 1.5b --ours

    # All layers (not just sample):
    python scripts/compare_weights.py --variant 7b --all-layers

Modes compared:
    Apple PSNR : HF bf16 vs Apple MLX dequantized (their pipeline)
    Ours PSNR  : HF bf16 vs our coreai-opt quantize→dequantize from fp32
    Delta      : Ours - Apple (positive = we're better, negative = we're worse)
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

VARIANT_CONFIG = {
    "1.5b": {
        "hf_path": "weights/fastvlm-1.5b",
        "mlx_path": "weights/fastvlm-1.5b-int8",  # HF MLX weights (Aug 2025)
        "bits": 8,
        "group_size": 64,
        "layer_count": 28,
    },
    "7b": {
        "hf_path": "weights/fastvlm-7b",
        "mlx_path": "weights/fastvlm-7b-int4",    # HF MLX weights (Aug 2025)
        "bits": 4,
        "group_size": 64,
        "layer_count": 28,
    },
}

ALL_MODULES = [
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]

SAMPLE_MODULES = [
    "mlp.down_proj",
    "mlp.gate_proj",
    "self_attn.q_proj",
    "self_attn.o_proj",
]

SAMPLE_LAYERS = [0, 3, 6, 9, 13, 17, 20, 24, 27]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    maxv = a.float().abs().max().item()
    if maxv == 0:
        return float("inf")
    return 20 * math.log10(maxv) - 10 * math.log10(mse)


def mlx_dequantize(w_q: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
                   bits: int, group_size: int) -> torch.Tensor:
    """
    Dequantize MLX affine-quantized weights back to fp32.

    MLX packs (32 // bits) values into each uint32, LSB first.
    Dequantization: w = scale * q + bias, per group of `group_size` input features.
    MLX uses signed scales — negative scale encodes groups whose range is
    centered below zero, as an optimization in the Metal quantization kernel.
    """
    values_per_uint32 = 32 // bits
    out_features, packed_cols = w_q.shape
    in_features = packed_cols * values_per_uint32
    mask = (1 << bits) - 1

    w_q_np = w_q.numpy().astype(np.uint32)
    unpacked = np.zeros((out_features, in_features), dtype=np.float32)
    for i in range(values_per_uint32):
        unpacked[:, i::values_per_uint32] = (w_q_np >> (i * bits)) & mask

    n_groups = in_features // group_size
    s = scales.numpy().astype(np.float32)
    b = biases.numpy().astype(np.float32)
    for g in range(n_groups):
        col = slice(g * group_size, (g + 1) * group_size)
        unpacked[:, col] = unpacked[:, col] * s[:, g:g+1] + b[:, g:g+1]

    return torch.from_numpy(unpacked)


def our_dequantize(w_bf16: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """
    Simulate our coreai-opt quantization scheme (fp16->int8/int4, PerBlock
    axis=1 block_size=64, ASYMMETRIC) by implementing the quantization
    math directly in numpy — identical to how mlx_dequantize works for
    Apple's weights, but applied from fp16 input (matching our Stage 3 pipeline).

    Uses the same asymmetric affine formula:
      scale  = (max - min) / (2^bits - 1)
      zero   = round(-min / scale)   [standard asymmetric zero-point]
      q      = clamp(round(w/scale) + zero, 0, 2^bits-1)
      w_dq   = (q - zero) * scale    == scale*q + bias  where bias = -zero*scale
    """
    import numpy as np

    # Cast bf16->fp16, matching our Stage 3 pipeline
    w = w_bf16.to(torch.float16).float().numpy()  # [out_f, in_f]
    out_f, in_f = w.shape
    n_groups = in_f // group_size
    n_levels = (1 << bits) - 1  # 255 for int8, 15 for int4
    result = np.zeros_like(w)

    for g in range(n_groups):
        col = slice(g * group_size, (g + 1) * group_size)
        block = w[:, col]  # [out_f, group_size]
        w_min = block.min(axis=1, keepdims=True)
        w_max = block.max(axis=1, keepdims=True)
        scale = (w_max - w_min) / n_levels
        scale = np.where(scale == 0, 1e-8, scale)
        zero = np.round(-w_min / scale).clip(0, n_levels)
        q = np.round(block / scale + zero).clip(0, n_levels)
        result[:, col] = (q - zero) * scale

    return torch.from_numpy(result).float()


def load_hf_weights(hf_path: str, layer_indices: list, modules: list) -> dict:
    weights = {}
    for fname in sorted(os.listdir(hf_path)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(hf_path, fname), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if not k.endswith(".weight"):
                    continue
                for layer_idx in layer_indices:
                    for mod in modules:
                        if f"layers.{layer_idx}.{mod}.weight" in k:
                            weights[k] = sf.get_tensor(k).float()
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["1.5b", "7b"], required=True)
    parser.add_argument("--ours", action="store_true",
                        help="Also run our coreai-opt quantization for comparison")
    parser.add_argument("--all-layers", action="store_true",
                        help="Compare all 28 layers (default: 9 sample layers)")
    parser.add_argument("--all-modules", action="store_true",
                        help="Compare all 7 module types (default: 4 sample modules)")
    args = parser.parse_args()

    cfg = VARIANT_CONFIG[args.variant]
    hf_path = cfg["hf_path"]
    mlx_path = cfg["mlx_path"]
    bits = cfg["bits"]
    group_size = cfg["group_size"]
    layer_count = cfg["layer_count"]

    layer_indices = list(range(layer_count)) if args.all_layers else SAMPLE_LAYERS
    modules = ALL_MODULES if args.all_modules else SAMPLE_MODULES

    print(f"\nVariant: {args.variant}  |  MLX bits: {bits}  |  group_size: {group_size}")
    print(f"Layers: {len(layer_indices)}  |  Modules: {len(modules)}")
    print(f"Comparing: Apple MLX dequantized{'  +  our coreai-opt fp32→int' + str(bits) if args.ours else ''}")

    print("\nLoading HF weights...")
    hf_weights = load_hf_weights(hf_path, layer_indices, modules)
    print(f"Loaded {len(hf_weights)} tensors")

    mlx_file = os.path.join(mlx_path, "model.safetensors")

    # Header
    if args.ours:
        print("\n" + "-" * 100)
        print(f"{'Layer/Module':<52} {'Apple PSNR':>12} {'Ours PSNR':>12} {'Delta':>8}  {'verdict'}")
        print("-" * 100)
    else:
        print("\n" + "-" * 80)
        print(f"{'Layer/Module':<52} {'Apple PSNR':>12}  {'shape'}")
        print("-" * 80)

    apple_psnrs = []
    our_psnrs = []

    # Per-layer stats for the layer-level summary
    layer_apple = {}  # layer_idx -> [psnr values]
    layer_ours = {}

    with safe_open(mlx_file, framework="pt", device="cpu") as sf:
        mlx_keys = set(sf.keys())

        for hf_key in sorted(hf_weights.keys()):
            hf_w = hf_weights[hf_key]

            # Extract layer index from key for per-layer aggregation
            # HF key format: model.layers.N.module.weight
            parts = hf_key.split(".")
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                layer_idx = -1

            # Apple MLX path
            mlx_base = "language_model." + hf_key
            w_key = mlx_base
            s_key = mlx_base.replace(".weight", ".scales")
            b_key = mlx_base.replace(".weight", ".biases")

            if w_key not in mlx_keys:
                print(f"{hf_key:<52} NOT FOUND in MLX")
                continue

            w_q = sf.get_tensor(w_key)
            scales = sf.get_tensor(s_key).float()
            biases = sf.get_tensor(b_key).float()
            apple_dq = mlx_dequantize(w_q, scales, biases, bits, group_size)

            if apple_dq.shape != hf_w.shape:
                print(f"{hf_key:<52} SHAPE MISMATCH")
                continue

            ap = psnr(hf_w, apple_dq)
            apple_psnrs.append(ap)
            layer_apple.setdefault(layer_idx, []).append(ap)

            if args.ours:
                try:
                    our_dq = our_dequantize(hf_w, bits, group_size)
                    op = psnr(hf_w, our_dq)
                except Exception as e:
                    op = float("nan")
                our_psnrs.append(op)
                layer_ours.setdefault(layer_idx, []).append(op)

                delta = op - ap if not math.isnan(op) else float("nan")
                verdict = "=" if abs(delta) < 1 else ("▲ ours" if delta > 0 else "▼ apple")
                print(f"{hf_key:<52} {ap:>12.1f} {op:>12.1f} {delta:>+8.1f}  {verdict}")
            else:
                shape_str = str(list(hf_w.shape))
                print(f"{hf_key:<52} {ap:>12.1f}  {shape_str}")

    # Summary
    finite_apple = [p for p in apple_psnrs if math.isfinite(p)]

    if args.ours:
        finite_ours = [p for p in our_psnrs if math.isfinite(p)]
        print("\n" + "=" * 100)
        print(f"{'SUMMARY':<52} {'Apple':>12} {'Ours':>12} {'Delta':>8}")
        print("-" * 100)
        def safemean(v): return sum(v)/len(v) if v else float("nan")
        def safemin(v): return min(v) if v else float("nan")
        def safemax(v): return max(v) if v else float("nan")
        am, om = safemean(finite_apple), safemean(finite_ours)
        print(f"{'Mean PSNR':<52} {am:>12.1f} {om:>12.1f} {om-am:>+8.1f}")
        print(f"{'Min PSNR':<52} {safemin(finite_apple):>12.1f} {safemin(finite_ours):>12.1f} {safemin(finite_ours)-safemin(finite_apple):>+8.1f}")
        print(f"{'Max PSNR':<52} {safemax(finite_apple):>12.1f} {safemax(finite_ours):>12.1f} {safemax(finite_ours)-safemax(finite_apple):>+8.1f}")

        # Per-layer summary (mean across modules)
        print(f"\n{'Per-layer mean PSNR (Apple vs Ours)':}")
        print(f"  {'Layer':>6} {'Apple':>10} {'Ours':>10} {'Delta':>8}")
        for layer_idx in sorted(layer_apple.keys()):
            a_vals = [p for p in layer_apple[layer_idx] if math.isfinite(p)]
            o_vals = [p for p in layer_ours.get(layer_idx, []) if math.isfinite(p)]
            a_mean = sum(a_vals) / len(a_vals) if a_vals else float("nan")
            o_mean = sum(o_vals) / len(o_vals) if o_vals else float("nan")
            delta = o_mean - a_mean if math.isfinite(o_mean) else float("nan")
            print(f"  {layer_idx:>6} {a_mean:>10.1f} {o_mean:>10.1f} {delta:>+8.1f}")
    else:
        print("\n" + "=" * 80)
        print(f"Apple MLX weight reconstruction PSNR ({args.variant}, {bits}-bit, group_size={group_size}):")
        print(f"  Mean: {sum(finite_apple)/len(finite_apple):.1f} dB  |  "
              f"Min: {min(finite_apple):.1f} dB  |  Max: {max(finite_apple):.1f} dB")
        print(f"  ({len(finite_apple)} weight tensors)")

        # Per-layer breakdown
        print(f"\n  Per-layer mean PSNR:")
        for layer_idx in sorted(layer_apple.keys()):
            vals = [p for p in layer_apple[layer_idx] if math.isfinite(p)]
            mean = sum(vals) / len(vals) if vals else float("nan")
            bar = "█" * int(max(0, mean - 30) / 2)
            print(f"  layer {layer_idx:>2}: {mean:>6.1f} dB  {bar}")


if __name__ == "__main__":
    main()
