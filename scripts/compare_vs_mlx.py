"""
compare_vs_mlx.py — Direct comparison of our Stage 2/3 pipeline outputs
against Apple's shipped MLX weights, tensor by tensor.

PURPOSE
-------
compare_weights.py measures per-component PSNR against the original HF
bf16 weights. This script answers a different question: how close is our
pipeline output to Apple's actual shipped artifacts?

For Stage 2 (fp16): compare our bf16->fp16 cast against Apple's MLX fp16
tensors. Should be near-identical or bit-identical since both are bf16->fp16
casts of the same source.

For Stage 3 (quantized): dequantize Apple's MLX int8/int4 weights to fp32,
dequantize our coreai-opt quantized weights to fp32, compare both. Reports:
  - PSNR(Apple dequantized, HF original)  — Apple's reconstruction quality
  - PSNR(Ours dequantized, HF original)   — our reconstruction quality
  - PSNR(Apple dequantized, Ours dequantized) — how similar the two schemes are

This directly addresses the question: does our quantization produce weights
that are similar to Apple's, or are we doing something structurally different?

USAGE
-----
  # Stage 2 comparison (fp16 tensors):
  python scripts/compare_vs_mlx.py --variant 0.5b --stage 2 \\
      --mlx-checkpoints-dir ~/git/tmorales2000/ml-fastvlm/checkpoints

  # Stage 3 comparison (quantized weights):
  python scripts/compare_vs_mlx.py --variant 1.5b --stage 3 \\
      --mlx-checkpoints-dir ~/git/tmorales2000/ml-fastvlm/checkpoints
  python scripts/compare_vs_mlx.py --variant 7b --stage 3 \\
      --mlx-checkpoints-dir ~/git/tmorales2000/ml-fastvlm/checkpoints
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "scripts")
from quantization import apply_quantization, QUANTIZATION_LEVELS  # noqa: E402
from fastvlm_decoder import FastVLMDecoderStateful, _load_decoder_weights  # noqa: E402
from transformers import AutoConfig  # noqa: E402

MLX_DIR_NAMES = {
    "0.5b": "llava-fastvithd_0.5b_stage3_llm.fp16",
    "1.5b": "llava-fastvithd_1.5b_stage3_llm.int8",
    "7b":   "llava-fastvithd_7b_stage3_llm.int4",
}

MLX_QUANTIZATION = {
    "0.5b": None,   # unquantized
    "1.5b": 8,      # int8
    "7b":   4,      # int4
}

SAMPLE_LAYERS = [0, 6, 13, 20, 27]
SAMPLE_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    ref = a.float().to(torch.float64)
    tst = b.float().to(torch.float64)
    mse = torch.mean((ref - tst) ** 2).item()
    if mse == 0:
        return float("inf")
    maxv = ref.abs().max().item()
    if maxv == 0:
        return float("inf")
    return 20 * math.log10(maxv) - 10 * math.log10(mse)


def mlx_dequantize(w_q: torch.Tensor, scales: torch.Tensor,
                   biases: torch.Tensor, bits: int, group_size: int = 64) -> torch.Tensor:
    values_per_uint32 = 32 // bits
    out_f, packed_cols = w_q.shape
    in_f = packed_cols * values_per_uint32
    mask = (1 << bits) - 1
    w_np = w_q.numpy().astype(np.uint32)
    unpacked = np.zeros((out_f, in_f), dtype=np.float32)
    for i in range(values_per_uint32):
        unpacked[:, i::values_per_uint32] = (w_np >> (i * bits)) & mask
    s = scales.numpy().astype(np.float32)
    b = biases.numpy().astype(np.float32)
    n_groups = in_f // group_size
    for g in range(n_groups):
        col = slice(g * group_size, (g + 1) * group_size)
        unpacked[:, col] = unpacked[:, col] * s[:, g:g+1] + b[:, g:g+1]
    return torch.from_numpy(unpacked)


def our_dequantize(w_fp16: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Quantize a fp16 weight using our coreai-opt scheme and dequantize back
    to fp32, using the fake-quantize forward pass to simulate the quantized
    weight representation.
    """
    from coreai_opt.quantization import Quantizer, QuantizerConfig
    from coreai_opt.quantization.spec import (
        PerBlockGranularity, QuantizationSpec, QuantizationScheme
    )
    from coreai_opt.quantization.config import ModuleQuantizerConfig

    dtype = torch.int8 if bits == 8 else torch.int4
    weight_spec = QuantizationSpec(
        dtype=dtype,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerBlockGranularity(axis=1, block_size=64),
    )
    linear_config = ModuleQuantizerConfig(
        op_input_spec={},
        op_output_spec={},
        op_state_spec={"weight": weight_spec},
    )
    out_f, in_f = w_fp16.shape
    layer = torch.nn.Linear(in_f, out_f, bias=False)
    layer.weight = torch.nn.Parameter(w_fp16.clone().float())
    layer.eval()
    config = QuantizerConfig(global_config=None).set_module_type(torch.nn.Linear, linear_config)
    quantizer = Quantizer(layer, config)
    prepared = quantizer.prepare((torch.zeros(1, in_f),))
    prepared.eval()
    with torch.no_grad():
        x = torch.eye(in_f)
        out_fp16 = layer(x).T
        out_quant = prepared(x).T
    return out_quant.float()


def load_hf_weights(weights_dir: str, layer_indices: list, modules: list) -> dict:
    from glob import glob
    import os
    weights = {}
    for fname in sorted(os.listdir(weights_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(weights_dir, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                if not k.endswith(".weight"):
                    continue
                for layer_idx in layer_indices:
                    for mod in modules:
                        if f"layers.{layer_idx}.{mod}.weight" in k:
                            weights[k] = f.get_tensor(k).float()
    return weights


def compare_stage2(variant: str, weights_dir: str, mlx_path: str) -> None:
    """Compare our bf16->fp16 Stage 2 output against Apple's MLX fp16 tensors."""
    print(f"\n{'─'*72}")
    print(f"STAGE 2 COMPARISON — {variant} (bf16->fp16 vs Apple MLX fp16)")
    print("─" * 72)

    mlx_file = os.path.join(mlx_path, "model.safetensors")

    psnr_values = []
    with safe_open(mlx_file, framework="pt", device="cpu") as sf:
        mlx_keys = set(sf.keys())
        hf_weights = load_hf_weights(weights_dir, SAMPLE_LAYERS, SAMPLE_MODULES)

        print(f"\n{'Tensor':<55} {'PSNR (dB)':>10}  note")
        print("-" * 72)

        for hf_key, hf_w_bf16 in sorted(hf_weights.items()):
            # Our Stage 2: bf16->fp16
            our_fp16 = hf_w_bf16.to(torch.float16).float()

            # Apple Stage 2: load from MLX safetensors
            mlx_key = "language_model." + hf_key
            if mlx_key not in mlx_keys:
                print(f"{hf_key:<55} NOT FOUND")
                continue
            apple_tensor = sf.get_tensor(mlx_key)
            # For unquantized variant, weight is directly fp16
            if apple_tensor.dtype == torch.uint32:
                print(f"{hf_key:<55} QUANTIZED (use --stage 3)")
                continue
            apple_fp32 = apple_tensor.float()

            score = psnr(our_fp16, apple_fp32)
            psnr_values.append(score)
            note = "bit-identical" if score == float("inf") else (
                "near-identical" if score > 100 else ""
            )
            score_str = "inf" if score == float("inf") else f"{score:.1f}"
            print(f"{hf_key:<55} {score_str:>10}  {note}")

    finite = [p for p in psnr_values if math.isfinite(p)]
    inf_count = len(psnr_values) - len(finite)
    print(f"\nResult: {inf_count} bit-identical, {len(finite)} with finite PSNR")
    if finite:
        print(f"Finite PSNR: mean={sum(finite)/len(finite):.1f} dB  "
              f"min={min(finite):.1f} dB  max={max(finite):.1f} dB")


def compare_stage3(variant: str, weights_dir: str, mlx_path: str, bits: int) -> None:
    """
    Compare our Stage 3 quantized weights against Apple's MLX quantized weights,
    both dequantized to fp32 for comparison.
    """
    print(f"\n{'─'*72}")
    print(f"STAGE 3 COMPARISON — {variant} (int{bits}: Apple MLX vs Ours vs HF)")
    print("─" * 72)

    mlx_file = os.path.join(mlx_path, "model.safetensors")
    hf_weights = load_hf_weights(weights_dir, SAMPLE_LAYERS, SAMPLE_MODULES)

    print(f"\n{'Tensor':<50} {'Apple/HF':>10} {'Ours/HF':>10} {'Apple/Ours':>12}")
    print("-" * 86)

    apple_hf_psnrs = []
    ours_hf_psnrs = []
    apple_ours_psnrs = []

    with safe_open(mlx_file, framework="pt", device="cpu") as sf:
        mlx_keys = set(sf.keys())

        for hf_key, hf_w in sorted(hf_weights.items()):
            mlx_base = "language_model." + hf_key
            w_key = mlx_base
            s_key = mlx_base.replace(".weight", ".scales")
            b_key = mlx_base.replace(".weight", ".biases")

            if w_key not in mlx_keys:
                print(f"{hf_key:<50} NOT FOUND")
                continue
            w_q = sf.get_tensor(w_key)
            if w_q.dtype != torch.uint32:
                print(f"{hf_key:<50} NOT QUANTIZED in MLX")
                continue

            scales = sf.get_tensor(s_key).float()
            biases = sf.get_tensor(b_key).float()

            # Apple dequantized
            apple_dq = mlx_dequantize(w_q, scales, biases, bits)

            # Our dequantized (from fp16 input matching our Stage 3)
            try:
                our_dq = our_dequantize(hf_w.to(torch.float16), bits)
                if our_dq.shape != hf_w.shape:
                    print(f"{hf_key:<50} SHAPE MISMATCH")
                    continue
            except Exception as e:
                print(f"{hf_key:<50} OUR DEQUANT FAILED: {e}")
                continue

            p_apple_hf = psnr(hf_w, apple_dq)
            p_ours_hf = psnr(hf_w, our_dq)
            p_apple_ours = psnr(apple_dq, our_dq)

            apple_hf_psnrs.append(p_apple_hf)
            ours_hf_psnrs.append(p_ours_hf)
            apple_ours_psnrs.append(p_apple_ours)

            print(f"{hf_key:<50} {p_apple_hf:>10.1f} {p_ours_hf:>10.1f} {p_apple_ours:>12.1f}")

    def stats(vals):
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            return "no data"
        return f"mean={sum(finite)/len(finite):.1f} min={min(finite):.1f} max={max(finite):.1f}"

    print(f"\n{'─'*86}")
    print(f"Apple vs HF  : {stats(apple_hf_psnrs)}")
    print(f"Ours vs HF   : {stats(ours_hf_psnrs)}")
    print(f"Apple vs Ours: {stats(apple_ours_psnrs)}")
    print()
    if apple_ours_psnrs:
        finite_ao = [v for v in apple_ours_psnrs if math.isfinite(v)]
        mean_ao = sum(finite_ao)/len(finite_ao) if finite_ao else 0
        if mean_ao > 60:
            print("INTERPRETATION: Apple and our schemes produce very similar weights.")
        elif mean_ao > 40:
            print("INTERPRETATION: Apple and our schemes produce somewhat similar weights.")
        else:
            print("INTERPRETATION: Apple and our schemes produce substantially different weights.")


def main():
    ap = argparse.ArgumentParser(
        description="Compare our pipeline outputs against Apple's MLX weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", choices=["0.5b", "1.5b", "7b"], required=True)
    ap.add_argument("--stage", choices=["2", "3"], required=True,
                    help="Stage 2: compare fp16 tensors. Stage 3: compare quantized weights.")
    ap.add_argument("--mlx-checkpoints-dir", required=True,
                    help="Parent directory of Apple's MLX checkpoint directories.")
    args = ap.parse_args()

    weights_dir = f"weights/fastvlm-{args.variant}"
    mlx_path = os.path.join(
        os.path.expanduser(args.mlx_checkpoints_dir),
        MLX_DIR_NAMES[args.variant]
    )

    if not os.path.isdir(mlx_path):
        print(f"MLX checkpoint not found: {mlx_path}")
        sys.exit(1)

    bits = MLX_QUANTIZATION[args.variant]

    if args.stage == "2":
        compare_stage2(args.variant, weights_dir, mlx_path)
    else:
        if bits is None:
            print(f"Variant {args.variant} has no quantization — use --stage 2")
            sys.exit(1)
        compare_stage3(args.variant, weights_dir, mlx_path, bits)


if __name__ == "__main__":
    main()
