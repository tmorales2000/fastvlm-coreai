#!/usr/bin/env python3
"""
scan_quantization_sensitivity.py — Discover optimal quantization recipe for FastVLM.

Measures per-layer quantization sensitivity by temporarily quantizing one layer
at a time, running calibration images through the model, and measuring the impact
on output quality. Produces conservative (mostly int8) and aggressive (mostly int4)
quantization recipes for use with export_fastvlm.py --quantize-recipe.

The generated recipe covers BOTH the decoder AND the vision tower — Apple's
reference implementation skips the vision tower, but sensitivity analysis may
reveal layers that can be safely quantized, reducing model size further.

Usage:
    # Generate recipes for FastVLM 0.5B using benchmark images as calibration
    python scripts/scan_quantization_sensitivity.py --variant 0.5b

    # Use a custom calibration image directory
    python scripts/scan_quantization_sensitivity.py --variant 1.5b \\
        --calibration-dir path/to/images

    # Control number of calibration images (default: all available)
    python scripts/scan_quantization_sensitivity.py --variant 0.5b \\
        --max-calibration-images 16

    # Local sensitivity only (faster — no end-to-end KL divergence)
    python scripts/scan_quantization_sensitivity.py --variant 0.5b --local-only

Output:
    quantization_recipes/fastvlm-{variant}.json

    Contains two recipes:
      "conservative" — mostly int8, sensitive layers kept fp16
      "aggressive"   — mostly int4, int8 for moderately sensitive, fp16 for critical

    Use with:
      python scripts/export_fastvlm.py --variant 0.5b \\
          --quantize-recipe quantization_recipes/fastvlm-0.5b.json::conservative
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ── Model registry (mirrors probe_vlm_config.py) ─────────────────────────────

MODEL_REGISTRY = {
    ("fastvlm", "0.5b"): ("apple/FastVLM-0.5B", "fastvlm-0.5b"),
    ("fastvlm", "1.5b"): ("apple/FastVLM-1.5B", "fastvlm-1.5b"),
    ("fastvlm", "7b"):   ("apple/FastVLM-7B",   "fastvlm-7b"),
}

DEFAULT_CALIBRATION_DIR = REPO_ROOT / "test_assets" / "images"
DEFAULT_OUTPUT_DIR      = REPO_ROOT / "quantization_recipes"

# ── Sensitivity thresholds ────────────────────────────────────────────────────
# Local sensitivity: Frobenius norm ratio of (W_fp16 - W_quant) / W_fp16
# KL sensitivity: KL divergence of output logits before/after quantization

# Layer is "critical" (keep fp16) if above these thresholds
CRITICAL_LOCAL_THRESHOLD       = 0.08   # >8% weight error → fp16
CRITICAL_KL_THRESHOLD          = 0.20   # >0.20 nats KL divergence → fp16

# Layer is "sensitive" (use int8 in aggressive recipe) if above these
SENSITIVE_LOCAL_THRESHOLD      = 0.04   # >4% weight error → int8 in aggressive
SENSITIVE_KL_THRESHOLD         = 0.05   # >0.05 nats KL divergence → int8 in aggressive

# Layers always kept at fp16 regardless of sensitivity
ALWAYS_FP16_PATTERNS = [
    "lm_head",                    # final projection — always fp16
    "model.embed_tokens",         # embedding table — always fp16
    "model.norm",                 # final norm — always fp16
    ".norm",                      # all LayerNorm/RMSNorm layers
]

# Vision tower layers to include in scan (Apple skips these — we don't)
VISION_TOWER_PATTERNS = [
    "model.vision_tower",
    "model.mm_projector",
]


@dataclass
class LayerSensitivity:
    name: str
    param_count: int
    local_sensitivity: float        # Frobenius norm ratio
    kl_sensitivity: float = 0.0    # KL divergence on logits (0 if --local-only)
    dtype: str = "fp16"            # original dtype
    recommended_int8: str = "fp16" # conservative recipe assignment
    recommended_int4: str = "fp16" # aggressive recipe assignment

    @property
    def is_vision(self) -> bool:
        return any(p in self.name for p in VISION_TOWER_PATTERNS)

    @property
    def is_always_fp16(self) -> bool:
        return any(p in self.name for p in ALWAYS_FP16_PATTERNS)


def quantize_int8_fake(weight: torch.Tensor) -> torch.Tensor:
    """Simulate int8 weight quantization (per-channel, symmetric)."""
    scale = weight.abs().max(dim=-1, keepdim=True).values / 127.0
    scale = scale.clamp(min=1e-8)
    quantized = (weight / scale).round().clamp(-128, 127)
    return quantized * scale


def quantize_int4_fake(weight: torch.Tensor) -> torch.Tensor:
    """Simulate int4 weight quantization (per-channel, symmetric, 16-element groups)."""
    W = weight
    group_size = 16
    orig_shape = W.shape
    if W.numel() >= group_size and W.shape[-1] >= group_size:
        W = W.reshape(-1, group_size)
        scale = W.abs().max(dim=-1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        quantized = (W / scale).round().clamp(-8, 7)
        return (quantized * scale).reshape(orig_shape)
    else:
        # Too small for group quantization — use per-tensor
        scale = W.abs().max() / 7.0
        scale = scale.clamp(min=1e-8)
        return (W / scale).round().clamp(-8, 7) * scale


def local_sensitivity(weight: torch.Tensor, bits: int = 8) -> float:
    """Frobenius norm ratio: ||W - W_quant|| / ||W||"""
    W = weight.float()
    W_q = (quantize_int8_fake if bits == 8 else quantize_int4_fake)(W)
    error = (W - W_q).norm().item()
    total = W.norm().item()
    return error / (total + 1e-8)


def load_calibration_images(
    calibration_dir: Path,
    image_processor,
    max_images: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Load and preprocess calibration images using the model's own image processor."""
    from PIL import Image

    image_paths = sorted(
        p for p in calibration_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[:max_images]

    if not image_paths:
        print(f"  ⚠ No images found in {calibration_dir}", file=sys.stderr)
        return []

    pixel_values_list = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            pv = image_processor(images=img, return_tensors="pt")["pixel_values"]
            pixel_values_list.append(pv.to(device=device, dtype=dtype))
        except Exception as e:
            print(f"  ⚠ Skipping {p.name}: {e}", file=sys.stderr)

    print(f"  Loaded {len(pixel_values_list)} calibration images from {calibration_dir}")
    return pixel_values_list


def get_reference_logits(
    model,
    pixel_values_list: list[torch.Tensor],
    tokenizer,
    device: torch.device,
) -> list[torch.Tensor]:
    """Get reference logits from unquantized model for each calibration image."""
    model.eval()
    IMAGE_TOKEN_INDEX = -200
    prompt = "\nDescribe this image."

    before_tok = tokenizer("USER: ", return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
    after_tok = tokenizer(f"\nDescribe this image.\nASSISTANT:",
                         return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(device)
    sentinel = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=torch.long, device=device)
    input_ids = torch.cat([before_tok, sentinel, after_tok], dim=1)

    logits_list = []
    with torch.no_grad():
        for pv in pixel_values_list:
            out = model(
                input_ids=input_ids,
                images=[pv[0]],
                output_hidden_states=False,
            )
            # Take logits at the last position
            logits_list.append(out.logits[0, -1, :].float().cpu())

    return logits_list


def kl_divergence(
    logits_ref: torch.Tensor,
    logits_quant: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """KL divergence between reference and quantized model output distributions."""
    p = F.softmax(logits_ref / temperature, dim=-1)
    q = F.softmax(logits_quant / temperature, dim=-1)
    # KL(P || Q) = sum(P * log(P/Q))
    kl = (p * (p / (q + 1e-10) + 1e-10).log()).sum().item()
    return max(0.0, kl)  # numerical clamp


def scan_layer(
    model,
    layer_name: str,
    param: nn.Parameter,
    ref_logits_list: list[torch.Tensor],
    pixel_values_list: list[torch.Tensor],
    tokenizer,
    device: torch.device,
    local_only: bool,
    bits: int = 8,
) -> tuple[float, float]:
    """
    Measure sensitivity of quantizing one layer.
    Returns (local_sensitivity, kl_sensitivity).
    """
    local_sens_8  = local_sensitivity(param.data, bits=8)
    local_sens_4  = local_sensitivity(param.data, bits=4)
    local_sens    = local_sens_8 if bits == 8 else local_sens_4

    if local_only or not ref_logits_list:
        return local_sens, 0.0

    # Temporarily quantize this one layer
    original_data = param.data.clone()
    quant_fn = quantize_int8_fake if bits == 8 else quantize_int4_fake
    param.data = quant_fn(param.data.float()).to(param.data.dtype)

    # Measure KL on all calibration images
    kl_values = []
    model.eval()
    IMAGE_TOKEN_INDEX = -200
    before_tok = tokenizer("USER: ", return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
    after_tok = tokenizer("\nDescribe this image.\nASSISTANT:",
                         return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(device)
    sentinel = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=torch.long, device=device)
    input_ids = torch.cat([before_tok, sentinel, after_tok], dim=1)

    with torch.no_grad():
        for pv, ref_logits in zip(pixel_values_list, ref_logits_list):
            try:
                out = model(
                    input_ids=input_ids,
                    images=[pv[0]],
                    output_hidden_states=False,
                )
                quant_logits = out.logits[0, -1, :].float().cpu()
                kl = kl_divergence(ref_logits, quant_logits)
                kl_values.append(kl)
            except Exception:
                pass

    # Restore original weights
    param.data = original_data

    avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
    return local_sens, avg_kl


def assign_dtype(
    sens: LayerSensitivity,
    recipe_type: str,  # "conservative" or "aggressive"
) -> str:
    """Assign quantization dtype based on sensitivity scores."""
    if sens.is_always_fp16:
        return "fp16"

    # Use combined score (weight local sensitivity more if no KL)
    has_kl = sens.kl_sensitivity > 0
    score = (
        0.5 * sens.local_sensitivity + 0.5 * sens.kl_sensitivity
        if has_kl else sens.local_sensitivity
    )
    kl = sens.kl_sensitivity

    if recipe_type == "conservative":
        # Aggressive thresholds — keep more fp16
        if score > CRITICAL_LOCAL_THRESHOLD or kl > CRITICAL_KL_THRESHOLD:
            return "fp16"
        elif score > SENSITIVE_LOCAL_THRESHOLD or kl > SENSITIVE_KL_THRESHOLD:
            return "fp16"
        else:
            return "int8"

    else:  # aggressive
        # Permissive thresholds — quantize more
        if score > CRITICAL_LOCAL_THRESHOLD or kl > CRITICAL_KL_THRESHOLD:
            return "fp16"
        elif score > SENSITIVE_LOCAL_THRESHOLD or kl > SENSITIVE_KL_THRESHOLD:
            return "int8"
        else:
            return "int4"


def print_sensitivity_table(sensitivities: list[LayerSensitivity]) -> None:
    """Print a sorted sensitivity table."""
    sorted_sens = sorted(sensitivities, key=lambda s: s.local_sensitivity, reverse=True)

    print(f"\n{'Layer':<60} {'Params':>10} {'Local':>8} {'KL':>8} {'Conserv':>10} {'Aggress':>10} {'Vision':>7}")
    print("-" * 115)
    for s in sorted_sens:
        vision_tag = "✓" if s.is_vision else ""
        fp16_tag   = "⚑" if s.is_always_fp16 else ""
        name_short = s.name[-58:] if len(s.name) > 58 else s.name
        print(
            f"  {name_short:<58} {s.param_count:>10,} "
            f"{s.local_sensitivity:>8.4f} {s.kl_sensitivity:>8.4f} "
            f"{s.recommended_int8:>10} {s.recommended_int4:>10} "
            f"{vision_tag+fp16_tag:>7}"
        )


def build_recipe(sensitivities: list[LayerSensitivity]) -> dict:
    """Build the recipe JSON from sensitivity results."""
    conservative = {}
    aggressive   = {}
    stats_c = {"fp16": 0, "int8": 0, "int4": 0}
    stats_a = {"fp16": 0, "int8": 0, "int4": 0}

    for s in sensitivities:
        conservative[s.name] = s.recommended_int8
        aggressive[s.name]   = s.recommended_int4
        stats_c[s.recommended_int8] = stats_c.get(s.recommended_int8, 0) + 1
        stats_a[s.recommended_int4] = stats_a.get(s.recommended_int4, 0) + 1

    return {
        "generated_by": "scan_quantization_sensitivity.py",
        "conservative": {
            "_summary": stats_c,
            "_description": "Mostly int8. Sensitive layers kept fp16. Lower quality risk.",
            "layers": conservative,
        },
        "aggressive": {
            "_summary": stats_a,
            "_description": "Mostly int4. int8 for moderately sensitive. Maximum compression.",
            "layers": aggressive,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="fastvlm",
                        help="Model family (default: fastvlm)")
    parser.add_argument("--variant", default="0.5b",
                        choices=["0.5b", "1.5b", "7b"],
                        help="Model variant (default: 0.5b)")
    parser.add_argument("--calibration-dir", type=Path,
                        default=DEFAULT_CALIBRATION_DIR,
                        help=f"Directory of calibration images (default: {DEFAULT_CALIBRATION_DIR})")
    parser.add_argument("--max-calibration-images", type=int, default=16,
                        help="Maximum number of calibration images to use (default: 16)")
    parser.add_argument("--local-only", action="store_true",
                        help="Use local sensitivity only (faster — skip end-to-end KL divergence)")
    parser.add_argument("--include-vision", action="store_true",
                        help="Include vision tower in sensitivity scan (experimental)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for recipe JSON (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps",
                        help="Device (default: mps)")
    args = parser.parse_args()

    key = (args.model.lower(), args.variant.lower())
    if key not in MODEL_REGISTRY:
        print(f"ERROR: Unknown model/variant: {args.model}/{args.variant}", file=sys.stderr)
        sys.exit(1)

    hf_model_id, weights_subdir = MODEL_REGISTRY[key]
    weights_dir = REPO_ROOT / "weights" / weights_subdir

    if not weights_dir.exists():
        print(f"ERROR: Weights not found at {weights_dir}", file=sys.stderr)
        print(f"Download with: hf download {hf_model_id} --local-dir {weights_dir}",
              file=sys.stderr)
        sys.exit(1)

    device = torch.device(
        args.device if args.device == "mps" and torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16

    print(f"\nQuantization Sensitivity Scanner")
    print(f"{'='*60}")
    print(f"  Model:   FastVLM {args.variant.upper()}")
    print(f"  Device:  {device}")
    print(f"  Mode:    {'local only' if args.local_only else 'local + end-to-end KL'}")
    print(f"  Vision:  {'included' if args.include_vision else 'excluded (use --include-vision)'}")
    print()

    # Load model and tokenizer
    print("Loading model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(weights_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load calibration images
    print("\nLoading calibration images...")
    image_processor = model.get_vision_tower().image_processor
    pixel_values_list = load_calibration_images(
        args.calibration_dir, image_processor,
        args.max_calibration_images, device, dtype
    )

    # Get reference logits
    ref_logits_list = []
    if not args.local_only and pixel_values_list:
        print("\nComputing reference logits...")
        t0 = time.time()
        ref_logits_list = get_reference_logits(model, pixel_values_list, tokenizer, device)
        print(f"  Done in {time.time()-t0:.1f}s ({len(ref_logits_list)} images)")

    # Collect quantizable layers
    print("\nScanning layers...")
    quantizable = []
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip 1D params (biases, norms)
        if param.numel() < 512:
            continue  # skip tiny layers
        is_vision = any(p in name for p in VISION_TOWER_PATTERNS)
        if is_vision and not args.include_vision:
            continue
        quantizable.append((name, param))

    print(f"  Found {len(quantizable)} quantizable weight tensors")
    if args.include_vision:
        vision_count = sum(
            1 for n, _ in quantizable
            if any(p in n for p in VISION_TOWER_PATTERNS)
        )
        print(f"  ({vision_count} vision tower, {len(quantizable)-vision_count} decoder)")

    # Scan each layer
    sensitivities: list[LayerSensitivity] = []
    t_total = time.time()

    for i, (name, param) in enumerate(quantizable):
        is_always_fp16 = any(p in name for p in ALWAYS_FP16_PATTERNS)

        if is_always_fp16:
            # Don't scan — always fp16
            s = LayerSensitivity(
                name=name,
                param_count=param.numel(),
                local_sensitivity=1.0,  # treat as maximally sensitive
                kl_sensitivity=1.0,
                dtype=str(param.dtype).replace("torch.", ""),
                recommended_int8="fp16",
                recommended_int4="fp16",
            )
        else:
            local_8, kl_8 = scan_layer(
                model, name, param,
                ref_logits_list, pixel_values_list,
                tokenizer, device, args.local_only, bits=8
            )
            s = LayerSensitivity(
                name=name,
                param_count=param.numel(),
                local_sensitivity=local_8,
                kl_sensitivity=kl_8,
                dtype=str(param.dtype).replace("torch.", ""),
            )
            s.recommended_int8 = assign_dtype(s, "conservative")
            s.recommended_int4 = assign_dtype(s, "aggressive")

        sensitivities.append(s)

        elapsed = time.time() - t_total
        eta = elapsed / (i + 1) * (len(quantizable) - i - 1)
        print(
            f"  [{i+1:3d}/{len(quantizable)}] {name[-50:]:<50} "
            f"local={s.local_sensitivity:.4f} "
            f"{'kl='+f'{s.kl_sensitivity:.4f}' if not args.local_only else ''}"
            f"  → int8:{s.recommended_int8:5s} int4:{s.recommended_int4:5s}"
            f"  ETA:{eta:.0f}s",
            flush=True
        )

    print(f"\nScan complete in {time.time()-t_total:.1f}s")

    # Print summary table
    print_sensitivity_table(sensitivities)

    # Build and save recipe
    recipe = build_recipe(sensitivities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"fastvlm-{args.variant}.json"
    output_path.write_text(json.dumps(recipe, indent=2))
    print(f"\nRecipe written to: {output_path}")

    # Print summary
    c = recipe["conservative"]["_summary"]
    a = recipe["aggressive"]["_summary"]
    print(f"\nConservative recipe: {c.get('fp16',0)} fp16, {c.get('int8',0)} int8, {c.get('int4',0)} int4")
    print(f"Aggressive recipe:   {a.get('fp16',0)} fp16, {a.get('int8',0)} int8, {a.get('int4',0)} int4")
    print(f"\nUse with:")
    print(f"  python scripts/export_fastvlm.py --variant {args.variant} \\")
    print(f"      --quantize-recipe {output_path}::conservative")
    print(f"  python scripts/export_fastvlm.py --variant {args.variant} \\")
    print(f"      --quantize-recipe {output_path}::aggressive")


if __name__ == "__main__":
    main()
