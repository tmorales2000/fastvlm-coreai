"""
verify_runtime.py — Python round-trip verification via coreai.runtime.

Loads the compiled .aimodel and calls all three functions (vision_encode,
project, decode) using coreai.runtime. No Xcode required.

Usage:
    python scripts/verify_runtime.py [--variant 1.5b] [--platform macos]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from coreai.authoring import AIModelAsset
from coreai.runtime import NDArray
from transformers import AutoConfig

sys.path.insert(0, "scripts")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    mse = ((a_f - b_f) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(b_f.max() ** 2 / mse)


async def verify(variant: str = "1.5b", platform: str = "macos") -> None:
    asset_path = Path(f"exports/fastvlm-{variant}/fastvlm_{platform}.aimodel")

    if not asset_path.exists():
        print(f"ERROR: {asset_path} not found.")
        print(
            f"Run: xcrun coreai-build compile "
            f"exports/fastvlm-{variant}/fastvlm.aimodel "
            f"--platform {platform.capitalize()} "
            f"--output {asset_path}"
        )
        sys.exit(1)

    weights_dir = f"weights/fastvlm-{variant}"
    config = AutoConfig.from_pretrained(weights_dir)
    text_cfg = getattr(config, "text_config", config)
    img_size = config.vision_config.image_size

    print(f"Loading {asset_path}...")
    asset = AIModelAsset.load(asset_path)

    async with asset.executable() as model:
        fn_names = model.function_names
        print(f"Functions: {fn_names}")

        # Verify expected functions are present
        expected = {"vision_encode", "project", "decode"}
        found = set(fn_names)
        if not expected.issubset(found):
            print(f"ERROR: Missing functions: {expected - found}")
            sys.exit(1)

        # ── vision_encode ─────────────────────────────────────────────────────
        fn_v = model.load_function("vision_encode")
        pixels = np.random.randn(1, 3, img_size, img_size).astype(np.float16)
        out_v = await fn_v({"pixel_values": NDArray(pixels)})
        features = out_v["image_features"].numpy()  # materialize inside ctx
        print(f"vision_encode: {features.shape}  dtype={features.dtype}  ✓")

        # ── project ───────────────────────────────────────────────────────────
        fn_p = model.load_function("project")
        out_p = await fn_p({"image_embeddings": NDArray(features)})
        projected = out_p["projected_embeddings"].numpy()
        print(f"project:       {projected.shape}  dtype={projected.dtype}  ✓")

        # ── decode (single step) ──────────────────────────────────────────────
        fn_d = model.load_function("decode")
        inp = np.array([[1]], dtype=np.int32)
        pos = np.array([[0]], dtype=np.int32)
        out_d = await fn_d({
            "input_ids":    NDArray(inp),
            "position_ids": NDArray(pos),
        })
        logits = out_d["logits"].numpy()
        print(f"decode:        {logits.shape}  dtype={logits.dtype}  ✓")

        # ── Optional PSNR against PyTorch reference ───────────────────────────
        # Uncomment and complete once re-authored models are verified:
        #
        # from fastvlm_decoder import FastVLMDecoderStateful
        # pt_model = FastVLMDecoderStateful.from_weights(text_cfg, weights_dir)
        # pt_model.eval()
        # with torch.no_grad():
        #     input_ids_t = torch.tensor(inp, dtype=torch.int32)
        #     pos_ids_t = torch.tensor(pos, dtype=torch.int32)
        #     pt_logits = pt_model(input_ids_t, pos_ids_t).numpy()
        # score = psnr(logits, pt_logits)
        # print(f"Decoder PSNR vs PyTorch fp32: {score:.1f} dB")
        #
        # Threshold: > 40 dB to pass

        print(f"\nAll three functions executed successfully ✓")
        print(f"variant={variant}  platform={platform}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Python round-trip verification via coreai.runtime"
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument("--platform", default="macos", choices=["macos", "ios"])
    args = ap.parse_args()
    asyncio.run(verify(args.variant, args.platform))
