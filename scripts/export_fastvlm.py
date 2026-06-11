"""
export_fastvlm.py — Export FastVLM to a single CoreAI .aimodel with three functions.

Produces one .aimodel asset containing three callable functions:
  - vision_encode : pixel tensor → patch embeddings
  - project       : patch embeddings → projected embeddings
  - decode        : token IDs + KV state → logits

Usage:
    python scripts/export_fastvlm.py [--variant 1.5b]

Output:
    exports/fastvlm-{variant}/fastvlm.aimodel
"""

import argparse
import sys
from pathlib import Path

import coreai_torch
import torch
from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_decoder import FastVLMDecoderStateful, MAX_SEQ_LEN
from fastvlm_projector import FastVLMProjector
from fastvlm_vision_encoder import FastVLMVisionEncoder


# ─── Confirmed ExternalizeSpec configurations ─────────────────────────────────
# composite_op_name values confirmed from live docs at apple.github.io/coreai-torch
# composite_attrs confirmed from individual composite op reference pages.
#
# Note: if target_class not found in a module, a UserWarning is emitted
# (not an error) — safe to pass a superset of specs to all three functions.

EXTERN_SPECS = [
    ExternalizeSpec(
        target_class=RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    ),
    ExternalizeSpec(
        target_class=SDPA,
        composite_op_name="scaled_dot_product_attention",  # confirmed
        composite_attrs=["scale", "is_causal", "window_size"],
    ),
    ExternalizeSpec(
        target_class=RoPE,
        composite_op_name="rope",
        composite_attrs=["base", "dims", "interleaved"],
    ),
]


def export_fastvlm(variant: str = "1.5b") -> Path:
    weights_dir = f"weights/fastvlm-{variant}"
    out_dir = Path(f"exports/fastvlm-{variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {weights_dir}...")
    config = AutoConfig.from_pretrained(weights_dir)
    text_cfg = getattr(config, "text_config", config)
    img_size = config.vision_config.image_size
    print(f"image_size={img_size}, vocab_size={text_cfg.vocab_size}")

    # ── Load re-authored models ───────────────────────────────────────────────
    print("Loading vision encoder...")
    vision_enc = FastVLMVisionEncoder.from_weights(config, weights_dir).eval()

    print("Loading projector...")
    projector = FastVLMProjector.from_weights(config, weights_dir).eval()

    print("Loading decoder...")
    decoder = FastVLMDecoderStateful.from_weights(text_cfg, weights_dir).eval()

    # ── Example inputs ────────────────────────────────────────────────────────
    pixels = torch.zeros(1, 3, img_size, img_size, dtype=torch.float16)
    patch_seq_len = (img_size // 14) ** 2  # approximation — verify from model
    patch_emb = torch.zeros(1, patch_seq_len, config.mm_hidden_size, dtype=torch.float16)
    input_ids = torch.zeros(1, 1, dtype=torch.int32)
    pos_ids = torch.zeros(1, 1, dtype=torch.int32)

    seq_len_dim = torch.export.Dim(
        "seq_len", min=1, max=text_cfg.max_position_embeddings
    )

    # ── TorchConverter — one instance, three functions ────────────────────────
    converter = TorchConverter()

    # Function 1: vision_encode
    print("Staging vision_encode...")
    converter.add_pytorch_module(
        vision_enc,
        export_fn=lambda m: torch.export.export(
            m, args=(pixels,)
        ).run_decompositions(get_decomp_table()),
        input_names=["pixel_values"],
        output_names=["image_features"],
        externalize_modules=EXTERN_SPECS,
        entrypoint_name="vision_encode",   # confirmed parameter name
    )

    # Function 2: project
    print("Staging project...")
    converter.add_pytorch_module(
        projector,
        export_fn=lambda m: torch.export.export(
            m, args=(patch_emb,)
        ).run_decompositions(get_decomp_table()),
        input_names=["image_embeddings"],
        output_names=["projected_embeddings"],
        entrypoint_name="project",         # confirmed parameter name
    )

    # Function 3: decode (with KV cache state)
    print("Staging decode...")
    converter.add_pytorch_module(
        decoder,
        export_fn=lambda m: torch.export.export(
            m,
            args=(input_ids, pos_ids),
            dynamic_shapes={"position_ids": {1: seq_len_dim}},
        ).run_decompositions(get_decomp_table()),
        input_names=["input_ids", "position_ids"],
        # state_names: one name per state, covers BOTH input AND mutation output
        state_names=["k_cache", "v_cache"],
        output_names=["logits"],
        externalize_modules=EXTERN_SPECS,
        entrypoint_name="decode",          # confirmed parameter name
    )

    # ── Convert and save ──────────────────────────────────────────────────────
    print("Converting to CoreAI program...")
    program = converter.to_coreai()
    program.optimize()

    asset_path = out_dir / "fastvlm.aimodel"
    program.save_asset(asset_path)
    print(f"\nSaved: {asset_path}")
    print(f"Contains functions: vision_encode, project, decode")
    return asset_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Export FastVLM to CoreAI .aimodel"
    )
    ap.add_argument(
        "--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"]
    )
    export_fastvlm(ap.parse_args().variant)
