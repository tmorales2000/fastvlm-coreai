"""
verify_runtime.py — End-to-end runtime verification for the FastVLM VLM bundle.

Loads the full 3-component bundle (vision.aimodel, embed.aimodel, {variant}.aimodel)
and runs a complete inference cycle against the PyTorch reference models:

  1. vision_encode  — pixel_values → image_features
  2. project        — image_features → projected_features
  3. embed_tokens   — all_token_ids (incl. image placeholders) → embeddings
  4. scatter_merge  — replace <image> positions with projected_features
  5. decode prefill — merged_inputs_embeds → logits (KV cache fills)
  6. decode steps   — N single-token decode steps with stateful KV

Compares each stage's output against the PyTorch reference model and reports
PSNR. All stages should pass at > 40 dB for fp16 models.

USAGE:
  python scripts/verify_runtime.py --variant 0.5b --bundle-path exports/fastvlm-0.5b.vlmasset
"""

import argparse
import asyncio
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from coreai.runtime import AIModel, NDArray
from transformers import AutoConfig, AutoTokenizer

from fastvlm_decoder import FastVLMDecoder, FastVLMEmbedTokens
from fastvlm_projector import FastVLMProjector
from fastvlm_vision_encoder import FastVLMVisionEncoder

IMAGE_TOKEN = "<image>"
NUM_IMAGE_TOKENS = 256


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_ndarray(t: torch.Tensor, dtype: torch.dtype | None = None) -> NDArray:
    if dtype is not None:
        t = t.to(dtype=dtype)
    return NDArray(t.detach().cpu().contiguous().numpy())


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref  = ref.astype(np.float32)
    test = test.astype(np.float32)
    mse  = np.mean((ref - test) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((np.max(np.abs(ref)) ** 2) / mse)


def _scatter_merge(
    text_embeds: np.ndarray,    # [1, L, hidden]
    image_embeds: np.ndarray,   # [1, 256, hidden]
    token_ids: list[int],
    image_token_id: int,
) -> np.ndarray:
    """Replace <image> placeholder positions with projected image embeddings."""
    merged = text_embeds.copy()
    img_idx = 0
    for pos, tok in enumerate(token_ids):
        if tok == image_token_id:
            merged[0, pos, :] = image_embeds[0, img_idx, :]
            img_idx += 1
    return merged


# ─── Main verification ────────────────────────────────────────────────────────

async def verify_runtime(
    variant: str,
    bundle_path: Path,
    decode_steps: int = 3,
    seed: int = 42,
) -> bool:
    torch.manual_seed(seed)
    weights_dir = Path(__file__).parent.parent / "weights" / f"fastvlm-{variant}"

    # Load configs
    config = AutoConfig.from_pretrained(str(weights_dir), trust_remote_code=True)
    text_cfg = getattr(config, "text_config", None) or config
    hidden = text_cfg.hidden_size
    vocab_size = text_cfg.vocab_size

    # Load tokenizer with <image> token
    tok = AutoTokenizer.from_pretrained(
        str(bundle_path / "tokenizer"), trust_remote_code=True
    )
    image_token_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"[INFO] image_token_id: {image_token_id}")

    # ── Build PyTorch reference models ────────────────────────────────────────
    print("\n[INFO] Building PyTorch reference models...")

    vision_ref  = FastVLMVisionEncoder.from_weights(config=config, weights_dir=str(weights_dir)).eval()
    proj_ref    = FastVLMProjector.from_weights(config=config, weights_dir=str(weights_dir)).eval()
    embed_ref   = FastVLMEmbedTokens.from_weights(str(weights_dir)).eval()
    decoder_ref = FastVLMDecoder.from_weights(text_cfg, str(weights_dir)).eval()

    # ── Load compiled models ──────────────────────────────────────────────────
    print("[INFO] Loading compiled bundle...")
    vision_model  = await AIModel.load(bundle_path / "vision.aimodel")
    embed_model   = await AIModel.load(bundle_path / "embed.aimodel")
    decoder_model = await AIModel.load(bundle_path / f"fastvlm-{variant}.aimodel")

    encode_fn  = vision_model.load_function("encode_image")
    project_fn = vision_model.load_function("project")
    embed_fn   = embed_model.load_function("embed_tokens")
    decode_fn  = decoder_model.load_function("main")

    # State dict for stateful KV cache (persistent across calls)
    desc  = decode_fn.desc
    state = {
        name: NDArray.from_descriptor(descriptor=desc.state_descriptor(name=name))
        for name in desc.state_names
    }

    all_passed = True
    image_size = getattr(config, "image_size", 1024)

    # ── Stage 1: vision_encode ────────────────────────────────────────────────
    print("\n[INFO] Stage 1: vision_encode")
    torch.manual_seed(seed)
    pixel_values = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        ref_features = vision_ref(pixel_values)

    rt_out = await encode_fn(
        inputs={"pixel_values": _to_ndarray(pixel_values)}, state={}
    )
    rt_features = rt_out["image_features"].numpy()
    psnr = _psnr(ref_features.numpy(), rt_features)
    print(f"  vision_encode PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    # ── Stage 2: project ──────────────────────────────────────────────────────
    print("\n[INFO] Stage 2: project")
    with torch.no_grad():
        ref_projected = proj_ref(ref_features)

    rt_out = await project_fn(
        inputs={"x": NDArray(rt_features)}, state={}
    )
    rt_projected = rt_out["projected_features"].numpy()
    psnr = _psnr(ref_projected.numpy(), rt_projected)
    print(f"  project PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    # ── Stage 3: embed_tokens ─────────────────────────────────────────────────
    print("\n[INFO] Stage 3: embed_tokens")
    # Build a token sequence: 256 <image> placeholders + some text tokens
    text_tokens = [151644, 1, 2, 3, 4, 5]  # <|im_start|> + dummy text
    all_tokens  = ([image_token_id] * NUM_IMAGE_TOKENS) + text_tokens
    token_tensor = torch.tensor([all_tokens], dtype=torch.int32)

    with torch.no_grad():
        ref_embeds = embed_ref(token_tensor)  # [1, 262, hidden]

    rt_out = await embed_fn(
        inputs={"input_ids": _to_ndarray(token_tensor, dtype=torch.int32)}, state={}
    )
    rt_embeds = rt_out["embeddings"].numpy()
    psnr = _psnr(ref_embeds.numpy(), rt_embeds)
    print(f"  embed_tokens PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    # ── Stage 4: scatter_merge ────────────────────────────────────────────────
    print("\n[INFO] Stage 4: scatter_merge")
    # Reference: scatter image embeddings into text embeddings
    ref_merged = ref_embeds.numpy().copy()
    ref_proj_np = ref_projected.numpy()  # [1, 256, hidden]
    img_idx = 0
    for pos, tok_id in enumerate(all_tokens):
        if tok_id == image_token_id:
            ref_merged[0, pos, :] = ref_proj_np[0, img_idx, :]
            img_idx += 1

    rt_merged = _scatter_merge(rt_embeds, rt_projected, all_tokens, image_token_id)
    psnr = _psnr(ref_merged, rt_merged)
    print(f"  scatter_merge PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    # ── Stage 5: decode prefill ───────────────────────────────────────────────
    seq_len = len(all_tokens)
    print(f"\n[INFO] Stage 5: decode prefill (seq_len={seq_len})")

    ref_merged_t = torch.from_numpy(ref_merged)
    pos_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        ref_logits = decoder_ref(ref_merged_t, pos_ids)

    rt_out = await decode_fn(
        inputs={
            "inputs_embeds": NDArray(rt_merged.astype(np.float16)),
            "position_ids":  _to_ndarray(pos_ids, dtype=torch.int32),
        },
        state=state,
    )
    rt_logits = rt_out["logits"].numpy()
    psnr = _psnr(ref_logits.numpy(), rt_logits)
    print(f"  prefill logits PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    # ── Stage 6: decode loop ──────────────────────────────────────────────────
    print(f"\n[INFO] Stage 6: decode loop ({decode_steps} steps)")
    current_seq_len = seq_len

    # Reset decoder_ref KV cache for decode steps
    decoder_ref.k_cache = torch.zeros_like(decoder_ref.k_cache)
    decoder_ref.v_cache = torch.zeros_like(decoder_ref.v_cache)

    for step in range(decode_steps):
        current_seq_len += 1
        next_id  = torch.randint(1, min(vocab_size, 1000), (1, 1), dtype=torch.int32)
        next_pos = torch.tensor([[current_seq_len - 1]], dtype=torch.int32)

        # Reference: embed single token, pass as inputs_embeds
        with torch.no_grad():
            next_embed = embed_ref(next_id)  # [1, 1, hidden]
            ref_logits = decoder_ref(next_embed, next_pos)

        # Runtime
        rt_embed_out = await embed_fn(
            inputs={"input_ids": _to_ndarray(next_id, dtype=torch.int32)}, state={}
        )
        rt_embed = rt_embed_out["embeddings"].numpy()

        rt_out = await decode_fn(
            inputs={
                "inputs_embeds": NDArray(rt_embed.astype(np.float16)),
                "position_ids":  _to_ndarray(next_pos, dtype=torch.int32),
            },
            state=state,  # SAME dict — KV cache persists across calls
        )
        rt_logits = rt_out["logits"].numpy()
        psnr = _psnr(ref_logits.numpy(), rt_logits)
        print(f"  step {step+1}/{decode_steps} PSNR: {psnr:.1f} dB")
        all_passed &= psnr > 40

    print()
    if all_passed:
        print("[PASS] All stages match PyTorch reference (> 40 dB PSNR).")
    else:
        print("[FAIL] One or more stages below 40 dB threshold.")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["0.5b", "1.5b", "7b"], required=True)
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=None,
        help="Path to .vlmasset bundle (default: exports/fastvlm-{variant}.vlmasset)"
    )
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundle_path = args.bundle_path or (
        Path(__file__).parent.parent / "exports" / f"fastvlm-{args.variant}.vlmasset"
    )

    passed = asyncio.run(verify_runtime(
        variant=args.variant,
        bundle_path=bundle_path,
        decode_steps=args.decode_steps,
        seed=args.seed,
    ))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
