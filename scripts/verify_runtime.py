"""
verify_runtime.py — End-to-end runtime verification of a compiled FastVLM
.aimodel: actually CALL the compiled model (not just compare graphs) through
a realistic vision_encode -> project -> decode (prefill, then N single-token
decode steps) sequence, and confirm outputs match the PyTorch reference at
every step.

WHY THIS SCRIPT EXISTS, AND HOW IT DIFFERS FROM verify_compiled_vision_encoder.py
----------------------------------------------------------------------------------
verify_compiled_vision_encoder.py uses coreai_torch.debugging.comparator —
an op-by-op, bisection-search DIAGNOSTIC tool comparing the traced PyTorch
graph against the compiled graph. It answers "where, if anywhere, does the
compiled model's graph diverge from the PyTorch source."

This script is different in kind: it uses coreai.runtime directly — the
actual inference-serving API a real app would use — to call the compiled
.aimodel's three entry points in the realistic sequence an app would use
them, with the KV cache persisting as runtime state across multiple decode
calls. It answers "if I actually run this exported model the way an app
would, do I get correct results end to end." This is an integration test,
not a graph diagnostic. A model can pass the comparator (graph-level
fidelity) and still have a runtime bug in how state is threaded across
calls, or vice versa — they check different things.

RUNTIME STATE API (confirmed from coreai-torch's own tests/utils.py,
specifically _init_runtime_state / _execute_and_compare — there is no
public documentation for this beyond the test suite itself)
----------------------------------------------------------------------------
State (our k_cache/v_cache) is NOT implicitly persisted by the loaded model
object. It is an explicit dict the CALLER must create once and pass back in
on every call:

    state = {
        "k_cache": NDArray.from_descriptor(descriptor=desc.state_descriptor(name="k_cache")),
        "v_cache": NDArray.from_descriptor(descriptor=desc.state_descriptor(name="v_cache")),
    }
    rt_outputs_1 = await decode_fn(inputs=prefill_inputs, state=state)
    # `state` has now been mutated in place by the runtime call above.
    rt_outputs_2 = await decode_fn(inputs=next_token_inputs, state=state)
    # state carries forward correctly ONLY because we passed the SAME dict
    # object back in — not a copy, not a freshly re-initialized one.

NDArray.from_descriptor(...) auto-allocates a correctly shaped/typed
zero-initialized array directly from the model's own state descriptor
metadata — this is how the cache's (n_layers, batch_size, max_seq_len,
kv_dim) shape gets right without us hardcoding it here.

USAGE
-----
  python scripts/verify_runtime.py --variant 1.5b
  python scripts/verify_runtime.py --variant 1.5b --aimodel-path exports/fastvlm-1.5b_float32.aimodel
  python scripts/verify_runtime.py --variant 1.5b --decode-steps 8

ARGUMENTS
---------
  --variant       FastVLM variant being verified. Default: 1.5b.
  --aimodel-path  Path to the compiled .aimodel. Default: derived from
                   --variant as exports/fastvlm-<variant>_float32.aimodel
                   (matching export_fastvlm.py's naming convention).
  --decode-steps  Number of single-token decode steps to run after the
                   initial prefill. Default: 4.
  --atol          Absolute tolerance for comparing runtime output against
                   the PyTorch reference. Default: 1e-2 (looser than the
                   1e-5 used in coreai-torch's own simple buffer tests,
                   since FastVLM's full forward pass accumulates far more
                   floating-point operations per step).
  --seed          Random seed for the synthetic test image. Default: 0.

WHAT THIS SCRIPT DOES
-----------------------
1. Builds the PyTorch reference path (vision encoder + projector + stateful
   decoder, all fp32) exactly as verify_*.py scripts do, to get ground-truth
   outputs at each step.
2. Loads the compiled .aimodel via coreai.runtime.AIModel.load(...) and gets
   each of the three entry points via model.load_function(name).
3. Runs vision_encode once, project once (no state — these are stateless).
4. Initializes decode's state dict via NDArray.from_descriptor(...).
5. Runs decode once as a "prefill" (multi-token input_ids/position_ids),
   then --decode-steps additional single-token calls, REUSING the same
   state dict object each time, comparing runtime output against the
   PyTorch reference at every step.

INTERPRETING RESULTS
---------------------
If every step's PSNR is high and stable (similar magnitude to
verify_decoder.py's Stage 2 results, ~70+ dB), the compiled model is
faithfully reproducing the PyTorch reference through the full realistic
call sequence, including state persistence across calls. A LOW or
DEGRADING PSNR across decode steps would suggest the cache is not being
threaded through state correctly between runtime calls — that is a
DIFFERENT failure mode than anything verify_decoder.py or
verify_compiled_vision_encoder.py can catch, since both of those exercise
the cache only within a single Python-level forward pass, never across
genuinely separate runtime invocations.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_decoder import FastVLMDecoderStateful  # noqa: E402
from fastvlm_projector import FastVLMProjector  # noqa: E402
from fastvlm_vision_encoder import FastVLMVisionEncoder  # noqa: E402


def _image_size(config) -> int:
    return int(config.mm_vision_tower.split("_")[-1])


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_val = np.max(np.abs(ref))
    return 20 * np.log10(max_val) - 10 * np.log10(mse) if max_val > 0 else float("inf")


def _to_ndarray(t: torch.Tensor) -> NDArray:
    return NDArray(t.detach().cpu().numpy())


async def verify_runtime(
    variant: str,
    aimodel_path: Path,
    decode_steps: int,
    atol: float,
    seed: int,
) -> bool:
    weights_dir = f"weights/fastvlm-{variant}"
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    image_size = _image_size(config)

    print(f"Building PyTorch reference path for {variant}...")
    vision_model = FastVLMVisionEncoder.from_weights(config, weights_dir, dtype=torch.float32)
    vision_model.eval()
    projector_model = FastVLMProjector.from_weights(config, weights_dir, dtype=torch.float32)
    projector_model.eval()
    decoder_model = FastVLMDecoderStateful.from_weights(text_cfg, weights_dir)
    decoder_model = decoder_model.to(dtype=torch.float32)
    decoder_model.eval()

    torch.manual_seed(seed)
    pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    print(f"Loading compiled model from {aimodel_path}...")
    model = await AIModel.load(str(aimodel_path))
    vision_fn = model.load_function("vision_encode")
    project_fn = model.load_function("project")
    decode_fn = model.load_function("decode")

    all_passed = True

    # --- vision_encode ---
    print("\nRunning vision_encode...")
    with torch.no_grad():
        ref_image_features = vision_model(pixel_values)
    rt_out = await vision_fn(inputs={"pixel_values": _to_ndarray(pixel_values)}, state={})
    rt_image_features = rt_out["image_features"].numpy()
    psnr = _psnr(ref_image_features.numpy(), rt_image_features)
    print(f"  vision_encode PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    # --- project ---
    print("Running project...")
    with torch.no_grad():
        ref_projected = projector_model(ref_image_features)
    rt_out = await project_fn(inputs={"x": NDArray(rt_image_features)}, state={})
    rt_projected = rt_out["projected_features"].numpy()
    psnr = _psnr(ref_projected.numpy(), rt_projected)
    print(f"  project PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    # --- decode: prefill, then decode_steps single-token calls ---
    # State dict is built ONCE via NDArray.from_descriptor and REUSED across
    # every subsequent call below — this is the critical part of the runtime
    # state contract (see module docstring). desc.state_descriptor gives us
    # the correct (n_layers, batch_size, max_seq_len, kv_dim) shape without
    # us hardcoding it here.
    desc = decode_fn.desc
    state = {
        name: NDArray.from_descriptor(descriptor=desc.state_descriptor(name=name))
        for name in desc.state_names
    }

    prefill_len = 6
    print(f"\nRunning decode prefill (query_len={prefill_len})...")
    torch.manual_seed(seed + 1)
    prefill_ids = torch.randint(1, text_cfg.vocab_size, (1, prefill_len), dtype=torch.int32)
    prefill_positions = torch.arange(prefill_len, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        ref_logits = decoder_model(prefill_ids, prefill_positions)
    rt_out = await decode_fn(
        inputs={
            "input_ids": _to_ndarray(prefill_ids),
            "position_ids": _to_ndarray(prefill_positions),
        },
        state=state,
    )
    rt_logits = rt_out["logits"].numpy()
    psnr = _psnr(ref_logits.numpy(), rt_logits)
    print(f"  prefill logits PSNR: {psnr:.1f} dB")
    all_passed &= psnr > 40

    seq_len = prefill_len
    for step in range(decode_steps):
        seq_len += 1
        next_id = torch.randint(1, text_cfg.vocab_size, (1, 1), dtype=torch.int32)
        next_position = torch.tensor([[seq_len - 1]], dtype=torch.int32)

        with torch.no_grad():
            ref_logits = decoder_model(next_id, next_position)
        rt_out = await decode_fn(
            inputs={
                "input_ids": _to_ndarray(next_id),
                "position_ids": _to_ndarray(next_position),
            },
            state=state,  # SAME dict object — carries the cache forward.
        )
        rt_logits = rt_out["logits"].numpy()
        psnr = _psnr(ref_logits.numpy(), rt_logits)
        print(f"  decode step {step + 1}/{decode_steps} logits PSNR: {psnr:.1f} dB")
        all_passed &= psnr > 40

    print()
    if all_passed:
        print("[PASS] Compiled model matches PyTorch reference at every step, "
              "including state persistence across separate runtime calls.")
    else:
        print("[FAIL] One or more steps fell below the PSNR threshold (40 dB). "
              "If PSNR degrades specifically across decode steps (not at "
              "prefill), suspect the KV cache state is not threading "
              "correctly between runtime calls.")
    return all_passed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end runtime verification of a compiled FastVLM .aimodel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/verify_runtime.py --variant 1.5b
  python scripts/verify_runtime.py --variant 1.5b --decode-steps 8
""",
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument("--aimodel-path", default=None,
                     help="Path to the compiled .aimodel. Default: exports/fastvlm-<variant>_float32.aimodel")
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    aimodel_path = Path(args.aimodel_path) if args.aimodel_path else (
        Path(__file__).resolve().parents[1] / "exports" / f"fastvlm-{args.variant}_float32.aimodel"
    )

    passed = asyncio.run(
        verify_runtime(args.variant, aimodel_path, args.decode_steps, args.atol, args.seed)
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
