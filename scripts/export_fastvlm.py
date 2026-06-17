"""
export_fastvlm.py — Export FastVLM components to a single Core AI .aimodel.

WHAT THIS SCRIPT DOES
----------------------
Stages three components — vision_encode, project, decode — into one
TorchConverter and produces a single .aimodel with three callable entry
points (via entrypoint_name), matching the per-function call frequency
mismatch in a VLM: the vision encoder runs once per image, the projector
runs once per image, the decoder runs once per token. No example in Apple's
own coreai-models repo is a VLM, so this multi-entrypoint pattern is not
directly precedented there, but TorchConverter.add_pytorch_module's
entrypoint_name parameter exists precisely to support staging multiple
programs into one converter and is the documented mechanism for this case.

CURRENTLY IMPLEMENTED: vision_encode only.
project and decode entrypoints are added once this script is validated
end-to-end for the vision encoder — see TODO markers below.

WHY fp32 IS THE DEFAULT (not fp16)
-------------------------------------
Every export.py in coreai-models defaults to dtype=float32, even though
most accept float16 as an option:
  clip, clap, edsr, efficient-sam, pvt, roberta, sam3, t5, wav2vec2, whisper,
  yolo — ALL default to float32. depth-anything doesn't even offer float16.
This is not incidental — fp32 is the safe, universal choice Apple hands to
TorchConverter; precision/hardware-target mapping happens later, inside the
Core AI compiler (AIProgram.optimize()), which is independent of what dtype
the PyTorch trace used. coreai_torch.TorchConverter has no precision
parameter at all — confirmed by reading converter.py: add_pytorch_module
and add_exported_program take only input_names/output_names/state_names/
entrypoint_name, nothing precision-related.

THIS MATTERS CONCRETELY FOR FastVLM'S VISION ENCODER
---------------------------------------------------------
probe_activations.py found that FastViTHD's conv_exp output reaches
~252,866 (0.5b) / ~55,630 (1.5b) / ~12,740 (7b) max abs at network.9 on
1024x1024 input — 0.5b and 1.5b exceed the fp16 ceiling of 65504. Naively
casting the PyTorch module to fp16 before tracing reproduces this overflow
(confirmed: verify_vision_encoder.py Stage 2 fails with NaN for 0.5b/1.5b).
Tracing in fp32 (this script's default) avoids baking that overflow into
the exported graph. Whether the Core AI compiler's own fp16 lowering (during
optimize()) handles this safely is a SEPARATE open question — verify with
coreai_torch.debugging.comparator.create_comparator_for_programs() against
the compiled AIProgram once exported, feeding the same risk inputs
probe_activations.py used. Do not assume; verify.

EXPORT RECIPE (matches every model in coreai-models, e.g. models/clip/export.py)
-------------------------------------------------------------------------------
  model.to(dtype)
  with torch.autocast(device_type="cpu", dtype=dtype):
      exported = torch.export.export(model, args=(), kwargs=example_inputs)
  exported = exported.run_decompositions(get_decomp_table())
  converter.add_pytorch_module(model, export_fn=..., entrypoint_name=...)
  ...
  coreai_program = converter.to_coreai()
  coreai_program.optimize()
  coreai_program.save_asset(path, metadata)

USAGE
-----
  python scripts/export_fastvlm.py --variant 1.5b
  python scripts/export_fastvlm.py --variant 1.5b --dtype float16
  python scripts/export_fastvlm.py --variant 0.5b --output-dir ./exports/

ARGUMENTS
---------
  --variant     FastVLM variant to export. Default: 1.5b.
                Choices: 0.5b, 1.5b, 7b.
  --dtype       Torch dtype to trace the model in. Default: float32,
                matching every model in coreai-models. Pass float16 only
                after confirming via the comparator (see above) that the
                compiled model handles this variant's activation scale safely.
                Choices: float32, float16.
  --output-dir  Where to write the .aimodel. Default: exports/ in repo root.
  --overwrite   Overwrite an existing .aimodel at the output path.
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from transformers import AutoConfig

import sys
sys.path.insert(0, "scripts")
from fastvlm_vision_encoder import FastVLMVisionEncoder  # noqa: E402


def _image_size(config) -> int:
    """Image size is encoded in mm_vision_tower (e.g. 'mobileclip_l_1024')."""
    return int(config.mm_vision_tower.split("_")[-1])


def _vision_example_inputs(image_size: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.randn(1, 3, image_size, image_size).to(dtype)}


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "exports")


def _asset_path(output_dir: str, variant: str, dtype: torch.dtype) -> Path:
    dtype_name = str(dtype).split(".")[-1]
    return Path(output_dir) / f"fastvlm-{variant}_{dtype_name}.aimodel"


def _save_asset(coreai_program, model_path: Path, overwrite: bool) -> None:
    if model_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{model_path} already exists. Pass --overwrite to replace it."
            )
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(model_path, _build_aimodel_metadata())


def _build_aimodel_metadata() -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "Apple (FastVLM); re-authored by Agentive Intent LLC"
    metadata.license = "See FastVLM repository license"
    metadata.model_description = (
        "FastVLM vision encoder (FastViTHD), re-authored for Core AI export. "
        "Produces [B, H*W, C] patch embeddings matching "
        "MobileCLIPVisionTower.feature_select() in the original model."
    )
    metadata.creation_date = int(time.time())
    return metadata


def export_vision_encoder(
    variant: str,
    dtype: torch.dtype,
    output_dir: str,
    overwrite: bool,
) -> Path:
    weights_dir = f"weights/fastvlm-{variant}"
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    image_size = _image_size(config)

    print(f"[INFO] Sourcing vision encoder ({variant}, dtype={dtype})...")
    model = FastVLMVisionEncoder.from_weights(config, weights_dir, dtype=dtype)
    model.eval()

    example_inputs = _vision_example_inputs(image_size, dtype)

    print("[INFO] Model sourced. Running torch export with decompositions...")
    # autocast is a no-op (and emits a noisy warning) when dtype is float32 —
    # CPU autocast only supports promoting/demoting to bfloat16 or float16,
    # there's nothing to autocast TO when the target is float32 itself.
    if dtype == torch.float32:
        exported = torch.export.export(model, args=(), kwargs=example_inputs)
    else:
        with torch.autocast(device_type="cpu", dtype=dtype):
            exported = torch.export.export(model, args=(), kwargs=example_inputs)
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["pixel_values"],
        output_names=["image_features"],
        entrypoint_name="vision_encode",
    )

    # TODO: once vision_encode is validated end-to-end (export + comparator
    # verification against the compiled .aimodel), add:
    #   converter.add_exported_program(..., entrypoint_name="project")
    #   converter.add_exported_program(
    #       ..., state_names=["k_cache", "v_cache"], entrypoint_name="decode"
    #   )
    # then call converter.to_coreai() once for all three.

    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    model_path = _asset_path(output_dir, variant, dtype)
    _save_asset(coreai_program, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")
    return model_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export FastVLM vision encoder to a Core AI .aimodel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/export_fastvlm.py --variant 1.5b
  python scripts/export_fastvlm.py --variant 1.5b --dtype float16
  python scripts/export_fastvlm.py --variant 0.5b --overwrite
""",
    )
    ap.add_argument(
        "--variant",
        default="1.5b",
        choices=["0.5b", "1.5b", "7b"],
        help="FastVLM variant to export. (default: 1.5b)",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Torch dtype to trace in. (default: float32, matching coreai-models convention)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the .aimodel asset. (default: <repo-root>/exports/)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing .aimodel asset at the output path.",
    )
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    output_dir = args.output_dir or _default_output_dir()

    export_vision_encoder(args.variant, dtype, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
