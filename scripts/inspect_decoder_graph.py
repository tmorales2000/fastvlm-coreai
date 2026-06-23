"""
inspect_decoder_graph.py — Inspect the exported decoder graph for custom ops.

Useful for debugging TorchConverter failures: shows exactly what op names
appear in the exported graph, so register_torch_lowering() can be called
with the correct qualified name.

USAGE
-----
  python scripts/inspect_decoder_graph.py --variant 1.5b
"""

import argparse
import sys

import torch
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_decoder import FastVLMDecoderStateful, MAX_SEQ_LEN  # noqa: E402
from coreai_torch import get_decomp_table  # noqa: E402


def inspect(variant: str) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    print(f"Loading decoder ({variant})...")
    model = FastVLMDecoderStateful.from_weights(text_cfg, weights_dir)
    model.eval()

    example_inputs = {
        "input_ids": torch.randint(1, text_cfg.vocab_size, (1, 8), dtype=torch.int32),
        "position_ids": torch.arange(8, dtype=torch.int32).unsqueeze(0),
    }
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "position_ids": {1: seq_len_dim},
    }

    print("Exporting...")
    exported = torch.export.export(
        model, args=(), kwargs=example_inputs, dynamic_shapes=dynamic_shapes
    )
    exported = exported.run_decompositions(get_decomp_table())

    # Collect all unique op targets
    all_targets: dict[str, int] = {}
    custom_targets: dict[str, int] = {}

    for node in exported.graph.nodes:
        if node.op == "call_function":
            target = str(node.target)
            all_targets[target] = all_targets.get(target, 0) + 1
            # Flag anything that isn't a plain aten op
            if not target.startswith("aten.") and "prims" not in target:
                custom_targets[target] = custom_targets.get(target, 0) + 1

    print(f"\nTotal unique op targets: {len(all_targets)}")
    print(f"Total nodes: {sum(all_targets.values())}")

    print("\nNon-aten ops (custom / composite / higher-order):")
    for t, count in sorted(custom_targets.items()):
        print(f"  {count:4d}x  {t}")

    print("\nTop 10 aten ops by count:")
    aten_ops = {t: c for t, c in all_targets.items() if t.startswith("aten.")}
    for t, count in sorted(aten_ops.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:4d}x  {t}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    args = ap.parse_args()
    inspect(args.variant)
