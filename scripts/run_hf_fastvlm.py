"""
run_hf_fastvlm.py — Run FastVLM inference using the original HuggingFace weights.

This is the ground truth baseline for comparing against the CoreAI export.
Uses the original llava_qwen.py model code via trust_remote_code=True.

Run this alongside llm-runner on the same image/prompt to verify the CoreAI
export produces output consistent with the original HF model.

USAGE:
  python scripts/run_hf_fastvlm.py --variant 0.5b --image ~/test.jpeg \
    --prompt "Describe this image."

  # Compare with CoreAI export:
  python scripts/run_hf_fastvlm.py --variant 0.5b --image test_assets/images/afghan_girl.jpg \
    --prompt "Describe this portrait. What color are the subject's eyes?"

  # Use temperature 0 for deterministic output (recommended for benchmarking):
  python scripts/run_hf_fastvlm.py --variant 0.5b --image ~/test.jpeg \
    --prompt "Describe this image." --temperature 0

NOTES:
  - Runs on CPU by default (safe on all machines, ~30-60 sec for 0.5B)
  - Use --device mps for GPU acceleration on Apple Silicon (~5-10 sec)
  - The HF model uses trust_remote_code=True (runs llava_qwen.py)
  - Uses the LLaVA-style prompt format: USER: <image>\\nprompt\\nASSISTANT:
    which matches FastVLM's training format
  - Output is the ground truth for verifying CoreAI export quality
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--variant", choices=["0.5b", "1.5b", "7b"], default="0.5b")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. 0 = greedy (default, recommended for benchmarking)")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                        help="Device to run on (cpu is safest, mps for speed on Apple Silicon)")
    args = parser.parse_args()

    weights_dir = Path(__file__).parent.parent / "weights" / f"fastvlm-{args.variant}"
    if not weights_dir.exists():
        print(f"ERROR: Weights not found at {weights_dir}", file=sys.stderr)
        print(f"Download with: hf download apple/FastVLM-{args.variant.upper()} "
              f"--local-dir {weights_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading FastVLM {args.variant.upper()} from {weights_dir}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Image: {args.image}")
    print(f"[INFO] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"[INFO] Temperature: {args.temperature} ({'greedy' if args.temperature == 0 else 'sampling'})")
    print()

    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    dtype = torch.float16
    device = args.device

    # Load config to determine correct model class
    config = AutoConfig.from_pretrained(str(weights_dir), trust_remote_code=True)
    model_class_name = config.architectures[0] if config.architectures else "unknown"
    print(f"[INFO] Model architecture: {model_class_name}")

    # Load processor
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(weights_dir), trust_remote_code=True)
    print(f"[INFO] Processor loaded in {time.time()-t0:.1f}s")

    # Load model using auto class (resolves trust_remote_code model class)
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(weights_dir),
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
    except Exception:
        # Fallback: load model class directly from llava_qwen.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llava_qwen",
            weights_dir / "llava_qwen.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ModelClass = getattr(mod, model_class_name, None)
        if ModelClass is None:
            print(f"ERROR: Could not find {model_class_name} in llava_qwen.py", file=sys.stderr)
            sys.exit(1)
        model = ModelClass.from_pretrained(
            str(weights_dir),
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    print(f"[INFO] Model loaded in {time.time()-t0:.1f}s")

    # Build LLaVA-style prompt (matching FastVLM training format)
    image = Image.open(args.image).convert("RGB")
    full_prompt = f"USER: <image>\n{args.prompt}\nASSISTANT:"

    # Process inputs
    inputs = processor(text=full_prompt, images=image, return_tensors="pt")
    if device == "mps" and torch.backends.mps.is_available():
        inputs = {k: v.to("mps") if hasattr(v, "to") else v for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    print(f"[INFO] Prompt tokens: {prompt_len}")
    print()

    # Generate
    t0 = time.time()
    do_sample = args.temperature > 0
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Decode — only the generated tokens
    generated_ids  = output[0][prompt_len:]
    response       = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = len(generated_ids)

    print("=" * 64)
    print(f"FastVLM {args.variant.upper()} (HF reference, {device})")
    print("=" * 64)
    print(response)
    print()
    print(f"Generated {generated_tokens} tokens in {elapsed:.1f}s "
          f"({generated_tokens/elapsed:.1f} tok/sec)")


if __name__ == "__main__":
    main()
