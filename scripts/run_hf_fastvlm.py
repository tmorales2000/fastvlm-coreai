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
  - Runs on MPS (Apple Silicon GPU) by default — ~5-10 sec for 0.5B
  - Use --device cpu as fallback if MPS causes issues
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
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps",
                        help="Device to run on. Default: mps (GPU on Apple Silicon). Use cpu if mps fails.")
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

    # Load tokenizer from local weights
    from transformers import AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    # image_processor loaded from model after model loads (uses model's own config)
    print(f"[INFO] Tokenizer loaded in {time.time()-t0:.1f}s")

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
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location("llava_qwen", weights_dir / "llava_qwen.py")
        mod  = importlib.util.module_from_spec(spec)
        _sys.modules["llava_qwen"] = mod   # register before exec for timm @register_model
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

    # Use the model's own image processor (loaded from model_cfg["image_cfg"])
    # This handles FastVLM's pad aspect ratio and exact preprocessing correctly.
    image_processor = model.get_vision_tower().image_processor

    # Process image
    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]

    # Tokenize prompt. FastVLM uses IMAGE_TOKEN_INDEX = -200 as sentinel in
    # input_ids to mark image injection position — not a real vocabulary token.
    IMAGE_TOKEN_INDEX = -200
    before_tok = tokenizer("USER: ", return_tensors="pt", add_special_tokens=False)
    after_tok  = tokenizer(f"\n{args.prompt}\nASSISTANT:", return_tensors="pt", add_special_tokens=False)

    input_ids = torch.cat([
        before_tok["input_ids"],
        torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=torch.long),
        after_tok["input_ids"],
    ], dim=1)
    attention_mask = torch.ones_like(input_ids)

    target_device = torch.device(device if (device == "mps" and torch.backends.mps.is_available()) else "cpu")
    pixel_values  = pixel_values.to(dtype=dtype, device=target_device)
    input_ids     = input_ids.to(target_device)
    attention_mask = attention_mask.to(target_device)

    prompt_len = input_ids.shape[1]
    print(f"[INFO] Prompt tokens: {prompt_len}")
    print()

    # Generate — pass input_ids positionally as LlavaQwen2ForCausalLM.generate()
    # expects it that way (calls embed_tokens(inputs) internally)
    t0 = time.time()
    do_sample = args.temperature > 0
    with torch.no_grad():
        output = model.generate(
            input_ids,
            images=[pixel_values[0]],  # pass as list to use batched encode path in llava_qwen
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Decode — only the generated tokens
    generated_ids  = output[0][prompt_len:]
    response       = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
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
