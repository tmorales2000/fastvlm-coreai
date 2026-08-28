"""
fastvlm_fixtures.py — Realistic decoder input fixtures for FastVLM verification.

Provides the full multimodal pipeline from image + prompt to decoder inputs_embeds,
factored out of run_hf_fastvlm.py for reuse by:

  - verify_decoder.py  Phase 4 (recipe quality with realistic inputs)
  - scan_quantization_sensitivity.py  (per-layer sensitivity on real inputs)
  - verify_runtime.py  (end-to-end comparison against CoreAI runtime)

The key function is build_decoder_fixture(), which runs:
  image → preprocessing → vision encoder → projector → scatter-merge
  prompt → tokenizer → embed_tokens
  → combined inputs_embeds + position_ids

These are the actual decoder inputs the CoreAI export will receive at
inference time — the same distribution the model was trained on.

FIXTURE CACHING
---------------
Fixtures are expensive to generate (~2-5s per image on MPS due to vision
encoding). The cache stores the decoder inputs to disk so verify_decoder
and the scanner can reuse them without re-running the vision pipeline.

Cache key: hash of (variant, image_path, prompt, preprocessing_config).
Cache location: test_assets/fixtures/fastvlm-{variant}-{hash}.pt

USAGE
-----
  from fastvlm_fixtures import build_decoder_fixture, DecoderFixture

  fixture = build_decoder_fixture(
      variant="0.5b",
      image_path="test_assets/images/great_wave.jpg",
      prompt="Describe exactly what you see in this image.",
      use_cache=True,
  )
  # fixture.inputs_embeds: [1, seq_len, hidden_size] float16
  # fixture.position_ids:  [1, seq_len] int32
  # fixture.prompt_tokens: int (number of text tokens)
  # fixture.image_tokens:  int (number of image tokens = 256)
  # fixture.text:          str (decoded prompt for reference)
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Fixture schema version — bump this when the pipeline changes in a way
# that would produce different outputs from the same inputs:
#   - vision encoder/projector architecture change
#   - preprocessing change (image_strategy, image_size)
#   - chat template or tokenizer change
#   - scatter-merge implementation change
# Bumping invalidates all cached fixtures, forcing a rebuild.
FIXTURE_SCHEMA_VERSION = 2

# Default corpus — semantic images from test_assets/images/
# Excludes synthetic aspect-ratio fixtures (tall_narrow_circle, wide_short_square)
# which are degenerate for recipe quality testing.
CORPUS_IMAGES = [
    "test_assets/images/great_wave.jpg",
    "test_assets/images/earthrise.jpg",
    "test_assets/images/blue_marble.jpg",
    "test_assets/images/pale_blue_dot.png",
    "test_assets/images/pillars_of_creation.jpg",
    "test_assets/images/hubble_deep_field.jpg",
    "test_assets/images/girl_pearl_earring.jpg",
    "test_assets/images/migrant_mother.jpg",
    "test_assets/images/lunch_skyscraper.jpg",
]

# Fixed evaluation prompt — same across all images for cross-recipe comparison
DEFAULT_PROMPT = "Describe exactly what you see in this image."

# Cache directory
FIXTURE_CACHE_DIR = Path("test_assets/fixtures")

IMAGE_TOKEN_INDEX = -200  # Sentinel used by llava_qwen.py


@dataclass
class DecoderFixture:
    """Decoder inputs produced by the full FastVLM multimodal pipeline.

    inputs_embeds and position_ids are the exact tensors the decoder
    receives at inference time. They include the scatter-merged image
    features at the image token positions.

    Sequence layout (chat template dependent):
      [0 .. before_tokens-1]           — tokens before <image> in prompt
      [before_tokens .. image_end-1]   — image embedding tokens (256 for FastVLM)
      [image_end .. seq_len-1]         — tokens after <image> in prompt

    Use image_start/image_end for accurate position slicing in evaluation.
    image_tokens is a count (= image_end - image_start) retained for compatibility.
    """
    inputs_embeds: torch.Tensor   # [1, seq_len, hidden_size] float16
    position_ids:  torch.Tensor   # [1, seq_len] int32
    image_tokens:  int            # number of image token positions (256 for FastVLM)
    image_start:   int            # index of first image token in sequence
    image_end:     int            # index after last image token (= image_start + image_tokens)
    text_tokens:   int            # number of text token positions (before + after image)
    prompt:        str            # original prompt string
    image_path:    str            # source image path
    variant:       str            # model variant ("0.5b", "1.5b", "7b")
    device:        str            # device the tensors are on

    @property
    def seq_len(self) -> int:
        return self.inputs_embeds.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.inputs_embeds.shape[2]


def _fixture_cache_key(
    variant: str,
    image_path: str,
    prompt: str,
) -> str:
    """Deterministic cache key for a fixture.

    Includes FIXTURE_SCHEMA_VERSION so pipeline changes automatically
    invalidate stale caches without requiring manual deletion.
    """
    content = f"v{FIXTURE_SCHEMA_VERSION}|{variant}|{Path(image_path).resolve()}|{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _fixture_cache_path(variant: str, image_path: str, prompt: str) -> Path:
    key = _fixture_cache_key(variant, image_path, prompt)
    return FIXTURE_CACHE_DIR / f"fastvlm-{variant}-{key}.pt"


def _load_cached_fixture(cache_path: Path) -> Optional[DecoderFixture]:
    """Load a cached fixture from disk. Returns None if not found or invalid."""
    if not cache_path.is_file():
        return None
    try:
        data = torch.load(cache_path, weights_only=True, map_location="cpu")
        return DecoderFixture(**data)
    except Exception:
        return None


def _save_cached_fixture(fixture: DecoderFixture, cache_path: Path) -> None:
    """Save a fixture to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Move to CPU for caching
    data = {
        "inputs_embeds": fixture.inputs_embeds.cpu(),
        "position_ids":  fixture.position_ids.cpu(),
        "image_tokens":  fixture.image_tokens,
        "image_start":   fixture.image_start,
        "image_end":     fixture.image_end,
        "text_tokens":   fixture.text_tokens,
        "prompt":        fixture.prompt,
        "image_path":    fixture.image_path,
        "variant":       fixture.variant,
        "device":        "cpu",
    }
    torch.save(data, cache_path)


def _load_hf_model(weights_dir: Path, dtype: torch.dtype, device: str):
    """Load the HF FastVLM model. Mirrors run_hf_fastvlm.py loading logic."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(weights_dir), trust_remote_code=True)
    model_class_name = config.architectures[0] if config.architectures else "unknown"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(weights_dir),
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llava_qwen", weights_dir / "llava_qwen.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["llava_qwen"] = mod
        spec.loader.exec_module(mod)
        ModelClass = getattr(mod, model_class_name)
        model = ModelClass.from_pretrained(
            str(weights_dir),
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    return model


def build_decoder_fixture(
    variant: str,
    image_path: str | Path,
    prompt: str = DEFAULT_PROMPT,
    device: str = "mps",
    use_cache: bool = True,
    verbose: bool = False,
) -> DecoderFixture:
    """Build realistic decoder inputs from image + prompt via the full HF pipeline.

    This runs the complete FastVLM multimodal pipeline:
      image → preprocessing → vision encoder → projector
      prompt → tokenizer → embed_tokens
      → scatter-merge → inputs_embeds + position_ids

    The returned fixture is the exact input the decoder receives at inference
    time — the same distribution the model was trained on.

    Args:
        variant: Model variant ("0.5b", "1.5b", "7b").
        image_path: Path to input image.
        prompt: Text prompt (default: DEFAULT_PROMPT).
        device: Compute device ("mps" or "cpu").
        use_cache: If True, load from/save to fixture cache. Default: True.
        verbose: Print timing information.

    Returns:
        DecoderFixture with inputs_embeds and position_ids on CPU.
    """
    image_path = str(Path(image_path).resolve())

    # Check cache first
    if use_cache:
        cache_path = _fixture_cache_path(variant, image_path, prompt)
        cached = _load_cached_fixture(cache_path)
        if cached is not None:
            if verbose:
                print(f"[fixture] Cache hit: {cache_path.name}")
            return cached
        if verbose:
            print(f"[fixture] Cache miss — building fixture...")

    weights_dir = Path(__file__).parent.parent / "weights" / f"fastvlm-{variant}"
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_dir}\n"
            f"Download: hf download apple/FastVLM-{variant.upper()} "
            f"--local-dir {weights_dir}"
        )

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from PIL import Image
    from transformers import AutoTokenizer

    dtype = torch.float16
    target_device = torch.device(
        "mps" if device == "mps" and torch.backends.mps.is_available() else "cpu"
    )

    # Load model and tokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    model = _load_hf_model(weights_dir, dtype, device)
    if verbose:
        print(f"[fixture] Model loaded in {time.time()-t0:.1f}s")

    # Image preprocessing
    image_processor = model.get_vision_tower().image_processor
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(dtype=dtype, device=target_device)

    # Prompt → token IDs with IMAGE_TOKEN_INDEX sentinel
    messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert "<image>" in full_prompt
    before_str, after_str = full_prompt.split("<image>", 1)
    before_tok = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)
    after_tok  = tokenizer(after_str,  return_tensors="pt", add_special_tokens=False)

    input_ids = torch.cat([
        before_tok["input_ids"],
        torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=torch.long),
        after_tok["input_ids"],
    ], dim=1).to(target_device)

    text_tokens = before_tok["input_ids"].shape[1] + after_tok["input_ids"].shape[1]

    # Run the full multimodal forward pass to get inputs_embeds
    # We intercept at the decoder input level by using prepare_inputs_labels_for_multimodal
    t0 = time.time()
    with torch.no_grad():
        # Call the model's multimodal preparation — this runs vision encoder,
        # projector, and scatter-merge, returning the combined inputs_embeds
        (
            _,           # input_ids (None when using embeds)
            _,           # position_ids
            _,           # attention_mask
            _,           # past_key_values
            inputs_embeds,
            _,           # labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            images=[pixel_values[0]],
        )

    if verbose:
        print(f"[fixture] Vision pipeline in {time.time()-t0:.2f}s")

    seq_len      = inputs_embeds.shape[1]
    # before_tokens: tokens before <image> sentinel in the original input_ids
    before_tokens = before_tok["input_ids"].shape[1]
    after_tokens  = after_tok["input_ids"].shape[1]
    image_tokens  = seq_len - before_tokens - after_tokens
    image_start   = before_tokens
    image_end     = before_tokens + image_tokens
    text_tokens   = before_tokens + after_tokens

    # Build position_ids — sequential, matching how the decoder expects them
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

    fixture = DecoderFixture(
        inputs_embeds = inputs_embeds.cpu().to(torch.float16),
        position_ids  = position_ids.cpu(),
        image_tokens  = image_tokens,
        image_start   = image_start,
        image_end     = image_end,
        text_tokens   = text_tokens,
        prompt        = prompt,
        image_path    = image_path,
        variant       = variant,
        device        = "cpu",
    )

    if use_cache:
        _save_cached_fixture(fixture, cache_path)
        if verbose:
            print(f"[fixture] Cached to {cache_path.name}")

    return fixture


def build_corpus_fixtures(
    variant: str,
    images: list[str] = CORPUS_IMAGES,
    prompt: str = DEFAULT_PROMPT,
    device: str = "mps",
    use_cache: bool = True,
    verbose: bool = True,
) -> list[DecoderFixture]:
    """Build fixtures for the full evaluation corpus.

    Loads the model once and generates fixtures for all images.
    Cache is checked per-image so partially-cached corpora are handled.

    Args:
        variant: Model variant.
        images: List of image paths. Defaults to CORPUS_IMAGES (9 images).
        prompt: Fixed evaluation prompt.
        device: Compute device.
        use_cache: Use fixture cache (strongly recommended).
        verbose: Print progress.

    Returns:
        List of DecoderFixture, one per image.
    """
    # Check which images need building
    needs_build = []
    fixtures = {}
    for img in images:
        if use_cache:
            cache_path = _fixture_cache_path(variant, img, prompt)
            cached = _load_cached_fixture(cache_path)
            if cached is not None:
                fixtures[img] = cached
                if verbose:
                    print(f"[corpus] Cache hit: {Path(img).name}")
                continue
        needs_build.append(img)

    if not needs_build:
        return [fixtures[img] for img in images if img in fixtures]

    if verbose:
        print(f"[corpus] Building {len(needs_build)} fixture(s) — loading model...")

    # Load model once for all uncached images
    weights_dir = Path(__file__).parent.parent / "weights" / f"fastvlm-{variant}"
    dtype = torch.float16
    target_device = torch.device(
        "mps" if device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    model = _load_hf_model(weights_dir, dtype, device)
    image_processor = model.get_vision_tower().image_processor

    from PIL import Image

    messages_template = [{"role": "user", "content": f"<image>\n{prompt}"}]
    full_prompt = tokenizer.apply_chat_template(
        messages_template, tokenize=False, add_generation_prompt=True
    )
    before_str, after_str = full_prompt.split("<image>", 1)
    before_tok = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)
    after_tok  = tokenizer(after_str,  return_tensors="pt", add_special_tokens=False)
    text_tokens = before_tok["input_ids"].shape[1] + after_tok["input_ids"].shape[1]

    input_ids_base = torch.cat([
        before_tok["input_ids"],
        torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=torch.long),
        after_tok["input_ids"],
    ], dim=1)

    for img_path in needs_build:
        if verbose:
            print(f"[corpus] Building: {Path(img_path).name}")
        t0 = time.time()

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            if verbose:
                print(f"[corpus] MISSING: {img_path} — skipping")
            continue

        pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(dtype=dtype, device=target_device)
        input_ids    = input_ids_base.clone().to(target_device)

        with torch.no_grad():
            (_, _, _, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=[pixel_values[0]],
            )

        seq_len       = inputs_embeds.shape[1]
        before_tokens = before_tok["input_ids"].shape[1]
        after_tokens  = after_tok["input_ids"].shape[1]
        image_tokens  = seq_len - before_tokens - after_tokens
        image_start   = before_tokens
        image_end     = before_tokens + image_tokens
        text_tokens_n = before_tokens + after_tokens
        position_ids  = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

        fixture = DecoderFixture(
            inputs_embeds = inputs_embeds.cpu().to(torch.float16),
            position_ids  = position_ids.cpu(),
            image_tokens  = image_tokens,
            image_start   = image_start,
            image_end     = image_end,
            text_tokens   = text_tokens_n,
            prompt        = prompt,
            image_path    = str(Path(img_path).resolve()),
            variant       = variant,
            device        = "cpu",
        )
        fixtures[img_path] = fixture

        if use_cache:
            cache_path = _fixture_cache_path(variant, img_path, prompt)
            _save_cached_fixture(fixture, cache_path)

        if verbose:
            print(f"[corpus]   {seq_len} tokens ({image_tokens} image + "
                  f"{text_tokens} text) in {time.time()-t0:.1f}s")

    return [fixtures[img] for img in images if img in fixtures]
