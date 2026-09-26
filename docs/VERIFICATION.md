# FastVLM CoreAI — Verification Guide

Two verification layers catch different classes of problems:

**Layer 1 — PyTorch verification** (three scripts, no CoreAI runtime needed)
Runs fast. Catches re-authoring bugs and quantization quality issues before
committing to a full export.

**Layer 2 — CoreAI runtime verification** (`verify_runtime.py`)
Requires a compiled `.aimodel` bundle and `llm-runner`. Catches export
pipeline bugs, MLIR lowering issues, and compiled model drift.

---

## Device Strategy

Each verification phase runs on the device that produces the most meaningful
result for its purpose. This is not a performance choice — it is a correctness
choice.

| Script / Phase | Device | Reason |
|----------------|--------|--------|
| verify_decoder Phase 1 | CPU | IEEE 754 strict fp32 — MPS rounding produces lower PSNR scores that don't indicate bugs |
| verify_decoder Phase 2 | MPS | Deployment accuracy — measures fp16 on the hardware the model actually runs on |
| verify_decoder Phase 3 | MPS | Fast KV cache correctness test |
| verify_decoder Phase 4 prepare | CPU | coreai_opt RoPEImpl creates freqs buffer on CPU at init; MPS causes device mismatch |
| verify_decoder Phase 4 infer | MPS | Fast corpus evaluation |
| verify_projector (both phases) | CPU | Architecture correctness; projector is small, no performance concern |
| verify_vision_encoder Phase 1 | CPU | Architecture correctness — IEEE 754 fp32 |
| verify_vision_encoder Phase 2 | MPS | fp16 overflows on CPU at 1024×1024 (186 conv2d ops); real images required |

### MPS correctness requirements

Two non-obvious requirements apply whenever moving models to MPS:

- **`.eval()` must be called AFTER `.to(device)`** — calling `.eval()` on CPU
  then moving to MPS leaves MPS-specific state incorrectly initialized. Manifests
  as cosine=0.50 instead of cosine=1.0 for identical models. All verify scripts
  chain `.to(device).eval()`.

- **`load_state_dict(assign=True)` overwrites device placement** — weights loaded
  from safetensors are CPU tensors. `assign=True` replaces parameters with those
  CPU tensors, undoing any prior `.to(device)`. Always call `.to(device)` after
  `load_state_dict`.

- **MPS has no float64** — metrics.py calls `.cpu()` before `.double()` throughout.
  Any metric computation that calls `.double()` directly on a MPS tensor will crash.

### Real images vs random inputs

Random N(0,1) inputs are catastrophically misleading for vision encoder evaluation:

| Input | Cosine similarity | PSNR |
|-------|-------------------|------|
| Random N(0,1) pixels | 0.04 | 28.6 dB |
| Real preprocessed image | 1.0000 | 82.4 dB |

The FastViTHD conv network saturates into activation regimes it never encounters
during training when fed random pixels. All Phase 2 evaluations use real corpus
images through the HF image processor.

---

## Layer 1A: verify_decoder.py — Four Phases

### Prerequisites

Build decoder fixtures for the variant you want to verify. Fixtures capture real
multimodal decoder inputs (image → vision encoder → projector → scatter-merge)
and are cached for reuse.

```bash
# Build fixtures for one variant (~5s per image on MPS, cached forever)
python scripts/build_fixtures.py --variant 0.5b

# Build for all downloaded variants
python scripts/build_fixtures.py

# Preview corpus and check which images exist
python scripts/build_fixtures.py --list
```

Fixtures are stored in `test_assets/fixtures/`. They are cached until
`FIXTURE_SCHEMA_VERSION` changes. If that happens, run `build_fixtures.py --force`.

### Four phases

| Phase | Device | What it tests | Gate |
|-------|--------|--------------|------|
| 1 — Architecture correctness | CPU | Re-authored decoder vs HF Qwen2 in fp32. Real embedding-table inputs. | >80 dB PASS, 50–80 dB MARGINAL (exits nonzero), <50 dB FAIL |
| 2 — FP16 fidelity | MPS | Decoder fp32 vs fp16 on real fixture inputs. Baseline for Phase 4. | Informational (MEASURED) |
| 3 — KV cache correctness | MPS | Incremental cached decode vs full-pass reference. | >40 dB PSNR |
| 4 — Compression quality | CPU prepare → MPS infer | Compressed vs fp16 on 9-image corpus at final generation position. | Top-5 overlap ≥80% mean, ≥60% worst |

### Usage

```bash
# Full four-phase verification (no compression)
python scripts/verify_decoder.py --variant 0.5b

# Full verification with compression
python scripts/verify_decoder.py --variant 0.5b --compression 4bit
python scripts/verify_decoder.py --variant 0.5b --compression 8bit

# Full verification with YAML recipe
python scripts/verify_decoder.py --variant 0.5b \
    --compression-config quantization_recipes/fastvlm-0.5b-aggressive.yaml

# Run individual phases
python scripts/verify_decoder.py --variant 0.5b --stage correctness
python scripts/verify_decoder.py --variant 0.5b --stage fidelity
python scripts/verify_decoder.py --variant 0.5b --stage cache
python scripts/verify_decoder.py --variant 0.5b --compression 4bit --stage compression
```

### Interpreting Phase 4 output

**RECOMMEND** — export and validate with `verify_runtime.py`.

**CAUTION** — worst-case image degrades significantly. Consider `8bit`
or a mixed-precision YAML recipe from `compression_scanner.py`.

**INCONCLUSIVE** — fixtures unavailable. Run `build_fixtures.py` first.

**Key metric notes:**

- **Top-5 overlap** is the primary gate. It asks "does the compressed model
  preserve the FP16 baseline's high-ranking candidate token set?" A top-1 flip
  between candidates already favored by the FP16 baseline is generally less
  concerning than introducing a different candidate set.
- **Top-1 agreement** is context only. Low top-1 combined with high top-5
  indicates compression is reordering candidates already favored by the FP16
  baseline — generally less concerning.
- **Margin preservation** is negative when the compressed model reverses the
  fp16 top-1/top-2 ordering. Not necessarily a problem if top-5=100%.
- **PSNR** is reported as context. It is **not the gate**. The 21.7 dB PSNR
  for 1.5B int4 fails any reasonable PSNR threshold but the model produces
  clean output at 115 tok/sec. Behavioral metrics are the evidence.

### When to run

| Situation | Run |
|-----------|-----|
| After any change to `fastvlm_decoder.py` | All phases |
| Before exporting a new compression preset | Phase 4 |
| New variant added | All phases, Phase 4 for each compression |
| After `FIXTURE_SCHEMA_VERSION` bumps | `build_fixtures.py --force`, then all phases |

---

## Layer 1B: verify_projector.py — Two Phases

Verifies the re-authored FastVLMProjector (mlp2x_gelu) against the HF
mm_projector. Both phases run on CPU.

| Phase | What it tests | Gate |
|-------|--------------|------|
| 1 — Architecture correctness | Re-authored FastVLMProjector vs HF mm_projector (nn.Sequential) in fp32. Both loaded from bf16 checkpoint. | >60 dB PASS, 40–60 dB MARGINAL (exits nonzero), <40 dB FAIL |
| 2 — FP16 fidelity | Port fp32 vs port fp16. Measures bf16→fp16 cast cost. | Informational (MEASURED) |

**Expected results:** Phase 1 typically produces **inf dB (bit-identical)** — the
projector is two `nn.Linear` layers and a GELU with no structural re-authoring,
so bit identity is the correct result. Phase 2 typically produces ~92 dB.

```bash
python scripts/verify_projector.py --variant 0.5b
python scripts/verify_projector.py --variant 0.5b --stage correctness
python scripts/verify_projector.py --variant 0.5b --stage fidelity
```

---

## Layer 1C: verify_vision_encoder.py — Two Phases

Verifies the re-authored FastVLMVisionEncoder (FastViTHD) against the full HF
model. Phase 1 on CPU; Phase 2 on MPS with real corpus images.

| Phase | Device | What it tests | Gate |
|-------|--------|--------------|------|
| 1 — Architecture correctness | CPU | Re-authored encoder vs full HF model at fp32. Confirms ANELayerNorm and ANEAttention substitutions are functionally equivalent. | >70 dB PASS, 50–70 dB MARGINAL (exits nonzero), <50 dB FAIL |
| 2 — FP16 fidelity | MPS | Port fp32 vs port fp16 on 9 real corpus images through HF image processor. | Informational (MEASURED) |

**Phase 2 requirements:**
- **MPS only** — fp16 overflows on CPU at 1024×1024 (FastViTHD has 186 conv2d
  ops; fp16 saturates at network.9 with values approaching 65504 ceiling).
- **Real images required** — random inputs produce cosine=0.04 (catastrophically
  misleading); real preprocessed images produce cosine≈1.0, PSNR≈81.5 dB.
- **Both models on same device** — cross-device comparison (CPU fp32 vs MPS fp16)
  produces spurious divergence from different floating-point implementations.

**Expected results:** Phase 1 ~138 dB. Phase 2 ~81.5 dB mean PSNR, cosine=1.0000
mean across 9 corpus images.

```bash
python scripts/verify_vision_encoder.py --variant 0.5b
python scripts/verify_vision_encoder.py --variant 0.5b --stage correctness
python scripts/verify_vision_encoder.py --variant 0.5b --stage fidelity
```

---

## Layer 2: verify_runtime.py

Requires the exported bundle and `llm-runner` from Apple's `coreai-models`.
Requires macOS 27 GM or later.

```bash
# Verify end-to-end CoreAI runtime vs PyTorch reference
python scripts/verify_runtime.py --variant 0.5b \
    --image test_assets/images/great_wave.jpg

# With decode steps
python scripts/verify_runtime.py --variant 0.5b \
    --image test_assets/images/great_wave.jpg \
    --decode-steps 5
```

Expected output for 0.5B fp16 with a real image:

```
  ✓ vision_encode PSNR:  71.9 dB  (PASS)
  ✓ project PSNR:        67.6 dB  (PASS)
  ✓ embed_tokens PSNR:   inf dB   (PASS — bit-identical)
  ✓ scatter_merge PSNR:  67.7 dB  (PASS)
  ✓ decode prefill PSNR: 50.2 dB  (PASS > 40 dB)
  ✓ decode step 1:       44.4 dB  (PASS > 40 dB)

[PASS] All stages match PyTorch reference.
```

---

## Threshold calibration note

The Phase 4 thresholds (top-5 ≥80% mean, ≥60% worst) are project heuristics
designed to rank and reject obviously degraded recipes. They are not empirically
established model-quality boundaries. As more CoreAI A/B data is collected —
comparing Phase 4 predictions against actual exported model behavior and
human-observed output quality — these thresholds should be calibrated.

---

## Metrics reference

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| PSNR (dB) | Signal-to-noise ratio of logit tensors | Higher is better. Inf = identical. Not the gate. |
| NRMSE | Normalized root mean square error | Lower is better. 0 = identical. |
| Cosine similarity | Direction of logit vector | ≥0.99 is excellent. |
| KL divergence | Distribution shift in token probabilities | Lower is better. 0 = identical. |
| Top-5 overlap | Fraction of positions where top-5 token sets match | ≥80% for Phase 4 gate. |
| Top-1 agreement | Fraction of positions where argmax matches | Context only — not gated. |
| Margin preservation | Ratio of compressed/fp16 margin on reference top-2 tokens | >0 preserves ordering. |

---

## Fixture corpus

Nine public domain images covering diverse scene types, used for Phase 2
(FP16 fidelity) and Phase 4 (compression quality) in all three verify scripts.

| Image | Scene type |
|-------|-----------|
| `great_wave.jpg` | Artwork / historical |
| `earthrise.jpg` | Space / landscape |
| `blue_marble.jpg` | Space / Earth |
| `pale_blue_dot.png` | Space / minimal |
| `pillars_of_creation.jpg` | Space / nebula |
| `hubble_deep_field.jpg` | Space / stars |
| `girl_pearl_earring.jpg` | Portrait |
| `migrant_mother.jpg` | Portrait / documentary |
| `lunch_skyscraper.jpg` | Architecture / people |

Download with: `python scripts/fetch_test_images.py`
