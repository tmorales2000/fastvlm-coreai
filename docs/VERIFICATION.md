# FastVLM CoreAI — Verification Guide

Two verification layers catch different classes of problems:

**Layer 1 — PyTorch verification** (`verify_decoder.py`)
No CoreAI runtime needed. Runs fast. Catches re-authoring bugs and
quantization quality issues before committing to a full export.

**Layer 2 — CoreAI runtime verification** (`verify_runtime.py`)
Requires a compiled `.aimodel` bundle and `llm-runner`. Catches
export pipeline bugs, MLIR lowering issues, and compiled model drift.

---

## Layer 1: verify_decoder.py

### Prerequisites

Build decoder fixtures for the variant you want to verify.
Fixtures capture real multimodal decoder inputs (image → vision encoder →
projector → scatter-merge) and are cached for reuse.

```bash
# Build fixtures for one variant (~5s per image on MPS, cached forever)
python scripts/build_fixtures.py --variant 0.5b

# Build for all downloaded variants
python scripts/build_fixtures.py

# Preview corpus and check which images exist
python scripts/build_fixtures.py --list
```

Fixtures are stored in `test_assets/fixtures/`. They are cached until
`FIXTURE_SCHEMA_VERSION` changes (pipeline change). If that happens,
run `build_fixtures.py --force` to rebuild.

### Four phases

| Phase | What it tests | Gate |
|-------|--------------|------|
| 1 — Architecture correctness | Re-authored decoder vs HF Qwen2 in fp32. Random token sequence with real embedding-table inputs. | >80 dB PASS, 50–80 dB MARGINAL (exits nonzero), <50 dB FAIL |
| 2 — FP16 fidelity | Decoder in fp32 vs fp16 on real fixture inputs. Establishes the fp16 deployment baseline for Phase 4. | Informational (MEASURED) |
| 3 — KV cache correctness | Incremental cached decode vs full-pass reference. Catches cache offset, head reshape, and `mutable_slice_update` bugs. | >40 dB PSNR |
| 4 — Compression quality | Compressed vs fp16 on 9-image corpus. Evaluates the final generation position (first token decision). | Top-5 overlap ≥80% mean, ≥60% worst |

### Usage

```bash
# Full four-phase verification (no compression)
python scripts/verify_decoder.py --variant 0.5b

# Full verification with compression preset
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

```
  Corpus: 9 images
  Prompt: "Describe exactly what you see in this image."
  Evaluated at: final generation position (first token decision)

        Metric             Mean     Worst
  --------------------------------------------
  PSNR (dB)               32.3      31.5   (ref: 20 dB)
  Top-5 overlap          100.0%   100.0%
  Top-1 agreement         50.0%     0.0%   (context only)
  Margin preservation     -0.43    -4.54

  Top-5 ≥80%: 9/9 images pass

  Primary gate:
    mean  top-5 overlap : 100.0%  (threshold: ≥80%)
    worst top-5 overlap : 100.0%  (floor: ≥60%)

[RECOMMEND] Export 4bit recipe for Core AI runtime validation.
```

**RECOMMEND** — export and validate with `verify_runtime.py`.

**CAUTION** — worst-case image degrades significantly. Consider `8bit`
or a mixed-precision YAML recipe from `scan_quantization_sensitivity.py`.

**INCONCLUSIVE** — fixtures unavailable. Run `build_fixtures.py` first.

**Key metric notes:**
- **Top-5 overlap** is the primary gate. It asks "does the compressed model
  preserve the FP16 baseline's high-ranking candidate token set?" A top-1 flip
  between candidates already favored by the FP16 baseline is generally less
  concerning than introducing a substantially different candidate set. Top-5
  overlap does not by itself prove semantic equivalence of candidates.
- **Top-1 agreement** is context only. Low top-1 combined with high top-5
  indicates compression is primarily reordering candidates already favored by
  the FP16 baseline. This is generally less concerning than introducing a
  different candidate set, but does not prove semantic equivalence.
- **Margin preservation** is negative when the compressed model reverses
  the fp16 top-1/top-2 ordering. Not necessarily a problem if top-5=100%.
- **PSNR** is reported as context. It is not the gate. The 21.7 dB PSNR
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

## Layer 2: verify_runtime.py

Requires the exported bundle and `llm-runner` from Apple's `coreai-models`.
Run on macOS 26.5 — see [Known Issues](STATUS.md) for macOS 27 beta crash.

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

Without `--image`, vision stages show NaN (expected — fp16 saturation
with random pixel_values input). This is not a failure.

---

## Threshold calibration note

The Phase 4 thresholds (top-5 ≥80% mean, ≥60% worst) are project heuristics
designed to rank and reject obviously degraded recipes. They are not empirically
established model-quality boundaries. As more Core AI A/B data is collected —
comparing Phase 4 predictions against actual exported model behavior and
human-observed output quality — these thresholds should be calibrated
to reflect what the metrics actually predict.

---

## Metrics reference

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| PSNR (dB) | Signal-to-noise ratio of logit tensors | Higher is better. Inf = identical. |
| NRMSE | Normalized root mean square error | Lower is better. 0 = identical. |
| Cosine similarity | Direction of logit vector | ≥0.99 is excellent. |
| KL divergence | Distribution shift in token probabilities | Lower is better. 0 = identical. |
| Top-5 overlap | Fraction of positions where top-5 token sets match | ≥80% for compression gate. |
| Top-1 agreement | Fraction of positions where argmax matches | Context only — not gated. |
| Margin preservation | Ratio of compressed/fp16 margin on reference top-2 tokens | >0 preserves ordering. <0 reverses it. |

---

## Fixture corpus

Nine public domain images covering diverse scene types:

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
