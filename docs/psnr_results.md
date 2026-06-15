# PSNR Verification Results

## Decoder thresholds (two-stage verification)
- Stage 1 — fp32 port vs HF Qwen2ForCausalLM: **> 80 dB** pass (investigate < 50 dB)
- Stage 2 — fp16 cached decode vs full pass: **> 40 dB** pass (investigate < 40 dB)
- Stage 2 — fp16 overflow guard: max |logit| < 60000 (fp16 ceiling 65504)

## Projector thresholds (two-stage verification)
- Stage 1 — fp32 port vs original HF mm_projector: **> 80 dB** pass (inf dB = bit-identical)
- Stage 2 — fp16 vs fp32 self-consistency: **> 60 dB** pass (investigate < 50 dB)
- Stage 2 — fp16 overflow guard: max |output| < 60000 (fp16 ceiling 65504)

## Other component thresholds
- Vision encoder re-authored fp16 vs fp32: **> 70 dB** (investigate < 60 dB)
- After palettization vs fp16: **> 35 dB** (investigate < 30 dB)
- Compiled on-device vs fp32: **> 40 dB** (investigate < 35 dB)

---

## 1.5B

| Stage | PSNR | Pass? | Notes |
|-------|------|-------|-------|
| Decoder Stage 1 — fp32 port vs HF Qwen2 | 113.2 dB | ✓ | |
| Decoder Stage 2 — fp16 cached decode | 72.1 dB | ✓ | max logit 12 |
| Projector Stage 1 — fp32 port vs HF mm_projector | inf dB | ✓ | bit-identical |
| Projector Stage 2 — fp16 health | 90.9 dB | ✓ | max output 9.43 |
| Vision encoder re-authored fp16 vs fp32 | TBD | — | |
| Python runtime (macOS) vs fp32 | TBD | — | |
| After palettization vs fp16 (iOS) | TBD | — | |
| Compiled iOS on-device vs fp32 | TBD | — | |

---

## 0.5B

| Stage | PSNR | Pass? | Notes |
|-------|------|-------|-------|
| Decoder Stage 1 — fp32 port vs HF Qwen2 | 129.2 dB | ✓ | |
| Decoder Stage 2 — fp16 cached decode | 65.1 dB | ✓ | max logit 16 |
| Projector Stage 1 — fp32 port vs HF mm_projector | inf dB | ✓ | bit-identical |
| Projector Stage 2 — fp16 health | 93.2 dB | ✓ | max output 10.45 |
| Vision encoder re-authored fp16 vs fp32 | TBD | — | |
| Python runtime (macOS) vs fp32 | TBD | — | |
| After palettization vs fp16 (iOS) | TBD | — | |

---

## 7B

| Stage | PSNR | Pass? | Notes |
|-------|------|-------|-------|
| Decoder Stage 1 — fp32 port vs HF Qwen2 | 110.2 dB | ✓ | |
| Decoder Stage 2 — fp16 cached decode | 49.5 dB | ✓ | max logit 18 |
| Projector Stage 1 — fp32 port vs HF mm_projector | inf dB | ✓ | bit-identical |
| Projector Stage 2 — fp16 health | 95.9 dB | ✓ | max output 44.44 |
| Vision encoder re-authored fp16 vs fp32 | TBD | — | |
| Python runtime (macOS) vs fp32 | TBD | — | |
| After INT4 quantization vs fp16 (macOS) | TBD | — | |
