# PSNR Verification Results

PSNR thresholds:
- Decoder/projector re-authored fp16 vs fp32: **> 50 dB** (investigate < 40 dB)
- Vision encoder re-authored fp16 vs fp32: **> 70 dB** (investigate < 60 dB)
- After palettization vs fp16: **> 35 dB** (investigate < 30 dB)
- Compiled on-device vs fp32: **> 40 dB** (investigate < 35 dB)

---

## 1.5B

| Stage | PSNR | Pass? | Notes |
|-------|------|-------|-------|
| Decoder re-authored fp16 vs fp32 | 61.8 dB | ✓ | |
| Vision encoder re-authored fp16 vs fp32 | TBD | — | |
| Projector re-authored fp16 vs fp32 | TBD | — | |
| Python runtime (macOS) vs fp32 | TBD | — | |
| After palettization vs fp16 (iOS) | TBD | — | |
| Compiled iOS on-device vs fp32 | TBD | — | |

---

## 0.5B

| Stage | PSNR | Pass? | Notes |
|-------|------|-------|-------|
| Decoder re-authored fp16 vs fp32 | TBD | — | |
| Vision encoder re-authored fp16 vs fp32 | TBD | — | |
| Python runtime (macOS) vs fp32 | TBD | — | |
| After palettization vs fp16 (iOS) | TBD | — | |

---

## 7B

| Stage | PSNR | Pass? | Notes |
|-------|------|-------|-------|
| Decoder re-authored fp16 vs fp32 | TBD | — | |
| Vision encoder re-authored fp16 vs fp32 | TBD | — | |
| Python runtime (macOS) vs fp32 | TBD | — | |
| After INT4 quantization vs fp16 (macOS) | TBD | — | |
