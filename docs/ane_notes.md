# ANE Conversion Notes

## Issues found and fixed

<!-- Document each ANE incompatibility encountered and how it was resolved.
     Example format:

### SEBlock dynamic AvgPool2d
- Issue: AvgPool2d(h, w) with dynamic h/w — ANE requires static shapes
- Fix: Replaced with AdaptiveAvgPool2d(1)
- Stage: FastVLMVisionEncoder, stage N

### LayerNormChannel manual decomposition
- Issue: Manual mean/pow/sqrt decomposition prevents layer_norm composite op
- Fix: Replaced with nn.LayerNorm(C)
- Stage: AttentionBlock
-->

## Composite ops used

| Op | Class | composite_op_name | composite_attrs |
|----|-------|-------------------|----------------|
| RMSNorm | RMSNormImpl | rms_norm | ["axes", "eps"] |
| Attention | SDPA | scaled_dot_product_attention | ["scale", "is_causal", "window_size"] |
| RoPE | RoPE | rope | ["base", "dims", "interleaved"] |

## Ops on ANE vs GPU fallback

<!-- Document from Core AI Debugger inspection after compilation.
     Green = ANE, Yellow = GPU fallback, Red = CPU fallback.

### 1.5B macOS
- vision_encode: TBD
- project: TBD
- decode: TBD

### 1.5B iOS
- vision_encode: TBD
- project: TBD
- decode: TBD
-->

## Static shape decisions

<!-- Document fixed sizes chosen at export time:
  - image_size: TBD (from vision_config.image_size)
  - MAX_SEQ_LEN: TBD (from text_config.max_position_embeddings)
  - KV cache shape: [num_layers, 1, MAX_SEQ_LEN, head_dim * num_kv_heads]
-->
