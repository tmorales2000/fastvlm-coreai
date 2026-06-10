# FastVLM Architecture Notes

## Model variants

<!-- Fill in from discover_weights.py output -->

| Property | 0.5B | 1.5B | 7B |
|----------|------|------|----|
| hidden_size | TBD | TBD | TBD |
| num_hidden_layers | TBD | TBD | TBD |
| num_attention_heads | TBD | TBD | TBD |
| num_key_value_heads | TBD | TBD | TBD |
| intermediate_size | TBD | TBD | TBD |
| vocab_size | TBD | TBD | TBD |
| rope_theta | TBD | TBD | TBD |
| rms_norm_eps | TBD | TBD | TBD |
| mm_projector_type | TBD | TBD | TBD |
| mm_hidden_size | TBD | TBD | TBD |
| image_size | TBD | TBD | TBD |

## Component breakdown

- **vision_encoder**: FastViTHD — hybrid Conv2d/attention encoder
- **projector**: MLP (depth from mm_projector_type config, e.g. mlp2x_gelu = 2 layers)
- **decoder**: Qwen2 transformer with RoPE positional encoding

## Weight key sanitize() renames (from FastVLM.swift)

<!-- Document during conversion:
  - layer_scale_  →  layerScale
  - vision_model.network.N.M  →  vision_model.network.N.layers.M
  - mm_projector.N  →  mm_projector.layers.N
-->

## Image preprocessing

<!-- Document from preprocessor_config.json:
  - crop_size
  - mean / std normalization values
  - resize strategy
-->

## Three-function asset layout

| entrypoint_name | Input | Output |
|----------------|-------|--------|
| vision_encode | [1,3,H,W] fp16 | [1,h*w,C] patch embeddings |
| project | [1,h*w,C] embeddings | [1,h*w,D] projected |
| decode | input_ids + position_ids + KV state | logits [1,L,vocab] |
