"""
fastvlm_decoder.py — Re-authored FastVLM language decoder for CoreAI export.

The decoder is Qwen2. Reference:
  coreai-models/python/src/coreai_models/models/macos/qwen2.py

Key decisions:
  - RMSNormImpl, SDPA, RoPE are composite ops (externalized via ExternalizeSpec)
  - nn.LayerNorm and nn.BatchNorm2d handled automatically by get_decomp_table()
  - KV cache uses register_buffer + state_names (NOT readonly KV cache pattern)
  - q_proj, k_proj, v_proj are SEPARATE nn.Linear layers (NOT fused)
  - KV cache writes use aten.slice_scatter with scalar shape-symint indices

QKV FUSION DECISION
-------------------
An earlier version of this file fused q/k/v into a single qkv_proj Linear
(via _mutate_state_dict) to match Apple's Core AI reference implementation
(coreai-models/qwen2.py, USE_FUSED_KV = True). However, Apple's MLX
inference weights (llava-fastvithd_*b_stage3_llm.*) keep q/k/v UNFUSED and
independently quantized — verified by scripts/audit_weight_dtypes.py showing
self_attn.q_proj.weight, k_proj.weight, v_proj.weight each as separate
QUANTIZED tensors.

Fusing QKV before quantization merges three projections with different weight
distributions into one matrix, giving the quantizer a blunter instrument
to represent all three — particularly v_proj, which has inherently higher
weight variance and reconstructs 6-10 dB worse than q_proj in both 1.5B
and 7B per our compare_weights.py analysis. Keeping them unfused lets the
per-group scales for each projection adapt independently.

We depart from Apple's Core AI reference architecture here deliberately, to
match Apple's actual quantization practice. The coreai-torch SDPA composite
op takes separate q, k, v tensors regardless — qkv fusion was never a
coreai-torch requirement, only our earlier choice. Unfusing has no export
impact and should improve quantization quality, especially at int4 for 7B.

KV CACHE DESIGN
---------------
k_cache and v_cache have shape (num_hidden_layers, 1, MAX_SEQ_LEN, kv_dim)
where kv_dim = head_dim * num_key_value_heads (flattened head layout).

Each attention layer writes its new K/V vectors into the cache at
offset = seq_len - query_len, then reads back the full context [0:seq_len]
for attention. The cache write uses aten.slice_scatter with scalar symint
begin/end, avoiding auto_functionalized_v2 (from mutates_args custom ops)
and the MPSGraph runtime crash with tensor begin indices (FB23024751).

FORWARD SIGNATURE
-----------------
forward(input_ids, position_ids) -> logits

  input_ids    : (1, query_len)  int32  — tokens for this step
  position_ids : (1, seq_len)   int32  — ABSOLUTE positions for full context
                 seq_len >= query_len; grows by 1 each decode step
  logits       : (1, query_len, vocab_size)

position_ids is wider than input_ids during decode so each attention layer
can compute offset = seq_len - query_len without Python branching.
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from safetensors import safe_open

# Maximum context length for static KV cache shape at export time.
MAX_SEQ_LEN = 4096


# ─── KV cache write helper ───────────────────────────────────────────────────
#
# Uses aten.slice_scatter with scalar shape-symint begin/end rather than a
# mutates_args custom op. A mutates_args custom op produces auto_functionalized_v2
# in the exported graph, which coreai-torch 0.4.0 cannot lower (no handler in
# _higher_order_resolver). aten.slice_scatter is already in _aten_to_core_resolver
# and lowers cleanly. Scalar symint begin/end also avoids the MPSGraph runtime
# crash with runtime-tensor begin indices (FB23024751, apple/coreai-models#5).


# ─── Composite op wrappers ────────────────────────────────────────────────────


class FastVLMRMSNorm(nn.Module):
    """
    Holds the learnable scale (gamma) as nn.Parameter and passes it
    explicitly to RMSNormImpl.forward(x, weight).

    Required because RMSNormImpl does NOT hold the scale internally —
    it must be passed as a forward argument so it appears as a graph
    input on the composite op boundary after externalization.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.norm = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.weight)


class FastVLMAttention(nn.Module):
    """
    Qwen2 attention with separate q_proj, k_proj, v_proj (NOT fused).

    Keeping projections separate gives the quantizer independent per-group
    scales for each projection, matching Apple's MLX quantization scheme
    exactly (verified by audit_weight_dtypes.py: each proj is a separate
    QUANTIZED tensor in Apple's shipped MLX weights).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        q_dim = self.n_heads * self.head_dim
        kv_dim_proj = self.n_kv_heads * self.head_dim

        # Separate projections — intentionally NOT fused. See module docstring.
        self.q_proj = nn.Linear(dim, q_dim, bias=True)
        self.k_proj = nn.Linear(dim, kv_dim_proj, bias=True)
        self.v_proj = nn.Linear(dim, kv_dim_proj, bias=True)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)

        self.sdpa = SDPA(is_causal=True)
        self.rope = RoPE(base=float(getattr(config, "rope_theta", 1e6)))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim
        kv_dim = n_kv_heads * head_dim
        layer = self.layer_idx
        # Compute offset as a shape symint (not a runtime tensor) so MPSGraph
        # can lower the slice_update. The bug (FB23024751, apple/coreai-models#5)
        # is that MPSGraph crashes when begin is a runtime-tensor index but runs
        # correctly when begin is derived from shape dimensions (symints).
        # position_ids.shape[-1] == seq_len and x.shape[1] == query_len (== L).
        offset = position_ids.shape[-1] - x.shape[1]

        # Separate projections → reshape to head layout
        q = self.q_proj(x).reshape(B, L, n_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)

        # RoPE on q and k together (fused for efficiency, standard Qwen2 pattern)
        torch._check_is_size(seq_len)
        rope_positions = position_ids.narrow(-1, seq_len - L, L)
        qk = self.rope(
            torch.cat([q, k], dim=1), position_ids=rope_positions
        )
        q = qk.narrow(1, 0, n_heads)
        k = qk.narrow(1, n_heads, n_kv_heads)

        # Flatten heads for cache storage: (B, n_kv_heads, L, head_dim) -> (B, L, kv_dim)
        k_flat = k.permute(0, 2, 1, 3).reshape(B, L, kv_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(B, L, kv_dim)

        # Write new K/V into cache at [layer, :, offset:offset+L, :]
        # slice_scatter with scalar symint begin/end: no custom op, no tensors.
        # Cache shape: [num_layers, 1, MAX_SEQ_LEN, kv_dim]
        # k_flat/v_flat shape: [1, L, kv_dim] — insert at dim=0 for layer idx,
        # dim=2 for sequence position.
        # Cache shape: [num_layers, 1, MAX_SEQ_LEN, kv_dim]
        # k_flat shape: [1, L, kv_dim] -> unsqueeze to [1, 1, L, kv_dim]
        # We need to write k_flat into k_cache[layer, :, offset:offset+L, :].
        # Strategy: slice k_cache to the single layer row, scatter into it,
        # then scatter the updated row back into the full cache.
        k_flat_4d = k_flat.unsqueeze(0)  # [1, 1, L, kv_dim]
        v_flat_4d = v_flat.unsqueeze(0)

        # Step 1: extract the layer row [1, 1, MAX_SEQ_LEN, kv_dim]
        k_row = k_cache[layer:layer+1]
        v_row = v_cache[layer:layer+1]

        # Step 2: scatter new tokens into the row at dim=2 (sequence dim)
        k_row_new = torch.ops.aten.slice_scatter(
            k_row, k_flat_4d, dim=2, start=offset, end=offset + L
        )
        v_row_new = torch.ops.aten.slice_scatter(
            v_row, v_flat_4d, dim=2, start=offset, end=offset + L
        )

        # Step 3: scatter the updated row back into the full cache at dim=0
        k_cache.copy_(torch.ops.aten.slice_scatter(
            k_cache, k_row_new, dim=0, start=layer, end=layer + 1
        ))
        v_cache.copy_(torch.ops.aten.slice_scatter(
            v_cache, v_row_new, dim=0, start=layer, end=layer + 1
        ))

        # Read back full context [0:seq_len] and restore head-separated layout
        k_ctx = k_cache[layer].narrow(1, 0, seq_len)
        v_ctx = v_cache[layer].narrow(1, 0, seq_len)
        k = k_ctx.reshape(B, seq_len, n_kv_heads, head_dim).permute(0, 2, 1, 3).to(q.dtype)
        v = v_ctx.reshape(B, seq_len, n_kv_heads, head_dim).permute(0, 2, 1, 3).to(q.dtype)

        out = (
            self.sdpa(q, k, v)
            .permute(0, 2, 1, 3)
            .reshape(B, L, n_heads * head_dim)
        )
        return self.o_proj(out)


class FastVLMMLP(nn.Module):
    """SwiGLU MLP. silu is handled automatically by get_decomp_table()."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class FastVLMDecoderBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = FastVLMAttention(config, layer_idx)
        self.mlp = FastVLMMLP(config)
        self.input_layernorm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x), position_ids, k_cache, v_cache, seq_len
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# ─── Stateful decoder model ───────────────────────────────────────────────────


class FastVLMDecoderStateful(nn.Module):
    """
    Full Qwen2 decoder re-authored for CoreAI export with KV cache states.

    KV cache is registered as mutable buffers via register_buffer.
    The export pipeline passes state_names=["k_cache", "v_cache"] to
    TorchConverter.add_pytorch_module(), which maps both the input and
    in-place mutation output to a single state name each.

    Cache buffers must be zeroed before a new generation sequence begins.
    In the Swift runtime this is handled by the state reset API.
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FastVLMDecoderBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

        head_dim = config.hidden_size // config.num_attention_heads
        kv_dim = head_dim * config.num_key_value_heads
        self.register_buffer("k_cache", torch.zeros(config.num_hidden_layers, 1, MAX_SEQ_LEN, kv_dim))
        self.register_buffer("v_cache", torch.zeros_like(self.k_cache))

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        B, query_len = input_ids.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        # offset is NOT computed as a runtime tensor here — each attention layer
        # derives it from shape symints directly. See FastVLMAttention.forward.

        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, self.k_cache, self.v_cache, seq_len)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoderStateful":
        """Load from SafeTensors weights directory."""
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
        missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
        actual_missing = set(missing) - {"k_cache", "v_cache"}
        if actual_missing:
            raise RuntimeError(f"Unexpected missing keys: {actual_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        return model


# ─── Weight loading helpers ───────────────────────────────────────────────────

# Keys that identify decoder weights in the FastVLM checkpoint.
_DECODER_PREFIXES = (
    "model.layers.",
    "model.embed_tokens.",
    "model.norm.",
    "lm_head.",
)


def _load_decoder_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """
    Load decoder weights from SafeTensors, filtering to _DECODER_PREFIXES only.

    Excludes vision tower and projector weights. Casts bfloat16 (the storage
    dtype in Apple's checkpoint) to the target dtype on load.

    Note: _mutate_state_dict (QKV fusion) has been removed. Weights are
    loaded as-is from HF with separate q_proj/k_proj/v_proj, which now
    load directly into FastVLMAttention's separate q_proj/k_proj/v_proj
    nn.Linear modules without any key remapping needed.
    """
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    result = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not any(key.startswith(p) for p in _DECODER_PREFIXES):
                    continue
                t = f.get_tensor(key)
                if t.dtype != dtype:
                    t = t.to(dtype)
                result[key] = t
    return result
