"""
fastvlm_decoder.py — Re-authored FastVLM language decoder for CoreAI export.

Follows Apple's authoritative recipe from coreai-models:
  coreai-models/python/src/coreai_models/vlm/export.py
  coreai-models/python/src/coreai_models/models/macos/qwen3_vl.py
  coreai-models/python/src/coreai_models/primitives/macos/cache.py

DESIGN
======
1. INPUTS_EMBEDS NOT INPUT_IDS
   Decoder takes pre-computed embeddings. embed_tokens exported separately
   as embed.aimodel (asset role: embedding).

2. K_CACHE / V_CACHE AS EXPLICIT FORWARD ARGS
   Matching Qwen3VLForCausalLMEmbeddings.forward() exactly:
     forward(inputs_embeds, position_ids, k_cache, v_cache) -> logits
   State names: "k_cache" / "v_cache" matching KV_STATE_NAMES in vlm/export.py.
   At runtime the coreai-torch compiler renames these to keyCache/valueCache
   (camelCase) for Swift CoreAISequentialVLMEngine compatibility.

3. KVCACHE FROM coreai-models cache.py (mutable_slice_update)
   Imported directly from coreai_models.primitives.macos.cache.
   Uses mutable_slice_update which creates AutoFunctionalized nodes that
   remove_functionalization replaces with immutable_slice_update for MLIR.
   NOTE: cache_scatter.py (slice_scatter) is for inference/testing only —
   remove_functionalization only handles mutable_slice_update, not slice_scatter.

4. QKV UNFUSED — independent quantization scales per projection.

5. KV CACHE SHAPE
   At export: [n_layers, 1, n_kv_heads, max_ctx, head_dim] — fixed at export time.
   At runtime: compiled model exposes seq_dim as dynamic (-1) → GrowingKVCache
   in Swift. This is a coreai-torch optimization, not something we control.

FORWARD SIGNATURE
=================
forward(inputs_embeds, position_ids, k_cache, v_cache) -> logits

  inputs_embeds : [1, L, hidden]                             fp16
  position_ids  : [1, S]                                     int32  S = offset + L
  k_cache       : [n_layers, 1, n_kv_heads, max_ctx, head]   fp16
  v_cache       : same
  logits        : [1, L, vocab_size]                         fp16
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from safetensors import safe_open

# State names — must match KV_STATE_NAMES in coreai-models/vlm/export.py
KEY_CACHE_NAME   = "k_cache"
VALUE_CACHE_NAME = "v_cache"
KV_STATE_NAMES   = (KEY_CACHE_NAME, VALUE_CACHE_NAME)


# ─── KVCache — exact copy of cache.py (mutable_slice_update) ────────────────

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.cache import KVCache  # noqa: F401 — re-export

# KVCache is imported directly from coreai-models for exact API compatibility.
# cache.py uses mutable_slice_update which creates AutoFunctionalized nodes
# that remove_functionalization replaces with immutable_slice_update for MLIR.
# (cache_scatter.py uses slice_scatter and is for inference/testing only —
#  NOT for export, as remove_functionalization only handles mutable_slice_update.)


# ─── Composite op wrappers ────────────────────────────────────────────────────

class FastVLMRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.norm   = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.weight)


class FastVLMAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx  = layer_idx
        dim             = config.hidden_size
        self.n_heads    = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim   = dim // self.n_heads
        q_dim           = self.n_heads    * self.head_dim
        kv_dim          = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, q_dim,  bias=True)
        self.k_proj = nn.Linear(dim, kv_dim, bias=True)
        self.v_proj = nn.Linear(dim, kv_dim, bias=True)
        self.o_proj = nn.Linear(q_dim, dim,  bias=False)

        self.sdpa = SDPA(is_causal=True)
        self.rope = RoPE(base=float(getattr(config, "rope_theta", 1e6)))

    def forward(
        self,
        x:            torch.Tensor,
        position_ids: torch.Tensor,
        cache:        KVCache,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim

        seq_len   = position_ids.shape[-1]
        torch._check_is_size(L)
        torch._check_is_size(seq_len)
        offset = seq_len - L
        torch._check_is_size(offset)

        q = self.q_proj(x).reshape(B, L, n_heads,    head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)

        rope_positions = position_ids.narrow(-1, offset, L)
        qk = self.rope(torch.cat([q, k], dim=1), position_ids=rope_positions)
        q  = qk.narrow(1, 0, n_heads)
        k  = qk.narrow(1, n_heads, n_kv_heads)

        k_ctx, v_ctx = cache.update_and_fetch(
            self.layer_idx, offset, k, v, seq_len=seq_len, query_len=L
        )

        out = (
            self.sdpa(q, k_ctx.to(q.dtype), v_ctx.to(q.dtype))
            .permute(0, 2, 1, 3)
            .reshape(B, L, n_heads * head_dim)
        )
        return self.o_proj(out)


class FastVLMMLP(nn.Module):
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
        self.self_attn                = FastVLMAttention(config, layer_idx)
        self.mlp                      = FastVLMMLP(config)
        self.input_layernorm          = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x:            torch.Tensor,
        position_ids: torch.Tensor,
        cache:        KVCache,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


# ─── Decoder ─────────────────────────────────────────────────────────────────

class FastVLMDecoder(nn.Module):
    """
    FastVLM (Qwen2) decoder for CoreAI export.

    Matches Qwen3VLForCausalLMEmbeddings.forward() from Apple's recipe:
      forward(inputs_embeds, position_ids, k_cache, v_cache) -> logits

    - inputs_embeds replaces input_ids — embed_tokens is in embed.aimodel
    - k_cache/v_cache are explicit args, state_names=("k_cache","v_cache")
    - KVCache uses slice_scatter (cache_scatter.py pattern)
    - Fixed cache size (dynamic_shapes=None for caches per vlm/export.py)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FastVLMDecoderBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm    = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [1, L, hidden]                            fp16
        position_ids:  torch.Tensor,  # [1, S]                                    int32
        k_cache:       torch.Tensor,  # [n_layers, 1, n_kv_heads, max_ctx, head]  fp16
        v_cache:       torch.Tensor,  # same
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoder":
        """Load weights, excluding embed_tokens (in embed.aimodel)."""
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
        missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        return model


# ─── EmbedTokens (embed.aimodel) ─────────────────────────────────────────────

class FastVLMEmbedTokens(nn.Module):
    """
    Token embedding lookup, exported as embed.aimodel.
    Matches EmbedTokens from vlm/export.py exactly.
    Uses weight[input_ids] for Int32 index compatibility.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]

    @classmethod
    def from_weights(cls, weights_dir: str) -> "FastVLMEmbedTokens":
        st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
        for path in st_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "embed_tokens" in key:
                        return cls(f.get_tensor(key).to(torch.float16))
        raise FileNotFoundError(f"embed_tokens not found in {weights_dir}")


# Backward-compat alias
FastVLMDecoderStateful = FastVLMDecoder


# ─── Weight loading ───────────────────────────────────────────────────────────

_DECODER_PREFIXES = (
    "model.layers.",
    "model.norm.",
    "lm_head.",
)


def _load_decoder_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Load decoder weights, excluding embed_tokens (lives in embed.aimodel)."""
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
