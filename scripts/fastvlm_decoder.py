"""
fastvlm_decoder.py — Re-authored FastVLM language decoder for CoreAI export.

The decoder is Qwen2. Reference:
  coreai-models/python/src/coreai_models/models/macos/qwen2.py

Key decisions:
  - RMSNormImpl, SDPA, RoPE are composite ops (externalized via ExternalizeSpec)
  - nn.LayerNorm and nn.BatchNorm2d handled automatically by get_decomp_table()
  - KV cache uses register_buffer + state_names (NOT readonly KV cache pattern)
  - QKV projections fused in _mutate_state_dict before weight loading
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from safetensors import safe_open
from transformers import AutoConfig

# Maximum context length for static KV cache shape at export time.
# Set from text_config.max_position_embeddings in practice.
MAX_SEQ_LEN = 4096


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
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        total = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

        self.qkv_proj = nn.Linear(dim, total, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Composite ops — preserved by ExternalizeSpec during export
        self.sdpa = SDPA(is_causal=True)
        self.rope = RoPE(base=float(getattr(config, "rope_theta", 1e4)))

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(B, L, self.n_heads + 2 * self.n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        qk = qkv.narrow(1, 0, self.n_heads + self.n_kv_heads)
        v = qkv.narrow(1, self.n_heads + self.n_kv_heads, self.n_kv_heads)

        qk = self.rope(qk, position_ids=position_ids)
        q = qk.narrow(1, 0, self.n_heads)
        k = qk.narrow(1, self.n_heads, self.n_kv_heads)

        out = (
            self.sdpa(q, k, v)
            .permute(0, 2, 1, 3)
            .reshape(B, L, self.n_heads * self.head_dim)
        )
        return self.o_proj(out)


class FastVLMMLP(nn.Module):
    """SwiGLU MLP. silu is handled automatically by get_decomp_table()."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class FastVLMDecoderBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = FastVLMAttention(config, layer_idx)
        self.mlp = FastVLMMLP(config)
        self.input_layernorm = FastVLMRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = FastVLMRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids)
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

        # KV cache as mutable state buffers
        head_dim = config.hidden_size // config.num_attention_heads
        kv_dim = head_dim * config.num_key_value_heads
        cache_shape = (config.num_hidden_layers, 1, MAX_SEQ_LEN, kv_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoderStateful":
        """Load from SafeTensors weights directory.

        Decoder weights in Apple's original PyTorch checkpoint live directly at:
          model.layers.*, model.embed_tokens.*, model.norm.*, lm_head.*
        There is NO language_model. prefix — confirmed from discovery output.

        Loading steps:
          1. Load only decoder keys (filter by DECODER_PREFIXES)
          2. Fuse q/k/v -> qkv_proj (_mutate_state_dict, keys still have model. prefix)
          3. Strip model. prefix so keys match PyTorch module hierarchy
          4. load_state_dict with strict=True
        """
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        _mutate_state_dict(weights)
        # Strip model. prefix so keys match the module hierarchy:
        #   model.layers.N.* -> layers.N.*
        #   model.embed_tokens.* -> embed_tokens.*
        #   model.norm.* -> norm.*
        #   lm_head.* -> lm_head.* (no prefix to strip)
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
        model.load_state_dict(weights, assign=True, strict=True)
        return model


# ─── Weight loading helpers ───────────────────────────────────────────────────

# Keys that belong to the decoder in Apple's original PyTorch checkpoint.
# Confirmed from discovery output — no language_model. prefix exists.
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
    """Load only decoder weights from SafeTensors, casting bfloat16 -> dtype.

    Filters to DECODER_PREFIXES only — excludes vision tower and projector keys.
    Weights are stored as bfloat16 in Apple's checkpoint; cast during loading.
    Embeddings are kept as float32 for precision (skip cast).
    """
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    result = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not any(key.startswith(p) for p in _DECODER_PREFIXES):
                    continue
                t = f.get_tensor(key)
                if t.dtype != dtype and "embed_tokens" not in key:
                    t = t.to(dtype)
                result[key] = t
    return result


def _mutate_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """Fuse q_proj, k_proj, v_proj -> qkv_proj in-place.

    Called BEFORE stripping the model. prefix, so keys are still:
      model.layers.N.self_attn.q_proj.weight  (confirmed from discovery output)

    After fusion, replaces those three keys with:
      model.layers.N.self_attn.qkv_proj.weight
      model.layers.N.self_attn.qkv_proj.bias

    The model. prefix is stripped in from_weights() after this call.
    """
    layer_indices = set()
    for k in state_dict:
        if k.startswith("model.layers.") and ".self_attn.q_proj.weight" in k:
            idx = int(k.split(".")[2])
            layer_indices.add(idx)

    for i in sorted(layer_indices):
        weights, biases = [], []
        for proj in ["q_proj", "k_proj", "v_proj"]:
            wk = f"model.layers.{i}.self_attn.{proj}.weight"
            bk = f"model.layers.{i}.self_attn.{proj}.bias"
            if wk not in state_dict:
                break
            weights.append(state_dict.pop(wk))
            biases.append(state_dict.pop(bk))
        else:
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
                weights
            )
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"] = torch.cat(
                biases
            )
