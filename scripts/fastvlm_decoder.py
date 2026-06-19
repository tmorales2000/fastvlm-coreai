"""
fastvlm_decoder.py — Re-authored FastVLM language decoder for CoreAI export.

The decoder is Qwen2. Reference:
  coreai-models/python/src/coreai_models/models/macos/qwen2.py

Key decisions:
  - RMSNormImpl, SDPA, RoPE are composite ops (externalized via ExternalizeSpec)
  - nn.LayerNorm and nn.BatchNorm2d handled automatically by get_decomp_table()
  - KV cache uses register_buffer + state_names (NOT readonly KV cache pattern)
  - QKV projections fused in _mutate_state_dict before weight loading
  - KV cache writes use coreai::mutable_slice_update (not plain slice-assign)

KV CACHE DESIGN
---------------
k_cache and v_cache have shape (num_hidden_layers, 1, MAX_SEQ_LEN, kv_dim)
where kv_dim = head_dim * num_key_value_heads (flattened head layout).

Each attention layer writes its new K/V vectors into the cache at
offset = seq_len - query_len, then reads back the full context [0:seq_len]
for attention. The write MUST use coreai::mutable_slice_update — not a plain
slice-assign — so that TorchConverter correctly binds the buffer as mutable
state in the exported graph.

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


# ─── KV cache state mutation op ──────────────────────────────────────────────
#
# mutable_slice_update MUST be the coreai custom op — not a plain slice-assign.
# mutates_args=["x"] tells torch.export the buffer is mutated in place, so
# TorchConverter correctly binds it as mutable state via state_names.
# This is a verbatim copy of Apple's _ops.py from coreai_models/primitives/.
#
@torch.library.custom_op("coreai::mutable_slice_update", mutates_args=["x"])
def mutable_slice_update(
    x: torch.Tensor, update: torch.Tensor, begin: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    b = torch.split(begin, 1, dim=0)
    e = torch.split(end, 1, dim=0)
    slices = tuple(slice(bb.item(), ee.item()) for bb, ee in zip(b, e, strict=False))
    x[slices] = update
    return x.clone()


@mutable_slice_update.register_fake
def _mutable_slice_update_fake(x, update, begin, end):
    return x.clone()


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

        self.sdpa = SDPA(is_causal=True)
        self.rope = RoPE(base=float(getattr(config, "rope_theta", 1e6)))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        offset: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim
        kv_dim = n_kv_heads * head_dim
        layer = self.layer_idx

        qkv = (
            self.qkv_proj(x)
            .reshape(B, L, n_heads + 2 * n_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        qk = qkv.narrow(1, 0, n_heads + n_kv_heads)
        v  = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        # RoPE: narrow to the query positions within the full context
        torch._check_is_size(seq_len)
        rope_positions = position_ids.narrow(-1, seq_len - L, L)
        qk = self.rope(qk, position_ids=rope_positions)
        q = qk.narrow(1, 0, n_heads)
        k = qk.narrow(1, n_heads, n_kv_heads)

        # Flatten heads for cache storage: (B, n_kv_heads, L, head_dim) -> (B, L, kv_dim)
        k_flat = k.permute(0, 2, 1, 3).reshape(B, L, kv_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(B, L, kv_dim)

        # Write new K/V into cache at [layer, :, offset:offset+L, :]
        device = k_cache.device
        z = torch.zeros(1, dtype=torch.int32, device=device)
        begin = torch.cat([
            torch.tensor([layer], dtype=torch.int32, device=device),
            z,
            offset.reshape(1).to(torch.int32),
            z,
        ])
        end = torch.cat([
            torch.tensor([layer + 1], dtype=torch.int32, device=device),
            torch.tensor([k_cache.size(1)], dtype=torch.int32, device=device),
            (offset.reshape(1).to(torch.int32) + L),
            torch.tensor([kv_dim], dtype=torch.int32, device=device),
        ])
        mutable_slice_update(k_cache, k_flat.unsqueeze(0), begin, end)
        mutable_slice_update(v_cache, v_flat.unsqueeze(0), begin, end)

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
        offset: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x), position_ids, k_cache, v_cache, offset, seq_len
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
        offset = torch.tensor(seq_len - query_len, dtype=torch.int32)

        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, self.k_cache, self.v_cache, offset, seq_len)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoderStateful":
        """Load from SafeTensors weights directory."""
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        _mutate_state_dict(weights)
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


def _mutate_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """
    Fuse separate q_proj, k_proj, v_proj -> qkv_proj in-place.
    Matches the _mutate_state_dict pattern in coreai-models/qwen2.py.
    Fusion order is [q, k, v] — confirmed by Stage 1 PSNR 113.2 dB vs HF.
    """
    layer_indices = set()
    for k in state_dict:
        if k.startswith("model.layers.") and ".self_attn.q_proj.weight" in k:
            layer_indices.add(int(k.split(".")[2]))

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
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(weights)
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"] = torch.cat(biases)
