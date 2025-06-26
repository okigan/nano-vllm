import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .rotary_embedding import RotaryEmbedding
from .linear import ColumnParallelLinear
from .layernorm import RMSNorm
from nanovllm.utils.torch_compile_utils import optional_torch_compile


class SelfAttention(nn.Module):
    """Multi-head self attention with QKV projection and Q/K normalization"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 2048,
        bias: bool = True,
        rotary_base: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        
        # QKV projections
        q_proj_size = num_attention_heads * head_dim
        k_proj_size = num_key_value_heads * head_dim
        v_proj_size = num_key_value_heads * head_dim
        
        # QKV projections
        self.q_proj = ColumnParallelLinear(hidden_size, q_proj_size, bias=bias)
        self.k_proj = ColumnParallelLinear(hidden_size, k_proj_size, bias=bias)
        self.v_proj = ColumnParallelLinear(hidden_size, v_proj_size, bias=bias)
        
        # Output projection
        self.o_proj = ColumnParallelLinear(q_proj_size, hidden_size, bias=bias)
        
        # Q/K normalization layers (Qwen3 specific)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            rotary_dim=None,  # Let it default to head_dim
            max_position_embeddings=max_position_embeddings,
            base=rotary_base
        )

    @optional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Optimized self-attention forward pass
        """
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        # 1. QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # 3. Apply RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Rotary embeddings
        q = self.rotary_emb(q, cos, sin)
        k = self.rotary_emb(k, cos, sin)

        # 5. KV cache handling
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
        
        present = (k, v) if use_cache else None

        # 6. Grouped-query attention
        if self.num_key_value_heads < self.num_attention_heads:
            n_rep = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        # 7. Transpose for batched matmul: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 8. Attention computation using SDPA
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Use PyTorch's optimized SDPA for better performance
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(layer_past is None),
            scale=scale
        )

        # 9. Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim)
        
        output = self.o_proj(attn_output)
        return output, present
