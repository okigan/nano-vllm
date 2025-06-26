"""
Optimized attention layer with efficient KV cache implementation.

This fixes the major performance bottleneck where KV cache concatenation
was happening on every decode step, causing O(n²) complexity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from nanovllm.layers.rotary_embedding import RotaryEmbedding


class OptimizedGroupedQueryAttention(nn.Module):
    """
    Optimized Grouped-Query Attention with efficient KV cache.
    
    Key optimizations:
    1. Pre-allocated KV cache tensors to avoid concatenations
    2. In-place cache updates where possible
    3. Optimized memory layout for MPS
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias)
        
        # Layer norms for queries and keys
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        # Cache management
        self.max_cache_size = max_position_embeddings
        self._cached_k: Optional[torch.Tensor] = None
        self._cached_v: Optional[torch.Tensor] = None
        self._cache_seq_len = 0
        
    def _init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize pre-allocated cache tensors"""
        # [batch_size, max_seq_len, num_kv_heads, head_dim]
        self._cached_k = torch.zeros(
            batch_size, self.max_cache_size, self.num_key_value_heads, self.head_dim,
            device=device, dtype=dtype
        )
        self._cached_v = torch.zeros(
            batch_size, self.max_cache_size, self.num_key_value_heads, self.head_dim,
            device=device, dtype=dtype
        )
        self._cache_seq_len = 0
        
    def _update_cache(self, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Update cache in-place with new key/value tensors"""
        batch_size, seq_len = k.shape[0], k.shape[1]
        
        # Ensure cache is initialized
        if self._cached_k is None or self._cached_k.shape[0] != batch_size:
            self._init_cache(batch_size, k.device, k.dtype)
            
        # Check bounds
        end_pos = start_pos + seq_len
        if end_pos > self.max_cache_size:
            raise ValueError(f"Sequence length {end_pos} exceeds max cache size {self.max_cache_size}")
        
        # Update cache in-place (cache is guaranteed to be initialized at this point)
        assert self._cached_k is not None and self._cached_v is not None
        self._cached_k[:, start_pos:end_pos] = k
        self._cached_v[:, start_pos:end_pos] = v
        self._cache_seq_len = max(self._cache_seq_len, end_pos)
        
        # Return the relevant portion of the cache
        return (
            self._cached_k[:, :end_pos].clone(),
            self._cached_v[:, :end_pos].clone()
        )
        
    def clear_cache(self):
        """Clear the KV cache"""
        self._cached_k = None
        self._cached_v = None
        self._cache_seq_len = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Optimized forward pass with efficient KV cache handling.
        """
        device = hidden_states.device
        
        # [batch_size, seq_len, hidden_dim]
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        # 1. Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # 3. Apply layer norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Apply rotary embeddings
        q = self.rotary_emb(q, cos, sin)
        k = self.rotary_emb(k, cos, sin)

        # 5. Efficient KV cache handling
        if use_cache:
            if layer_past is not None:
                # Incremental decoding: use efficient cache update
                past_k, past_v = layer_past
                start_pos = past_k.shape[1]  # Where to insert new tokens
                full_k, full_v = self._update_cache(k, v, start_pos)
            else:
                # Prefill: initialize cache
                self.clear_cache()
                full_k, full_v = self._update_cache(k, v, 0)
                
            present = (full_k, full_v)
        else:
            # No cache
            if layer_past is not None:
                past_k, past_v = layer_past
                # Fall back to concatenation for compatibility
                full_k = torch.cat([past_k, k], dim=1)
                full_v = torch.cat([past_v, v], dim=1)
            else:
                full_k, full_v = k, v
            present = (full_k, full_v) if use_cache else None

        # 6. Handle grouped-query attention
        if self.num_key_value_heads < self.num_attention_heads:
            n_rep = self.num_attention_heads // self.num_key_value_heads
            full_k = full_k.repeat_interleave(n_rep, dim=2)
            full_v = full_v.repeat_interleave(n_rep, dim=2)

        # 7. Transpose for batched matmul: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        full_k = full_k.transpose(1, 2)
        full_v = full_v.transpose(1, 2)

        # 8. Attention computation
        attn_scores = torch.matmul(q, full_k.transpose(-1, -2)) * self.scale

        # 9. Causal masking
        if layer_past is None:
            # Full attention: apply causal mask
            q_len, k_len = q.size(-2), full_k.size(-2)
            mask = torch.triu(torch.ones((q_len, k_len), device=device, dtype=torch.bool), diagonal=1)
            attn_scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn_scores.dtype).min)

        # 10. Softmax and attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, full_v)

        # 11. Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim)
        
        output = self.o_proj(attn_output)
        return output, present
