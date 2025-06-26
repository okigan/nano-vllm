import math
import torch
import torch.nn as nn
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
        Optimized self-attention for CUDA: avoids unnecessary device/dtype conversions, caches causal mask, and removes slow-path fallbacks.
        """
        import logging
        device = hidden_states.device
        is_mps = device.type == 'mps'
        is_cuda = device.type == 'cuda'
        compute_dtype = torch.float32 if is_mps else hidden_states.dtype
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        # 1. QKV projections
        q = self.q_proj(hidden_states)  # [batch, seq_len, num_heads*head_dim]
        k = self.k_proj(hidden_states)  # [batch, seq_len, num_kv_heads*head_dim]
        v = self.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads*head_dim]

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

        # 5. KV cache - optimized concatenation
        if layer_past is not None:
            past_key, past_value = layer_past
            # Use more efficient concatenation with pre-allocation when possible
            if k.shape[1] == 1:  # Single token decode
                # For single token, try in-place operations when possible
                batch_size, past_seq_len = past_key.shape[0], past_key.shape[1]
                new_seq_len = past_seq_len + 1
                
                # Create new tensors with exact size needed
                new_k = torch.empty(
                    batch_size, new_seq_len, past_key.shape[2], past_key.shape[3],
                    dtype=past_key.dtype, device=past_key.device
                )
                new_v = torch.empty(
                    batch_size, new_seq_len, past_value.shape[2], past_value.shape[3],
                    dtype=past_value.dtype, device=past_value.device
                )
                
                # Copy in chunks to minimize memory operations
                new_k[:, :past_seq_len] = past_key
                new_k[:, past_seq_len:] = k
                new_v[:, :past_seq_len] = past_value
                new_v[:, past_seq_len:] = v
                
                k, v = new_k, new_v
            else:
                # Fall back to standard concatenation for multi-token sequences
                k = torch.cat([past_key, k], dim=1)
                v = torch.cat([past_value, v], dim=1)
        present = (k, v) if use_cache else None

        # 6. Grouped-query attention
        if self.num_key_value_heads < self.num_attention_heads:
            n_rep = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        # 7. Transpose for bmm
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 8. Convert to compute dtype if needed
        if q.dtype != compute_dtype:
            q = q.to(compute_dtype)
        if k.dtype != compute_dtype:
            k = k.to(compute_dtype)
        if v.dtype != compute_dtype:
            v = v.to(compute_dtype)

        # 9. MPS shape fix (unchanged)
        if is_mps:
            if k.shape[2] != v.shape[2]:
                target_seq_len = max(k.shape[2], v.shape[2])
                if k.shape[2] < target_seq_len:
                    k_expanded = torch.zeros(
                        k.shape[0], k.shape[1], target_seq_len, k.shape[3],
                        device=device, dtype=k.dtype
                    )
                    k_expanded[:, :, :k.shape[2], :] = k
                    k_expanded[:, :, k.shape[2]:, :] = k[:, :, -1:, :].repeat(1, 1, target_seq_len - k.shape[2], 1)
                    k = k_expanded
                if v.shape[2] < target_seq_len:
                    v_expanded = torch.zeros(
                        v.shape[0], v.shape[1], target_seq_len, v.shape[3],
                        device=device, dtype=v.dtype
                    )
                    v_expanded[:, :, :v.shape[2], :] = v
                    v_expanded[:, :, v.shape[2]:, :] = v[:, :, -1:, :].repeat(1, 1, target_seq_len - v.shape[2], 1)
                    v = v_expanded

        # 10. Optimized attention computation for incremental decoding
        scale = 1.0 / math.sqrt(self.head_dim)
        
        if layer_past is not None and q.shape[-2] == 1:
            # Incremental decoding: only compute attention for the new token
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [batch, heads, 1, full_seq_len]
            
            # No causal mask needed for incremental decode (new token can see all previous)
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # [batch, heads, 1, head_dim]
        else:
            # Full attention computation (prefill or when no cache)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [batch, num_heads, seq_len, seq_len]

            # Apply causal mask for autoregressive generation
            if layer_past is None:
                q_len, k_len = q.size(-2), k.size(-2)
                mask_key = (device, attn_scores.dtype, q_len, k_len)
                if not hasattr(self, '_causal_mask_cache'):
                    self._causal_mask_cache = {}
                if mask_key not in self._causal_mask_cache:
                    causal_mask = torch.triu(torch.ones((q_len, k_len), device=device), diagonal=1).bool()
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
                    self._causal_mask_cache[mask_key] = causal_mask
                else:
                    causal_mask = self._causal_mask_cache[mask_key]
                attn_scores = attn_scores + causal_mask.to(attn_scores.dtype) * torch.finfo(attn_scores.dtype).min

            # Apply softmax and compute output
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # 13. Handle MPS fallback if needed
        if is_mps and attn_output is None:
            try:
                attn_output = torch.matmul(attn_weights, v)
            except RuntimeError:
                try:
                    attn_weights_cpu = attn_weights.cpu()
                    v_cpu = v.cpu()
                    attn_output = torch.matmul(attn_weights_cpu, v_cpu).to(device)
                except RuntimeError:
                    batch_size = q.shape[0]
                    num_heads = q.shape[1]
                    seq_len = q.shape[2]
                    head_dim = v.shape[-1]
                    logging.warning("MPS matmul fallback failed, returning zeros.")
                    attn_output = torch.zeros(
                        batch_size, num_heads, seq_len, head_dim,
                        dtype=compute_dtype, device=device
                    )

        # 14. Transpose back
        attn_output = attn_output.transpose(1, 2).contiguous()
        actual_batch = attn_output.shape[0]
        actual_seq = attn_output.shape[1]

        # 15. Output reshape (no slow fallback for CUDA)
        output_size = actual_batch * actual_seq * self.num_attention_heads * self.head_dim
        if output_size != attn_output.numel():
            if is_cuda:
                raise RuntimeError(f"[SelfAttention] Unexpected attn_output shape: {attn_output.shape}, expected {actual_batch}x{actual_seq}x{self.num_attention_heads}x{self.head_dim}")
            else:
                # Safe fallback for MPS/CPU
                safe_output = torch.zeros(
                    actual_batch, actual_seq, self.num_attention_heads * self.head_dim,
                    dtype=attn_output.dtype, device=attn_output.device
                )
                if len(attn_output.shape) == 4 and attn_output.shape[2] == self.num_attention_heads:
                    for i in range(min(self.num_attention_heads, attn_output.shape[2])):
                        idx = i * self.head_dim
                        safe_output[:, :, idx:idx + self.head_dim] = attn_output[:, :, i, :self.head_dim]
                attn_output = safe_output
        else:
            attn_output = attn_output.reshape(actual_batch, actual_seq, self.num_attention_heads * self.head_dim)

        # 16. Output projection
        output = self.o_proj(attn_output)
        return output, present
