#!/usr/bin/env python3
"""
Targeted optimizations for nano-vllm based on cProfile analysis.

Key optimizations focus on the main bottlenecks:
1. Use SDPA (Scaled Dot-Product Attention) like HuggingFace  
2. Reduce tensor operations and memory allocations
3. Cache computation results
4. Minimize device/dtype conversions
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def optimize_attention_forward(original_forward):
    """
    Monkey patch the attention forward to use SDPA
    """
    def optimized_forward(
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
        Optimized attention using SDPA instead of manual attention computation
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        batch_size, seq_len = hidden_states.shape[:2]

        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states) 
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary embeddings - ensure types match
        if cos.dtype != dtype:
            cos = cos.to(dtype)
        if sin.dtype != dtype:
            sin = sin.to(dtype)
        
        from nanovllm.layers.rotary_embedding import apply_rotary_emb
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Handle KV cache
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
        
        present = (k, v) if use_cache else None

        # Grouped-query attention
        if self.num_key_value_heads < self.num_attention_heads:
            n_rep = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        # Transpose for SDPA: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's optimized SDPA instead of manual attention
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use SDPA with causal mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=(layer_past is None),  # Only causal for prefill
                scale=None  # Use default scale
            )
        else:
            # Fallback to manual attention if SDPA not available
            scale = 1.0 / (self.head_dim ** 0.5)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            
            if layer_past is None:  # Only apply causal mask for prefill
                q_len, k_len = q.size(-2), k.size(-2)
                causal_mask = torch.triu(torch.ones((q_len, k_len), device=device, dtype=torch.bool), diagonal=1)
                attn_scores.masked_fill_(causal_mask, torch.finfo(dtype).min)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_attention_heads * self.head_dim)

        # Output projection
        output = self.o_proj(attn_output)
        
        return output, present
    
    return optimized_forward


def optimize_mlp_forward(original_forward):
    """
    Optimize MLP forward pass
    """
    def optimized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized MLP forward with fused operations"""
        # Project to gate and up in parallel
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Fused SiLU and multiply
        intermediate = F.silu(gate) * up
        
        # Down projection
        return self.down_proj(intermediate)
    
    return optimized_forward


def apply_optimizations(model_runner):
    """
    Apply optimizations to a ModelRunner instance
    """
    print("Applying performance optimizations...")
    
    # Check if the model has the expected structure
    if not (hasattr(model_runner, 'model') and 
            hasattr(model_runner.model, 'model') and 
            hasattr(model_runner.model.model, 'layers')):
        print("Warning: Unexpected model structure, skipping optimizations")
        return model_runner
    
    optimized_layers = 0
    
    # Optimize each transformer layer
    for i, layer in enumerate(model_runner.model.model.layers):
        # Optimize attention
        if hasattr(layer, 'self_attn'):
            try:
                original_attn_forward = layer.self_attn.forward
                layer.self_attn.forward = optimize_attention_forward(original_attn_forward).__get__(layer.self_attn)
                optimized_layers += 1
            except Exception as e:
                print(f"Warning: Failed to optimize attention in layer {i}: {e}")
        
        # Optimize MLP
        if hasattr(layer, 'mlp'):
            try:
                original_mlp_forward = layer.mlp.forward
                layer.mlp.forward = optimize_mlp_forward(original_mlp_forward).__get__(layer.mlp)
            except Exception as e:
                print(f"Warning: Failed to optimize MLP in layer {i}: {e}")
    
    print(f"Successfully optimized {optimized_layers} layers")
    
    # Add optimized decode method
    original_decode = model_runner._run_decode
    model_runner._run_decode = _optimized_run_decode.__get__(model_runner)
    model_runner._original_run_decode = original_decode
    
    return model_runner


def _optimized_run_decode(self, seqs):
    """
    Optimized decode method with reduced tensor operations
    """
    next_token_ids = []
    
    for seq in seqs:
        seq_id = seq.seq_id
        
        # Check cache
        if seq_id not in self.kv_caches:
            print(f"WARNING: Sequence {seq_id} not found in KV cache")
            self.kv_caches[seq_id] = {
                'position': len(seq.token_ids),
                'tokens': seq.token_ids.copy(),
                'kv_cache': None
            }

        cache_entry = self.kv_caches[seq_id]
        current_tokens = seq.token_ids
        cached_tokens = cache_entry['tokens']
        
        # Check if we can do incremental decode
        if (cache_entry['kv_cache'] is not None and 
            len(current_tokens) == len(cached_tokens) + 1 and
            current_tokens[:-1] == cached_tokens):
            # Incremental decode
            new_token = current_tokens[-1]
            input_ids = torch.tensor([[new_token]], device=self.device, dtype=torch.long)
            position = len(cached_tokens)
            positions = torch.tensor([[position]], device=self.device, dtype=torch.long)
            
            logits, new_kv_cache = self.model.forward_with_cache(
                input_ids=input_ids,
                positions=positions,
                kv_caches=cache_entry['kv_cache'],
                use_cache=True
            )
            
            cache_entry['kv_cache'] = new_kv_cache
            cache_entry['tokens'] = current_tokens.copy()
            cache_entry['position'] = len(current_tokens)
            
            next_token_logits = logits[0, -1, :]
        else:
            # Full prefill
            input_ids = torch.tensor(current_tokens, device=self.device).unsqueeze(0)
            positions = torch.arange(0, len(current_tokens), dtype=torch.long, device=self.device).unsqueeze(0)
            
            logits, new_kv_cache = self.model.forward_with_cache(
                input_ids=input_ids,
                positions=positions,
                kv_caches=None,
                use_cache=True
            )
            
            cache_entry['kv_cache'] = new_kv_cache
            cache_entry['tokens'] = current_tokens.copy()
            cache_entry['position'] = len(current_tokens)
            
            next_token_logits = logits[0, -1, :]
        
        # Simplified repetition penalty
        self._apply_repetition_penalty_mps(next_token_logits, current_tokens)
        
        # Optimized sampling
        temperature = getattr(seq.sampling_params, 'temperature', 1.0)
        
        if temperature <= 1e-6:
            next_token_id = int(torch.argmax(next_token_logits).item())
        else:
            # Use more efficient sampling
            scaled_logits = next_token_logits / temperature
            probs = F.softmax(scaled_logits, dim=0)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())
        
        next_token_ids.append(next_token_id)
        
        # Update cache
        cache_entry['tokens'].append(next_token_id)
        cache_entry['position'] = len(cache_entry['tokens'])
        
    return next_token_ids


def optimize_model_runner(model_runner):
    """
    Apply all optimizations to a ModelRunner
    """
    return apply_optimizations(model_runner)
