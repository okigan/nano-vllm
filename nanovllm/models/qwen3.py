import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from typing import Optional

from ..layers.attention import SelfAttention
from ..layers.activation import SiluAndMul
from ..layers.linear import ColumnParallelLinear, RowParallelLinear
from ..layers.layernorm import RMSNorm
from ..layers.embed_head import VocabParallelEmbedding


class Qwen3MLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the MLP exactly matching HuggingFace.
        Steps:
        1. Apply gate and up projections
        2. Apply SiLU activation and multiply (gating mechanism)
        3. Apply down projection
        Handles dtype compatibility for all linear ops.
        """
        # Save original dtype
        orig_dtype = x.dtype

        # Use the dtype of the weights for all linear ops
        linear_dtype = self.gate_proj.weight.dtype
        x_proj = x.to(linear_dtype)

        # [batch_size, seq_len, hidden_dim]
        gate = self.gate_proj(x_proj)
        up = self.up_proj(x_proj)

        # SiLU activation and multiply (in linear_dtype)
        x_act = self.act(gate, up)

        # Down projection (in linear_dtype)
        out = self.down_proj(x_act)

        # Return to original dtype
        out = out.to(orig_dtype)
        return out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Update attention parameters to match HuggingFace model
        self.self_attn = SelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            bias=not getattr(config, "no_bias", True),
            rotary_base=getattr(config, "rope_theta", 10000.0),
        )
        
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the decoder layer, exactly matching HuggingFace implementation.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            cos: Cosine part of rotary embeddings [seq_len, 1, head_dim]
            sin: Sine part of rotary embeddings [seq_len, 1, head_dim]
            
        Returns:
            Updated hidden states [batch_size, seq_len, hidden_size]
        """
        # Save original input for residual connections
        residual = hidden_states
        
        # Apply input layer normalization first (pre-attention norm)
        # This follows the RMSNorm -> attention -> add residual pattern in Qwen3
        hidden_states = self.input_layernorm(hidden_states)
        
        # Get sequence length for attention mask
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Create parameters for attention with cache support
        kv_cache_params = {
            'cu_seqlens': torch.tensor([0, seq_len], device=hidden_states.device),
            'max_s': seq_len,
            'layer_past': layer_past,
            'use_cache': use_cache
        }
        
        # Apply self-attention - the attention module includes:
        # 1. Q/K/V projections
        # 2. Q/K normalization
        # 3. Rotary embeddings
        # 4. Attention computation
        # 5. Output projection
        attn_output, present = self.self_attn(
            hidden_states=hidden_states, 
            cos=cos, 
            sin=sin,
            **kv_cache_params
        )
        
        # Add residual connection after attention
        hidden_states = residual + attn_output
        
        # Save for residual connection after MLP
        residual = hidden_states
        
        # Apply post-attention layer normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP (feed-forward network)
        # MLP includes gate_proj, up_proj, SiLU activation, and down_proj
        mlp_output = self.mlp(hidden_states)
        
        # Add residual connection after MLP
        hidden_states = residual + mlp_output
        
        return hidden_states, present


class Qwen3Model(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_rotary_embedding(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation of rotary embeddings to exactly match HuggingFace Qwen3.
        
        This implementation is based on examining the HF Qwen3RotaryEmbedding class
        and making sure our outputs match exactly.
        
        Args:
            positions: Position tensor [batch_size, seq_len] or [seq_len]
            
        Returns:
            tuple of (cos, sin) tensors for rotary embeddings
        """
        device = positions.device
        # Get parameters from config as HF does
        base = getattr(self.config, "rope_theta", 10000.0)
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        attention_scaling = 1.0  # Default as in HF implementation
        
        # HF computes inv_freq like this (always in float)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        
        # Ensure positions are in the right format
        if positions.dim() <= 1:
            # Handle a sequence-only tensor (create batch dim)
            position_ids_expanded = positions.unsqueeze(0).unsqueeze(1).float()
        else:
            # Handle batch of positions
            position_ids_expanded = positions.unsqueeze(1).float()
            
        # Compute freqs
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids_expanded.shape[0], -1, 1)
        
        # Force float32 for precision
        with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=False):
            # This is exactly how HF computes it
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        
        return cos, sin

    def forward(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        # Get embeddings from input ids
        hidden_states = self.embed_tokens(input_ids)
        
        # Get rotary embeddings - critically important to generate these correctly
        cos, sin = self.get_rotary_embedding(positions)
        
        # Initialize cache list if using cache
        new_kv_caches = [] if use_cache else None
        
        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            # Get layer cache if available
            layer_past = kv_caches[i] if kv_caches is not None else None
            
            # Apply each transformer layer with the same rotary embeddings
            # This matches HuggingFace's implementation where the same
            # rotary embeddings are used for each layer
            hidden_states, present = layer(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                layer_past=layer_past,
                use_cache=use_cache
            )
            
            # Store new cache if using cache
            if use_cache and new_kv_caches is not None:
                new_kv_caches.append(present)
            
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, new_kv_caches


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model with a language modeling head.
    
    This follows the HuggingFace implementation but simplified for inference.
    """
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # Initialize the base model
        self.model = Qwen3Model(config)
        
        # Initialize the language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        
        # Tie weights between input embeddings and output embedding
        # Note: In HF models this is done with tie_weights() after initialization
        self.lm_head.weight = self.model.embed_tokens.weight
        
        # Set the model to evaluation mode by default for inference
        self.eval()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model, exactly matching HuggingFace's implementation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len] or [seq_len] 
            positions: Position IDs [batch_size, seq_len] or [seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        # Ensure input_ids has batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
            
        # Ensure positions has batch dimension and matches input_ids shape
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # [1, seq_len]
            
        # Ensure positions has batch dimension
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # [1, seq_len]
        
        # Get hidden states from the model (no cache support here for compatibility)
        hidden_states, _ = self.model(input_ids, positions, kv_caches=None, use_cache=False)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def forward_with_cache(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass with KV cache support for efficient incremental decoding.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len] or [seq_len] 
            positions: Position IDs [batch_size, seq_len] or [seq_len]
            kv_caches: Previous KV caches for incremental decoding
            use_cache: Whether to return and use KV caches
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            new_kv_caches: Updated KV caches for next iteration
        """
        # Ensure input_ids has batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
            
        # Ensure positions has batch dimension and matches input_ids shape
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # [1, seq_len]
        
        # Get hidden states from the model with cache support
        hidden_states, new_kv_caches = self.model(input_ids, positions, kv_caches=kv_caches, use_cache=use_cache)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits, new_kv_caches