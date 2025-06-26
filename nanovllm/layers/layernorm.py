
import torch
from torch import nn
from nanovllm.utils.torch_compile_utils import optional_torch_compile


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @optional_torch_compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """RMSNorm implementation exactly matching HuggingFace Qwen3 RMSNorm.
        
        HF Qwen3 implementation:
        ```
        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
        ```
        """
        # Save original dtype to restore later
        input_dtype = x.dtype
        
        # Compute variance in float32, exactly as HF does
        if x.dtype != torch.float32:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
        
        # Apply normalization - convert to float32, normalize, then back to input dtype
        x = x * torch.rsqrt(variance + self.eps)
        
        # Apply weight and return to original dtype, exactly as HF does
        if x.dtype != input_dtype:
            return self.weight * x.to(input_dtype)
        else:
            return self.weight * x

    @optional_torch_compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm with residual connection, exactly matching HuggingFace.
        
        This is used when we want to add a residual connection and then normalize.
        """
        orig_dtype = x.dtype
        
        # Ensure all tensors are float32 for precise computation
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if residual.dtype != torch.float32:
            residual = residual.to(torch.float32)
        if self.weight.dtype != torch.float32:
            weight = self.weight.to(torch.float32)
        else:
            weight = self.weight
        
        # Ensure residual matches x's shape by slicing to the same batch size
        if residual.shape[0] != x.shape[0]:
            residual = residual[:x.shape[0]]
        
        # Add residual (use non-inplace addition)
        x = x + residual
        
        # Save for return
        residual_out = x.clone()
        
        # Compute variance
        var = x.pow(2).mean(dim=-1, keepdim=True)
        
        # Apply normalization
        x = x * torch.rsqrt(var + self.eps)
        
        # Apply weight
        x = x * weight
        
        # Return to original dtype
        if x.dtype != orig_dtype:
            x = x.to(orig_dtype)
        if residual_out.dtype != orig_dtype:
            residual_out = residual_out.to(orig_dtype)
        
        return x, residual_out

    @optional_torch_compile
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
