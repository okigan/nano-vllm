import torch
from torch import nn


class Sampler(nn.Module):
    """Sampler module that handles temperature-based token sampling."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """Sample from logits using temperature-controlled sampling.
        
        Args:
            logits: The raw logits from the model, shape [batch_size, vocab_size]
            temperatures: The temperatures to use, shape [batch_size]
            
        Returns:
            The sampled token IDs, shape [batch_size]
        """
        # Ensure dimensions are correct
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if not torch.is_tensor(temperatures):
            if isinstance(temperatures, (int, float)):
                temperatures = torch.tensor([temperatures], device=logits.device)
            elif isinstance(temperatures, list):
                temperatures = torch.tensor(temperatures, device=logits.device)
        if temperatures.dim() > 1:
            temperatures = temperatures.squeeze()
        if temperatures.shape[0] == 1 and logits.shape[0] > 1:
            temperatures = temperatures.repeat(logits.shape[0])
        is_mps = logits.device.type == 'mps'
        device = logits.device
        greedy_indices = logits.argmax(dim=-1)
        if (temperatures <= 1e-6).all():
            return greedy_indices
        if is_mps:
            logits = logits.float().cpu()
            temperatures = temperatures.float().cpu()
        next_token_ids = [greedy_indices[i].item() if temperatures[i].item() <= 1e-6 else torch.multinomial(torch.softmax((logits[i] / temperatures[i].item()) - (logits[i] / temperatures[i].item()).max(), dim=-1), num_samples=1).item() for i in range(logits.shape[0])]
        return torch.tensor(next_token_ids, device=device)

