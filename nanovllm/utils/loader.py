import os
from torch import nn
import torch
import glob
from huggingface_hub import snapshot_download


def load_model(model: nn.Module, path: str):
    """
    Load model weights directly from safetensor files without HF overhead.
    
    Args:
        model: The model to load weights into
        path: Path to the directory containing safetensor files
    """
    print(f"[DEBUG] Loading model weights from {path}")
    
    # If path is a HuggingFace model ID, download it
    if not os.path.exists(path):
        print(f"Path {path} does not exist, attempting to download from HuggingFace Hub...")
        path = snapshot_download(repo_id=path, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
    
    # Get the target device from our model
    target_device = next(model.parameters()).device
    print(f"[DEBUG] Target device for model weights: {target_device}")
    
    # Load weights directly from safetensor files - much faster!
    from safetensors import safe_open
    
    safetensor_files = glob.glob(os.path.join(path, "*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensor files found in {path}")
    
    print(f"Found {len(safetensor_files)} safetensor files")
    
    # Get model's state dict for direct weight loading
    model_state_dict = model.state_dict()
    weights_loaded = 0
    
    # Load weights from each safetensor file
    for file_path in safetensor_files:
        # Load on CPU first, then move to target device (safetensors doesn't support MPS)
        load_device = "cpu" if "mps" in str(target_device) else str(target_device)
        with safe_open(file_path, framework="pt", device=load_device) as f:
            for weight_name in f.keys():
                # Use direct key mapping - keys match exactly!
                model_key = weight_name
                
                if model_key in model_state_dict:
                    # Load tensor and move to target device if needed
                    weight_tensor = f.get_tensor(weight_name)
                    if weight_tensor.device != target_device:
                        weight_tensor = weight_tensor.to(target_device)
                    model_param = model_state_dict[model_key]
                    
                    # Check shape compatibility
                    if weight_tensor.shape == model_param.shape:
                        # Direct copy - much faster than the complex mapping
                        model_param.copy_(weight_tensor)
                        weights_loaded += 1
                    else:
                        print(f"Shape mismatch for {model_key}: {weight_tensor.shape} vs {model_param.shape}")
                else:
                    print(f"Key {model_key} not found in model state dict")
    
    print(f"Successfully loaded {weights_loaded} weights from safetensor files")
    
    # Clear cache based on device type
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
