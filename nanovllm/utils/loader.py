
import os
from torch import nn
import torch
import re
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
# Patch for Qwen2/transformers NoneType error (see https://github.com/huggingface/transformers/issues/1239)
try:
    from transformers import modeling_utils
    if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
        modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
except Exception as e:
    print(f"[DEBUG] Could not patch ALL_PARALLEL_STYLES: {e}")


def load_model(model: nn.Module, path: str):
    """
    Load model weights from safetensor files.
    
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
    
    # Determine the appropriate dtype based on device
    load_dtype = torch.float32 if target_device.type == 'mps' else torch.float16
    print(f"[DEBUG] Using {load_dtype} for weight loading")
    
    # Use transformers to load the model first as reference
    print(f"Using transformers to load reference model from {path} with dtype {load_dtype}")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=load_dtype, trust_remote_code=True, config=config)
    # Patch config after instantiation to avoid NoneType errors in HF Qwen2 model
    for attr in ("tensor_parallel_style", "parallel_style"):
        if not hasattr(ref_model.config, attr) or getattr(ref_model.config, attr) is None:
            setattr(ref_model.config, attr, "none")
    
    # Confirm the reference model loaded correctly
    print(f"[DEBUG] Reference model loaded. Class: {type(ref_model).__name__}")
    print(f"[DEBUG] Reference model config: {ref_model.config}")
    
    # Verify the reference model has expected weights
    ref_state_dict = ref_model.state_dict()
    print(f"[DEBUG] Reference model has {len(ref_state_dict)} keys")
    
    # Sample a few reference model weights to verify they're loaded correctly
    sample_keys = list(ref_state_dict.keys())[:3]
    for key in sample_keys:
        tensor = ref_state_dict[key]
        print(f"[DEBUG] Ref model tensor '{key}': shape={tensor.shape}, dtype={tensor.dtype}, "
              f"mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
    
    # Get the model's state dict to access tensors directly
    state_dict = model.state_dict()
    print(f"[DEBUG] Target model has {len(state_dict)} keys")
    
    # Sample a few target model weights before loading
    sample_keys = list(state_dict.keys())[:3]
    for key in sample_keys:
        tensor = state_dict[key]
        print(f"[DEBUG] Target model tensor '{key}' BEFORE loading: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
    
    # Create mapping from reference model to our model
    # This handles key name differences between HF model and nano-vllm model
    key_mapping = {}
    unmatched_keys = []
    
    # Add specific attention mappings that need special handling for Qwen3
    attention_mappings = {
        # Critical Q/K/V/O mappings
        'q_proj': 'q_proj',  # Direct match
        'k_proj': 'k_proj',  # Direct match
        'v_proj': 'v_proj',  # Direct match
        'o_proj': 'o_proj',  # Direct match
        'self_attn.q_proj': 'self_attn.q_proj',  # Direct match
        'self_attn.k_proj': 'self_attn.k_proj',  # Direct match
        'self_attn.v_proj': 'self_attn.v_proj',  # Direct match
        'self_attn.o_proj': 'self_attn.o_proj',  # Direct match
    }
    
    # First, try direct matches
    for k_ref in ref_state_dict.keys():
        if k_ref in state_dict:
            key_mapping[k_ref] = k_ref
        else:
            unmatched_keys.append(k_ref)
    
    # Try to match remaining keys with simple transformations
    for k_ref in list(unmatched_keys):  # Use a copy for iteration safety
        # Try without 'model.' prefix
        if k_ref.startswith('model.'):
            k_without_prefix = k_ref[6:]  # Remove 'model.' prefix
            if k_without_prefix in state_dict:
                key_mapping[k_ref] = k_without_prefix
                unmatched_keys.remove(k_ref)
                continue
        
        # Apply attention-specific key mappings for query, key, value, output projections
        for attn_key, attn_val in attention_mappings.items():
            if attn_key in k_ref:
                # Try to find matching key in the state dict
                potential_key = k_ref.replace(attn_key, attn_val)
                if potential_key in state_dict:
                    key_mapping[k_ref] = potential_key
                    if k_ref in unmatched_keys:
                        unmatched_keys.remove(k_ref)
                    continue
                
                # Try without model prefix
                if k_ref.startswith('model.'):
                    potential_key = k_ref[6:].replace(attn_key, attn_val)
                    if potential_key in state_dict:
                        key_mapping[k_ref] = potential_key
                        if k_ref in unmatched_keys:
                            unmatched_keys.remove(k_ref)
                        continue
        
        # Try with pattern matching for layer names
        for k_model in state_dict.keys():
            # Match transformer blocks
            if ('layers' in k_ref and 'layers' in k_model and 
                k_ref.split('.')[-1] == k_model.split('.')[-1]):  # Same parameter type
                ref_match = re.search(r'layers\.(\d+)', k_ref)
                model_match = re.search(r'layers\.(\d+)', k_model)
                if ref_match and model_match:
                    # Same layer number
                    ref_layer = ref_match.group(1)
                    model_layer = model_match.group(1)
                    if ref_layer == model_layer:
                        key_mapping[k_ref] = k_model
                        if k_ref in unmatched_keys:
                            unmatched_keys.remove(k_ref)
                        continue
    
    # Check for keys in state_dict that don't have a mapping
    missing_mappings = [k for k in state_dict.keys() if not any(v == k for v in key_mapping.values())]
    if missing_mappings:
        print(f"WARNING: {len(missing_mappings)} keys in model don't have a mapping from reference model.")
        print(f"First few missing keys: {missing_mappings[:10]}")

    # Print all reference keys that were not matched
    if unmatched_keys:
        print(f"WARNING: {len(unmatched_keys)} reference keys were not matched to model keys.")
        print(f"First few unmatched reference keys: {unmatched_keys[:10]}")
    
    # Copy weights from reference state dict
    weights_copied = 0
    shape_mismatches = 0
    
    print("Transferring weights from reference model to nano-vllm model")
    for k_ref, k_model in key_mapping.items():
        ref_tensor = ref_state_dict[k_ref]
        model_tensor = state_dict[k_model]
        # Handle shape mismatches
        if ref_tensor.shape != model_tensor.shape:
            print(f"SHAPE MISMATCH: {k_ref} -> {k_model}: {ref_tensor.shape} vs {model_tensor.shape}")
            shape_mismatches += 1
            # Try to copy if embedding or lm_head with vocab mismatch
            if "embed_tokens" in k_model or "lm_head" in k_model:
                if ref_tensor.shape[0] >= model_tensor.shape[0]:  # Can truncate
                    try:
                        # Slice in dimension 0 (vocab dimension) and ensure correct device
                        device = model_tensor.device
                        truncated_tensor = ref_tensor[:model_tensor.shape[0], ...].to(device=device, dtype=model_tensor.dtype)
                        # Debug check for NaNs
                        if torch.isnan(truncated_tensor).any():
                            print(f"WARNING: NaN values found in truncated tensor for {k_model}")
                        model_tensor.copy_(truncated_tensor)
                        weights_copied += 1
                        continue
                    except Exception as e:
                        print(f"Failed to copy truncated weight: {e}")
            # Print a warning for all other shape mismatches
            else:
                print(f"WARNING: Skipping weight for {k_model} due to shape mismatch.")
        else:
            # Same shape, copy directly - ensure correct device and dtype
            device = model_tensor.device
            converted_tensor = ref_tensor.to(device=device, dtype=model_tensor.dtype)
            # Debug check for NaNs
            if torch.isnan(converted_tensor).any():
                print(f"WARNING: NaN values found in converted tensor for {k_model}")
            model_tensor.copy_(converted_tensor)
            weights_copied += 1
    
    print(f"Successfully transferred {weights_copied}/{len(key_mapping)} weights with {shape_mismatches} shape mismatches")
    
    # Verify a few model weights after loading
    for key in sample_keys:
        if key in state_dict:
            tensor = state_dict[key]
            print(f"[DEBUG] Target model tensor '{key}' AFTER loading: shape={tensor.shape}, dtype={tensor.dtype}, "
                f"mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
    
    # Check for nan/inf values
    nan_params = []
    inf_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
    
    if nan_params:
        print(f"WARNING: NaN values found in {len(nan_params)} parameters: {nan_params[:5]}")
    if inf_params:
        print(f"WARNING: Inf values found in {len(inf_params)} parameters: {inf_params[:5]}")
    
    # Clean up reference model to free memory
    del ref_model
    del ref_state_dict
    
    # Clean up cache based on device type
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def find_safetensors_files(model_name: str):
    pass
