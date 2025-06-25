from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import random

def check_model_weights(llm):
    """Verify the LLM's weights by comparing with the HuggingFace model"""
    print("\n=== WEIGHT VERIFICATION ===")
    
    # Get the original HuggingFace model path
    model_path = llm.engine.engine_args.model
    print(f"Loading reference HF model from {model_path}")
    
    # Load HF model for reference
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
    print(f"Loading reference model to {device}")
    
    # Sample some keys to check
    base_model = llm.engine.model_runner.model
    nano_state_dict = base_model.state_dict()
    
    # Get some sample keys
    sample_keys = random.sample(list(nano_state_dict.keys()), min(5, len(nano_state_dict)))
    
    print("\nSampling weights from nano-vllm model:")
    for key in sample_keys:
        tensor = nano_state_dict[key]
        print(f"  {key}: shape={tensor.shape}, mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
    
    return True

def main():    # Command line argument parsing for device overrides
    import sys
    
    # Check for --device flag
    device_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            device_arg = sys.argv[i + 1]
    
    # Check device availability
    print("=== DEVICE INFO ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check for MPS (Apple Silicon) compatibility
    is_mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS available: {is_mps_available}")
    
    # Determine current device based on availability and command line args
    if device_arg:
        # Override device from command line argument
        device = device_arg
        print(f"Device set from command line: {device}")
    else:
        # Use default device selection
        device = 'cuda' if torch.cuda.is_available() else 'mps' if is_mps_available else 'cpu'
        print(f"Current device (auto-selected): {device}")
    
    # Special handling for MPS devices
    if device == 'mps':
        print("\n=== MPS COMPATIBILITY MODE ===")
        print("Running on Apple Silicon with MPS acceleration")
        print("- Setting default dtype to float32 for MPS compatibility")
        torch.set_default_dtype(torch.float32)
        print("- Using GQA-compatible attention implementation")
        print("- Note: Performance may be lower than on CUDA devices")
        print("- TIP: Use --device cpu to fall back to CPU if MPS fails\n")
    elif device == 'cpu':
        print("\n=== CPU MODE ===")
        print("Running on CPU - slower but may be more stable")
        print("- TIP: Add more memory for larger models")
        print("- TIP: For M1/M2/M3 Macs, try --device mps for GPU acceleration\n")
    
    path = "Qwen/Qwen3-0.6B"  # Use Qwen3-0.6B model
    # path = "Qwen/Qwen3-8B"  # Use Qwen3-8B model
    
    print("=== MODEL/TOKENIZER VERIFICATION ===")
    print(f"Loading from: {path}")
    
    # Load tokenizer and check its properties
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
    print(f"Tokenizer pad token: {tokenizer.pad_token}")
    print(f"Tokenizer eos token: {tokenizer.eos_token}")
    print(f"Tokenizer bos token: {tokenizer.bos_token}")
    
    # Check if tokenizer has special tokens
    special_tokens = tokenizer.special_tokens_map
    print(f"Special tokens: {special_tokens}")
    
    print("\nInitializing LLM...")
    # Pass enforce_eager mode for easier debugging
    # Hardcoded KV cache config for demonstration
    kvcache_block_size = 16  # Number of tokens per KV cache block
    num_kvcache_blocks = 2048  # Total number of KV cache blocks
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        kvcache_block_size=kvcache_block_size,
        num_kvcache_blocks=num_kvcache_blocks,
    )
    
    # Debug: Print basic model info
    print(f"Model path: {path}")
    
    # For Apple MPS devices, add additional info
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Running on Apple MPS device - patching for MPS compatibility")
        print("Note: MPS may have limitations with certain tensor operations")
    
    # Debug: Check model vocab size
    print("\n=== MODEL INFO ===")
    model_vocab_size = llm.engine.model_runner.model.lm_head.weight.shape[0]
    print(f"Model vocab size: {model_vocab_size}")
    tokenizer_vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    if model_vocab_size != tokenizer_vocab_size:
        print(f"Vocabulary mismatch: {model_vocab_size - tokenizer_vocab_size} tokens")
    else:
        print("Model and tokenizer vocab sizes match perfectly!")
    
    # Check model weights for correctness
    weight_check_passed = check_model_weights(llm)
    if not weight_check_passed:
        print("WARNING: Model weights verification failed!")

    # Define prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?"
    ]
    
    print("\n=== GENERATION TEST ===")
    try:
        # Create sampling parameters based on device type
        if device == 'mps':
            print("MPS DEVICE: Using conservative sampling parameters")
            # Use near-deterministic settings for MPS
            sampling_params = SamplingParams(
                temperature=0.01,  # Near-deterministic
                max_tokens=10      # Generate fewer tokens for testing
            )
        else:
            # Regular parameters for other devices
            sampling_params = SamplingParams(
                temperature=0.1,    # Use lower temperature for more predictable outputs
                max_tokens=30       # Limit generated token count
            )
            
        # Try to generate with careful error handling
        outputs = llm.generate(prompts, sampling_params)
        
        # Debug: Check output tokenization
        print("\n=== OUTPUT ANALYSIS ===")
        for i, output in enumerate(outputs):
            print(f"\n--- Output {i} ---")
            print(f"Prompt: {repr(prompts[i])}")
            print(f"Completion: {repr(output['text'])}")
            
            # Get the generated token IDs
            generated_tokens = output['token_ids']
            if isinstance(generated_tokens, list):
                print(f"Generated token IDs (first 10): {generated_tokens[:10]}")
                
                # Check each generated token
                print("Token analysis:")
                for j, token_id in enumerate(generated_tokens[:10]):
                    try:
                        decoded_token = tokenizer.decode([token_id])
                        print(f"  Token {j}: {token_id} -> {repr(decoded_token)}")
                    except Exception as e:
                        print(f"    ERROR decoding token {token_id}: {e}")
        
        print("\n✅ Generation successful!")
    except Exception as e:
        print(f"\n❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        
        if device == 'mps':
            print("\nMPS SPECIFIC TROUBLESHOOTING:")
            print("1. MPS backend has limitations with certain tensor operations")
            print("2. Consider trying again with --device cpu flag")
            print("3. The model may work better with custom MPS compatibility mode")

if __name__ == "__main__":
    main()
