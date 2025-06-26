
def _import_generation_after_fix_deps():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from nanovllm.llm import LLM
    return torch, AutoTokenizer, AutoModelForCausalLM, LLM

def test_model_generations():
    """Compare HuggingFace and nano-vllm generations to verify the fix."""
    print("=== TESTING MODEL GENERATIONS AFTER CAUSAL MASK FIX ===")
    
    model_name = "Qwen/Qwen3-0.6B"
    prompt = "Write a haiku about a cat"
    
    print(f"Model: {model_name}")
    print(f"Prompt: '{prompt}'")
    
    # Load HuggingFace model
    print("\nLoading HuggingFace model...")
    torch, AutoTokenizer, AutoModelForCausalLM, LLM = _import_generation_after_fix_deps()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    
    # Generate with HF model
    print("Generating with HuggingFace model...")
    hf_input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Move input to the same device as model
    hf_input_ids = hf_input_ids.to(hf_model.device)
    hf_outputs = hf_model.generate(
        hf_input_ids,
        max_new_tokens=30,
        do_sample=False
    )
    hf_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
    print(f"HF Generated: '{hf_text}'")
    
    # Load nano-vllm model
    print("\nLoading nano-vllm model...")
    nano_model = LLM(model_name, device="cpu", dtype="float16")
    
    # Generate with nano-vllm
    print("Generating with nano-vllm model...")
    from nanovllm.sampling_params import SamplingParams
    params = SamplingParams(max_tokens=30, temperature=0.0)
    nano_outputs = nano_model.generate([prompt], sampling_params=params)
    generated_text = nano_outputs[0]["text"] 
    nano_text = f"{prompt}{generated_text}"  # Concatenate with f-string
    print(f"nano-vllm Generated: '{nano_text}'")
    
    # Compare outputs
    print("\n=== COMPARISON ===")
    if hf_text.strip() == nano_text.strip():
        print("✅ OUTPUTS MATCH! The fix was successful.")
    else:
        print("❌ OUTPUTS DIFFER. The models are still generating different text.")
        
        # Calculate what percentage of tokens match
        hf_tokens = tokenizer.encode(hf_text)
        nano_tokens = tokenizer.encode(nano_text)
        min_len = min(len(hf_tokens), len(nano_tokens))
        
        matches = sum(1 for i in range(min_len) if hf_tokens[i] == nano_tokens[i])
        match_percentage = (matches / min_len) * 100
        
        print(f"Token match percentage: {match_percentage:.2f}%")
        print("First diverging tokens:")
        
        # Find first position where tokens diverge
        for i in range(min_len):
            if hf_tokens[i] != nano_tokens[i]:
                print(f"Position {i}: HF='{tokenizer.decode([hf_tokens[i]])}', nano='{tokenizer.decode([nano_tokens[i]])}'")
                break

if __name__ == "__main__":
    test_model_generations()
