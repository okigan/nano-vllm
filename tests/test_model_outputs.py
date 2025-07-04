

def _import_model_outputs_deps():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    from nanovllm.utils.loader import load_model
    return torch, AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen3ForCausalLM, load_model

def test_model_outputs():
    """Test that nano-vllm and HF model outputs are similar (smoke test, small prompt)."""
    torch, AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen3ForCausalLM, load_model = _import_model_outputs_deps()
    model_name = "Qwen/Qwen3-0.6B"
    prompt = "Hi!"
    print(f"=== MODEL OUTPUT TEST FOR {model_name} ===\n")

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    hf_model.eval()

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    nano_model = Qwen3ForCausalLM(config)
    load_model(nano_model, model_name)
    nano_model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs: {input_ids[0].tolist()}")

    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)

    print("\n=== FORWARD PASS COMPARISON ===")
    with torch.no_grad():
        hf_output = hf_model(input_ids).logits
        print(f"HF output shape: {hf_output.shape}")
        print(f"HF logits norm: {torch.norm(hf_output).item():.6f}")
        nano_output = nano_model(input_ids, positions)
        print(f"nano output shape: {nano_output.shape}")
        print(f"nano logits norm: {torch.norm(nano_output).item():.6f}")
        logits_diff = torch.abs(hf_output - nano_output).mean().item()
        print(f"Mean absolute difference in logits: {logits_diff:.6f}")
        hf_next_token_logits = hf_output[0, -1, :]
        nano_next_token_logits = nano_output[0, -1, :]
        hf_top_k = torch.topk(hf_next_token_logits, 5)
        nano_top_k = torch.topk(nano_next_token_logits, 5)
        print("\nHF Top 5 Next Token Predictions:")
        for i, (token_id, score) in enumerate(zip(hf_top_k.indices.tolist(), hf_top_k.values.tolist())):
            token_text = tokenizer.decode([token_id])
            print(f"  {i+1}. Token {token_id} ('{token_text}'): {score:.4f}")
        print("\nnano Top 5 Next Token Predictions:")
        for i, (token_id, score) in enumerate(zip(nano_top_k.indices.tolist(), nano_top_k.values.tolist())):
            token_text = tokenizer.decode([token_id])
            print(f"  {i+1}. Token {token_id} ('{token_text}'): {score:.4f}")
        hf_set = set(hf_top_k.indices.tolist())
        nano_set = set(nano_top_k.indices.tolist())
        overlap = hf_set.intersection(nano_set)
        print(f"\nOverlap in top predictions: {len(overlap)}/{len(hf_set)}")
        print(f"Common tokens: {overlap}")

    print("\n=== GENERATION TEST ===")
    with torch.no_grad():
        hf_outputs = hf_model.generate(
            input_ids, 
            max_new_tokens=2,  # keep short for test
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        hf_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
    print(f"HF Generated: {repr(hf_text)}")

    print("\nGenerating with nano-vllm model...")
    curr_ids = input_ids[0].tolist()
    with torch.no_grad():
        for i in range(2):  # keep short for test
            input_tensor = torch.tensor(curr_ids).unsqueeze(0)
            positions = torch.arange(len(curr_ids)).unsqueeze(0)
            logits = nano_model(input_tensor, positions)
            next_token_id = logits[0, -1, :].argmax(dim=-1).item()
            curr_ids.append(next_token_id)
            token_text = tokenizer.decode([next_token_id])
            print(f"Token {i+1}: {next_token_id} -> {repr(token_text)}")
    nano_text = tokenizer.decode(curr_ids, skip_special_tokens=True)
    print(f"\nnano-vllm Generated: {repr(nano_text)}")
    print("\nGeneration match:", "✅" if hf_text == nano_text else "❌")


if __name__ == "__main__":
    test_model_outputs()
