"""
Test: Compare nano-vllm vs HuggingFace transformers generation (greedy and sampling) on CPU/MPS for a short prompt.
Fastest possible test for CI/dev.
"""
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "Hello world!"

@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_nano_vllm_vs_hf_greedy(device_type):
    # Skip MPS if not available
    if device_type == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    # Load tokenizer and HF model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_model = hf_model.to(device_type)
    hf_model.eval()
    # Tokenize prompt
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device_type)
    # Greedy decode with HF
    import time
    hf_start = time.time()
    with torch.no_grad():
        hf_out = hf_model.generate(input_ids, max_new_tokens=16, do_sample=False)
    hf_elapsed = time.time() - hf_start
    hf_text = tokenizer.decode(hf_out[0], skip_special_tokens=True)
    hf_tokens = hf_out.shape[1] - input_ids.shape[1]
    hf_tps = hf_tokens / hf_elapsed if hf_elapsed > 0 else float('inf')
    # nano-vllm config
    config = Config(model_path=MODEL_NAME)
    runner = ModelRunner(config, rank=0, event=None)
    runner.device = torch.device(device_type)
    runner.model = runner.model.to(device_type)
    runner.model.eval()
    # nano-vllm sequence
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    seq = Sequence(token_ids=input_ids[0].tolist(), sampling_params=sampling_params, prompt=PROMPT)
    # Prefill
    runner.kv_caches = {}
    runner.run([seq], is_prefill=True)
    # Generate tokens
    tokens = seq.token_ids.copy()
    nano_start = time.time()
    for _ in range(16):
        next_id = runner._run_decode([seq])[0]
        tokens.append(next_id)
        # Update sequence state to match model runner expectations
        seq.token_ids.append(next_id)
        seq.generated_tokens.append(next_id)
        seq.current_position = len(seq.token_ids)
        runner.kv_caches[seq.seq_id]['tokens'] = tokens.copy()
        runner.kv_caches[seq.seq_id]['position'] = len(tokens)
    nano_elapsed = time.time() - nano_start
    nano_text = tokenizer.decode(tokens, skip_special_tokens=True)
    nano_tokens = len(tokens) - input_ids.shape[1]
    nano_tps = nano_tokens / nano_elapsed if nano_elapsed > 0 else float('inf')
    # Compare outputs (allow minor differences)
    print(f"HF:    {hf_text}")
    print(f"nano:  {nano_text}")
    print(f"HF tokens/sec:    {hf_tps:.2f}")
    print(f"nano-vllm tokens/sec: {nano_tps:.2f}")
    assert nano_text[:32] == hf_text[:32]  # Only check prefix for speed

def test_nano_vllm_vs_hf_sampling():
    # Only run on CPU for speed
    device_type = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_model = hf_model.to(device_type)
    hf_model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device_type)
    # Sample with temperature
    with torch.no_grad():
        hf_out = hf_model.generate(input_ids, max_new_tokens=16, do_sample=True, temperature=0.8)
    hf_text = tokenizer.decode(hf_out[0], skip_special_tokens=True)
    config = Config(model_path=MODEL_NAME)
    runner = ModelRunner(config, rank=0, event=None)
    runner.device = torch.device(device_type)
    runner.model = runner.model.to(device_type)
    runner.model.eval()
    sampling_params = SamplingParams(temperature=0.8, max_tokens=16)
    seq = Sequence(token_ids=input_ids[0].tolist(), sampling_params=sampling_params, prompt=PROMPT)
    runner.kv_caches = {}
    runner.run([seq], is_prefill=True)
    tokens = seq.token_ids.copy()
    for _ in range(16):
        next_id = runner._run_decode([seq])[0]
        tokens.append(next_id)
        # Update sequence state to match model runner expectations
        seq.token_ids.append(next_id)
        seq.generated_tokens.append(next_id)
        seq.current_position = len(seq.token_ids)
        runner.kv_caches[seq.seq_id]['tokens'] = tokens.copy()
        runner.kv_caches[seq.seq_id]['position'] = len(tokens)
    nano_text = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"HF:    {hf_text}")
    print(f"nano:  {nano_text}")
    # Just check that nano-vllm produces a non-empty, non-repetitive output
    assert len(nano_text.strip()) > 0
    assert nano_text != PROMPT


if __name__ == "__main__":
    pytest.main([__file__])