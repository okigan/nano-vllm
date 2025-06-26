import os
import time
import traceback
from random import randint, seed
from nanovllm import LLM, SamplingParams

def main():
    try:
        seed(0)
        # Benchmark settings for large batch
        num_seqs = 32  # Reduced batch size for stability
        min_input_len = 100
        max_input_len = 1024
        min_output_len = 100
        max_output_len = 1024

        # Local path or Hugging Face path
        path = os.path.expanduser("Qwen/Qwen3-0.6B")
        print(f"Loading model from {path}...")
        llm = LLM(path, enforce_eager=False, max_model_len=4096)
        print("Model loaded successfully")

        print("Model: Qwen3-0.6B")
        print(f"Total Requests: {num_seqs} sequences")
        print(f"Input Length: Randomly sampled between {min_input_len}–{max_input_len} tokens")
        print(f"Output Length: Randomly sampled between {min_output_len}–{max_output_len} tokens")
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(min_input_len, max_input_len))]
            for _ in range(num_seqs)
        ]
        sampling_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(min_output_len, max_output_len))
            for _ in range(num_seqs)
        ]

        print("Running warmup...")
        result = llm.generate(["Testing warmup: "], SamplingParams())
        print(f"Warmup output: {result[0]}")
        
        print("Starting benchmark run...")
        t = time.time()
        results = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        elapsed = time.time() - t
        
        print("Generation completed successfully")
        for i, output in enumerate(results):
            print(f"Output {i}: length={len(output)}")
        
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / elapsed
        print(f"Total: {total_tokens} tokens, Time: {elapsed:.2f}s, Throughput: {throughput:.2f} tokens/s")
        
        print("Benchmark completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
