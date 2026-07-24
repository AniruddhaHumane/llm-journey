from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

if __name__ == "__main__":
    gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Warmup pass (forces PyTorch/CUDA kernels to initialize)

    vocab_size = 3
    SENTENCE = """Red hat had a small, white, and black stripe on it. It was a small, white, and black stripe on the back of the hat.

The hat was made of a white, black, and white striped fabric. The hat was made of a white, black, and white striped fabric. The hat was made of a white, black, and white striped fabric.

The hat was made of a white, black, and white striped fabric. The hat was made of a white,"""
    tokens = gpt_tokenizer.encode(SENTENCE, return_tensors="pt")
    with torch.no_grad():  # warmup
        _ = gpt_model(tokens)
    start_time = time.perf_counter()
    with torch.inference_mode():
        out = gpt_model(tokens, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdims=True)
    prefill_time = (time.perf_counter() - start_time) * 1000  # ms
    print("--- PREFILL PHASE ---")
    print(f"Prefill time ({tokens.shape[1]} tokens): {prefill_time:.3f} ms\n")

    curr_id = next_token
    curr_cache = past_key_values
    decode_cached_times = []
    for step in range(30):
        start_time = time.perf_counter()
        with torch.inference_mode():
            out = gpt_model(curr_id, past_key_values=curr_cache, use_cache=True)
            curr_cache = out.past_key_values
            curr_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdims=True)
        decode_cached_times.append((time.perf_counter() - start_time) * 1000)

    full_seq = torch.cat([tokens, curr_id], dim=-1)
    decode_uncached_times = []
    for step in range(30):
        start_time = time.perf_counter()
        with torch.inference_mode():
            out = gpt_model(full_seq, use_cache=False)
            new_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdims=True)
        decode_uncached_times.append((time.perf_counter() - start_time) * 1000)
        full_seq = torch.cat([full_seq, new_token], dim=-1)

    avg_cached = sum(decode_cached_times) / len(decode_cached_times)
    avg_nocache = sum(decode_uncached_times) / len(decode_uncached_times)

    print("--- DECODE PHASE (30 Tokens) ---")
    print(f"Avg Decode step WITH Cache : {avg_cached:.3f} ms / token")
    print(f"Avg Decode step NO Cache   : {avg_nocache:.3f} ms / token")
    print(f"Speedup Factor             : {avg_nocache / avg_cached:.2f}x")
