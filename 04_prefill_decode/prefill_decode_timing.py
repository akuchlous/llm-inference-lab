import gc
import time
import torch
from transformers import AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CONTEXT_LENGTHS = [128, 512, 1024, 2048, 4096]
DECODE_STEPS = 32

print("Loading model:", MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
).cuda().eval()

print()
print(
    f"{'Context':>7} | {'Prefill ms':>10} | {'Prefill tok/s':>14} | "
    f"{'Decode ms/tok':>13} | {'Decode tok/s':>12}"
)
print("-" * 72)

# Warm up CUDA/model kernels once before measuring.
warmup_ids = torch.randint(
    0, model.config.vocab_size, (1, 32), device="cuda"
)
with torch.inference_mode():
    _ = model.model(input_ids=warmup_ids, use_cache=True, return_dict=True)
torch.cuda.synchronize()
del warmup_ids, _
torch.cuda.empty_cache()

for context_len in CONTEXT_LENGTHS:
    gc.collect()
    torch.cuda.empty_cache()

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, context_len),
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        # PREFILL: process the whole prompt and build KV cache.
        torch.cuda.synchronize()
        start = time.perf_counter()

        out = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - start

        past_key_values = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:, :]
        logits = model.lm_head(last_hidden)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        del out, last_hidden, logits

        # DECODE: feed one new token at a time using the existing KV cache.
        decode_times = []
        for _ in range(DECODE_STEPS):
            torch.cuda.synchronize()
            step_start = time.perf_counter()

            out = model.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            last_hidden = out.last_hidden_state[:, -1:, :]
            logits = model.lm_head(last_hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - step_start)

            del out, last_hidden, logits

    prefill_ms = prefill_s * 1000
    prefill_tps = context_len / prefill_s
    decode_ms = (sum(decode_times) / len(decode_times)) * 1000
    decode_tps = 1000 / decode_ms

    print(
        f"{context_len:7d} | {prefill_ms:10.2f} | {prefill_tps:14.2f} | "
        f"{decode_ms:13.2f} | {decode_tps:12.2f}"
    )

    del input_ids, attention_mask, past_key_values, next_token
    gc.collect()
    torch.cuda.empty_cache()
