import gc
import time
import torch
from transformers import AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
BATCH_SIZES = [1, 2, 4, 8]
PROMPT_TOKENS = 1024
NEW_TOKENS = 64
GiB = 1024 ** 3

print("Loading model:", MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
).cuda().eval()

model_memory = torch.cuda.memory_allocated()
print(f"Model allocated memory: {model_memory / GiB:.3f} GiB")
print()

print(
    f"{'Batch':>5} | {'Elapsed(s)':>10} | {'Output tok':>10} | "
    f"{'Throughput tok/s':>16} | {'Peak above model':>16}"
)
print("-" * 72)

for batch_size in BATCH_SIZES:
    gc.collect()
    torch.cuda.empty_cache()

    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(batch_size, PROMPT_TOKENS),
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=model.config.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = batch_size * (output_ids.shape[1] - input_ids.shape[1])
    throughput = generated_tokens / elapsed
    peak_above_model = torch.cuda.max_memory_allocated() - model_memory

    print(
        f"{batch_size:5d} | "
        f"{elapsed:10.2f} | "
        f"{generated_tokens:10d} | "
        f"{throughput:16.2f} | "
        f"{peak_above_model / GiB:13.3f} GiB"
    )

    del input_ids, attention_mask, output_ids
    gc.collect()
    torch.cuda.empty_cache()
