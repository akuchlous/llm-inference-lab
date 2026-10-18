import gc
import torch
from transformers import AutoModelForCausalLM

GiB = 1024 ** 3
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CONTEXT_LENGTHS = [128, 512, 1024, 2048, 4096, 8192]
OUTPUT_TOKENS = 100


def gib(value):
    return value / GiB


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
)
model = model.cuda()
model.eval()

model_memory = torch.cuda.memory_allocated()
print(f"\nModel memory: {gib(model_memory):.3f} GiB")

for context_length in CONTEXT_LENGTHS:
    gc.collect()
    torch.cuda.empty_cache()

    input_ids = torch.randint(
        low=100,
        high=10000,
        size=(1, context_length),
        dtype=torch.long,
        device="cuda",
    )

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    print(f"\nRunning context length: {context_length:,} tokens")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=OUTPUT_TOKENS,
            do_sample=False,
        )

    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()

    print(f"Before inference: {gib(before):.3f} GiB")
    print(f"Peak memory:      {gib(peak):.3f} GiB")
    print(f"After inference:  {gib(after):.3f} GiB")
    print(f"Peak above model: {gib(peak - model_memory):.3f} GiB")

    del input_ids
    del outputs

print("\nExperiment complete.")
