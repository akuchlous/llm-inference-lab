import gc
import torch
from transformers import AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
SEQ_LEN = 8192
GiB = 1024 ** 3


def report(label):
    print(
        f"{label:<28} "
        f"allocated={torch.cuda.memory_allocated()/GiB:.3f} GiB  "
        f"reserved={torch.cuda.memory_reserved()/GiB:.3f} GiB"
    )


print("Loading model:", MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
).cuda().eval()

report("1. Model loaded")

input_ids = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(1, SEQ_LEN),
    device="cuda",
)

with torch.inference_mode():
    outputs = model.model(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )

past_key_values = outputs.past_key_values

del outputs
gc.collect()
report("2. KV cache alive")

# Delete the live KV tensors, but deliberately DO NOT call empty_cache().
del past_key_values, input_ids
gc.collect()
report("3. KV deleted, no empty_cache")

# Now return unused cached allocator blocks to CUDA.
torch.cuda.empty_cache()
report("4. After empty_cache")
