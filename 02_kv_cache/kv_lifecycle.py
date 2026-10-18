import gc
import torch
from transformers import AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
SEQ_LEN = 8192
GiB = 1024 ** 3


def show(label):
    torch.cuda.synchronize()
    print(
        f"{label:<24} "
        f"allocated={torch.cuda.memory_allocated()/GiB:6.3f} GiB  "
        f"reserved={torch.cuda.memory_reserved()/GiB:6.3f} GiB"
    )


print("Loading model:", MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
).cuda().eval()

show("1. Model loaded")

input_ids = torch.randint(
    0,
    model.config.vocab_size,
    (1, SEQ_LEN),
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
torch.cuda.empty_cache()
show("2. KV cache alive")

# Delete the request-specific state but keep the model loaded.
del past_key_values, input_ids
gc.collect()
torch.cuda.empty_cache()
show("3. KV cache deleted")

print()
print("Expected pattern:")
print("model only -> model + ~1 GiB KV -> back near model only")
