import torch
from transformers import AutoModelForCausalLM

GiB = 1024 ** 3
MODEL = "mistralai/Mistral-7B-v0.1"


def memory(label):
    free, total = torch.cuda.mem_get_info()
    print(f"\n=== {label} ===")
    print(f"Free GPU:          {free / GiB:.3f} GiB")
    print(f"PyTorch allocated: {torch.cuda.memory_allocated() / GiB:.3f} GiB")
    print(f"PyTorch reserved:  {torch.cuda.memory_reserved() / GiB:.3f} GiB")
    print(f"Peak allocated:    {torch.cuda.max_memory_allocated() / GiB:.3f} GiB")


memory("BEFORE MODEL")
print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
)
model = model.cuda()

memory("AFTER MODEL")

num_params = sum(p.numel() for p in model.parameters())
print("\n=== MODEL PARAMETERS ===")
print(f"Parameters: {num_params:,}")
print(f"Parameters: {num_params / 1e9:.3f} billion")

parameter_bytes = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)
print(f"Parameter memory: {parameter_bytes / GiB:.3f} GiB")

input("\nPress ENTER to exit...")
