import torch

GiB = 1024 ** 3

print("GPU:", torch.cuda.get_device_name(0))

free, total = torch.cuda.mem_get_info()

print()
print("=== GPU MEMORY BASELINE ===")
print(f"Total GPU memory:  {total / GiB:.3f} GiB")
print(f"Free GPU memory:   {free / GiB:.3f} GiB")
print(f"PyTorch allocated: {torch.cuda.memory_allocated() / GiB:.3f} GiB")
print(f"PyTorch reserved:  {torch.cuda.memory_reserved() / GiB:.3f} GiB")
