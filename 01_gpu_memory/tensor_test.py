import torch

GiB = 1024 ** 3


def memory(label):
    free, total = torch.cuda.mem_get_info()

    print(f"\n=== {label} ===")
    print(f"Free GPU:           {free / GiB:.3f} GiB")
    print(f"PyTorch allocated:  {torch.cuda.memory_allocated() / GiB:.3f} GiB")
    print(f"PyTorch reserved:   {torch.cuda.memory_reserved() / GiB:.3f} GiB")


memory("START")

x = torch.zeros(
    1_000_000_000,
    dtype=torch.float16,
    device="cuda",
)
memory("AFTER CREATING TENSOR")

del x
memory("AFTER del x")

y = torch.zeros(
    500_000_000,
    dtype=torch.float16,
    device="cuda",
)
memory("AFTER smaller y")

torch.cuda.empty_cache()
memory("AFTER empty_cache()")
