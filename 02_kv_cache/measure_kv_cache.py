import gc
import torch
from transformers import AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CONTEXT_LENGTHS = [128, 512, 1024, 2048, 4096, 8192]
GiB = 1024 ** 3
MiB = 1024 ** 2
KiB = 1024


def cache_tensor_bytes(past_key_values):
    """Return the exact bytes held by K/V tensors in the cache."""
    if hasattr(past_key_values, "to_legacy_cache"):
        layers = past_key_values.to_legacy_cache()
    else:
        layers = past_key_values

    total = 0
    for layer in layers:
        key, value = layer[:2]
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


print("Loading model:", MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
).cuda().eval()

model_memory = torch.cuda.memory_allocated()
print(f"Model allocated memory: {model_memory / GiB:.3f} GiB")
print()

print(
    f"{'Tokens':>8} | {'Exact KV':>10} | {'KV/token':>10} | "
    f"{'Allocator delta':>15} | {'Peak above model':>16}"
)
print("-" * 73)

for seq_len in CONTEXT_LENGTHS:
    gc.collect()
    torch.cuda.empty_cache()

    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, seq_len),
        device="cuda",
    )

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # Call the base transformer so we do not allocate full-vocabulary
        # logits for every prompt token. We only need the KV cache.
        outputs = model.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    exact_kv_bytes = cache_tensor_bytes(past_key_values)

    # Drop the final hidden state/output object while deliberately keeping
    # past_key_values alive. After empty_cache(), allocated memory above the
    # model should be dominated by the live KV tensors (plus tiny inputs/state).
    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    allocator_delta = torch.cuda.memory_allocated() - model_memory
    peak_above_model = torch.cuda.max_memory_allocated() - model_memory

    print(
        f"{seq_len:8d} | "
        f"{exact_kv_bytes / MiB:8.1f} MiB | "
        f"{exact_kv_bytes / seq_len / KiB:8.1f} KiB | "
        f"{allocator_delta / MiB:12.1f} MiB | "
        f"{peak_above_model / MiB:13.1f} MiB"
    )

    del past_key_values, input_ids
    gc.collect()
    torch.cuda.empty_cache()

print()
print("Theory for Mistral 7B BF16:")
print("2 (K+V) x 32 layers x 8 KV heads x 128 head dim x 2 bytes")
print("= 131,072 bytes = 128 KiB per token")
