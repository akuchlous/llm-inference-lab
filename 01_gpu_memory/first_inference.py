import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GiB = 1024 ** 3
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def memory(label):
    free, total = torch.cuda.mem_get_info()
    print(f"\n=== {label} ===")
    print(f"Free GPU:          {free / GiB:.3f} GiB")
    print(f"PyTorch allocated: {torch.cuda.memory_allocated() / GiB:.3f} GiB")
    print(f"PyTorch reserved:  {torch.cuda.memory_reserved() / GiB:.3f} GiB")
    print(f"Peak allocated:    {torch.cuda.max_memory_allocated() / GiB:.3f} GiB")


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("MODEL:", MODEL)
print("Tokenizer:", tokenizer.name_or_path)
print("Chat template exists:", tokenizer.chat_template is not None)

memory("BEFORE MODEL")
print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
)
model = model.cuda()
model.eval()

memory("AFTER MODEL")

messages = [
    {
        "role": "user",
        "content": (
            "Explain in simple terms why GPU memory "
            "is important for running large language models."
        ),
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
inputs = {key: value.cuda() for key, value in inputs.items()}

print("\nInput tokens:", inputs["input_ids"].shape[1])
torch.cuda.reset_peak_memory_stats()

print("\nGenerating...")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

memory("AFTER GENERATION")
print("\nOutput tokens:", outputs.shape[1])
print("\n=== MODEL OUTPUT ===")

input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
