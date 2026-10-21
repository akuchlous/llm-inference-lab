import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROMPTS = [
    "Explain in simple terms why KV cache grows with context length.",
    "A server has 46 GiB of VRAM. A model uses 13.5 GiB for weights and each active request uses 1 GiB of KV cache. Ignoring other overhead, how many such requests can fit concurrently? Show the arithmetic.",
    "What is the difference between model weights, activations, and KV cache during LLM inference?",
    "If a model has 32 query heads but only 8 KV heads, what memory benefit does grouped-query attention provide compared with 32 KV heads?",
]


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(model_name: str, precision: str):
    if precision == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
    elif precision == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    model.eval()
    return model


def generate_answers(model, tokenizer, precision: str, max_new_tokens: int):
    print(f"\n{'=' * 80}\n{precision.upper()} ANSWERS\n{'=' * 80}")

    for index, prompt in enumerate(PROMPTS, start=1):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        print(f"\n--- Prompt {index} ---")
        print(prompt)
        print(f"\n[{precision.upper()}]")
        print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BF16 and INT4-NF4 answer quality")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print("Decoding: deterministic greedy decoding (do_sample=False)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_model = load_model(args.model, "bf16")
    generate_answers(bf16_model, tokenizer, "bf16", args.max_new_tokens)
    del bf16_model
    cuda_cleanup()

    int4_model = load_model(args.model, "int4")
    generate_answers(int4_model, tokenizer, "int4", args.max_new_tokens)
    del int4_model
    cuda_cleanup()

    print("\n=== What to compare ===")
    print("1. Factual correctness")
    print("2. Whether arithmetic/reasoning is preserved")
    print("3. Missing or hallucinated details")
    print("4. Clarity and completeness")
    print("5. Whether differences are material, not merely wording changes")


if __name__ == "__main__":
    main()
