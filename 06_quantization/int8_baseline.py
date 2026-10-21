import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def mib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 2)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 baseline for quantization experiments")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print("Precision: INT8 weights via bitsandbytes")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    cuda_sync()
    load_seconds = time.perf_counter() - load_start

    model_alloc = torch.cuda.memory_allocated()
    model_reserved = torch.cuda.memory_reserved()
    param_count = sum(p.numel() for p in model.parameters())
    reported_footprint = model.get_memory_footprint() if hasattr(model, "get_memory_footprint") else 0

    print("\n=== Model load ===")
    print(f"Parameters: {param_count:,}")
    if reported_footprint:
        print(f"Model reported footprint: {gib(reported_footprint):.3f} GiB")
    print(f"CUDA allocated after load: {gib(model_alloc - baseline_alloc):.3f} GiB")
    print(f"CUDA reserved after load: {gib(model_reserved):.3f} GiB")
    print(f"Load time: {load_seconds:.2f} s")

    seed_text = (
        "Explain how GPU memory, transformer layers, attention, and KV cache interact "
        "during LLM inference. Focus on systems behavior and capacity planning. "
    )
    seed_ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    repeats = max(1, (args.prompt_tokens + len(seed_ids) - 1) // len(seed_ids))
    prompt_ids = (seed_ids * repeats)[: args.prompt_tokens]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    print("\n=== Workload ===")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"Requested output tokens: {args.max_new_tokens}")

    with torch.inference_mode():
        _ = model.generate(
            input_ids=input_ids[:, : min(128, input_ids.shape[1])],
            attention_mask=attention_mask[:, : min(128, attention_mask.shape[1])],
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    cuda_sync()

    torch.cuda.reset_peak_memory_stats()
    start_alloc = torch.cuda.memory_allocated()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    cuda_sync()
    elapsed = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated()

    generated_tokens = output.shape[1] - input_ids.shape[1]
    output_tps = generated_tokens / elapsed if elapsed else float("inf")
    peak_above_model = max(0, peak_alloc - model_alloc)

    print("\n=== Measured generation ===")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Output throughput: {output_tps:.2f} tok/s")
    print(f"Peak CUDA allocated: {gib(peak_alloc):.3f} GiB")
    print(f"Peak above loaded model: {mib(peak_above_model):.1f} MiB")
    print(f"Allocated at measured-run start: {gib(start_alloc):.3f} GiB")

    print("\n=== Baseline record ===")
    print(
        "precision=INT8 "
        f"params={param_count} "
        f"model_alloc_gib={gib(model_alloc - baseline_alloc):.3f} "
        f"prompt_tokens={input_ids.shape[1]} "
        f"output_tokens={generated_tokens} "
        f"elapsed_s={elapsed:.3f} "
        f"output_tok_s={output_tps:.2f} "
        f"peak_gib={gib(peak_alloc):.3f}"
    )


if __name__ == "__main__":
    main()
