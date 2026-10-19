# Lesson 04 Results — Prefill vs Decode

Hardware: NVIDIA A40
Model: `mistralai/Mistral-7B-Instruct-v0.3`
Precision: BF16
Batch size: 1
Decode steps measured: 32

| Context | Prefill time | Prefill throughput | Decode time/token | Decode throughput |
|---:|---:|---:|---:|---:|
| 128 | 65.85 ms | 1,943.69 tok/s | 32.09 ms | 31.16 tok/s |
| 512 | 91.82 ms | 5,575.83 tok/s | 30.82 ms | 32.45 tok/s |
| 1,024 | 174.78 ms | 5,858.86 tok/s | 31.07 ms | 32.19 tok/s |
| 2,048 | 338.39 ms | 6,052.12 tok/s | 31.90 ms | 31.35 tok/s |
| 4,096 | 686.22 ms | 5,968.92 tok/s | 33.51 ms | 29.84 tok/s |

## Observations

1. **Prefill time grows strongly with prompt length.** Moving from 128 to 4,096 input tokens increased prefill latency from about 66 ms to about 686 ms.
2. **Prefill throughput improves sharply for larger prompts and then plateaus near 6K tokens/s.** Very short prompts underutilize the GPU, while larger matrix operations make better use of the hardware.
3. **Decode latency stays nearly flat at batch size 1.** Across 128–4,096 tokens of context, decode remained around 31–34 ms per generated token.
4. **Longer context does still make decode somewhat slower.** Decode throughput fell from roughly 32.5 tok/s near 512 tokens to 29.8 tok/s at 4,096 tokens because each new token attends over and reads a larger KV cache.

## Engineering takeaway

Prefill and decode have different performance profiles:

- **Prefill** processes the input prompt in one large parallel pass and strongly affects time to first token (TTFT).
- **Decode** generates one token at a time using the existing KV cache and strongly affects time per output token (TPOT).

For this Mistral-7B BF16 batch-1 workload on an A40, long context primarily increases TTFT; per-token decode latency changes much more modestly.
