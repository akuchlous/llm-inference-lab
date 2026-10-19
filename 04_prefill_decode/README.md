# Lesson 04 — Prefill vs Decode

## Question

Where does inference latency come from, and how do input prompt length and output generation affect it differently?

## Mental model

Inference has two distinct phases:

1. **Prefill** — process all input/prompt tokens and build the KV cache.
2. **Decode** — generate output tokens one at a time using the existing KV cache.

This maps naturally to serving metrics:

- **TTFT (time to first token)** is dominated by prefill for long prompts.
- **TPOT (time per output token)** is dominated by decode.

## Experiment

Hardware: NVIDIA A40  
Model: `mistralai/Mistral-7B-Instruct-v0.3`  
Precision: BF16  
Batch size: 1  
Decode steps measured: 32

Run:

```bash
python 04_prefill_decode/prefill_decode_timing.py
```

## Measured output

| Context | Prefill time | Prefill throughput | Decode time/token | Decode throughput |
|---:|---:|---:|---:|---:|
| 128 | 65.85 ms | 1,943.69 tok/s | 32.09 ms | 31.16 tok/s |
| 512 | 91.82 ms | 5,575.83 tok/s | 30.82 ms | 32.45 tok/s |
| 1,024 | 174.78 ms | 5,858.86 tok/s | 31.07 ms | 32.19 tok/s |
| 2,048 | 338.39 ms | 6,052.12 tok/s | 31.90 ms | 31.35 tok/s |
| 4,096 | 686.22 ms | 5,968.92 tok/s | 33.51 ms | 29.84 tok/s |

## Interpretation

Increasing context from 128 to 4,096 tokens (32×) increased prefill latency from about **66 ms to 686 ms** — roughly 10×.

Decode stayed much flatter at roughly **31–34 ms/token**. Longer context still makes decode slightly more expensive because every new token reads/attends over a larger KV cache.

Prefill throughput improves sharply with larger prompts because larger matrix operations use the GPU more efficiently, then plateaus around ~6K input tokens/s in this setup.

## Engineering takeaway

Long prompts primarily hurt **time to first token**, while output generation speed changes more modestly for this batch-1 workload.

For a real serving system, TTFT is not exactly equal to prefill time: queueing, tokenization, scheduling, the first decode step, and network streaming also contribute. But prefill is usually the dominant model-compute component for long prompts.

`RESULTS.md` is retained as the raw experiment record; this README is the canonical lesson summary.
