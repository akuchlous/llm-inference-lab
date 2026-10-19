# Lesson 05 Results — vLLM Concurrency

Hardware: NVIDIA A40
Model: `mistralai/Mistral-7B-Instruct-v0.3`
Serving engine: vLLM 0.32.0
Output length: 64 tokens per request

| Concurrency | Elapsed | Output tokens | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 1.72 s | 64 | 37.19 tok/s |
| 2 | 1.84 s | 128 | 69.41 tok/s |
| 4 | 1.85 s | 256 | 138.19 tok/s |
| 8 | 1.89 s | 512 | 270.95 tok/s |

## Observations

1. Aggregate throughput scales strongly with concurrency: from **37.19 tok/s at concurrency 1** to **270.95 tok/s at concurrency 8**, about **7.3x higher throughput**.
2. Wall-clock completion time stays nearly flat: **1.72 s → 1.89 s** while the amount of output work increases 8x.
3. This is exactly the kind of workload where a serving engine can keep the GPU much busier than single-request inference.
4. Compared with the earlier Hugging Face fixed-batch experiment at batch 8 (~140.6 tok/s), this vLLM run reached ~271 tok/s, though this is not yet a strict apples-to-apples comparison because prompt construction, server overhead, and execution paths differ.

## Engineering takeaway

A production inference engine can exploit request concurrency much more effectively than a simple one-request-at-a-time loop. Continuous batching lets new and active requests share GPU decode steps dynamically, improving aggregate throughput while keeping per-batch wall-clock time nearly flat over this small concurrency range.
