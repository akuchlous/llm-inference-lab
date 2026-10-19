# Lesson 03 Results — Batching

Hardware: NVIDIA A40
Model: `mistralai/Mistral-7B-Instruct-v0.3`
Precision: BF16
Generated tokens: 64 per request

Model allocated memory: **13.500 GiB**

## Fixed 1K context batching

| Batch size | Elapsed | Output tokens | Throughput | Peak above model |
|---:|---:|---:|---:|---:|
| 1 | 2.52 s | 64 | 25.45 tok/s | 0.247 GiB |
| 2 | 2.41 s | 128 | 53.21 tok/s | 0.486 GiB |
| 4 | 2.83 s | 256 | 90.58 tok/s | 0.963 GiB |
| 8 | 3.67 s | 512 | 139.61 tok/s | 1.918 GiB |

## Context length × batch size

| Context | Batch | Elapsed | Output tokens | Throughput | Peak above model |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 2.47 s | 64 | 25.94 tok/s | 0.247 GiB |
| 1024 | 2 | 2.40 s | 128 | 53.39 tok/s | 0.486 GiB |
| 1024 | 4 | 2.81 s | 256 | 91.18 tok/s | 0.963 GiB |
| 1024 | 8 | 3.64 s | 512 | 140.56 tok/s | 1.918 GiB |
| 4096 | 1 | 2.78 s | 64 | 22.99 tok/s | 0.963 GiB |
| 4096 | 2 | 3.68 s | 128 | 34.79 tok/s | 1.918 GiB |
| 4096 | 4 | 5.37 s | 256 | 47.70 tok/s | 3.829 GiB |
| 4096 | 8 | 8.75 s | 512 | 58.53 tok/s | 7.650 GiB |

## Observations

1. **Throughput rises strongly with batching at 1K context.** Batch 8 delivered about 5.4x the aggregate throughput of batch 1 (140.56 vs 25.94 tok/s), although batch size increased 8x.
2. **Longer context reduces decode throughput.** At batch 8, moving from 1K to 4K context reduced throughput from 140.56 to 58.53 tok/s, about a 58% drop.
3. **Peak memory scales nearly linearly with both context length and batch size.** Going from 1K to 4K context makes peak-above-model memory almost exactly 4x larger at each batch size.
4. At 4K context, increasing batch size still improves aggregate throughput, but with diminishing returns: 22.99 → 34.79 → 47.70 → 58.53 tok/s.
5. The longer-context slowdown is expected because each decode step must attend over and read a larger KV cache. More context therefore increases not only capacity requirements but also the memory traffic and work needed per generated token.

## Engineering takeaway

Batching trades GPU memory for throughput, but context length changes the economics of batching. More active sequences improve GPU utilization, while longer sequences consume more KV-cache memory and make each decode step more expensive. A production serving system therefore has to reason about **context length × concurrency × memory headroom × throughput × latency** together rather than tuning batch size in isolation.

The synthetic experiment uses an explicit all-ones `attention_mask` because there is no padding. It also sets both `min_new_tokens` and `max_new_tokens` to 64 so every measurement performs the same amount of decode work.
