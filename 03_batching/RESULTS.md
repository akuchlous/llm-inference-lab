# Lesson 03 Results — Batching

Hardware: NVIDIA A40
Model: `mistralai/Mistral-7B-Instruct-v0.3`
Precision: BF16
Prompt length: 1,024 tokens per request
Generated tokens: 64 per request

Model allocated memory: **13.500 GiB**

| Batch size | Elapsed | Output tokens | Throughput | Peak above model |
|---:|---:|---:|---:|---:|
| 1 | 2.52 s | 64 | 25.45 tok/s | 0.247 GiB |
| 2 | 2.41 s | 128 | 53.21 tok/s | 0.486 GiB |
| 4 | 2.83 s | 256 | 90.58 tok/s | 0.963 GiB |
| 8 | 3.67 s | 512 | 139.61 tok/s | 1.918 GiB |

## Observations

1. **Throughput rises strongly with batching.** Batch 8 delivered about 5.5x the aggregate token throughput of batch 1 (139.61 vs 25.45 tok/s), even though batch size increased 8x.
2. **Latency for the whole batch rises much more slowly than work completed.** Batch 8 generated 512 total output tokens in 3.67 seconds versus 64 tokens in 2.52 seconds for batch 1.
3. **Peak memory scales almost linearly with batch size** for this fixed prompt length and output length: approximately 0.24 GiB of additional peak memory per sequence.
4. The throughput scaling is sub-linear: increasing batch size keeps the GPU busier, but eventually compute, memory bandwidth, scheduling, and other overheads prevent perfect scaling.

## Engineering takeaway

Batching trades GPU memory for throughput. More active sequences improve GPU utilization, but each sequence carries its own KV cache and working memory. Production inference therefore has to balance batch size/concurrency against latency and memory headroom.

The initial run emitted an attention-mask warning for batched generation. The experiment script was updated to pass an explicit all-ones `attention_mask` because these synthetic prompts contain no padding.
