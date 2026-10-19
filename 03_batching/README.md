# Lesson 03 — Batching, Context Length, and Throughput

## Question

How do batch size and prompt context length affect GPU throughput and peak memory during LLM inference?

## Mental model

Batching lets the GPU process multiple active sequences together. This improves utilization and aggregate throughput, but every sequence consumes its own KV cache and working memory.

Longer context increases the KV cache carried by each request and makes each decode step read/attend over more history.

So the serving tradeoff is:

**context length × concurrency → memory pressure → throughput/latency**

## Experiment 1 — Fixed 1K context

Hardware: NVIDIA A40  
Model: `mistralai/Mistral-7B-Instruct-v0.3`  
Precision: BF16  
Prompt length: 1,024 tokens per request  
Generated tokens: 64 per request  
Model allocated memory: **13.500 GiB**

Run:

```bash
python 03_batching/batching_experiment.py
```

### Measured output

| Batch size | Elapsed | Output tokens | Throughput | Peak above model |
|---:|---:|---:|---:|---:|
| 1 | 2.52 s | 64 | 25.45 tok/s | 0.247 GiB |
| 2 | 2.41 s | 128 | 53.21 tok/s | 0.486 GiB |
| 4 | 2.83 s | 256 | 90.58 tok/s | 0.963 GiB |
| 8 | 3.67 s | 512 | 139.61 tok/s | 1.918 GiB |

Batch 8 produced about **5.5×** the aggregate throughput of batch 1 while peak-above-model memory grew nearly linearly.

## Experiment 2 — Context length × batch size

Run:

```bash
python 03_batching/context_batch_grid.py
```

### Measured output

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

## Interpretation

At batch 8, moving from 1K to 4K context:

- increased peak-above-model memory from **1.918 GiB to 7.650 GiB** — almost exactly 4×,
- reduced aggregate throughput from **140.56 to 58.53 tok/s** — about a 58% drop.

The 4× memory scaling is consistent with each request carrying roughly 4× as much KV/context state. Decode also gets slower because each generated token must attend over and read a larger history.

## Engineering takeaway

Batch size cannot be tuned independently of context length. A production system needs to reason about **context length, active sequences, KV-cache capacity, memory headroom, throughput, and latency together**.

`RESULTS.md` is retained as the raw experiment record; this README is the canonical lesson summary.
