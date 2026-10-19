# Lesson 02 — KV Cache

This lesson connects transformer attention to real GPU memory usage.

## What we already know

During autoregressive generation:

1. Each transformer layer produces its own Key (K) and Value (V) vectors for every token.
2. A new token's Query (Q) attends to previous K/V entries from the **same layer**.
3. Previous K/V vectors do not need to be recomputed during normal decoding, so they are cached.
4. Each newly generated token adds new K/V entries at every transformer layer.
5. Therefore KV-cache memory grows approximately linearly with sequence length and also with concurrent active sequences.

A useful mental model is:

> KV cache trades GPU memory for avoiding repeated computation.

## Mistral 7B architecture values

For `mistralai/Mistral-7B-Instruct-v0.3`:

- Transformer layers: 32
- Attention heads: 32
- KV heads: 8
- Hidden size: 4096
- Head dimension: 4096 / 32 = 128
- BF16 bytes per value: 2

Mistral uses Grouped Query Attention (GQA), so it has fewer KV heads than query/attention heads. This reduces KV-cache memory.

## Theoretical KV-cache formula

For batch size 1, per token:

```text
KV bytes/token = 2 × layers × KV heads × head dimension × bytes/value
```

The leading `2` represents Key + Value.

For this model:

```text
2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB/token
```

Theoretical cache sizes:

- 1K tokens → 128 MiB
- 2K tokens → 256 MiB
- 4K tokens → 512 MiB
- 8K tokens → 1 GiB
- 32K tokens → 4 GiB

## Experiment 1 — Exact KV tensors

Run:

```bash
python 02_kv_cache/measure_kv_cache.py
```

Hardware: NVIDIA A40  
Model memory: **13.500 GiB**  
Precision: BF16

| Tokens | Exact KV | KV/token | Allocator delta | Peak above model |
|---:|---:|---:|---:|---:|
| 128 | 16.0 MiB | 128.0 KiB | 24.1 MiB | 40.2 MiB |
| 512 | 64.0 MiB | 128.0 KiB | 72.1 MiB | 130.4 MiB |
| 1,024 | 128.0 MiB | 128.0 KiB | 136.1 MiB | 252.6 MiB |
| 2,048 | 256.0 MiB | 128.0 KiB | 264.1 MiB | 497.2 MiB |
| 4,096 | 512.0 MiB | 128.0 KiB | 520.2 MiB | 986.2 MiB |
| 8,192 | 1,024.0 MiB | 128.0 KiB | 1,032.2 MiB | 1,964.2 MiB |

The exact K/V tensors match theory exactly at **128 KiB/token**.

At 8K context, exact KV is **1 GiB**, while peak-above-model memory is about **1.918 GiB**. The difference is temporary forward-pass memory, workspaces/activations, and allocator/runtime effects rather than persistent KV cache.

## Experiment 2 — KV lifecycle

`kv_lifecycle.py` showed:

```text
1. Model loaded          allocated=13.500 GiB  reserved=13.627 GiB
2. KV cache alive        allocated=14.508 GiB  reserved=15.033 GiB
3. KV cache deleted      allocated=13.508 GiB  reserved=13.752 GiB
```

The script explicitly calls `torch.cuda.empty_cache()` after deleting the KV cache, so unused reserved blocks are returned to CUDA.

## Experiment 3 — PyTorch caching allocator

`allocator_cache_behavior.py` separated deletion from `empty_cache()`:

```text
1. Model loaded               allocated=13.500 GiB  reserved=13.627 GiB
2. KV cache alive             allocated=14.508 GiB  reserved=15.971 GiB
3. KV deleted, no empty_cache allocated=13.508 GiB  reserved=15.971 GiB
4. After empty_cache          allocated=13.508 GiB  reserved=13.752 GiB
```

This proves the distinction between:

- **allocated** — live tensor memory,
- **reserved** — GPU memory owned by PyTorch's caching allocator,
- **returned to CUDA** — unused blocks released by `empty_cache()`.

## Engineering takeaway

KV-cache capacity can be predicted directly from architecture and precision, but GPU capacity planning also needs headroom for transient inference memory.

Model weights are shared across requests, while each active sequence needs its own KV state. Long context and high concurrency therefore multiply memory pressure quickly.

`RESULTS.md` is retained as the raw experiment record; this README is the canonical lesson summary.
