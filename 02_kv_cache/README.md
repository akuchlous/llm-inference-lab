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
2 × 32 × 8 × 128 × 2
```

The next step is to calculate this value by hand, convert it to KiB/token, and then predict KV-cache memory for:

- 1,024 tokens
- 8,192 tokens
- 32,768 tokens

## Why compare theory with Lesson 01?

Lesson 01 measured approximately 1.918 GiB of **peak memory above model weights** for an 8,192-token prompt plus generation.

That measured value is not pure KV cache. It also includes temporary inference tensors, workspaces/activations, allocator effects, and runtime overhead.

By calculating theoretical KV memory first, we can separate:

```text
measured peak memory
        ↓
KV cache
+ temporary inference memory
+ allocator/runtime overhead
```

## Questions this lesson should answer

- How many bytes of KV cache does one token require?
- Why does the number of transformer layers multiply KV memory?
- Why does GQA reduce KV memory?
- How closely does theoretical KV memory explain the A40 experiment?
- What happens to KV memory when context length or concurrency increases?

## Engineering connection

Understanding this calculation is foundational for later topics:

**KV cache → concurrency → batching → throughput → GPU capacity → inference cost**

For enterprise serving, model weights are often shared across requests while each active request needs its own KV state. That means KV-cache capacity can become a practical limit on how many long-context requests a GPU can serve simultaneously.
