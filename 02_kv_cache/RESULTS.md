# Lesson 02 Results — KV Cache

Hardware: NVIDIA A40
Model memory: 13.500 GiB
Model: Mistral 7B class, BF16 KV cache

## Theory

KV bytes per token:

`2 (K+V) × 32 layers × 8 KV heads × 128 head dim × 2 bytes = 131,072 bytes = 128 KiB/token`

## Measurements

| Tokens | Exact KV | KV/token | Allocator delta | Peak above model |
|---:|---:|---:|---:|---:|
| 128 | 16.0 MiB | 128.0 KiB | 24.1 MiB | 40.2 MiB |
| 512 | 64.0 MiB | 128.0 KiB | 72.1 MiB | 130.4 MiB |
| 1024 | 128.0 MiB | 128.0 KiB | 136.1 MiB | 252.6 MiB |
| 2048 | 256.0 MiB | 128.0 KiB | 264.1 MiB | 497.2 MiB |
| 4096 | 512.0 MiB | 128.0 KiB | 520.2 MiB | 986.2 MiB |
| 8192 | 1024.0 MiB | 128.0 KiB | 1032.2 MiB | 1964.2 MiB |

## Key finding

The exact K/V tensor size matches the theoretical prediction exactly: **128 KiB per token** at every tested context length.

At 8192 tokens, the KV tensors occupy exactly **1024 MiB (1 GiB)**. The allocator delta is only slightly larger, at 1032.2 MiB, while peak memory above the model reaches 1964.2 MiB.

This demonstrates that **peak inference memory is not the same thing as KV-cache memory**. The additional peak is transient memory used while computing the forward pass, including activations/intermediate tensors, attention workspaces, and runtime/allocator effects.

## Engineering takeaway

KV-cache capacity can be predicted directly from architecture and precision, but GPU capacity planning must also leave headroom for transient inference memory. A server that allocates VRAM based only on model weights + theoretical KV size can run out of memory during actual inference.
