# Lesson 01 Results — GPU Memory

Date: October 17, 2026

## Hardware

- GPU: NVIDIA A40
- Advertised VRAM: 48 GB
- PyTorch-visible VRAM: 44.422 GiB
- NVIDIA driver: 580.159.03
- Driver-supported CUDA: 13.0
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- PyTorch CUDA build: 12.8

## Baseline

```text
Total GPU memory:  44.422 GiB
Free GPU memory:   44.158 GiB
PyTorch allocated: 0.000 GiB
PyTorch reserved:  0.000 GiB
```

Approximately 0.264 GiB was unavailable even with no PyTorch allocation.

## PyTorch allocator experiment

A 1,000,000,000-element FP16 tensor is approximately 1.863 GiB.

```text
START
Free:      44.158 GiB
Allocated:  0.000 GiB
Reserved:   0.000 GiB

AFTER TENSOR
Free:      42.252 GiB
Allocated:  1.863 GiB
Reserved:   1.863 GiB

AFTER del x
Free:      42.252 GiB
Allocated:  0.000 GiB
Reserved:   1.863 GiB

AFTER empty_cache()
Free:      44.115 GiB
Allocated:  0.000 GiB
Reserved:   0.000 GiB
```

## Mistral 7B model load

Base model: `mistralai/Mistral-7B-v0.1`

- Parameters: 7,241,732,096
- Precision: BF16
- Parameter memory: 13.489 GiB
- PyTorch allocated after load: 13.489 GiB
- PyTorch reserved after load: 13.615 GiB
- `nvidia-smi` memory used while resident: about 13.880 GiB

## First instruction-model inference

Model: `mistralai/Mistral-7B-Instruct-v0.3`

Before model load:

```text
Free GPU:          44.158 GiB
PyTorch allocated:  0.000 GiB
PyTorch reserved:   0.000 GiB
Peak allocated:     0.000 GiB
```

After model load:

```text
Free GPU:          30.531 GiB
PyTorch allocated: 13.500 GiB
PyTorch reserved:  13.627 GiB
Peak allocated:    13.500 GiB
```

Input tokens: 19

After generation of up to 100 new tokens:

```text
Free GPU:          30.435 GiB
PyTorch allocated: 13.508 GiB
PyTorch reserved:  13.664 GiB
Peak allocated:    13.523 GiB
```

Total sequence length reported after generation: 119 tokens.

## Context-length experiment

Batch size: 1  
Output tokens: 100  
Model memory: about 13.500 GiB

| Context | Before inference | Peak | After inference | Peak above model |
|---:|---:|---:|---:|---:|
| 128 | 13.500 GiB | 13.540 GiB | 13.508 GiB | 0.039 GiB |
| 512 | 13.508 GiB | 13.628 GiB | 13.508 GiB | 0.127 GiB |
| 1,024 | 13.508 GiB | 13.747 GiB | 13.508 GiB | 0.247 GiB |
| 2,048 | 13.508 GiB | 13.986 GiB | 13.508 GiB | 0.486 GiB |
| 4,096 | 13.508 GiB | 14.464 GiB | 13.508 GiB | 0.963 GiB |
| 8,192 | 13.508 GiB | 15.419 GiB | 13.509 GiB | 1.918 GiB |

The growth is approximately linear. From 4,096 to 8,192 input tokens, measured peak-above-model memory increased by about 0.955 GiB for 4,096 extra tokens, roughly 244 KiB per token of observed peak growth.

This is not a measurement of pure KV-cache bytes/token. It includes other inference-time memory too.

## Main takeaway

The first experiment established three separate categories to reason about during inference:

1. **Model weights** — mostly fixed for a loaded model.
2. **Request-dependent persistent state** — especially KV cache.
3. **Transient/runtime memory** — temporary tensors, workspaces, allocator/runtime overhead.

The next experiment will isolate the KV-cache theory and compare it with these measurements.
