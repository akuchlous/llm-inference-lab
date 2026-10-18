# Lesson 01 — Where Does GPU Memory Go When Running an LLM?

This lesson starts from first principles: before optimizing inference, understand what occupies GPU memory and how that changes when a model is loaded and used.

## Questions

- How much GPU memory is visible before PyTorch allocates anything?
- What is the difference between PyTorch **allocated** and **reserved** memory?
- How much memory do Mistral 7B BF16 weights actually occupy?
- How much extra GPU memory does a small generation request need?
- How does peak memory change as prompt/context length increases?

## Hardware and software

- GPU: NVIDIA A40
- `nvidia-smi` memory: 46,068 MiB
- PyTorch-visible total: 44.422 GiB
- Driver: 580.159.03
- Driver-supported CUDA: 13.0
- PyTorch: 2.8.0+cu128
- PyTorch CUDA build: 12.8

The advertised 48 GB, `nvidia-smi` MiB value, and PyTorch GiB value differ because vendors and tools use different units and some device memory is not exposed to the PyTorch allocator.

## Experiment 1 — Baseline

Run:

```bash
python baseline.py
```

Measured baseline:

- Total GPU memory: **44.422 GiB**
- Free GPU memory: **44.158 GiB**
- PyTorch allocated: **0 GiB**
- PyTorch reserved: **0 GiB**

Even with no PyTorch allocation, about 0.264 GiB was unavailable. That is expected device/runtime overhead outside the PyTorch allocator.

## Experiment 2 — PyTorch allocated vs. reserved

`tensor_test.py` allocates a large FP16 tensor and observes what happens when the Python object is deleted.

A tensor with 1,000,000,000 FP16 values contains:

```text
1,000,000,000 × 2 bytes = 2,000,000,000 bytes ≈ 1.863 GiB
```

Observed behavior:

| State | PyTorch allocated | PyTorch reserved |
|---|---:|---:|
| Start | 0 GiB | 0 GiB |
| After large tensor | 1.863 GiB | 1.863 GiB |
| After `del x` | 0 GiB | 1.863 GiB |
| After `empty_cache()` | 0 GiB | 0 GiB |

### Mental model

- **allocated** = memory currently occupied by live tensors
- **reserved** = memory PyTorch has retained in its caching allocator for fast reuse
- `del` can release the tensor while PyTorch keeps the GPU block reserved
- `torch.cuda.empty_cache()` releases unused reserved blocks back to the CUDA allocator

## Experiment 3 — Model weights

`model_memory.py` loads Mistral 7B in BF16.

Prediction before loading:

```text
~7B parameters × 2 bytes/parameter ≈ 14 GB
```

Measured model:

- Parameters: **7,241,732,096**
- Exact parameter memory: **13.489 GiB**
- PyTorch allocated after load: **13.489 GiB**
- PyTorch reserved after load: **13.615 GiB**
- `nvidia-smi` usage while model was resident: about **13.880 GiB**

The weight calculation matches the measured PyTorch allocation closely. `nvidia-smi` is slightly higher because it includes memory outside the model tensors themselves, such as CUDA/runtime/library state.

## Experiment 4 — First inference

`first_inference.py` loads `mistralai/Mistral-7B-Instruct-v0.3` in BF16 and generates up to 100 new tokens.

Measured:

- Model allocated: **13.500 GiB**
- Input tokens: **19**
- Peak allocated during generation: **13.523 GiB**
- Allocated after generation: **13.508 GiB**

For this tiny request, the weights dominate GPU memory. The request itself adds relatively little peak memory.

## Experiment 5 — Context length vs. peak memory

`context_memory.py` holds model, batch size, and output length constant while varying the artificial input context length.

Model memory was about **13.500 GiB**.

| Context tokens | Peak memory | Peak above model |
|---:|---:|---:|
| 128 | 13.540 GiB | 0.039 GiB |
| 512 | 13.628 GiB | 0.127 GiB |
| 1,024 | 13.747 GiB | 0.247 GiB |
| 2,048 | 13.986 GiB | 0.486 GiB |
| 4,096 | 14.464 GiB | 0.963 GiB |
| 8,192 | 15.419 GiB | 1.918 GiB |

The growth is strikingly close to linear.

From 4,096 to 8,192 tokens, the extra 4,096 tokens increased measured peak-above-model memory by about 0.955 GiB, or roughly 244 KiB per additional token **for this observed peak-memory slope**.

That number is not pure KV-cache size. Peak memory includes KV cache plus temporary inference tensors, activations/workspaces, allocator effects, and runtime overhead.

## Key concepts learned

1. Model weights can dominate memory for small inference requests.
2. PyTorch allocated memory and reserved memory are different quantities.
3. Deleting a tensor does not necessarily return its GPU memory immediately.
4. BF16 model memory can be predicted accurately from parameter count.
5. Context length produces approximately linear growth in measured peak memory in this experiment.
6. Peak memory and steady-state memory are not the same thing.
7. The next major question is how much of the context-dependent memory is specifically the **KV cache**.

## Next lesson

[Lesson 02 — KV Cache](../02_kv_cache/README.md)

We will derive Mistral's theoretical KV-cache bytes per token and compare the theory with the A40 measurements from this lesson.
