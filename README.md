# LLM Inference Lab

A hands-on learning repository for understanding and optimizing LLM inference from the GPU up.

The goal is not just to learn how to call an LLM API. It is to build enough systems understanding to design, operate, secure, benchmark, and optimize private enterprise LLM infrastructure.

## Learning path

### Inference fundamentals

**tokens → attention → KV cache → GPU memory → batching → prefill/decode → throughput → latency → inference cost**

### One level higher

**serving engines → continuous batching → quantization → sparse/MoE models → multi-GPU inference → model selection → routing → caching → context management → RAG → observability → evaluation → reliability → economics**

## Long-term goal

Be able to take an enterprise workload, security requirements, model choices, hardware budget, and SLA and:

- design a private/on-prem LLM inference platform,
- secure it appropriately,
- benchmark the workload,
- determine whether it is memory-, compute-, or communication-bound,
- tune serving and GPU configuration,
- and explain quality, latency, throughput, reliability, and cost tradeoffs.

## Setup

### Labs 01–04 and 06

Use a CUDA-capable PyTorch environment, then install the common dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is deliberately not pinned in `requirements.txt` because GPU images differ in CUDA/driver compatibility. On Runpod, start from a PyTorch/CUDA image that already supports the GPU.

### Lab 05 — vLLM

vLLM should be installed in a separate environment because its wheels are coupled to particular PyTorch/CUDA builds:

```bash
pip install --upgrade uv
uv venv --python 3.12 --seed .venv-vllm
source .venv-vllm/bin/activate
uv pip install -r requirements-vllm.txt --torch-backend=auto
```

## Completed labs

| Lesson | Topic | Key result | Status |
|---|---|---|---|
| [01](01_gpu_memory/README.md) | GPU memory fundamentals | Mistral-7B BF16 weights use ~13.5 GiB; peak memory grows nearly linearly with context | Complete |
| [02](02_kv_cache/README.md) | KV cache | Exact Mistral KV cache = **128 KiB/token** | Complete |
| [03](03_batching/README.md) | Batching + context | Batch 8 at 1K context = **140.56 tok/s**; 4K context = **58.53 tok/s** | Complete |
| [04](04_prefill_decode/README.md) | Prefill vs decode | 4K prefill ≈ **686 ms**; decode ≈ **33.5 ms/token** | Complete |
| [05](05_vllm/README.md) | vLLM concurrency and saturation | **3,817 tok/s at concurrency 256**, but with a clear latency/TTFT saturation knee after ~128 | Complete |
| [06](06_quantization/README.md) | Quantization / memory efficiency | BF16 baseline first; then lower-precision memory, latency, throughput, capacity, and quality comparisons | In progress |
| 07 | Sparse activation / Mixture of Experts | Total vs active parameters, top-k expert routing, load balance, and serving implications | Planned |
| 08 | Multi-GPU inference | Tensor/expert parallelism, communication costs, placement, and scaling | Planned |
| 09 | Capacity and cost optimization | Convert workload + SLA into GPU count, concurrency targets, and serving economics | Planned |

Each lesson README is the canonical record containing the question, mental model, experiment, measured output, interpretation, and engineering takeaway. Some folders also retain `RESULTS.md` as a compact raw-results record.

## Hardware used so far

Experiments have been run on NVIDIA A40 GPUs (~46,068 MiB reported by `nvidia-smi`). Driver/CUDA versions vary across disposable Runpod instances, so exact software versions are recorded in lesson notes when relevant.

Primary model:

- `mistralai/Mistral-7B-Instruct-v0.3`
- BF16 for the Hugging Face experiments

## Current measured findings

### GPU memory

- Mistral-7B BF16 model allocation: ~**13.5 GiB**
- 1 billion FP16/BF16 values: ~**1.86 GiB**
- PyTorch `allocated` and `reserved` memory are different; deleting tensors does not necessarily return cached blocks to CUDA immediately.

### KV cache

For Mistral-7B BF16 with 32 layers, 8 KV heads, and head dimension 128:

```text
2 (K+V) × 32 × 8 × 128 × 2 bytes = 128 KiB/token
```

At 8K context, exact KV tensors occupy ~**1 GiB** per sequence.

### Batching

At 1K context:

```text
batch 1 → 25.94 tok/s
batch 2 → 53.39 tok/s
batch 4 → 91.18 tok/s
batch 8 → 140.56 tok/s
```

At batch 8, increasing context from 1K to 4K increased peak-above-model memory from **1.918 GiB to 7.650 GiB** and reduced throughput from **140.56 to 58.53 tok/s**.

### Prefill vs decode

At batch 1:

```text
128-token prompt  → prefill 65.85 ms, decode 32.09 ms/token
4096-token prompt → prefill 686.22 ms, decode 33.51 ms/token
```

Long prompts strongly increase TTFT through prefill. Per-token decode latency also rises because each new token attends over a larger KV cache, but the measured increase was much more modest over this context range.

### vLLM

With 64 output tokens/request:

```text
concurrency 1   →   37.10 tok/s, TTFT ~44 ms
concurrency 64  → 1762.97 tok/s, TTFT ~283 ms
concurrency 128 → 2928.97 tok/s, TTFT ~500 ms
concurrency 256 → 3816.89 tok/s, TTFT ~810 ms
```

From concurrency 128 to 256, throughput improved only about **30%**, while TTFT worsened about **62%** and average total latency about **55%**. This is the first clear saturation knee for this workload.

At high concurrency, representative `nvidia-smi` samples showed GPU utilization near **100%**, memory-controller activity around **96–100%**, VRAM around **42.3 / 46.1 GiB**, and power near **300 W**. These observations show the device was heavily saturated, though they are not by themselves enough to distinguish compute-bound from memory-bandwidth-bound execution.

## Upcoming optimization chapters

### 06 — Quantization

Lab 1 is now runnable from `06_quantization/bf16_baseline.py`. It establishes the BF16 control measurement before lower-precision variants are introduced.

Compare BF16 with lower-precision representations and measure what changes in model memory, KV-cache headroom, throughput, TTFT/TPOT, serving capacity, and answer quality.

### 07 — Sparse activation / Mixture of Experts

Study models where **total parameters are larger than active parameters per token**. Topics include router decisions, top-k expert selection, expert placement, load balancing, memory footprint, communication overhead, and expert parallelism.

A key systems question becomes: **how many parameters are active for each token, where do those parameters live, and how much communication is required to reach them?**

## Workflow

GitHub is the source of truth. Runpod instances are disposable:

```text
edit / commit in GitHub
        ↓
git pull on GPU pod
        ↓
run experiment
        ↓
record measured output in lesson README
```

The repository is intended to remain reproducible and useful as a technical portfolio, not just as scratch code.
