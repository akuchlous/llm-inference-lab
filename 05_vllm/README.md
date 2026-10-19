# Lesson 05 — vLLM Serving and Concurrency

## Question

How does a production-oriented serving engine such as vLLM use concurrent requests to improve aggregate GPU throughput, and what happens to per-request latency as concurrency rises?

## Mental model

A simple fixed batch groups requests together ahead of time. vLLM uses continuous batching so active requests can share GPU work dynamically as requests arrive and finish.

The objective is to keep the GPU busy while managing KV-cache memory and request scheduling efficiently.

## Environment

Hardware: NVIDIA A40  
Model: `mistralai/Mistral-7B-Instruct-v0.3`  
Serving engine: vLLM 0.32.0  
Output length: 64 tokens/request

Because vLLM ships against specific PyTorch/CUDA combinations, use a fresh environment:

```bash
pip install --upgrade uv
uv venv --python 3.12 --seed .venv-vllm
source .venv-vllm/bin/activate
uv pip install -r requirements-vllm.txt --torch-backend=auto
```

Start the server:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --tokenizer-mode mistral \
  --load-format mistral \
  --config-format mistral
```

In another terminal, run:

```bash
python 05_vllm/vllm_concurrency_benchmark.py
python 05_vllm/vllm_latency_benchmark.py
```

## Small-concurrency throughput experiment

| Concurrency | Elapsed | Output tokens | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 1.72 s | 64 | 37.19 tok/s |
| 2 | 1.84 s | 128 | 69.41 tok/s |
| 4 | 1.85 s | 256 | 138.19 tok/s |
| 8 | 1.89 s | 512 | 270.95 tok/s |

Concurrency 8 delivered **270.95 tok/s**, about **7.3×** the throughput of concurrency 1 at **37.19 tok/s**.

Wall-clock time changed only modestly from **1.72 s to 1.89 s** while the output work increased 8×. The serving engine is using concurrent work to keep the GPU much more fully utilized.

This should not yet be read as a strict vLLM-vs-Hugging-Face speed comparison because the earlier experiments differ in prompt construction, execution path, and server overhead. The clean conclusion is that **vLLM scales aggregate throughput strongly with concurrent requests in this workload**.

## Saturation experiment

The latency benchmark was then extended to concurrency levels from 1 through 256.

| Concurrency | Avg TTFT | Avg total latency | P95 total latency | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 43.84 ms | 1.73 s | 1.73 s | 37.10 tok/s |
| 2 | 72.34 ms | 1.84 s | 1.83 s | 68.92 tok/s |
| 4 | 70.76 ms | 1.86 s | 1.86 s | 137.38 tok/s |
| 8 | 91.96 ms | 1.89 s | 1.90 s | 269.81 tok/s |
| 16 | 120.64 ms | 2.00 s | 2.01 s | 508.89 tok/s |
| 32 | 179.24 ms | 2.03 s | 2.05 s | 998.07 tok/s |
| 64 | 282.56 ms | 2.28 s | 2.32 s | 1762.97 tok/s |
| 128 | 500.08 ms | 2.73 s | 2.77 s | 2928.97 tok/s |
| 256 | 809.75 ms | 4.22 s | 4.29 s | 3816.89 tok/s |

### Interpretation

The A40 shows excellent batching efficiency through moderate concurrency. From concurrency 1 to 64, throughput rises from about **37 tok/s to 1,763 tok/s** while average completion latency only moves from **1.73 s to 2.28 s**.

At higher concurrency, the tradeoff becomes much clearer. From **128 to 256 concurrent requests**, throughput rises from **2,929 to 3,817 tok/s** — only about a **30% gain** — while average TTFT rises from **500 ms to 810 ms** (about **62%**) and average total latency rises from **2.73 s to 4.22 s** (about **55%**).

That region is the first clear saturation knee for this specific workload. The GPU is still producing more total tokens at concurrency 256, but each extra unit of concurrency buys substantially less throughput and costs substantially more user-visible latency.

TTFT is approximately the sum of queue/scheduler delay, prompt tokenization, prefill, first-token decode, and streaming overhead. In this local benchmark, tokenization and network overhead are small, so the increase in TTFT at high concurrency is primarily a sign that requests spend longer waiting for GPU scheduling and prefill/first-token work.

The saturation point is not a universal property of vLLM. It depends on the complete workload and hardware combination, including model size, precision, prompt length, output length, GPU compute, GPU memory bandwidth, VRAM capacity, and serving configuration.

## Engineering takeaway

Production inference optimization is a multi-objective problem:

- **throughput** — total tokens served per second,
- **TTFT** — how quickly users see the response begin,
- **TPOT** — how quickly tokens stream after the first token,
- **tail latency** — what slower users experience,
- **KV-cache / VRAM capacity** — how many active sequences can fit.

The goal is not maximum throughput at any cost. It is to find the highest sustainable concurrency that still satisfies latency and reliability SLAs.

For this A40 + Mistral-7B BF16 + 64-token-output workload, the measurements suggest that **128 concurrent requests are still efficient, while moving to 256 enters a noticeably less efficient region**. The exact production operating point would depend on the required TTFT and latency SLA.

`RESULTS.md` is retained as the raw experiment record; this README is the canonical lesson summary.
