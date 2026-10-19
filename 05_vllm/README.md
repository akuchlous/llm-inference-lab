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

## Throughput experiment

| Concurrency | Elapsed | Output tokens | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 1.72 s | 64 | 37.19 tok/s |
| 2 | 1.84 s | 128 | 69.41 tok/s |
| 4 | 1.85 s | 256 | 138.19 tok/s |
| 8 | 1.89 s | 512 | 270.95 tok/s |

Concurrency 8 delivered **270.95 tok/s**, about **7.3×** the throughput of concurrency 1 at **37.19 tok/s**.

Wall-clock time changed only modestly from **1.72 s to 1.89 s** while the output work increased 8×. The serving engine is using concurrent work to keep the GPU much more fully utilized.

This should not yet be read as a strict vLLM-vs-Hugging-Face speed comparison because the earlier experiments differ in prompt construction, execution path, and server overhead. The clean conclusion is that **vLLM scales aggregate throughput strongly with concurrent requests in this workload**.

## Latency experiment

| Concurrency | Avg TTFT | Avg total latency | P95 total latency |
|---:|---:|---:|---:|
| 1 | 93.63 ms | 1.77 s | 1.77 s |
| 2 | 52.21 ms | 1.83 s | 1.82 s |
| 4 | 72.90 ms | 1.86 s | 1.86 s |
| 8 | 92.32 ms | 1.89 s | 1.90 s |

### Interpretation

Over this small concurrency range, individual-request completion latency barely changed: average total latency moved from **1.77 s at concurrency 1** to **1.89 s at concurrency 8**, while aggregate throughput increased more than 7×.

TTFT stayed in roughly the **50–95 ms** range and did not increase monotonically. At these small absolute times, warm kernels, scheduler timing, request arrival alignment, and measurement noise can easily move TTFT by tens of milliseconds. The main result is therefore not that concurrency 2 is intrinsically faster than concurrency 1; it is that **TTFT remained low and broadly stable while throughput scaled strongly**.

This is a favorable operating region: the GPU has enough unused parallel capacity that adding concurrent requests increases utilization much more than it hurts user-visible latency.

At higher concurrency, the expectation is that the system will eventually reach saturation. At that point queueing grows, TTFT and tail latency rise, and throughput gains begin to flatten. Finding that saturation point is a core production capacity-planning exercise.

## Engineering takeaway

Production inference optimization is a multi-objective problem:

- **throughput** — total tokens served per second,
- **TTFT** — how quickly users see the response begin,
- **TPOT** — how quickly tokens stream after the first token,
- **tail latency** — what slower users experience,
- **KV-cache / VRAM capacity** — how many active sequences can fit.

The goal is not maximum throughput at any cost. It is to find the highest sustainable concurrency that still satisfies latency and reliability SLAs.

`RESULTS.md` is retained as the raw experiment record; this README is the canonical lesson summary.
