# Lesson 05 — vLLM Serving and Concurrency

## Question

How does a production-oriented serving engine such as vLLM use concurrent requests to improve aggregate GPU throughput?

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
```

## Measured output

| Concurrency | Elapsed | Output tokens | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 1.72 s | 64 | 37.19 tok/s |
| 2 | 1.84 s | 128 | 69.41 tok/s |
| 4 | 1.85 s | 256 | 138.19 tok/s |
| 8 | 1.89 s | 512 | 270.95 tok/s |

## Interpretation

Concurrency 8 delivered **270.95 tok/s**, about **7.3×** the throughput of concurrency 1 at **37.19 tok/s**.

Wall-clock time changed only modestly from **1.72 s to 1.89 s** while the output work increased 8×. The serving engine is using concurrent work to keep the GPU much more fully utilized.

This should not yet be read as a strict vLLM-vs-Hugging-Face speed comparison because the earlier experiments differ in prompt construction, execution path, and server overhead. The clean conclusion is that **vLLM scales aggregate throughput strongly with concurrent requests in this workload**.

## Engineering takeaway

Throughput is not the only serving metric. A production system must also watch per-request TTFT, TPOT, queueing latency, memory utilization, and tail latency as concurrency rises.

`RESULTS.md` is retained as the raw experiment record; this README is the canonical lesson summary.
