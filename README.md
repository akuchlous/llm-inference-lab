# LLM Inference Lab

A hands-on learning repository for understanding and optimizing LLM inference from the GPU up.

The goal is not just to learn how to call an LLM API. It is to build enough systems understanding to design, operate, secure, and optimize private enterprise LLM infrastructure.

## Learning path

### Inference fundamentals

**tokens → attention → KV cache → GPU memory → batching → throughput → latency → inference cost**

### One level higher

**model selection → routing → caching → context management → RAG → observability → evaluation → reliability → economics**

## Long-term goal

Be able to take an enterprise workload, security requirements, model choices, hardware budget, and SLA and:

- design a private/on-prem LLM inference platform,
- secure it appropriately,
- benchmark the workload,
- determine whether it is memory-, compute-, or communication-bound,
- tune serving and GPU configuration,
- and explain the quality, latency, throughput, reliability, and cost tradeoffs.

## Labs

| Lesson | Topic | Status |
|---|---|---|
| [01](01_gpu_memory/README.md) | GPU memory fundamentals | Complete |
| 02 | KV cache | Next |
| 03 | Prefill vs. decode | Planned |
| 04 | Batching and concurrency | Planned |
| 05 | vLLM serving | Planned |
| 06 | Quantization | Planned |

Each lesson gets its own `README.md` containing the question, mental model, experiment, measurements, interpretation, and engineering takeaway.

## Environment used for Lab 01

- NVIDIA A40
- Driver 580.159.03
- CUDA driver capability 13.0
- Python 3.12.3
- PyTorch 2.8.0+cu128
- Mistral 7B / Mistral 7B Instruct

## Workflow

GitHub is the source of truth. Code can be edited locally, pulled onto a disposable GPU instance such as Runpod, executed there, and the resulting measurements committed back to this repository.
