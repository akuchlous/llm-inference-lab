# Lesson 06 — Quantization

## Question

What do we gain, and what do we lose, when we reduce model precision?

The experiments in this lesson compare the same model family and workload across precisions so that memory, latency, throughput, capacity, and quality tradeoffs can be measured rather than assumed.

## Mental model

Quantization reduces the number of bits used to represent model values.

```text
BF16 = 16 bits/value = 2 bytes/value
INT8 =  8 bits/value = 1 byte/value
INT4 =  4 bits/value = 0.5 byte/value
```

A 7B-parameter model therefore has a rough raw-weight footprint of:

```text
BF16 ~14 GB
INT8  ~7 GB
INT4 ~3.5 GB
```

Quantization does **not** reduce the parameter count. It stores those parameters more compactly, usually with scale/metadata and specialized kernels.

Lower precision guarantees lower weight-memory usage, but does not guarantee a proportional speedup. Runtime performance depends on the GPU architecture, kernel implementation, batch size, and whether the workload is limited by compute, memory bandwidth, or capacity.

## Lab 1 — BF16 baseline

The first experiment establishes a control group before introducing quantization.

### Runpod setup

Use an NVIDIA GPU pod with a CUDA-capable PyTorch image. From the terminal:

```bash
cd /llm-inference-lab
git pull
pip install -r requirements.txt
```

Then run:

```bash
python 06_quantization/bf16_baseline.py
```

The default workload is:

- model: `mistralai/Mistral-7B-Instruct-v0.3`
- precision: BF16
- prompt: 1024 tokens
- output: 64 tokens
- batch: 1

You can override the workload, for example:

```bash
python 06_quantization/bf16_baseline.py \
  --prompt-tokens 4096 \
  --max-new-tokens 128
```

### What the script records

The script prints:

- GPU name
- parameter count
- raw parameter memory
- CUDA memory allocated after model load
- CUDA memory reserved after model load
- model load time
- prompt and output token counts
- measured generation time
- output tokens/second
- peak CUDA allocated memory
- peak allocation above the loaded model

The last line is a compact baseline record intended to make BF16/INT8/INT4 comparison easy.

## Experimental rule

When comparing quantization formats, keep these fixed unless the experiment explicitly changes one:

```text
same GPU
same model family
same prompt token count
same output token count
same batch/concurrency
same serving path
```

Only precision/quantization should change.

## Planned comparison

| Metric | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| Parameter memory | measure | — | — |
| CUDA memory after load | measure | — | — |
| Peak GPU memory | measure | — | — |
| Output tok/s | measure | — | — |
| TTFT / TPOT | later serving lab | — | — |
| Maximum practical concurrency | later serving lab | — | — |
| Quality | baseline | compare | compare |

## Next experiment

After the BF16 numbers are recorded, add one quantized representation and run the exact same workload. Do not change multiple variables at once.

## Results

Run the baseline on the GPU and record the measured output here. GitHub remains the canonical record for the lab.
