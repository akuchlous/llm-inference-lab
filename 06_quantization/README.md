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
| Parameter memory | 13.500 GiB | — | — |
| CUDA memory after load | 13.500 GiB | — | — |
| Peak GPU memory | 13.747 GiB | — | — |
| Output tok/s | 33.59 | — | — |
| TTFT / TPOT | later serving lab | — | — |
| Maximum practical concurrency | later serving lab | — | — |
| Quality | baseline | compare | compare |

## Measured BF16 baseline

Environment: same Mistral-7B model family used in earlier labs, BF16, batch 1.

```text
=== Model load ===
Parameters: 7,248,023,552
Parameter bytes: 13.500 GiB
CUDA allocated after load: 13.500 GiB
CUDA reserved after load: 13.504 GiB
Load time: 3.93 s

=== Workload ===
Prompt tokens: 1024
Requested output tokens: 64

=== Measured generation ===
Generated tokens: 64
Elapsed: 1.905 s
Output throughput: 33.59 tok/s
Peak CUDA allocated: 13.747 GiB
Peak above loaded model: 252.7 MiB
Allocated at measured-run start: 13.508 GiB
```

Compact record:

```text
precision=BF16 params=7248023552 param_gib=13.500 model_alloc_gib=13.500 prompt_tokens=1024 output_tokens=64 elapsed_s=1.905 output_tok_s=33.59 peak_gib=13.747
```

### Interpretation

The raw parameter calculation and CUDA allocation agree almost exactly: about **13.5 GiB** is occupied by the BF16 model weights. The 1K-context, 64-token generation peaked at **13.747 GiB**, or about **252.7 MiB above the loaded model**. Throughput for this batch-1 workload was **33.59 output tokens/s**.

These values are now the control group. The next experiment should change only the model-weight representation and repeat the same 1024-token prompt / 64-token output workload.

## Next experiment

Run an INT8-weight version of the same model/workload and compare model allocation, peak memory, and output throughput against this BF16 baseline.

GitHub remains the canonical record for the lab.
