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

## Lab setup

Use an NVIDIA GPU pod with a CUDA-capable PyTorch image. From the terminal:

```bash
cd /llm-inference-lab
git pull
./run.sh
```

The default workload for both baselines is:

- model: `mistralai/Mistral-7B-Instruct-v0.3`
- prompt: 1024 tokens
- output: 64 tokens
- batch: 1

Run BF16:

```bash
python 06_quantization/bf16_baseline.py
```

Run INT8 weights:

```bash
python 06_quantization/int8_baseline.py
```

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

## Measured comparison

| Metric | BF16 | INT8 weights | INT4 |
|---|---:|---:|---:|
| Parameters | 7,248,023,552 | 7,248,023,552 | — |
| CUDA memory after load | 13.500 GiB | 7.006 GiB | — |
| Peak GPU memory | 13.747 GiB | 7.692 GiB | — |
| Peak above loaded model | 252.7 MiB | 702.9 MiB | — |
| Elapsed for 64 output tokens | 1.905 s | 7.524 s | — |
| Output throughput | 33.59 tok/s | 8.51 tok/s | — |

## BF16 baseline

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

## INT8-weight baseline

The INT8 experiment used bitsandbytes `load_in_8bit=True`. The weights are quantized, while the execution path remains mixed precision. On this A40 setup, bitsandbytes reported that BF16 inputs to `MatMul8bitLt` were cast to FP16 during the quantized matmul path.

```text
=== Measured generation ===
Generated tokens: 64
Elapsed: 7.524 s
Output throughput: 8.51 tok/s
Peak CUDA allocated: 7.692 GiB
Peak above loaded model: 702.9 MiB
Allocated at measured-run start: 7.014 GiB
```

Compact record:

```text
precision=INT8 params=7248023552 model_alloc_gib=7.006 prompt_tokens=1024 output_tokens=64 elapsed_s=7.524 output_tok_s=8.51 peak_gib=7.692
```

## Interpretation

The memory result is strong: INT8 reduced loaded-model CUDA allocation from **13.500 GiB to 7.006 GiB**, a reduction of about **48%**. Peak memory during generation fell from **13.747 GiB to 7.692 GiB**.

The speed result went in the opposite direction. For this batch-1 A40 + Hugging Face + bitsandbytes workload, output throughput fell from **33.59 tok/s to 8.51 tok/s**, or roughly **4× slower**.

This is the key systems lesson: **quantization is a representation and capacity optimization first; it is not automatically a latency or throughput optimization**. Real speed depends on whether the GPU and serving stack have efficient kernels for the chosen quantization format and workload.

The INT8 run also used more temporary memory above the loaded model (about **703 MiB** versus **253 MiB** for BF16), consistent with the fact that quantized execution still requires mixed-precision activations, temporary buffers, and quantization/dequantization work.

## Next experiment

Add an INT4-weight version of the same model/workload and measure whether the additional weight-memory reduction improves capacity, and what happens to batch-1 throughput on this A40.

GitHub remains the canonical record for the lab.
