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

Quantization does **not** reduce the model's logical parameter count. It stores those parameters more compactly, usually with scale/metadata and specialized kernels. Packed quantized tensors can make a naive `sum(p.numel())` report fewer stored tensor elements, so that value is not directly comparable to the original logical parameter count.

Lower precision guarantees lower weight-memory usage, but does not guarantee a proportional speedup. Runtime performance depends on the GPU architecture, kernel implementation, batch size, and whether the workload is limited by compute, memory bandwidth, or capacity.

## Lab setup

Use an NVIDIA GPU pod with a CUDA-capable PyTorch image. From the terminal:

```bash
cd /llm-inference-lab
git pull
./run.sh
```

The default workload for all three baselines is:

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

Run INT4 NF4 weights:

```bash
python 06_quantization/int4_nf4_baseline.py
```

Run the BF16 vs INT4 quality comparison:

```bash
python 06_quantization/quality_compare.py
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

| Metric | BF16 | INT8 weights | INT4 NF4 |
|---|---:|---:|---:|
| Logical model | Mistral-7B | Mistral-7B | Mistral-7B |
| CUDA memory after load | 13.500 GiB | 7.006 GiB | 3.858 GiB |
| CUDA reserved after load | 13.504 GiB | — | 6.734 GiB |
| Peak GPU memory | 13.747 GiB | 7.692 GiB | 4.195 GiB |
| Peak above loaded model | 252.7 MiB | 702.9 MiB | 344.7 MiB |
| Elapsed for 64 output tokens | 1.905 s | 7.524 s | 2.304 s |
| Output throughput | 33.59 tok/s | 8.51 tok/s | 27.77 tok/s |

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

## INT4 NF4 baseline

```text
=== Model load ===
Parameters: 3,758,362,624
Model reported footprint: 3.750 GiB
CUDA allocated after load: 3.858 GiB
CUDA reserved after load: 6.734 GiB
Load time: 4.40 s

=== Workload ===
Prompt tokens: 1024
Requested output tokens: 64

=== Measured generation ===
Generated tokens: 64
Elapsed: 2.304 s
Output throughput: 27.77 tok/s
Peak CUDA allocated: 4.195 GiB
Peak above loaded model: 344.7 MiB
Allocated at measured-run start: 3.866 GiB
```

Compact record:

```text
precision=INT4_NF4 params=3758362624 model_alloc_gib=3.858 prompt_tokens=1024 output_tokens=64 elapsed_s=2.304 output_tok_s=27.77 peak_gib=4.195
```

The `3,758,362,624` value above is a count of the packed tensors exposed by the quantized implementation, not evidence that the logical model became a 3.8B model. It is still the Mistral-7B model; the weights are represented in a packed 4-bit form.

## Quality comparison: BF16 vs INT4 NF4

Both variants were run with deterministic greedy decoding on the same four prompts covering KV cache, VRAM arithmetic, model-state concepts, and grouped-query attention.

### Observed result

There was **no clear evidence that INT4 was materially worse than BF16 on this small sample**.

- **KV-cache prompt:** both BF16 and INT4 incorrectly interpreted “KV cache” initially as a generic key-value application cache rather than transformer key/value state. This is a model/prompting failure shared by both precisions, not evidence of quantization damage.
- **VRAM arithmetic:** INT4 was actually more direct. It correctly computed `46 - 13.5 = 32.5 GiB` available for request KV cache before the output truncation. The BF16 answer took an unnecessary GiB-to-MB detour and had not reached the final concurrency number within the captured output.
- **Weights / activations / KV cache:** both answers were broadly reasonable but incomplete because the generation limit cut them off before finishing the KV-cache explanation.
- **Grouped-query attention:** both correctly recognized that fewer KV heads reduce memory by sharing K/V projections across query heads. Both outputs were truncated before explicitly stating the full `32 / 8 = 4×` KV-head memory reduction.

### Important limitation

This is **not a rigorous quality benchmark**. Four prompts are far too few to establish that INT4 has no quality loss. Also, several responses were truncated by the generation-token limit, which makes completeness difficult to compare.

The useful systems conclusion from this experiment is narrower:

> On these four deterministic prompts, aggressive INT4 NF4 quantization produced no obvious qualitative collapse relative to BF16. The dominant errors were shared by both versions.

A real quality study would use a larger task set with objective scoring (for example math/reasoning accuracy, code tests, factual QA, or perplexity), rather than judging a handful of free-form responses.

## Interpretation

The memory reduction is dramatic:

```text
BF16  13.500 GiB
INT8   7.006 GiB
INT4   3.858 GiB
```

INT4 NF4 reduced live CUDA model allocation by about **71% versus BF16** while retaining **27.77 tok/s**, about **83% of the BF16 batch-1 throughput** in this experiment. In contrast, the bitsandbytes INT8 path reached only **8.51 tok/s**.

This demonstrates why precision alone does not predict speed. The INT4 kernel path happened to be much more efficient than the INT8 path on this A40/workload, even though INT4 performs more aggressive compression.

`CUDA reserved after load` is larger than live allocation because PyTorch's caching allocator has claimed reusable memory blocks from CUDA. For INT4, about **3.858 GiB** was actively allocated to live tensors after load while **6.734 GiB** was held by the allocator. Reserved memory is allocator capacity, not all live model data.

The first quality check did not show an obvious INT4 degradation, but it also highlighted why quality needs a structured benchmark rather than a few subjective prompts.

## Lesson takeaway

Quantization changes three different dimensions that must be measured independently:

```text
precision ↓
   ├── model memory ↓ dramatically
   ├── speed → depends on kernels/hardware/workload
   └── quality → must be benchmarked; cannot be inferred from bit width alone
```

For this A40 experiment, INT4 NF4 was the strongest capacity/performance tradeoff of the three tested representations.

GitHub remains the canonical record for the lab.
