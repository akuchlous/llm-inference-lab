# Lesson 07 — Sparse Activation / Mixture of Experts

## Core idea

A dense transformer sends every token through the same feed-forward sub-network. A Mixture-of-Experts (MoE) layer contains multiple expert feed-forward networks plus a learned router.

For each token, the router scores all experts and selects only the top-k experts.

```text
token hidden state
      ↓
    router
      ↓
expert scores
      ↓
 top-k experts
      ↓
expert outputs combined
```

This means:

```text
total model parameters != active parameters per token
```

A model can therefore have much larger total capacity without activating every expert for every token.

## Important routing mental model

Routing is normally **per token, per MoE layer** — not per prompt.

For example, tokens from one prompt might route like this at one MoE layer:

```text
"The" -> E2 + E6
"cat" -> E1 + E4
"sat" -> E6 + E5
"on"  -> E2 + E0
```

With top-2 routing, each token is assigned to two experts.

The same token may also choose different experts at later MoE layers because its hidden representation changes as it passes through the transformer:

```text
token "cat"

MoE layer 1 -> E2 + E5
MoE layer 2 -> E1 + E7
MoE layer 3 -> E4 + E6
```

## How the router learns

The router is a learned part of the model. During training it learns a mapping roughly like:

```text
token hidden state
-> router scores
-> top-k experts
```

The router is not normally programmed with human rules such as `math -> expert 3`. Expert specialization emerges during training.

Training also commonly includes load-balancing pressure so the router does not send nearly every token to the same expert.

## Expert parallelism across GPUs

Experts can be placed on different GPUs:

```text
GPU0 -> Experts 1, 2
GPU1 -> Experts 3, 4
GPU2 -> Experts 5, 6
GPU3 -> Experts 7, 8
```

If a token is currently on GPU0 but the router selects Expert 6 on GPU2, the hidden-state data is moved to GPU2, Expert 6 runs there, and its output is returned/combined.

The expert weights normally remain resident on their assigned GPUs. Moving hidden-state data is much cheaper than moving expert weights for every token.

This introduces new serving concerns:

- expert placement across GPUs
- token movement between GPUs
- interconnect bandwidth / NVLink
- expert load imbalance
- queueing when popular experts become hotspots

So MoE performance depends not only on FLOPs and memory bandwidth, but also on communication and routing balance.

## Lab 1 — Router and load balance

This lab uses a toy PyTorch router rather than downloading a large MoE model. The goal is to make top-k routing and expert load visible.

Run:

```bash
cd /llm-inference-lab
git pull
python 07_sparse_moe/router_balance.py
```

Default setup:

```text
10,000 token hidden states
512-dimensional hidden state
8 experts
top-2 routing
```

For top-2 routing with 10,000 tokens:

```text
10,000 tokens x 2 experts = 20,000 expert assignments
```

With 8 experts, perfect balance would be:

```text
20,000 / 8 = 2,500 assignments/expert
12.5% of assignments per expert
```

### Measured balanced-router run

```text
Expert 0: 2434 assignments   12.17%  (0.97x ideal)
Expert 1: 2419 assignments   12.10%  (0.97x ideal)
Expert 2: 2436 assignments   12.18%  (0.97x ideal)
Expert 3: 2141 assignments   10.71%  (0.86x ideal)
Expert 4: 2523 assignments   12.62%  (1.01x ideal)
Expert 5: 2571 assignments   12.86%  (1.03x ideal)
Expert 6: 2705 assignments   13.53%  (1.08x ideal)
Expert 7: 2771 assignments   13.86%  (1.11x ideal)

busiest expert / ideal: 1.11x
busiest / least-used expert: 1.29x
```

This run was reasonably balanced. No expert was dramatically hotter than the others.

## Lab 2 — Deliberately create a hotspot

Run:

```bash
python 07_sparse_moe/router_balance.py --bias-expert 3 --bias 2.0
```

This artificially makes Expert 3 more attractive to the router.

### Measured hotspot run

```text
Expert 0: 1342 assignments    6.71%  (0.54x ideal)
Expert 1: 1322 assignments    6.61%  (0.53x ideal)
Expert 2: 1359 assignments    6.79%  (0.54x ideal)
Expert 3: 9919 assignments   49.59%  (3.97x ideal)
Expert 4: 1440 assignments    7.20%  (0.58x ideal)
Expert 5: 1490 assignments    7.45%  (0.60x ideal)
Expert 6: 1525 assignments    7.62%  (0.61x ideal)
Expert 7: 1603 assignments    8.02%  (0.64x ideal)

busiest expert / ideal: 3.97x
busiest / least-used expert: 7.50x
```

Expert 3 received almost half of all assignments even though the ideal share was only 12.5%.

If Expert 3 lives on one GPU, that GPU can become the bottleneck while GPUs holding the other experts remain underused.

The key systems lesson is:

> MoE efficiency depends on routing balance, not just total available GPU compute.

A cluster can have plenty of aggregate FLOPs and still suffer poor latency/throughput if one expert becomes hot.

## Next lab — Expert capacity and overflow

The next experiment adds a finite processing capacity to each expert:

```bash
python 07_sparse_moe/expert_capacity.py --bias-expert 3 --bias 2.0
```

The question is what happens when a hot expert receives more token assignments than it can process in the current batch. Depending on the real MoE implementation, excess tokens may queue, be rerouted, or be dropped.

This lab has been prepared but not yet run.

## Current takeaways

```text
MoE = more total model capacity without activating every expert per token
routing = per token, per MoE layer
router = learned during training
experts can live on different GPUs
hidden states move to selected experts
cross-GPU communication becomes part of inference
unbalanced routing can create GPU hotspots
load balancing is therefore a core MoE systems problem
```
