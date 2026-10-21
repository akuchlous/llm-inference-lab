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

This means total model parameters can be much larger than the parameters actively used for one token.

## Systems implications

MoE introduces new serving concerns:

- expert placement across GPUs
- token movement between GPUs
- interconnect bandwidth
- expert load imbalance
- queueing when popular experts become hotspots

The expert weights normally stay resident on their assigned GPUs. Hidden-state data is routed to the GPUs holding the selected experts; moving token representations is far cheaper than moving expert weights for every token.

## Lab 1 — Router and load balance

This first lab uses a toy PyTorch router rather than downloading a large MoE model. The goal is to make top-k routing and expert load visible.

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

The script prints the first few token routes and the number of assignments sent to each expert.

A second mode can deliberately create a hotspot by biasing one expert:

```bash
python 07_sparse_moe/router_balance.py --bias-expert 3
```

That will let us compare a relatively balanced router with an intentionally imbalanced one and see why expert load balancing matters for serving performance.

## What to observe

For top-2 routing with 10,000 tokens, there are 20,000 expert assignments total. With 8 experts, a perfectly balanced distribution would be about 2,500 assignments per expert.

The important measurements are:

```text
busiest expert / ideal
busiest expert / least-used expert
```

A large ratio means some expert — and potentially the GPU holding it — receives significantly more work than the others.

Measured output will be added here after the lab is run.
