import argparse
from collections import Counter

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy top-k MoE router / load-balance experiment")
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--bias-expert",
        type=int,
        default=None,
        help="Optionally bias one expert index to create an artificial hotspot.",
    )
    parser.add_argument("--bias", type=float, default=2.5)
    args = parser.parse_args()

    if args.top_k > args.experts:
        raise ValueError("top-k cannot exceed number of experts")

    torch.manual_seed(args.seed)

    # Pretend these are token hidden states from a transformer layer.
    hidden = torch.randn(args.tokens, args.hidden_size)

    # A real MoE router is a learned projection from hidden state -> expert scores.
    router = torch.nn.Linear(args.hidden_size, args.experts, bias=True)

    with torch.no_grad():
        if args.bias_expert is not None:
            if not (0 <= args.bias_expert < args.experts):
                raise ValueError("--bias-expert must be a valid expert index")
            router.bias[args.bias_expert] += args.bias

        logits = router(hidden)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_experts = torch.topk(probs, k=args.top_k, dim=-1)

    counts = Counter(top_experts.flatten().tolist())
    total_assignments = args.tokens * args.top_k
    ideal = total_assignments / args.experts

    print("=== MoE router experiment ===")
    print(f"tokens={args.tokens} hidden_size={args.hidden_size} experts={args.experts} top_k={args.top_k}")
    if args.bias_expert is not None:
        print(f"artificial hotspot: expert={args.bias_expert} bias=+{args.bias}")
    else:
        print("artificial hotspot: none")

    print("\nFirst 8 token routes:")
    for i in range(min(8, args.tokens)):
        pairs = [
            f"E{expert}({prob:.3f})"
            for expert, prob in zip(top_experts[i].tolist(), top_probs[i].tolist())
        ]
        print(f"token {i:>2}: " + ", ".join(pairs))

    print("\nExpert assignment counts:")
    max_count = 0
    min_count = total_assignments
    for expert in range(args.experts):
        count = counts.get(expert, 0)
        max_count = max(max_count, count)
        min_count = min(min_count, count)
        pct = 100.0 * count / total_assignments
        vs_ideal = count / ideal if ideal else 0.0
        print(f"Expert {expert}: {count:>6} assignments  {pct:6.2f}%  ({vs_ideal:.2f}x ideal)")

    imbalance_ratio = max_count / ideal if ideal else 0.0
    spread_ratio = max_count / max(1, min_count)

    print("\nSummary:")
    print(f"total expert assignments: {total_assignments}")
    print(f"ideal assignments/expert: {ideal:.1f}")
    print(f"busiest expert / ideal: {imbalance_ratio:.2f}x")
    print(f"busiest / least-used expert: {spread_ratio:.2f}x")

    print("\nMental model:")
    print("token hidden state -> router scores -> top-k experts -> expert workload")
    print("If one expert receives far more assignments, it can become a serving hotspot.")


if __name__ == "__main__":
    main()
