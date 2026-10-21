import argparse
import math
from collections import Counter

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy MoE expert-capacity / overflow experiment")
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bias-expert", type=int, default=None)
    parser.add_argument("--bias", type=float, default=2.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    hidden = torch.randn(args.tokens, args.hidden_size)
    router = torch.nn.Linear(args.hidden_size, args.experts, bias=True)

    with torch.no_grad():
        if args.bias_expert is not None:
            router.bias[args.bias_expert] += args.bias
        probs = torch.softmax(router(hidden), dim=-1)
        _, top_experts = torch.topk(probs, k=args.top_k, dim=-1)

    counts = Counter(top_experts.flatten().tolist())
    total_assignments = args.tokens * args.top_k
    ideal = total_assignments / args.experts
    capacity = math.ceil(ideal * args.capacity_factor)

    print("=== MoE expert-capacity experiment ===")
    print(f"tokens={args.tokens} experts={args.experts} top_k={args.top_k}")
    print(f"ideal assignments/expert={ideal:.1f}")
    print(f"capacity_factor={args.capacity_factor:.2f}")
    print(f"capacity/expert={capacity}")

    if args.bias_expert is None:
        print("artificial hotspot: none")
    else:
        print(f"artificial hotspot: expert={args.bias_expert} bias=+{args.bias}")

    total_overflow = 0
    print("\nPer-expert load:")
    for expert in range(args.experts):
        count = counts.get(expert, 0)
        overflow = max(0, count - capacity)
        total_overflow += overflow
        print(
            f"Expert {expert}: assignments={count:>5}  capacity={capacity:>5}  "
            f"overflow={overflow:>5}"
        )

    overflow_pct = 100.0 * total_overflow / total_assignments
    print("\nSummary:")
    print(f"total assignments={total_assignments}")
    print(f"total overflow={total_overflow}")
    print(f"overflow share={overflow_pct:.2f}%")

    print("\nMental model:")
    print("router assigns tokens -> each expert has finite batch capacity")
    print("assignments above capacity must be queued, rerouted, or dropped depending on implementation")


if __name__ == "__main__":
    main()
