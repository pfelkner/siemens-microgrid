"""CLI demo: instance -> feasible enumeration -> GM-QAOA -> exact-optimum comparison.

Round-1 view of the Benders master: the cost diagonal contains only the
direct (z-only) costs; optimality cuts would sharpen it in later rounds.

Usage:
    uv run python -m qc.run_poc
    uv run python -m qc.run_poc --start 645 --slots 2            # natural outage at slot 646
    uv run python -m qc.run_poc --force-outage 1 --p-values 2 4 6 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qc.grover_mixer import enumerate_feasible, expected_feasible_count
from qc.instance import decode, direct_costs, int_to_bits, load_instance
from qc.qaoa import gm_qaoa, sample_best


def main() -> int:
    ap = argparse.ArgumentParser(description="GM-QAOA PoC for microgrid dispatch master problem")
    ap.add_argument("--data", default="artifacts/data/all_data.csv")
    ap.add_argument("--start", type=int, default=645,
                    help="window start slot (default 645: contains the first natural outage)")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--force-outage", type=int, default=None, metavar="T",
                    help="pin grid_available[T]=0 inside the window")
    ap.add_argument("--p-values", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default="artifacts/results/qaoa_poc.png")
    args = ap.parse_args()

    inst = load_instance(args.data, start=args.start, T=args.slots,
                         force_outage=args.force_outage)
    print(f"instance: T={inst.T}, n_bits={inst.n_bits}, g_avail={inst.g_avail.tolist()}")

    feas = enumerate_feasible(inst)
    print(f"feasible states: {len(feas)} (expected {expected_feasible_count(inst)}) "
          f"of {2 ** inst.n_bits}")

    costs = direct_costs(int_to_bits(feas, inst.n_bits), inst)
    c_min = costs.min()
    min_set = costs == c_min
    uniform_p = min_set.sum() / len(feas)
    print(f"direct-cost optimum: {c_min:.2f} $ ({min_set.sum()} states), "
          f"uniform baseline P(min set) = {uniform_p:.4f}")

    rng = np.random.default_rng(args.seed)
    probs_by_p = {}
    for p in args.p_values:
        probs = gm_qaoa(costs, p=p)
        probs_by_p[p] = probs
        print(f"p={p:>2}: P(min-cost set) = {probs[min_set].sum():.4f} "
              f"(amplification x{probs[min_set].sum() / uniform_p:.2f})")

    p_final = max(probs_by_p, key=lambda p: float(probs_by_p[p][min_set].sum()))
    best = sample_best(probs_by_p[p_final], feas, costs, rng, shots=args.shots)
    best_cost = costs[np.where(feas == best)[0][0]]
    print(f"\nsampled best (best p={p_final}, {args.shots} shots): z={best} "
          f"cost={best_cost:.2f} $ (exact optimum {c_min:.2f} $)")
    for t, slot in enumerate(decode(best, inst)):
        print(f"  slot {t} ({'online' if inst.g_avail[t] else 'OUTAGE'}): {slot}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        probs = probs_by_p[p_final]
        ax.scatter(costs, probs, s=12, alpha=0.6, label=f"GM-QAOA p={p_final}")
        ax.axhline(1.0 / len(feas), color="gray", ls="--", lw=1, label="uniform 1/|F|")
        ax.set_xlabel("direct cost of z ($)")
        ax.set_ylabel("probability")
        ax.set_title(f"GM-QAOA distribution over {len(feas)} feasible states "
                     f"(T={inst.T}, g_avail={inst.g_avail.tolist()})")
        ax.legend()
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"plot saved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
