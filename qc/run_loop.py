"""CLI: the full Benders loop — GM-QAOA master + Gurobi subproblem — with ground truth.

Usage:
    uv run python -m qc.run_loop                       # default window (natural outage)
    uv run python -m qc.run_loop --start 0 --slots 2   # online-only window
    uv run python -m qc.run_loop --skip-brute-force    # loop only, no exact reference
"""

from __future__ import annotations

import argparse

import numpy as np

from qc.benders import benders_loop, brute_force_optimum
from qc.instance import decode, load_instance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid Benders loop for microgrid dispatch")
    ap.add_argument("--data", default="artifacts/data/all_data.csv")
    ap.add_argument("--start", type=int, default=645,
                    help="window start slot (default 645: contains the first natural outage)")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--force-outage", type=int, default=None, metavar="T",
                    help="pin grid_available[T]=0 inside the window")
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--gap-tol", type=float, default=1e-4)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-brute-force", action="store_true",
                    help="skip the exact enumeration reference (one LP per feasible state)")
    args = ap.parse_args(argv)

    inst = load_instance(args.data, start=args.start, T=args.slots,
                         force_outage=args.force_outage)
    print(f"instance: T={inst.T}, n_bits={inst.n_bits}, g_avail={inst.g_avail.tolist()}")

    result = benders_loop(inst, max_rounds=args.max_rounds, gap_tol=args.gap_tol,
                          shots=args.shots, seed=args.seed)

    print(f"\n{'rnd':>3} {'z':>8} {'subproblem':<11} {'Q(z)':>10} {'UB':>10} "
          f"{'LB':>10} {'gap':>10} {'|F|':>5} {'cut':>4}")
    for r in result.rounds:
        q = f"{r.q:.2f}" if r.q is not None else "—"
        lb = f"{r.lb:.2f}" if np.isfinite(r.lb) else "-inf"
        gap = f"{r.gap:.4f}" if np.isfinite(r.gap) else "inf"
        removed = f"-{r.n_removed}" if r.n_removed else "opt"
        print(f"{r.round:>3} {r.z:>8} {r.status:<11} {q:>10} {r.ub:>10.2f} "
              f"{lb:>10} {gap:>10} {r.n_states:>5} {removed:>4}")

    print(f"\ntermination: {result.termination} after {len(result.rounds)} rounds, "
          f"gap = {result.gap:.6f}")
    if result.best_z is None:
        print("no feasible configuration found (discrete problem infeasible)")
        return 1

    print(f"best: z={result.best_z}, total = {result.best_value:.2f} $ "
          f"(LB {result.lb:.2f} $)")
    for t, slot in enumerate(decode(result.best_z, inst)):
        print(f"  slot {t} ({'online' if inst.g_avail[t] else 'OUTAGE'}): {slot}")
    x = result.best_x
    print(f"  x*: imp={np.round(x['p_imp'], 1)} exp={np.round(x['p_exp'], 1)} "
          f"ch={np.round(x['p_ch'], 1)} dis={np.round(x['p_dis'], 1)} "
          f"soc={np.round(x['soc'], 1)} peak={x['p_peak']:.1f}")

    if not args.skip_brute_force:
        z_star, v_star, _ = brute_force_optimum(inst)
        ok = abs(result.best_value - v_star) <= max(10 * args.gap_tol, 1e-3)
        print(f"\nground truth (brute force over all feasible z): {v_star:.2f} $ "
              f"(z*={z_star}) -> loop {'MATCHES' if ok else 'DIFFERS'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
