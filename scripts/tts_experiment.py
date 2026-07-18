"""Measure S(T) = sum over Benders rounds of shots_99 — the TTS shot budget.

Runs the hybrid loop with sampler="exact" (the shots->infinity limit: each
round takes the exact master argmin, i.e. classical exact Benders), so the
round trajectory is deterministic per (T, window) — no seeds, no error bars
from sampling. Along that ideal trajectory each round records the exact
QAOA probability p_opt of hitting a master-argmin state and the implied
shots_99 = ln(0.01)/ln(1 - p_opt). Then

    S(T)   = sum_r shots_99(r)                  (measured here, per window)
    TTS(T) = S(T) x t_circuit(T)                (t_circuit analytic, qc/depth.py)

The LP-subproblem time per round is deliberately NOT counted: it is
polynomial in T and asymptotically negligible against the exponential shot
budget — the conservative assumption in favor of the quantum side.

Windows per T: one outage window (slot 646 = last window slot) plus online
windows at fixed starts, all from the standard data CSV.

Run:  uv run python -m scripts.tts_experiment --ts 1 2 3 4
      -> artifacts/tts_scaling.csv (per-run summary)
         artifacts/tts_rounds.csv  (per-round detail)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from qc.benders import benders_loop
from qc.instance import load_instance

DATA = "artifacts/data/all_data.csv"
OUT_SUMMARY = Path("artifacts/tts_scaling.csv")
OUT_ROUNDS = Path("artifacts/tts_rounds.csv")

ONLINE_STARTS = [0, 200, 400]   # online-only windows (grid available throughout)
OUTAGE_SLOT = 646               # first natural outage slot in the dataset


def run_one(t: int, start: int, kind: str, data: str, max_rounds: int,
            gap_tol: float, p: int) -> tuple[dict, list[dict]]:
    inst = load_instance(data, start=start, T=t)
    t0 = time.perf_counter()
    result = benders_loop(inst, max_rounds=max_rounds, gap_tol=gap_tol,
                          p=p, sampler="exact")
    wall = time.perf_counter() - t0

    detail = [{
        "t": t, "start": start, "window": kind, "round": r.round,
        "status": r.status, "n_states": r.n_states + r.n_removed,
        "p_opt": r.p_opt, "shots_99": r.shots_99,
    } for r in result.rounds]

    summary = {
        "t": t, "start": start, "window": kind,
        "n_states_initial": detail[0]["n_states"] if detail else 0,
        "rounds": len(result.rounds),
        "S": sum(r.shots_99 for r in result.rounds),
        "termination": result.termination,
        "gap": result.gap,
        "best_value": result.best_value,
        "sim_wall_s": wall,
    }
    return summary, detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="T values (default 1-5; T=5 peaks at ~5 GB for the bit matrix)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--gap-tol", type=float, default=1e-4)
    ap.add_argument("--p", type=int, default=6, help="QAOA layers (default 6)")
    args = ap.parse_args(argv)

    summaries, details = [], []
    for t in args.ts:
        windows = [(OUTAGE_SLOT - t + 1, "outage")] + \
                  [(s, "online") for s in ONLINE_STARTS]
        for start, kind in windows:
            summary, detail = run_one(t, start, kind, args.data,
                                      args.max_rounds, args.gap_tol, args.p)
            summaries.append(summary)
            details.extend(detail)
            print(f"T={t} start={start:>4} ({kind:>6}): rounds={summary['rounds']:>2} "
                  f"S={summary['S']:>10.1f} termination={summary['termination']} "
                  f"({summary['sim_wall_s']:.1f}s)")

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((OUT_SUMMARY, summaries), (OUT_ROUNDS, details)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
