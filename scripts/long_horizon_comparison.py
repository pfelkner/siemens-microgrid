"""Heuristic vs. MILP over a long, realistic horizon (no hybrid -- out of scope).

Motivation: on an isolated T=5 window a single bad or lucky local decision
dominates the whole score, so r_greedy can swing to extremes (near 1, or even
negative). Over a long horizon those local mistakes average out and the
monthly demand charge is amortized over its real billing period (not a
75-minute slice), so this gives a more representative "how good is the
trivial controller in real operation" number.

The hybrid solver is NOT included here: it is capped at ~T=5 (statevector
simulation of 8*T qubits) and there is no rolling-horizon/MPC driver for it
yet (doc/aproximation_ratio_instruction.md SS6 explicitly flags that as a
separate, not-yet-built experiment). This script only compares the greedy
heuristic (classical/heuristic_dispatch.py, pure NumPy, no qubit limit)
against the classical MILP optimum.

Window: T=2880 (30 days), start=0. NOTE: a size-limited Gurobi license
(the free pip-wheel one, ~2000 vars/constraints) rejects models past
T~125 ("Model too large for size-limited license"); an unrestricted license
is required to actually solve this at T=2880.

Run:  .venv/bin/python -m scripts.long_horizon_comparison
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from classical.deterministic_solver import build_and_solve
from classical.heuristic_dispatch import approx_ratio, objective, passive, sweep_pstar
from subproblem.feasible_start_x import Params

DATA_CSV = "artifacts/data/all_data.csv"
OUT_CSV = Path("artifacts/results/long_horizon_comparison.csv")

START = 0
T = 2880      # 30 days at 15-min slots


def main() -> int:
    df = pd.read_csv(DATA_CSV).iloc[START:START + T].reset_index(drop=True)
    pv = df["p_kw"].to_numpy(float)
    load = df["load_kw"].to_numpy(float)
    tou = df["tou_usd_kwh"].to_numpy(float)
    ga = df["grid_available"].to_numpy(int)
    n_outage = int((ga == 0).sum())

    params = Params()
    c_ref = objective(passive(pv, load, ga, params), tou, params)["total"]
    _, best_obj, p_star_best = sweep_pstar(pv, load, ga, tou, params)
    c_greedy = best_obj["total"]

    _, info, _ = build_and_solve([df], mip_gap=1e-4, quiet=True)
    c_opt = info["total_cost"]

    r_greedy = approx_ratio(c_ref, c_greedy, c_opt)

    print(f"window: start={START}, T={T} ({T * 0.25 / 24:.0f} days), "
          f"{n_outage} outage slots")
    print(f"C_ref={c_ref:.2f}  C_greedy={c_greedy:.2f} (P*={p_star_best:.1f})  "
          f"C_opt={c_opt:.2f}  (mip_gap={info['mip_gap']:.2e}, "
          f"runtime={info['runtime_s']:.2f}s)")
    print(f"r_greedy = {r_greedy:.4f}  (gap to opt = {c_greedy - c_opt:+.2f} $)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "start": START, "T": T, "n_outage": n_outage,
        "C_ref": c_ref, "C_greedy": c_greedy, "C_opt": c_opt,
        "p_star_best": p_star_best, "r_greedy": r_greedy,
        "gap_greedy": c_greedy - c_opt, "milp_mip_gap": info["mip_gap"],
        "milp_runtime_s": info["runtime_s"],
    }
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
