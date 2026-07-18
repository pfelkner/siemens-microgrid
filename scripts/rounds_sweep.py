"""Sweep the hybrid Benders loop over T and max_rounds, aggregating trial statistics.

For each T, runs `--trials` independent benders_loop() trials at each
max_rounds budget (same seed set reused across budgets for a given T) and
aggregates against the exact brute_force_optimum() reference. Produces
artifacts/qc_rounds_sweep.csv with columns:

  t, n_feasible, max_rounds, cost_classical,
  cost_quantum_mean, cost_quantum_std,
  approx_ratio_mean, approx_ratio_std,
  converged_frac, rounds_mean, runtime_trial_mean_s,
  n_trials_ok, n_trials

converged_frac / rounds_mean are computed over all trials (a trial that
never converges runs the full max_rounds budget). cost_quantum/approx_ratio
mean+std are computed only over "ok" trials (best_z found, i.e. at least one
feasible discrete configuration was solved) -- a trial that stays infeasible
throughout has no cost_quantum to average.

Run:  uv run python -m scripts.rounds_sweep --ts 1 2 3 4 5 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from qc.benders import benders_loop, brute_force_optimum
from qc.grover_mixer import enumerate_feasible
from qc.instance import load_instance

DEFAULT_TS = [1, 2, 3, 4, 5]
DEFAULT_ROUNDS_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
OUT_CSV = Path("artifacts/qc_rounds_sweep.csv")


def _run_trial(inst, max_rounds: int, gap_tol: float, shots: int, seed: int) -> dict:
    t0 = time.perf_counter()
    result = benders_loop(inst, max_rounds=max_rounds, gap_tol=gap_tol,
                          shots=shots, seed=seed)
    return {
        "runtime": time.perf_counter() - t0,
        "n_rounds": len(result.rounds),
        "converged": result.termination == "gap",
        "best_value": result.best_value if result.best_z is not None else None,
        "pid": os.getpid(),
    }


def sweep_t(t: int, rounds_grid: list[int], n_trials: int, gap_tol: float,
           shots: int, data: str, start: int, executor: ProcessPoolExecutor | None) -> list[dict]:
    inst = load_instance(data, start=start, T=t)
    n_feasible = len(enumerate_feasible(inst))
    _, cost_classical, _ = brute_force_optimum(inst, executor=executor)

    rows = []
    for max_rounds in rounds_grid:
        costs_ok, ratios_ok, rounds_used, runtimes, converged_count, n_ok = [], [], [], [], 0, 0

        if executor is not None:
            trial_results = list(executor.map(
                _run_trial, [inst] * n_trials, [max_rounds] * n_trials,
                [gap_tol] * n_trials, [shots] * n_trials, range(n_trials)))
        else:
            trial_results = [_run_trial(inst, max_rounds, gap_tol, shots, trial)
                             for trial in range(n_trials)]

        pids = set()
        for seed, tr in enumerate(trial_results):
            runtimes.append(tr["runtime"])
            rounds_used.append(tr["n_rounds"])
            pids.add(tr["pid"])
            if tr["converged"]:
                converged_count += 1
            if tr["best_value"] is not None:
                n_ok += 1
                costs_ok.append(tr["best_value"])
                ratios_ok.append(tr["best_value"] / cost_classical)
            if t >= 4:
                print(f"    T={t} max_rounds={max_rounds:>3} seed={seed:>2} "
                      f"rounds={tr['n_rounds']:>3} converged={tr['converged']} "
                      f"best_value={tr['best_value']} runtime={tr['runtime']:.3f}s "
                      f"pid={tr['pid']}", flush=True)

        rows.append({
            "t": t,
            "n_feasible": n_feasible,
            "max_rounds": max_rounds,
            "cost_classical": cost_classical,
            "cost_quantum_mean": float(np.mean(costs_ok)) if costs_ok else float("nan"),
            "cost_quantum_std": float(np.std(costs_ok)) if costs_ok else float("nan"),
            "approx_ratio_mean": float(np.mean(ratios_ok)) if ratios_ok else float("nan"),
            "approx_ratio_std": float(np.std(ratios_ok)) if ratios_ok else float("nan"),
            "converged_frac": converged_count / n_trials,
            "rounds_mean": float(np.mean(rounds_used)),
            "rounds_std": float(np.std(rounds_used)),
            "runtime_trial_mean_s": float(np.mean(runtimes)),
            "n_trials_ok": n_ok,
            "n_trials": n_trials,
        })
        print(f"  T={t} max_rounds={max_rounds:>3} converged={converged_count}/{n_trials} "
              f"ok={n_ok}/{n_trials} runtime/trial={np.mean(runtimes):.3f}s "
              f"n_workers_used={len(pids)}", flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Benders round-budget sweep over T")
    p.add_argument("--ts", type=int, nargs="+", default=DEFAULT_TS)
    p.add_argument("--rounds-grid", type=int, nargs="+", default=DEFAULT_ROUNDS_GRID)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--gap-tol", type=float, default=1e-4)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--data", default="artifacts/data/all_data.csv")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output", default=str(OUT_CSV))
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes for trials (1 = sequential, no pool)")
    args = p.parse_args()

    out = Path(args.output)
    existing_rows = []
    if out.exists():
        with open(out, newline="") as f:
            existing_rows = list(csv.DictReader(f))
    existing_rows = [r for r in existing_rows if int(r["t"]) not in set(args.ts)]

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        new_rows = []
        for t in sorted(args.ts):
            print(f"[sweep] T={t}", flush=True)
            new_rows += sweep_t(t, args.rounds_grid, args.trials, args.gap_tol,
                                args.shots, args.data, args.start, executor)
    finally:
        if executor is not None:
            executor.shutdown()

    all_rows = existing_rows + [{k: v for k, v in r.items()} for r in new_rows]
    all_rows.sort(key=lambda r: (int(r["t"]), int(r["max_rounds"])))

    fieldnames = ["t", "n_feasible", "max_rounds", "cost_classical",
                 "cost_quantum_mean", "cost_quantum_std",
                 "approx_ratio_mean", "approx_ratio_std",
                 "converged_frac", "rounds_mean", "rounds_std", "runtime_trial_mean_s",
                 "n_trials_ok", "n_trials"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
