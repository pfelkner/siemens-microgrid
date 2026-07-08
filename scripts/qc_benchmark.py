"""Benchmark GM-QAOA Benders loop: approximation ratio vs window size T.

For each T in 1..t_max, runs the Benders loop `--trials` times with different
seeds and compares against the Gurobi optimal. Reports the approximation ratio
ρ = cost_quantum / cost_classical alongside scaling metrics.

A random-feasible-state baseline is included for small T (|F| <= random_max)
to calibrate how much of the quantum advantage comes from above-chance selection.

GPU memory scales as 24 bytes * |F|: for all-online windows
  T=1: |F|=27  (~0 MB)   T=4: |F|=531k  (~12 MB)
  T=2: |F|=729 (~0 MB)   T=5: |F|=14.3M (~344 MB)
  T=3: |F|=19k (~0 MB)   T=6: |F|=387M  (~9 GB, likely OOM on 8 GB card)

Run:
    uv run python -m scripts.qc_benchmark --gpu
    uv run python -m scripts.qc_benchmark --gpu --t-max 5 --trials 30 --out my.csv
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from classical.deterministic_solver import build_and_solve
from qc.benders import benders_loop
from qc.grover_mixer import enumerate_feasible, expected_feasible_count
from qc.instance import Instance, direct_costs, int_to_bits
from subproblem.feasible_start_x import Params
from subproblem.subproblem import solve_subproblem
from qc.benders import build_sub_instance

FIELDS = [
    "t", "n_bits", "n_feasible", "vram_estimate_mb",
    "cost_classical", "runtime_gurobi_s",
    "cost_quantum_mean", "cost_quantum_std",
    "cost_random_mean", "cost_random_std",
    "approx_ratio_mean", "approx_ratio_std",
    "converged_frac", "rounds_mean",
    "runtime_trial_mean_s", "runtime_trial_std_s",
    "n_trials_ok", "n_trials",
]


def _gurobi_optimum(data: pd.DataFrame, t: int, start: int) -> tuple[float, float]:
    """Run Gurobi on the window; return (cost, runtime_s)."""
    window = data.iloc[start:start + t].reset_index(drop=True)
    t0 = time.perf_counter()
    model, _, _ = build_and_solve(
        [window], scenario_probs=None, time_limit=None,
        mip_gap=0.0, log_file="", quiet=True,
    )
    return float(model.ObjVal), time.perf_counter() - t0  # type: ignore[arg-type]


def _random_baseline(inst: Instance, states: np.ndarray,
                     rng: np.random.Generator, n: int) -> tuple[float, float]:
    """Cost of n random feasible states via Gurobi subproblem; return (mean, std)."""
    bits = int_to_bits(states, inst.n_bits)
    direct = direct_costs(bits, inst)
    params = Params()
    costs = []
    idx = rng.integers(0, len(states), size=n)
    for i in idx:
        sub = build_sub_instance(inst, int(states[i]), params)
        res = solve_subproblem(sub)
        if res.feasible:
            costs.append(float(direct[i]) + float(res.q_value or 0.0))
    if not costs:
        return float("nan"), float("nan")
    return float(np.mean(costs)), float(np.std(costs))


def run_t(
    t: int, start: int, data: pd.DataFrame,
    trials: int, max_rounds: int, shots: int, p: int,
    use_gpu: bool, random_n: int, seed: int,
) -> dict:
    chunk = data.iloc[start:start + t].reset_index(drop=True)
    g = chunk["grid_available"].to_numpy(dtype=int)
    inst = Instance(
        p_pv=chunk["p_kw"].to_numpy(dtype=float),
        p_load=chunk["load_kw"].to_numpy(dtype=float),
        tou=chunk["tou_usd_kwh"].to_numpy(dtype=float),
        g_avail=g,
    )

    n_feasible = expected_feasible_count(inst)
    vram_mb = n_feasible * 24 / 1024 ** 2

    print(f"  T={t}  |F|={n_feasible:,}  vram≈{vram_mb:.1f} MB", flush=True)

    # classical optimum
    cost_cl, t_gurobi = _gurobi_optimum(data, t, start)
    print(f"    Gurobi: {cost_cl:.4f} $ in {t_gurobi:.2f}s", flush=True)

    # random baseline (only for small instances)
    rng_base = np.random.default_rng(seed)
    if n_feasible <= random_n:
        states = enumerate_feasible(inst)
        cost_rand_mean, cost_rand_std = _random_baseline(inst, states, rng_base, min(50, n_feasible))
    else:
        cost_rand_mean, cost_rand_std = float("nan"), float("nan")

    # quantum trials
    q_costs, q_rounds, q_times = [], [], []
    converged = 0
    for trial in range(trials):
        t0 = time.perf_counter()
        try:
            result = benders_loop(
                inst, max_rounds=max_rounds, gap_tol=1e-4,
                shots=shots, seed=seed + trial, p=p, use_gpu=use_gpu,
            )
        except Exception as exc:
            print(f"    trial {trial} FAILED: {exc}", flush=True)
            if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                raise  # propagate OOM to caller
            continue
        elapsed = time.perf_counter() - t0
        if result.best_z is not None:
            q_costs.append(result.best_value)
            q_rounds.append(len(result.rounds))
            q_times.append(elapsed)
            if result.termination == "gap":
                converged += 1
        print(f"    trial {trial+1}/{trials}: cost={result.best_value:.4f}  "
              f"rounds={len(result.rounds)}  term={result.termination}  "
              f"{elapsed:.2f}s", flush=True)

    n_ok = len(q_costs)
    if n_ok == 0:
        q_mean = q_std = approx_mean = approx_std = float("nan")
        rounds_mean = rt_mean = rt_std = float("nan")
    else:
        q_mean = float(np.mean(q_costs))
        q_std = float(np.std(q_costs))
        ratios = [c / cost_cl for c in q_costs] if cost_cl != 0 else [float("nan")] * n_ok
        approx_mean = float(np.mean(ratios))
        approx_std = float(np.std(ratios))
        rounds_mean = float(np.mean(q_rounds))
        rt_mean = float(np.mean(q_times))
        rt_std = float(np.std(q_times))

    return {
        "t": t,
        "n_bits": inst.n_bits,
        "n_feasible": n_feasible,
        "vram_estimate_mb": round(vram_mb, 2),
        "cost_classical": round(cost_cl, 6),
        "runtime_gurobi_s": round(t_gurobi, 3),
        "cost_quantum_mean": round(q_mean, 6) if not np.isnan(q_mean) else "nan",
        "cost_quantum_std": round(q_std, 6) if not np.isnan(q_std) else "nan",
        "cost_random_mean": round(cost_rand_mean, 6) if not np.isnan(cost_rand_mean) else "nan",
        "cost_random_std": round(cost_rand_std, 6) if not np.isnan(cost_rand_std) else "nan",
        "approx_ratio_mean": round(approx_mean, 6) if not np.isnan(approx_mean) else "nan",
        "approx_ratio_std": round(approx_std, 6) if not np.isnan(approx_std) else "nan",
        "converged_frac": round(converged / trials, 4) if trials > 0 else "nan",
        "rounds_mean": round(rounds_mean, 2) if not np.isnan(rounds_mean) else "nan",
        "runtime_trial_mean_s": round(rt_mean, 3) if not np.isnan(rt_mean) else "nan",
        "runtime_trial_std_s": round(rt_std, 3) if not np.isnan(rt_std) else "nan",
        "n_trials_ok": n_ok,
        "n_trials": trials,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="artifacts/data/all_data.csv")
    ap.add_argument("--start", type=int, default=0,
                    help="window start slot in the data CSV (default 0 = all online)")
    ap.add_argument("--t-max", type=int, default=5,
                    help="largest window size to benchmark (default 5)")
    ap.add_argument("--trials", type=int, default=10,
                    help="independent Benders runs per T (different seeds)")
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--p", type=int, default=6, help="QAOA layers")
    ap.add_argument("--gpu", action="store_true", help="run QAOA on GPU via CuPy")
    ap.add_argument("--random-max", type=int, default=20_000,
                    help="|F| threshold below which the random baseline is computed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/results/qc_benchmark.csv")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    if args.start + args.t_max > len(data):
        ap.error(f"--start {args.start} + --t-max {args.t_max} exceeds data length {len(data)}")

    print(f"QC benchmark: T=1..{args.t_max}, {args.trials} trials, "
          f"{'GPU' if args.gpu else 'CPU'}, start={args.start}")

    rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for t in range(1, args.t_max + 1):
            print(f"\n[T={t}]", flush=True)
            try:
                row = run_t(
                    t=t, start=args.start, data=data,
                    trials=args.trials, max_rounds=args.max_rounds,
                    shots=args.shots, p=args.p,
                    use_gpu=args.gpu, random_n=args.random_max,
                    seed=args.seed,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                if "out of memory" in tb.lower() or "oom" in tb.lower():
                    print(f"  GPU OOM at T={t}, stopping sweep.")
                else:
                    print(f"  ERROR at T={t}: {exc}\n{tb}")
                break

            rows.append(row)
            writer.writerow(row)
            f.flush()

    if not rows:
        print("No results produced.")
        return 1

    print(f"\n=== Results -> {out_path} ===")
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 160,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(df[["t", "n_feasible", "cost_classical", "cost_quantum_mean",
                   "approx_ratio_mean", "approx_ratio_std",
                   "converged_frac", "rounds_mean", "runtime_trial_mean_s"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
