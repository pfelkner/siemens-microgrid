"""Benchmark GM-QAOA Benders loop: approximation ratio vs window size T.

Two modes:

1. T-sweep (default): for each T in 1..t_max, runs the Benders loop `--trials`
   times and compares against Gurobi. Reports ρ = cost_quantum / cost_classical.
   Output: artifacts/results/qc_benchmark.csv

2. Rounds-sweep (--rounds-sweep): fixes one or more T values and sweeps over
   max_rounds to show how ρ converges as the Benders budget grows.
   Output: artifacts/results/qc_rounds_sweep.csv

GPU memory scales as 24 bytes * |F|: for all-online windows
  T=1: |F|=27  (~0 MB)   T=4: |F|=531k  (~12 MB)
  T=2: |F|=729 (~0 MB)   T=5: |F|=14.3M (~344 MB)
  T=3: |F|=19k (~0 MB)   T=6: |F|=387M  (~9 GB, likely OOM on 8 GB card)

Run:
    uv run python -m scripts.qc_benchmark --gpu
    uv run python -m scripts.qc_benchmark --gpu --t-max 5 --trials 30
    uv run python -m scripts.qc_benchmark --gpu --rounds-sweep --sweep-t 4 5 --trials 30
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
from qc.benders import benders_loop, build_sub_instance
from qc.grover_mixer import enumerate_feasible, expected_feasible_count
from qc.instance import Instance, direct_costs, int_to_bits
from subproblem.feasible_start_x import Params
from subproblem.subproblem import solve_subproblem

FIELDS_TSWEEP = [
    "t", "n_bits", "n_feasible", "vram_estimate_mb",
    "cost_classical", "runtime_gurobi_s",
    "cost_quantum_mean", "cost_quantum_std",
    "cost_random_mean", "cost_random_std",
    "approx_ratio_mean", "approx_ratio_std",
    "converged_frac", "rounds_mean",
    "runtime_trial_mean_s", "runtime_trial_std_s",
    "n_trials_ok", "n_trials",
]

FIELDS_RSWEEP = [
    "t", "n_feasible", "max_rounds", "force_outage",
    "cost_classical",
    "cost_quantum_mean", "cost_quantum_std",
    "approx_ratio_mean", "approx_ratio_std",
    "converged_frac", "rounds_mean",
    "runtime_trial_mean_s",
    "n_trials_ok", "n_trials",
]

DEFAULT_ROUNDS = [5, 10, 15, 20, 25, 30, 35, 40, 50]


def _make_instance(data: pd.DataFrame, start: int, t: int,
                   force_outage: int | None = None) -> Instance:
    chunk = data.iloc[start:start + t].reset_index(drop=True)
    g_avail = chunk["grid_available"].to_numpy(dtype=int)
    if force_outage is not None:
        g_avail = g_avail.copy()
        g_avail[force_outage] = 0
    return Instance(
        p_pv=chunk["p_kw"].to_numpy(dtype=float),
        p_load=chunk["load_kw"].to_numpy(dtype=float),
        tou=chunk["tou_usd_kwh"].to_numpy(dtype=float),
        g_avail=g_avail,
    )


def _gurobi_optimum(data: pd.DataFrame, t: int, start: int,
                    force_outage: int | None = None) -> tuple[float, float]:
    """Run Gurobi on the window; return (cost, runtime_s)."""
    window = data.iloc[start:start + t].reset_index(drop=True)
    if force_outage is not None:
        window = window.copy()
        window.loc[force_outage, "grid_available"] = 0
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
    for i in rng.integers(0, len(states), size=n):
        sub = build_sub_instance(inst, int(states[i]), params)
        res = solve_subproblem(sub)
        if res.feasible:
            costs.append(float(direct[i]) + float(res.q_value or 0.0))
    if not costs:
        return float("nan"), float("nan")
    return float(np.mean(costs)), float(np.std(costs))


def _run_trials(
    inst: Instance, trials: int, max_rounds: int, shots: int, p: int,
    use_gpu: bool, seed: int, cost_cl: float, label: str,
) -> dict:
    """Run `trials` Benders loops; return aggregated stats dict."""
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
                raise
            continue
        elapsed = time.perf_counter() - t0
        if result.best_z is not None:
            q_costs.append(result.best_value)
            q_rounds.append(len(result.rounds))
            q_times.append(elapsed)
            if result.termination == "gap":
                converged += 1
        print(f"    {label} trial {trial+1}/{trials}: cost={result.best_value:.4f}  "
              f"rounds={len(result.rounds)}  term={result.termination}  "
              f"{elapsed:.2f}s", flush=True)

    n_ok = len(q_costs)
    if n_ok == 0:
        return dict(
            cost_quantum_mean="nan", cost_quantum_std="nan",
            approx_ratio_mean="nan", approx_ratio_std="nan",
            converged_frac="nan", rounds_mean="nan",
            runtime_trial_mean_s="nan", n_trials_ok=0,
        )
    q_mean = float(np.mean(q_costs))
    q_std = float(np.std(q_costs))
    ratios = [c / cost_cl for c in q_costs] if cost_cl != 0 else [float("nan")] * n_ok
    return dict(
        cost_quantum_mean=round(q_mean, 6),
        cost_quantum_std=round(q_std, 6),
        approx_ratio_mean=round(float(np.mean(ratios)), 6),
        approx_ratio_std=round(float(np.std(ratios)), 6),
        converged_frac=round(converged / trials, 4),
        rounds_mean=round(float(np.mean(q_rounds)), 2),
        runtime_trial_mean_s=round(float(np.mean(q_times)), 3),
        n_trials_ok=n_ok,
    )


# ── T-sweep mode ──────────────────────────────────────────────────────────────

def run_t(
    t: int, start: int, data: pd.DataFrame,
    trials: int, max_rounds: int, shots: int, p: int,
    use_gpu: bool, random_n: int, seed: int,
    force_outage: int | None = None,
) -> dict:
    inst = _make_instance(data, start, t, force_outage=force_outage)
    n_feasible = expected_feasible_count(inst)
    vram_mb = n_feasible * 24 / 1024 ** 2
    print(f"  T={t}  |F|={n_feasible:,}  vram≈{vram_mb:.1f} MB", flush=True)

    cost_cl, t_gurobi = _gurobi_optimum(data, t, start, force_outage=force_outage)
    print(f"    Gurobi: {cost_cl:.4f} $ in {t_gurobi:.2f}s", flush=True)

    rng_base = np.random.default_rng(seed)
    if n_feasible <= random_n:
        states = enumerate_feasible(inst)
        cost_rand_mean, cost_rand_std = _random_baseline(inst, states, rng_base, min(50, n_feasible))
    else:
        cost_rand_mean, cost_rand_std = float("nan"), float("nan")

    stats = _run_trials(inst, trials, max_rounds, shots, p, use_gpu, seed, cost_cl,
                        label=f"T={t}")
    return {
        "t": t,
        "n_bits": inst.n_bits,
        "n_feasible": n_feasible,
        "vram_estimate_mb": round(vram_mb, 2),
        "cost_classical": round(cost_cl, 6),
        "runtime_gurobi_s": round(t_gurobi, 3),
        "cost_random_mean": round(cost_rand_mean, 6) if not np.isnan(cost_rand_mean) else "nan",
        "cost_random_std": round(cost_rand_std, 6) if not np.isnan(cost_rand_std) else "nan",
        "n_trials": trials,
        **stats,
    }


# ── Rounds-sweep mode ─────────────────────────────────────────────────────────

def run_rounds_sweep(
    t: int, start: int, data: pd.DataFrame,
    rounds_list: list[int], trials: int, shots: int, p: int,
    use_gpu: bool, seed: int,
    force_outage: int | None = None,
) -> list[dict]:
    inst = _make_instance(data, start, t, force_outage=force_outage)
    n_feasible = expected_feasible_count(inst)
    vram_mb = n_feasible * 24 / 1024 ** 2
    print(f"\n  T={t}  |F|={n_feasible:,}  vram≈{vram_mb:.1f} MB", flush=True)

    cost_cl, t_gurobi = _gurobi_optimum(data, t, start, force_outage=force_outage)
    print(f"    Gurobi: {cost_cl:.4f} $ in {t_gurobi:.2f}s", flush=True)

    rows = []
    for max_rounds in rounds_list:
        print(f"\n  [T={t}, max_rounds={max_rounds}]", flush=True)
        try:
            stats = _run_trials(inst, trials, max_rounds, shots, p, use_gpu, seed,
                                cost_cl, label=f"rounds={max_rounds}")
        except Exception as exc:
            tb = traceback.format_exc()
            if "out of memory" in tb.lower() or "oom" in tb.lower():
                print(f"  GPU OOM at T={t} rounds={max_rounds}, skipping rest.")
                break
            print(f"  ERROR: {exc}")
            continue
        rows.append({
            "t": t,
            "n_feasible": n_feasible,
            "max_rounds": max_rounds,
            "force_outage": force_outage,
            "cost_classical": round(cost_cl, 6),
            "n_trials": trials,
            **stats,
        })
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="artifacts/data/all_data.csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--p", type=int, default=6, help="QAOA layers")
    ap.add_argument("--gpu", action="store_true", help="run QAOA on GPU via CuPy")
    ap.add_argument("--seed", type=int, default=0)

    # T-sweep mode
    ap.add_argument("--t-max", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=100)
    ap.add_argument("--random-max", type=int, default=20_000)
    ap.add_argument("--out", default="artifacts/results/qc_benchmark.csv")

    # Rounds-sweep mode
    ap.add_argument("--rounds-sweep", action="store_true",
                    help="sweep max_rounds for fixed T values instead of sweeping T")
    ap.add_argument("--sweep-t", type=int, nargs="+", default=[4, 5],
                    help="T values to use in rounds-sweep mode (default: 4 5)")
    ap.add_argument("--rounds-list", type=int, nargs="+", default=DEFAULT_ROUNDS,
                    help="max_rounds values to sweep (default: 5 10 15 20 25 30 35 40 50)")
    ap.add_argument("--out-rounds", default="artifacts/results/qc_rounds_sweep.csv")
    ap.add_argument("--force-outage", type=int, default=None, metavar="SLOT",
                    help="pin grid_available[SLOT]=0 inside the window (slot index within window)")

    args = ap.parse_args(argv)
    data = pd.read_csv(args.data)

    if args.rounds_sweep:
        return _main_rounds_sweep(args, data)
    else:
        return _main_tsweep(args, data)


def _main_tsweep(args: argparse.Namespace, data: pd.DataFrame) -> int:
    if args.start + args.t_max > len(data):
        print(f"ERROR: --start {args.start} + --t-max {args.t_max} exceeds data length {len(data)}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"QC benchmark (T-sweep): T=1..{args.t_max}, {args.trials} trials, "
          f"max_rounds={args.max_rounds}, {'GPU' if args.gpu else 'CPU'}")

    rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS_TSWEEP)
        writer.writeheader()
        for t in range(1, args.t_max + 1):
            print(f"\n[T={t}]", flush=True)
            try:
                row = run_t(t=t, start=args.start, data=data, trials=args.trials,
                            max_rounds=args.max_rounds, shots=args.shots, p=args.p,
                            use_gpu=args.gpu, random_n=args.random_max, seed=args.seed,
                            force_outage=args.force_outage)
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


def _main_rounds_sweep(args: argparse.Namespace, data: pd.DataFrame) -> int:
    out_path = Path(args.out_rounds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outage_str = f", force_outage={args.force_outage}" if args.force_outage is not None else ""
    print(f"QC benchmark (rounds-sweep): T={args.sweep_t}, "
          f"rounds={args.rounds_list}, {args.trials} trials, "
          f"{'GPU' if args.gpu else 'CPU'}{outage_str}")

    rows: list[dict] = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS_RSWEEP)
        writer.writeheader()
        for t in args.sweep_t:
            new_rows = run_rounds_sweep(
                t=t, start=args.start, data=data,
                rounds_list=args.rounds_list, trials=args.trials,
                shots=args.shots, p=args.p, use_gpu=args.gpu, seed=args.seed,
                force_outage=args.force_outage,
            )
            rows.extend(new_rows)
            for row in new_rows:
                writer.writerow(row)
            f.flush()

    if not rows:
        print("No results produced.")
        return 1

    print(f"\n=== Results -> {out_path} ===")
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 160,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(df[["t", "max_rounds", "cost_classical", "cost_quantum_mean",
                   "approx_ratio_mean", "converged_frac",
                   "rounds_mean", "runtime_trial_mean_s"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
