"""Approximation-ratio sweep across diverse T=5 windows.

Runs the GM-QAOA Benders loop (rounds-sweep variant, max_rounds=40) and the
rule-based heuristic on the same set of windows, then computes the normalized
approximation ratio per the instruction in doc/aproximation_ratio_instruction.md:

    r = (C_ref - C_method) / (C_ref - C_opt)   in [0, 1]

    r=0  passive controller  (no strategy)
    r=1  classical MILP optimum

Output:
    artifacts/results/approx_ratio_results.csv  — per-window numbers
    artifacts/results/approx_ratio_summary.csv  — mean ± std across windows

Run:
    ./.venv/bin/python -m scripts.approx_ratio_sweep
    ./.venv/bin/python -m scripts.approx_ratio_sweep --gpu
    ./.venv/bin/python -m scripts.approx_ratio_sweep --trials 10 --gpu
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from classical.heuristic_dispatch import (
    Params, approx_ratio, from_csv, objective, passive, sweep_pstar,
)
from scripts.qc_benchmark import run_rounds_sweep

DATA = "artifacts/data/all_data.csv"

# ── Window definitions ────────────────────────────────────────────────────────
# Each entry: (label, start_slot, force_outage_within_window | None)
# Covers evening-peak, PV-surplus, high-load, and outage scenarios.
WINDOWS: list[tuple[str, int, int | None]] = [
    # Evening peak: TOU=0.4 $/kWh, demand charge dominates
    ("peak_eve_d1",   65,    None),   # day 1 evening (slots 65–69, TOU=0.4)
    ("peak_eve_d30",  2945,  None),   # day 30 evening (same hour, different regime)
    # PV surplus: battery charges + exports
    ("pv_surplus_a",  6860,  None),   # strong midday PV (~480 kW avg)
    ("pv_surplus_b",  10700, None),   # second PV peak (seasonal variation)
    # High load: stress demand charge
    ("high_load_a",   1085,  None),   # near data peak load
    ("high_load_b",   22785, None),   # later-season high load
    # Outage windows: resiliency revenue exercised
    ("outage_nat",    644,   None),   # natural outage at slot 646 (offset 2)
    ("outage_forced", 200,   2),      # force outage at offset 2 in a normal window
]

RESULT_FIELDS = [
    "label", "start", "force_outage",
    "C_opt", "C_ref", "C_greedy", "C_hybrid_mean", "C_hybrid_std",
    "r_passive", "r_greedy", "r_hybrid_mean", "r_hybrid_std",
    "gap_to_opt_hybrid_mean", "converged_frac", "rounds_mean",
    "runtime_qc_s",
]


def _heuristic(data_path: str, start: int, slots: int,
               force_outage: int | None, c_opt: float) -> dict:
    """Run passive + greedy heuristic; return cost and ratio dict."""
    pv, load, ga, tou = from_csv(data_path, start, slots)  # type: ignore[misc]
    if force_outage is not None:
        ga = ga.copy()
        ga[force_outage] = 0
    p = Params()
    anchor = objective(passive(pv, load, ga, p), tou, p)
    _, best_obj, _ = sweep_pstar(pv, load, ga, tou, p)
    c_ref = anchor["total"]
    c_greedy = best_obj["total"]
    return {
        "C_ref": c_ref,
        "C_greedy": c_greedy,
        "r_passive": 0.0,
        "r_greedy": approx_ratio(c_ref, c_greedy, c_opt),
    }


def run_window(
    label: str, start: int, force_outage: int | None,
    data: pd.DataFrame, data_path: str,
    trials: int, shots: int, p_layers: int,
    use_gpu: bool, seed: int,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Window: {label}  start={start}  force_outage={force_outage}", flush=True)

    t0 = time.perf_counter()
    rows = run_rounds_sweep(
        t=5, start=start, data=data,
        rounds_list=[40],
        trials=trials, shots=shots, p=p_layers,
        use_gpu=use_gpu, seed=seed,
        force_outage=force_outage,
    )
    runtime_qc = time.perf_counter() - t0

    if not rows:
        print(f"  WARNING: no QC results for {label}", flush=True)
        return {}

    row = rows[0]  # only max_rounds=40
    c_opt = row["cost_classical"]
    c_hybrid_mean = row.get("cost_quantum_mean", float("nan"))
    c_hybrid_std = row.get("cost_quantum_std", float("nan"))

    if isinstance(c_hybrid_mean, str):
        c_hybrid_mean = float("nan")
    if isinstance(c_hybrid_std, str):
        c_hybrid_std = float("nan")

    h = _heuristic(data_path, start, 5, force_outage, c_opt)
    c_ref = h["C_ref"]

    denom = c_ref - c_opt
    if abs(denom) < 1e-9:
        r_hybrid_mean = float("nan")
        r_hybrid_std = float("nan")
    else:
        r_hybrid_mean = (c_ref - c_hybrid_mean) / denom if not np.isnan(c_hybrid_mean) else float("nan")
        r_hybrid_std = c_hybrid_std / abs(denom) if not np.isnan(c_hybrid_std) else float("nan")

    result = {
        "label": label,
        "start": start,
        "force_outage": force_outage,
        "C_opt": round(c_opt, 4),
        "C_ref": round(c_ref, 4),
        "C_greedy": round(h["C_greedy"], 4),
        "C_hybrid_mean": round(c_hybrid_mean, 4) if not np.isnan(c_hybrid_mean) else "nan",
        "C_hybrid_std": round(c_hybrid_std, 4) if not np.isnan(c_hybrid_std) else "nan",
        "r_passive": 0.0,
        "r_greedy": round(h["r_greedy"], 4),
        "r_hybrid_mean": round(r_hybrid_mean, 4) if not np.isnan(r_hybrid_mean) else "nan",
        "r_hybrid_std": round(r_hybrid_std, 4) if not np.isnan(r_hybrid_std) else "nan",
        "gap_to_opt_hybrid_mean": round(c_hybrid_mean - c_opt, 4) if not np.isnan(c_hybrid_mean) else "nan",
        "converged_frac": row.get("converged_frac", "nan"),
        "rounds_mean": row.get("rounds_mean", "nan"),
        "runtime_qc_s": round(runtime_qc, 1),
    }

    print(f"  C_opt={c_opt:.2f}  C_ref={c_ref:.2f}  C_greedy={h['C_greedy']:.2f}"
          f"  C_hybrid={c_hybrid_mean:.2f}", flush=True)
    print(f"  r: passive=0.000  greedy={h['r_greedy']:.3f}"
          f"  hybrid={r_hybrid_mean:.3f} ± {r_hybrid_std:.3f}  MILP=1.000", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--trials", type=int, default=20,
                    help="Benders trials per window (default 20)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--p", type=int, default=6, help="QAOA layers")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/results/approx_ratio_results.csv")
    ap.add_argument("--out-summary", default="artifacts/results/approx_ratio_summary.csv")
    args = ap.parse_args(argv)

    data = pd.read_csv(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Approximation-ratio sweep: {len(WINDOWS)} windows, T=5, "
          f"max_rounds=40, {args.trials} trials, {'GPU' if args.gpu else 'CPU'}")
    estimated_min = len(WINDOWS) * args.trials * 20 / 60
    print(f"Estimated runtime: ~{estimated_min:.0f} min")

    results = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for label, start, force_outage in WINDOWS:
            row = run_window(
                label=label, start=start, force_outage=force_outage,
                data=data, data_path=args.data,
                trials=args.trials, shots=args.shots, p_layers=args.p,
                use_gpu=args.gpu, seed=args.seed,
            )
            if row:
                results.append(row)
                writer.writerow(row)
                f.flush()

    if not results:
        print("No results produced.")
        return 1

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"=== Results -> {out_path} ===")
    display_cols = ["label", "C_opt", "C_ref", "r_greedy", "r_hybrid_mean",
                    "r_hybrid_std", "gap_to_opt_hybrid_mean", "converged_frac"]
    with pd.option_context("display.max_columns", None, "display.width", 160,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(df[display_cols].to_string(index=False))

    # Summary statistics over all valid windows
    r_hybrid_vals: pd.Series = pd.to_numeric(df["r_hybrid_mean"], errors="coerce").dropna()  # type: ignore[assignment]
    r_greedy_vals: pd.Series = pd.to_numeric(df["r_greedy"], errors="coerce").dropna()  # type: ignore[assignment]
    summary = {
        "n_windows": len(results),
        "r_passive_mean": 0.0,
        "r_greedy_mean": round(r_greedy_vals.mean(), 4),
        "r_greedy_std": round(r_greedy_vals.std(), 4),
        "r_hybrid_mean": round(r_hybrid_vals.mean(), 4),
        "r_hybrid_std": round(r_hybrid_vals.std(), 4),
        "r_milp_mean": 1.0,
    }
    sum_path = Path(args.out_summary)
    pd.DataFrame([summary]).to_csv(sum_path, index=False)

    print(f"\n=== Summary (passive=0 → greedy → hybrid → MILP=1) ===")
    print(f"  passive : r = 0.000")
    print(f"  greedy  : r = {summary['r_greedy_mean']:.3f} ± {summary['r_greedy_std']:.3f}")
    print(f"  hybrid  : r = {summary['r_hybrid_mean']:.3f} ± {summary['r_hybrid_std']:.3f}")
    print(f"  MILP    : r = 1.000")
    print(f"\nSummary -> {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
