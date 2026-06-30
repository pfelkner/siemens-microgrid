"""Compare classical MILP, deterministic DP, and stochastic DP schedules.

Usage:
    uv run python compare.py               # uses artifacts/results/ schedules
    uv run python compare.py --detail      # also prints per-slot max deviations
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("artifacts/results")
DT = 0.25  # h per slot


def load_schedule(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scenario" in df.columns:
        # stochastic: return scenario 0 (or expected if caller wants)
        df = df[df["scenario"] == 0].drop(columns="scenario").reset_index(drop=True)
    return df


def cost_breakdown(sched: pd.DataFrame, tou: pd.Series) -> dict:
    energy = float((tou * sched["Grid_Import"] * DT).sum())
    export = float((0.05 * sched["Grid_Export"] * DT).sum())
    peak   = float(sched["Grid_Import"].max())
    demand = 15.0 * peak
    served = int(sched.get("Outage_Served", pd.Series(dtype=float)).fillna(0).sum()) \
             if "Outage_Served" in sched.columns else 0
    resiliency = 225.0 * served
    return {
        "energy":     energy,
        "demand":     demand,
        "resiliency": resiliency,
        "export":     export,
        "total":      energy + demand - resiliency - export,
        "peak_kw":    peak,
        "served":     served,
    }


def balance_error(sched: pd.DataFrame) -> float:
    online = sched["grid_available"] == 1
    resid = (sched["p_pv_kw"] + sched["Grid_Import"] - sched["Grid_Export"]
             + sched["BESS_Discharge"] - sched["BESS_Charge"] - sched["p_load_kw"])
    return float(resid[online].abs().max()) if online.any() else 0.0


def soc_stats(sched: pd.DataFrame) -> tuple[float, float]:
    return float(sched["BESS_SoC"].min()), float(sched["BESS_SoC"].max())


def print_table(rows: list[dict], detail: bool) -> None:
    names = [r["name"] for r in rows]
    w = max(len(n) for n in names) + 2

    header = f"{'Solver':<{w}}  {'Total $':>10}  {'Energy $':>10}  {'Demand $':>10}  "
    header += f"{'Resil $':>8}  {'Export $':>9}  {'Peak kW':>8}  {'Served':>6}  {'BalErr':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        c = r["costs"]
        print(
            f"{r['name']:<{w}}  {c['total']:>10.2f}  {c['energy']:>10.2f}  "
            f"{c['demand']:>10.2f}  {c['resiliency']:>8.2f}  {c['export']:>9.2f}  "
            f"{c['peak_kw']:>8.1f}  {c['served']:>6d}  {r['bal_err']:>9.2e}"
        )

    if detail and len(rows) >= 2:
        print()
        ref = rows[0]
        for other in rows[1:]:
            if len(ref["sched"]) != len(other["sched"]):
                continue
            for col in ["Grid_Import", "BESS_SoC", "BESS_Charge", "BESS_Discharge"]:
                diff = (ref["sched"][col] - other["sched"][col]).abs()
                print(f"  {ref['name']} vs {other['name']}  {col:20s}  "
                      f"max_diff={diff.max():.3f}  mean_diff={diff.mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args()

    res = Path(args.results)
    solvers = [
        ("Classical MILP (det)",    res / "schedule_classical.csv"),
        ("DP det",                  res / "schedule_dp.csv"),
        ("DP stoch M=5 (s0)",       res / "schedule_dp_stochastic.csv"),
        ("DP stoch M=5 (expected)", res / "schedule_dp_expected.csv"),
    ]

    base_tou = pd.read_csv("all_data.csv").set_index("timestamp")["tou_usd_kwh"]

    rows = []
    for name, path in solvers:
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        sched = load_schedule(path)
        tou = sched["tou_usd_kwh"] if "tou_usd_kwh" in sched.columns \
              else sched["timestamp"].map(base_tou).reset_index(drop=True)
        rows.append({
            "name":    name,
            "window":  (sched["timestamp"].iloc[0], sched["timestamp"].iloc[-1]),
            "T":       len(sched),
            "costs":   cost_breakdown(sched, tou),
            "bal_err": balance_error(sched),
            "soc":     soc_stats(sched),
            "sched":   sched,
        })

    if not rows:
        print("No schedules found. Run the solvers first.")
        return

    # Group by (start_ts, T) so we only compare same-window schedules.
    from itertools import groupby
    rows.sort(key=lambda r: (r["window"][0], r["T"]))
    groups = [(k, list(v)) for k, v in
              groupby(rows, key=lambda r: (r["window"][0], r["T"]))]

    for (start, T), group in groups:
        print(f"\n{'='*20} Schedule comparison  T={T} slots  start={start} {'='*20}\n")
        print_table(group, args.detail)
        print()
        for r in group:
            lo, hi = r["soc"]
            outage = int((r["sched"]["grid_available"] == 0).sum())
            print(f"  {r['name']:30s}  SoC ∈ [{lo:.1f}, {hi:.1f}] kWh  "
                  f"outage_slots={outage}  served={r['costs']['served']}")

        milp = next((r for r in group if "MILP" in r["name"]), None)
        dp   = next((r for r in group if r["name"] == "DP det"), None)
        if milp and dp:
            gap = dp["costs"]["total"] - milp["costs"]["total"]
            gap_pct = 100 * gap / abs(milp["costs"]["total"])
            print(f"\n  DP det optimality gap vs MILP: ${gap:+.2f} ({gap_pct:+.1f}%)")
            print(f"  Breakdown: demand diff = "
                  f"${dp['costs']['demand'] - milp['costs']['demand']:+.2f}  "
                  f"energy diff = ${dp['costs']['energy'] - milp['costs']['energy']:+.2f}")


if __name__ == "__main__":
    main()
