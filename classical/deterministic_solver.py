"""Deterministic MILP solver for the Siemens microgrid dispatch problem.

This module intentionally handles only the single-scenario classical baseline.
Use `classical.stochastic_solver` for the multi-scenario two-stage MILP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gurobipy as gp
import pandas as pd

from ._milp_core import (
    BESS_CAP,
    BESS_PMAX,
    DEMAND_CHARGE,
    DT,
    ETA,
    ETA_RT,
    EXPORT_RATE,
    FRAC_EDGE,
    FRAC_MID,
    GRID_PMAX,
    RESILIENCY_PER_MIN,
    RESILIENCY_PER_SLOT,
    SOC_HIGH_TH,
    SOC_INIT,
    SOC_LOW_TH,
    append_summary,
    build_microgrid_milp,
    check_peak_cover,
    run_sanity_checks,
    status_str,
)


def _single_dataframe(df_or_list: pd.DataFrame | list[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df_or_list, list):
        if len(df_or_list) != 1:
            raise ValueError(
                "classical.deterministic_solver is deterministic-only; "
                "use classical.stochastic_solver for multiple scenarios"
            )
        return df_or_list[0]
    return df_or_list


def build_and_solve(
    df: pd.DataFrame | list[pd.DataFrame],
    scenario_probs: list[float] | None = None,
    time_limit: float | None = None,
    mip_gap: float = 1e-4,
    log_file: str = "",
    quiet: bool = True,
    resiliency_per_slot: float = RESILIENCY_PER_SLOT,
    export_rate: float = EXPORT_RATE,
    soc_init: float = SOC_INIT,
    peak_floor: float = 0.0,
    peak_mode: str = "demand_charge",
    penalty_rate: float = 0.0,
) -> tuple[gp.Model, dict, list[pd.DataFrame]]:
    """Build and solve the deterministic single-scenario MILP.

    The first argument may still be a one-item list for compatibility with older
    callers. Passing more than one dataframe is rejected so stochastic runs do
    not accidentally go through the deterministic module.
    """
    df_single = _single_dataframe(df)
    if scenario_probs is not None and scenario_probs not in ([1.0], [1]):
        raise ValueError("deterministic solver accepts only scenario_probs=None or [1.0]")
    return build_microgrid_milp(
        [df_single],
        scenario_probs=None,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_file=log_file,
        quiet=quiet,
        resiliency_per_slot=resiliency_per_slot,
        export_rate=export_rate,
        soc_init=soc_init,
        peak_floor=peak_floor,
        peak_mode=peak_mode,
        penalty_rate=penalty_rate,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Deterministic classical MILP microgrid solver")
    p.add_argument("--data", default="artifacts/data/all_data.csv")
    p.add_argument("--slots", type=int, default=2880,
                   help="Number of 15-min slots to optimize (capped at file length)")
    p.add_argument("--resiliency-per-min", type=float, default=RESILIENCY_PER_MIN,
                   help="Resiliency revenue in $/min (band 10-20; use 0.10 for pre-patch behavior)")
    p.add_argument("--export-rate", type=float, default=EXPORT_RATE,
                   help="Export tariff in $/kWh (use 0.0 for pre-patch behavior)")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    out_schedule = Path("artifacts/results/schedule_classical.csv")
    out_summary = Path("artifacts/results/results_classical.csv")
    gurobi_log = Path("artifacts/results/gurobi.log")
    out_schedule.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    n = min(args.slots, len(df))
    if n < args.slots and not args.quiet:
        print(f"[info] requested {args.slots} slots but data has {len(df)}; using {n}.")
    df = df.iloc[:n].reset_index(drop=True)

    if gurobi_log.exists():
        gurobi_log.unlink()

    _, info, schedules = build_and_solve(
        df,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        log_file=str(gurobi_log),
        quiet=args.quiet,
        resiliency_per_slot=args.resiliency_per_min * 15.0,
        export_rate=args.export_rate,
    )

    schedules[0].to_csv(out_schedule, index=False)

    all_errors: list[str] = []
    all_warnings: list[str] = []
    errs, warns = run_sanity_checks(schedules[0])
    all_errors += errs
    all_warnings += warns
    all_errors += check_peak_cover(info, schedules)

    for w in all_warnings:
        print(w)
    for e in all_errors:
        print(f"ERROR: {e}", file=sys.stderr)

    append_summary(out_summary, info)

    if not args.quiet:
        print(
            f"[done] T={info['T']} M=1 (deterministic) status={info['status_str']} "
            f"runtime={info['runtime_s']:.2f}s gap={info['mip_gap']:.2e} "
            f"total=${info['total_cost']:.2f} "
            f"(energy=${info['energy_cost']:.2f} + demand=${info['demand_cost']:.2f} "
            f"- resiliency=${info['resiliency_revenue']:.2f} "
            f"- export=${info['export_revenue']:.2f}) "
            f"served={info['served_count']}/{info['outage_slots']}"
        )

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
