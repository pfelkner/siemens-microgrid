"""Stochastic two-stage MILP solver for the Siemens microgrid dispatch problem.

This module handles the multi-scenario case explicitly: `peak_import` is the
first-stage scalar shared across scenarios; all dispatch variables are
second-stage and scenario-specific. Use `classical.deterministic_solver` for the
single-scenario deterministic baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gurobipy as gp
import pandas as pd

from ._milp_core import (
    EXPORT_RATE,
    RESILIENCY_PER_MIN,
    RESILIENCY_PER_SLOT,
    SOC_INIT,
    append_summary,
    build_microgrid_milp,
    check_peak_cover,
    run_sanity_checks,
)
from .scenarios import generate_scenarios


def build_and_solve(
    df_list: list[pd.DataFrame],
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
    """Build and solve the stochastic multi-scenario MILP."""
    if len(df_list) <= 1:
        raise ValueError(
            "classical.stochastic_solver expects at least two scenarios; "
            "use classical.deterministic_solver for deterministic runs"
        )
    return build_microgrid_milp(
        df_list,
        scenario_probs=scenario_probs,
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


def build_from_forecast(
    df: pd.DataFrame,
    n_scenarios: int,
    *,
    scenario_probs: list[float] | None = None,
    pv_noise_sigma: float = 0.15,
    load_noise_sigma: float = 0.05,
    seed: int = 0,
    homogeneous: bool = False,
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
    """Generate scenarios from a base forecast, then solve the stochastic MILP."""
    if n_scenarios <= 1:
        raise ValueError("n_scenarios must be > 1 for stochastic_solver")
    df_list = generate_scenarios(
        df,
        n_scenarios,
        pv_noise_sigma=pv_noise_sigma,
        load_noise_sigma=load_noise_sigma,
        seed=seed,
        homogeneous=homogeneous,
    )
    return build_and_solve(
        df_list,
        scenario_probs=scenario_probs,
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


def write_stochastic_schedules(
    base: Path,
    schedules: list[pd.DataFrame],
    scenario_probs: list[float] | None = None,
) -> tuple[Path, Path]:
    """Write per-scenario and expected schedules using the old output names."""
    stoch_path = base.parent / (base.stem + "_stochastic.csv")
    parts = []
    for s, sched in enumerate(schedules):
        sc = sched.copy()
        sc.insert(0, "scenario", s)
        parts.append(sc)
    pd.concat(parts, ignore_index=True).to_csv(stoch_path, index=False)

    probs = scenario_probs if scenario_probs is not None else [1.0 / len(schedules)] * len(schedules)
    exp_path = base.parent / (base.stem + "_expected.csv")
    num_cols = [
        "p_pv_kw", "p_load_kw", "Grid_Import", "Grid_Export",
        "BESS_Charge", "BESS_Discharge", "BESS_SoC",
    ]
    expected = schedules[0].copy()
    for col in num_cols:
        expected[col] = sum(prob * sc[col] for prob, sc in zip(probs, schedules))
    expected.to_csv(exp_path, index=False)
    return stoch_path, exp_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stochastic classical MILP microgrid solver")
    p.add_argument("--data", default="artifacts/data/all_data.csv")
    p.add_argument("--slots", type=int, default=2880,
                   help="Number of 15-min slots to optimize (capped at file length)")
    p.add_argument("--scenarios", type=int, default=5,
                   help="Number of stochastic scenarios; must be > 1")
    p.add_argument("--scenarios-seed", type=int, default=0)
    p.add_argument("--pv-sigma", type=float, default=0.15,
                   help="PV AR(1) noise std (fraction of forecast value)")
    p.add_argument("--load-sigma", type=float, default=0.05,
                   help="Load AR(1) noise std (fraction of forecast value)")
    p.add_argument("--homogeneous", action="store_true",
                   help="Use constant-sigma AR(1) scenarios (Step-3 back-compat)")
    p.add_argument("--resiliency-per-min", type=float, default=RESILIENCY_PER_MIN,
                   help="Resiliency revenue in $/min (band 10-20; use 0.10 for pre-patch behavior)")
    p.add_argument("--export-rate", type=float, default=EXPORT_RATE,
                   help="Export tariff in $/kWh (use 0.0 for pre-patch behavior)")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.scenarios <= 1:
        raise SystemExit("stochastic_solver requires --scenarios > 1")

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

    _, info, schedules = build_from_forecast(
        df,
        args.scenarios,
        pv_noise_sigma=args.pv_sigma,
        load_noise_sigma=args.load_sigma,
        seed=args.scenarios_seed,
        homogeneous=args.homogeneous,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        log_file=str(gurobi_log),
        quiet=args.quiet,
        resiliency_per_slot=args.resiliency_per_min * 15.0,
        export_rate=args.export_rate,
    )

    write_stochastic_schedules(out_schedule, schedules)

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for s, sched in enumerate(schedules):
        errs, warns = run_sanity_checks(sched)
        prefix = f"s{s}: "
        all_errors += [prefix + e for e in errs]
        all_warnings += [prefix + w for w in warns]
    all_errors += check_peak_cover(info, schedules)

    for w in all_warnings:
        print(w)
    for e in all_errors:
        print(f"ERROR: {e}", file=sys.stderr)

    append_summary(out_summary, info)

    if not args.quiet:
        print(
            f"[done] T={info['T']} M={args.scenarios} (stochastic) status={info['status_str']} "
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
