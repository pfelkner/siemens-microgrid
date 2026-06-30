"""Two-stage stochastic DP solver for the Siemens microgrid dispatch problem.

Stochastic counterpart of dp_solver.py. Same signed-trajectory reformulation,
extended to M scenarios with the two-stage structure of classical_solver.py:

  - First stage (here-and-now, shared):  the demand-charge peak cap.
  - Second stage (wait-and-see, per s):  the whole SoC trajectory / dispatch.

The decomposition that makes this cheap: the ONLY thing coupling scenarios is the
first-stage peak. The deterministic DP already sweeps that peak as its outer
variable. So for a fixed cap the M scenarios separate completely — solve M
independent inner DPs from the same soc_init, take the probability-weighted mean
of their optima, add c_dem*cap, minimize over caps. Cost is just M independent
deterministic DPs:  O(M * peak_levels * T * L^2).

This is the classical *reference* for the stochastic quantum approach
(quantum_approach.md §7): the quantum solver runs M independent GM-QAOA inner
solves per outer step — M x shots, NOT M x qubits — and this file is what we
validate it against.

Reuses build_stage_costs / _dp_forward / _recover_schedule / validate from
dp_solver.py; only the cap sweep and summary change (scenario averaging).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dp_solver import (
    BESS_CAP, DEMAND_CHARGE, EXPORT_RATE, GRID_PMAX, RESILIENCY_PER_SLOT,
    SOC_INIT, INF, _EPS,
    build_stage_costs, _dp_forward, _recover_schedule, _summarize, validate,
    solve_dp,
)
from scenarios import generate_scenarios


def solve_dp_stochastic(
    df_list: list[pd.DataFrame],
    scenario_probs: list[float] | None = None,
    soc_levels: int = 41,
    peak_levels: int = 41,
    soc_init: float = SOC_INIT,
    export_rate: float = EXPORT_RATE,
    resiliency_rate: float = RESILIENCY_PER_SLOT,
    serve_tol: float = 2.0,
) -> tuple[list[pd.DataFrame], dict]:
    """Solve the two-stage stochastic dispatch by DP.

    Returns (schedules, info): one schedule DataFrame per scenario plus an info
    dict whose cost fields are probability-weighted expectations (matching the
    MILP's expected-cost objective). demand_cost is billed once on the shared
    first-stage peak.
    """
    M = len(df_list)
    T = len(df_list[0])
    probs = scenario_probs if scenario_probs is not None else [1.0 / M] * M
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError(f"scenario_probs must sum to 1 (got {sum(probs)})")

    levels = np.linspace(0.0, BESS_CAP, soc_levels)
    i0 = int(np.argmin(np.abs(levels - soc_init)))            # shared start

    # Per-scenario stage costs (second stage is independent given the cap).
    stage = [build_stage_costs(df, levels, export_rate, resiliency_rate, serve_tol)
             for df in df_list]

    # First-stage sweep: shared cap forced on EVERY scenario, demand billed once.
    caps = np.linspace(0.0, GRID_PMAX, peak_levels)
    best_cap, best_total = None, INF
    for cap in caps:
        exp_second = 0.0
        feasible = True
        for s in range(M):
            base, imp = stage[s]
            dp, _ = _dp_forward(base, imp, cap, i0, track=False)
            m = float(dp.min())
            if m >= INF / 2:                                  # cap infeasible for s
                feasible = False
                break
            exp_second += probs[s] * m
        if not feasible:
            continue
        total = exp_second + DEMAND_CHARGE * cap
        if total < best_total:
            best_total, best_cap = total, float(cap)

    if best_cap is None:
        raise RuntimeError(
            "No cap feasible for all scenarios — refine --soc-levels/--peak-levels "
            "or check data."
        )

    # Recover each scenario's trajectory under the winning cap.
    schedules: list[pd.DataFrame] = []
    for s in range(M):
        base, imp = stage[s]
        dp, parents = _dp_forward(base, imp, best_cap, i0, track=True)
        j = int(np.argmin(dp))
        path = np.empty(T, dtype=np.int32)
        for t in range(T - 1, -1, -1):
            path[t] = j
            j = int(parents[t][j])
        prev = np.concatenate([[i0], path[:-1]])
        schedules.append(_recover_schedule(df_list[s], levels, prev, path))

    info = _summarize_stochastic(schedules, df_list, probs, export_rate,
                                 resiliency_rate, best_cap, T, soc_levels,
                                 peak_levels)
    return schedules, info


def _summarize_stochastic(schedules, df_list, probs, export_rate,
                          resiliency_rate, cap, T, soc_levels, peak_levels) -> dict:
    """Expected cost decomposition. Demand billed once on the realized first-stage
    peak (= max import across scenarios, which is <= cap and what the MILP's
    peak_import variable would settle on)."""
    M = len(schedules)
    energy = export = resiliency = 0.0
    served_total = 0.0
    for s, sched in enumerate(schedules):
        tou = df_list[s]["tou_usd_kwh"].to_numpy(dtype=float)
        energy += probs[s] * float((tou * sched["Grid_Import"].to_numpy() * 0.25).sum())
        export += probs[s] * float((export_rate * sched["Grid_Export"].to_numpy() * 0.25).sum())
        served_s = int(sched["Outage_Served"].fillna(0).sum())
        resiliency += probs[s] * resiliency_rate * served_s
        served_total += probs[s] * served_s

    realized_peak = max(float(sc["Grid_Import"].max()) for sc in schedules)
    demand = DEMAND_CHARGE * realized_peak                     # first-stage, once
    outage_slots = int((df_list[0]["grid_available"] == 0).sum())
    return {
        "T": T, "M": M,
        "soc_levels": soc_levels, "peak_levels": peak_levels,
        "soc_step_kwh": BESS_CAP / (soc_levels - 1),
        "peak_cap_kw": cap,
        "peak_import_kw": realized_peak,
        "total_cost": energy + demand - resiliency - export,
        "energy_cost": energy,
        "demand_cost": demand,
        "export_revenue": export,
        "resiliency_revenue": resiliency,
        "expected_served": served_total,
        "outage_slots": outage_slots,
    }


def _selfcheck() -> None:
    """M=1 must reproduce the deterministic solver exactly; M>1 must stay feasible."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "p_kw":   [0, 0, 50, 200, 300, 100, 0, 0],
        "load_kw":[150, 160, 140, 120, 130, 200, 210, 180],
        "tou_usd_kwh": [0.1, 0.1, 0.2, 0.3, 0.3, 0.4, 0.4, 0.2],
        "grid_available": [1, 1, 1, 1, 0, 0, 1, 1],
    })

    # M=1 reduction: must match dp_solver.solve_dp bit-for-bit on total cost.
    det_sched, det_info = solve_dp(df, soc_levels=41, peak_levels=21)
    sto_scheds, sto_info = solve_dp_stochastic([df], soc_levels=41, peak_levels=21)
    assert len(sto_scheds) == 1
    assert abs(sto_info["total_cost"] - det_info["total_cost"]) < 1e-6, \
        f"M=1 mismatch: {sto_info['total_cost']} vs {det_info['total_cost']}"
    assert not validate(sto_scheds[0]), "M=1 schedule infeasible"

    # M=3 scenarios: every scenario schedule must be feasible-by-construction.
    scenarios = generate_scenarios(df, 3, pv_noise_sigma=0.2, load_noise_sigma=0.1,
                                   seed=0, homogeneous=True)
    scheds, info = solve_dp_stochastic(scenarios, soc_levels=41, peak_levels=21)
    for s, sc in enumerate(scheds):
        errs = validate(sc)
        assert not errs, f"scenario {s} infeasible: {errs}"
    assert info["peak_import_kw"] <= GRID_PMAX + _EPS
    print(f"[selfcheck] OK  M=1 reduction matches deterministic "
          f"(${det_info['total_cost']:.2f})  |  M=3 expected total="
          f"${info['total_cost']:.2f}  peak={info['peak_import_kw']:.1f}kW")


def main() -> int:
    p = argparse.ArgumentParser(description="Stochastic two-stage DP microgrid solver")
    p.add_argument("--data", default="all_data.csv")
    p.add_argument("--slots", type=int, default=2880)
    p.add_argument("--scenarios", type=int, default=10,
                   help="Number of equiprobable scenarios (1 = deterministic)")
    p.add_argument("--scenarios-seed", type=int, default=0)
    p.add_argument("--pv-sigma", type=float, default=0.15)
    p.add_argument("--load-sigma", type=float, default=0.05)
    p.add_argument("--homogeneous", action="store_true")
    p.add_argument("--soc-levels", type=int, default=41)
    p.add_argument("--peak-levels", type=int, default=41)
    p.add_argument("--export-rate", type=float, default=EXPORT_RATE)
    p.add_argument("--resiliency-per-min", type=float, default=15.0)
    p.add_argument("--serve-tol", type=float, default=2.0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    df = pd.read_csv(args.data)
    n = min(args.slots, len(df))
    df = df.iloc[:n].reset_index(drop=True)

    M = args.scenarios
    if M == 1:
        df_list = [df]
    else:
        df_list = generate_scenarios(
            df, M, pv_noise_sigma=args.pv_sigma, load_noise_sigma=args.load_sigma,
            seed=args.scenarios_seed, homogeneous=args.homogeneous,
        )

    scheds, info = solve_dp_stochastic(
        df_list,
        soc_levels=args.soc_levels,
        peak_levels=args.peak_levels,
        export_rate=args.export_rate,
        resiliency_rate=args.resiliency_per_min * 15.0,
        serve_tol=args.serve_tol,
    )

    out = Path("artifacts/results")
    out.mkdir(parents=True, exist_ok=True)
    if M == 1:
        scheds[0].to_csv(out / "schedule_dp.csv", index=False)
    else:
        parts = []
        for s, sc in enumerate(scheds):
            c = sc.copy(); c.insert(0, "scenario", s); parts.append(c)
        pd.concat(parts, ignore_index=True).to_csv(out / "schedule_dp_stochastic.csv", index=False)
        num = ["p_pv_kw", "p_load_kw", "Grid_Import", "Grid_Export",
               "BESS_Charge", "BESS_Discharge", "BESS_SoC"]
        expected = scheds[0].copy()
        prob = 1.0 / M
        for col in num:
            expected[col] = sum(sc[col] for sc in scheds) * prob
        expected.to_csv(out / "schedule_dp_expected.csv", index=False)

    all_errs = []
    for s, sc in enumerate(scheds):
        all_errs += [f"s{s}: {e}" for e in validate(sc)]
    for e in all_errs:
        print(f"ERROR: {e}", file=sys.stderr)

    if not args.quiet:
        print(
            f"[done] T={info['T']} M={info['M']} L={info['soc_levels']} "
            f"(step={info['soc_step_kwh']:.1f}kWh) "
            f"E[total]=${info['total_cost']:.2f} "
            f"(energy=${info['energy_cost']:.2f} + demand=${info['demand_cost']:.2f} "
            f"- resiliency=${info['resiliency_revenue']:.2f} "
            f"- export=${info['export_revenue']:.2f}) "
            f"peak={info['peak_import_kw']:.1f}kW "
            f"E[served]={info['expected_served']:.1f}/{info['outage_slots']} "
            f"feasible={'YES' if not all_errs else 'NO'}"
        )
    return 1 if all_errs else 0


if __name__ == "__main__":
    sys.exit(main())
