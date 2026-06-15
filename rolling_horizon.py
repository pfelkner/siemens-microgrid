"""Rolling-horizon (MPC) driver with peak-commitment + exceedance penalty + ratchet.

Implements the "commit a peak, pay a penalty if you exceed, ratchet up next month"
billing on top of the existing MILP (classical_solver.build_and_solve).

How it works
------------
A billing period (~1 month) is solved as a sequence of overlapping windows:
each window plans `window_slots` ahead but only the first `step_slots` are
*implemented* (committed); then we advance, carrying forward the battery SoC and
the running realized peak. This reuses the solver's `soc_init` and `peak_floor`
hooks (added on feature/siemens-feedback).

Cost accounting for the month, given an exogenous committed peak P_commit:

    base    = commit_charge * P_commit                       # committed capacity
    penalty = penalty_rate  * max(0, realized_peak - P_commit) # exceedance
    total   = base + energy_cost + penalty - resiliency - export

Inside each window solve we run the solver in peak_mode="commit_penalty", which
prices only the *marginal* exceedance above peak_floor = max(P_commit, running
peak). Summed over the windows this equals penalty_rate * max(0, realized_peak -
P_commit); the base charge is added once here in the driver.

The committed peak is *exogenous*: `--sweep` evaluates a range of P_commit values
and plots total cost vs commitment to reveal the optimal trade-off. The ratchet
(next month's commitment) is reported via `ratchet_next_commit`.

Run the pure-logic self-test (no Gurobi/pandas needed):
    python rolling_horizon.py --self-test
"""

from __future__ import annotations

import argparse
import sys

SLOTS_PER_DAY = 96  # 15-min slots in 24 h (matches scenarios.py)


# ----------------------------------------------------------------------------
# Pure cost/ratchet arithmetic (no pandas/gurobi — unit-tested in _self_test)
# ----------------------------------------------------------------------------
def exceedance_penalty(realized_peak: float, p_commit: float, penalty_rate: float) -> float:
    """$ penalty for exceeding the committed peak (0 if under)."""
    return penalty_rate * max(0.0, realized_peak - p_commit)


def ratchet_next_commit(realized_peak: float, p_commit: float, factor: float = 1.0) -> float:
    """Next period's committed peak. Ratchets up to (factor * realized peak),
    never below the current commitment. factor=1.0 = reset exactly to realized peak."""
    return max(p_commit, factor * realized_peak)


def monthly_total(base_charge: float, energy_cost: float, penalty: float,
                  resiliency_rev: float, export_rev: float) -> float:
    """Total monthly cost (revenues subtracted)."""
    return base_charge + energy_cost + penalty - resiliency_rev - export_rev


# ----------------------------------------------------------------------------
# Per-window accrual from an implemented block (needs numpy/pandas)
# ----------------------------------------------------------------------------
def _accrue_block(block, tou_block, dt: float, export_rate: float,
                  resiliency_per_slot: float, tol: float = 1e-4):
    """Recompute realized cost contributions from an implemented schedule block.

    Returns (energy_cost, export_rev, resiliency_rev, block_peak, soc_end).
    Resiliency is recovered from the schedule: an outage slot counts as served
    iff the island power balance holds (load fully met without the grid).
    """
    import numpy as np

    imp  = block["Grid_Import"].to_numpy(dtype=float)
    exp  = block["Grid_Export"].to_numpy(dtype=float)
    pv   = block["p_pv_kw"].to_numpy(dtype=float)
    load = block["p_load_kw"].to_numpy(dtype=float)
    ch   = block["BESS_Charge"].to_numpy(dtype=float)
    dis  = block["BESS_Discharge"].to_numpy(dtype=float)
    gav  = block["grid_available"].to_numpy(dtype=int)

    energy_cost = float((tou_block * imp * dt).sum())
    export_rev  = float((export_rate * exp * dt).sum())

    resid = pv + dis - ch - load
    served_mask = (gav == 0) & (np.abs(resid) < tol)
    resiliency_rev = float(resiliency_per_slot * int(served_mask.sum()))

    block_peak = float(imp.max()) if len(imp) else 0.0
    soc_end = float(block["BESS_SoC"].iloc[-1])
    return energy_cost, export_rev, resiliency_rev, block_peak, soc_end


# ----------------------------------------------------------------------------
# Rolling-horizon simulation of one billing period at a fixed committed peak
# ----------------------------------------------------------------------------
def simulate_month(
    df_month,
    p_commit: float,
    penalty_rate: float,
    *,
    window_slots: int,
    step_slots: int,
    commit_charge: float,
    ratchet_factor: float = 1.0,
    resiliency_per_slot: float,
    export_rate: float,
    M: int = 1,
    scenario_seed: int = 0,
    mip_gap: float = 1e-4,
    time_limit: float | None = None,
    quiet: bool = True,
    log_file: str = "gurobi_rolling.log",
):
    """Roll a single billing period forward at one committed peak. Returns
    (record dict, implemented-schedule DataFrame)."""
    import pandas as pd
    from classical_solver import build_and_solve, DT, SOC_INIT

    if window_slots < step_slots:
        raise ValueError("window_slots must be >= step_slots (lookahead >= implemented block)")

    T = len(df_month)
    soc = SOC_INIT
    running_peak = 0.0
    energy_tot = export_tot = resil_tot = 0.0
    impl_rows = []
    n_windows = 0

    for start in range(0, T, step_slots):
        w_end = min(start + window_slots, T)
        window_df = df_month.iloc[start:w_end].reset_index(drop=True)
        peak_floor = max(p_commit, running_peak)

        if M == 1:
            df_list = [window_df]
        else:
            from scenarios import generate_scenarios
            df_list = generate_scenarios(window_df, M, seed=scenario_seed + start)

        _, _info, scheds = build_and_solve(
            df_list, None, time_limit, mip_gap, log_file, quiet,
            resiliency_per_slot=resiliency_per_slot,
            export_rate=export_rate,
            soc_init=soc,
            peak_floor=peak_floor,
            peak_mode="commit_penalty",
            penalty_rate=penalty_rate,
        )

        # Implement only the first step_slots of the window (MPC: plan long, commit short)
        sched = scheds[0]  # realized path (scenario 0)
        impl_len = min(step_slots, w_end - start)
        block = sched.iloc[:impl_len].reset_index(drop=True)
        tou_block = window_df["tou_usd_kwh"].to_numpy(dtype=float)[:impl_len]

        e, x, r, pk, soc = _accrue_block(
            block, tou_block, DT, export_rate, resiliency_per_slot
        )
        energy_tot += e
        export_tot += x
        resil_tot += r
        running_peak = max(running_peak, pk)
        impl_rows.append(block)
        n_windows += 1

    base_charge = commit_charge * p_commit
    penalty = exceedance_penalty(running_peak, p_commit, penalty_rate)
    total = monthly_total(base_charge, energy_tot, penalty, resil_tot, export_tot)
    next_commit = ratchet_next_commit(running_peak, p_commit, ratchet_factor)
    impl_sched = pd.concat(impl_rows, ignore_index=True) if impl_rows else None

    record = {
        "p_commit":           p_commit,
        "realized_peak":      running_peak,
        "exceeded":           running_peak > p_commit + 1e-6,
        "base_charge":        base_charge,
        "energy_cost":        energy_tot,
        "penalty":            penalty,
        "resiliency_revenue": resil_tot,
        "export_revenue":     export_tot,
        "total_cost":         total,
        "next_commit":        next_commit,
        "n_windows":          n_windows,
    }
    return record, impl_sched


def sweep_commitments(df_month, commits, penalty_rate, **kwargs):
    """Run simulate_month over a list of committed-peak values; return records."""
    records = []
    for pc in commits:
        rec, _ = simulate_month(df_month, float(pc), penalty_rate, **kwargs)
        flag = "EXCEEDED" if rec["exceeded"] else "ok"
        print(
            f"  P_commit={pc:8.1f} kW -> total=${rec['total_cost']:10.2f}  "
            f"(base=${rec['base_charge']:.0f}  energy=${rec['energy_cost']:.0f}  "
            f"penalty=${rec['penalty']:.0f}  peak={rec['realized_peak']:.1f}  {flag})"
        )
        records.append(rec)
    return records


def plot_sweep(records, out_png: str) -> None:
    """Plot total cost (and its components) vs committed peak; mark the optimum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pcs    = [r["p_commit"] for r in records]
    totals = [r["total_cost"] for r in records]
    base   = [r["base_charge"] for r in records]
    pen    = [r["penalty"] for r in records]
    best   = min(records, key=lambda r: r["total_cost"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pcs, totals, "o-", color="navy", lw=2, label="total cost")
    ax.plot(pcs, base, "--", color="darkorange", label="base demand charge")
    ax.plot(pcs, pen, "--", color="crimson", label="exceedance penalty")
    ax.axvline(best["p_commit"], color="green", ls=":", lw=2,
               label=f"optimal commit = {best['p_commit']:.0f} kW")
    ax.set_xlabel("Committed peak  P_commit  (kW)")
    ax.set_ylabel("Cost ($ / billing period)")
    ax.set_title("Peak-commitment trade-off: total cost vs committed peak")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"\nWrote sweep plot -> {out_png}")
    print(f"Optimal commitment: {best['p_commit']:.1f} kW  "
          f"(total ${best['total_cost']:.2f}, "
          f"{'exceeds' if best['exceeded'] else 'within'} commitment, "
          f"next-month commit -> {best['next_commit']:.1f} kW)")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",          default="all_data.csv")
    p.add_argument("--days",          type=int, default=None,
                   help="Limit horizon to first N days (default: all rows in the file)")
    p.add_argument("--window-days",   type=float, default=3.0, help="MPC lookahead window (days)")
    p.add_argument("--step-days",     type=float, default=1.0, help="Implemented block per window (days)")
    p.add_argument("--penalty-rate",  type=float, default=30.0,
                   help="$/kW exceedance penalty (PLACEHOLDER — confirm with supervisors)")
    p.add_argument("--commit-charge", type=float, default=15.0,
                   help="$/kW base charge on the committed peak (default = DEMAND_CHARGE)")
    p.add_argument("--ratchet-factor", type=float, default=1.0,
                   help="Next-month commit = max(commit, factor * realized peak)")
    p.add_argument("--scenarios",     type=int, default=1,
                   help="Scenarios per window (M). 1 = deterministic forecast (v1 default)")
    p.add_argument("--scenarios-seed", type=int, default=0)
    # Either a single commitment, or a sweep MIN MAX N
    p.add_argument("--p-commit",      type=float, default=None,
                   help="Single committed peak (kW); skip to use --sweep")
    p.add_argument("--sweep",         type=float, nargs=3, default=None,
                   metavar=("MIN", "MAX", "N"),
                   help="Sweep committed peak over N values in [MIN, MAX]")
    p.add_argument("--mip-gap",       type=float, default=1e-4)
    p.add_argument("--time-limit",    type=float, default=None)
    p.add_argument("--quiet",         action="store_true", default=True)
    p.add_argument("--out-plot",      default="peak_commit_sweep.png")
    p.add_argument("--out-csv",       default="peak_commit_sweep.csv")
    p.add_argument("--self-test",     action="store_true",
                   help="Run pure-logic self-test (no Gurobi/pandas) and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return 0

    import numpy as np
    import pandas as pd
    import classical_solver as cs

    df = pd.read_csv(args.data)
    if args.days is not None:
        df = df.iloc[: args.days * SLOTS_PER_DAY].reset_index(drop=True)

    window_slots = int(round(args.window_days * SLOTS_PER_DAY))
    step_slots   = int(round(args.step_days * SLOTS_PER_DAY))

    sim_kwargs = dict(
        window_slots=window_slots,
        step_slots=step_slots,
        commit_charge=args.commit_charge,
        ratchet_factor=args.ratchet_factor,
        resiliency_per_slot=cs.RESILIENCY_PER_SLOT,
        export_rate=cs.EXPORT_RATE,
        M=args.scenarios,
        scenario_seed=args.scenarios_seed,
        mip_gap=args.mip_gap,
        time_limit=args.time_limit,
        quiet=args.quiet,
    )

    n_days = len(df) / SLOTS_PER_DAY
    print(f"Rolling horizon: {len(df)} slots (~{n_days:.1f} days), "
          f"window={args.window_days}d step={args.step_days}d, "
          f"penalty=${args.penalty_rate}/kW, commit-charge=${args.commit_charge}/kW, M={args.scenarios}")

    if args.sweep is not None:
        lo, hi, n = args.sweep
        commits = list(np.linspace(lo, hi, int(n)))
        print(f"Sweeping P_commit over {int(n)} values in [{lo:.0f}, {hi:.0f}] kW:")
        records = sweep_commitments(df, commits, args.penalty_rate, **sim_kwargs)
        pd.DataFrame(records).to_csv(args.out_csv, index=False)
        print(f"Wrote sweep table -> {args.out_csv}")
        plot_sweep(records, args.out_plot)
    else:
        pc = args.p_commit if args.p_commit is not None else 300.0
        rec, _ = simulate_month(df, pc, args.penalty_rate, **sim_kwargs)
        print("\nMonthly result:")
        for k, v in rec.items():
            print(f"  {k:20s}: {v}")

    return 0


def _self_test() -> None:
    """Verify the pure cost/ratchet arithmetic without Gurobi or pandas."""
    # exceedance penalty
    assert exceedance_penalty(360, 300, 30) == 1800.0      # 60 kW over * $30
    assert exceedance_penalty(280, 300, 30) == 0.0         # under commitment
    assert exceedance_penalty(300, 300, 30) == 0.0         # exactly at commitment
    # ratchet
    assert ratchet_next_commit(360, 300) == 360.0          # ratchet up to realized peak
    assert ratchet_next_commit(280, 300) == 300.0          # stayed under -> unchanged
    assert abs(ratchet_next_commit(360, 300, 1.1) - 396.0) < 1e-9  # with 10% headroom factor
    # monthly total: base + energy + penalty - resiliency - export
    assert abs(monthly_total(4500, 1000, 1800, 200, 50)
               - (4500 + 1000 + 1800 - 200 - 50)) < 1e-9
    # limiting case: huge penalty makes any exceedance dominate (commit must cover peak)
    assert exceedance_penalty(360, 300, 1e9) == 6e10
    print("rolling_horizon self-test OK")


if __name__ == "__main__":
    raise SystemExit(main())
