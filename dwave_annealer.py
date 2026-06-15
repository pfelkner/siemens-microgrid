"""D-Wave annealer (simulated) for the microgrid QUBO.

Pipeline: build_qubo (qubo_model.py) -> BQM -> SA / SteepestDescent
-> decode -> evaluate

Rolling-window MPC is always used: 3-day forecast horizon, commit 1 day per step.
For real hardware, set DWAVE_API_TOKEN and pass --use-leap or --use-qpu.

Usage:
    python dwave_annealer.py
    python dwave_annealer.py --total-slots 480 --bits 4 --window 288 --step 96
    python dwave_annealer.py --use-leap --total-slots 480
"""

from __future__ import annotations

import argparse
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qubo_model import SOC_INIT, build_qubo, decode, evaluate


_COLORS = {
    "pv":       "#f9c74f",
    "load":     "#264653",
    "import":   "#e76f51",
    "export":   "#2a9d8f",
    "bess_ch":  "#457b9d",
    "bess_dis": "#a8dadc",
    "soc":      "#6a0572",
}


def _ts_axis(ax: plt.Axes, schedule: pd.DataFrame) -> None:
    n = len(schedule)
    ax.set_xlim(0, n)
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    if n - 1 not in ticks:
        ticks.append(n - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{(i * 15) // 60:02d}:{(i * 15) % 60:02d}" for i in ticks],
        fontsize=7, rotation=45)


def plot_schedule(schedule: pd.DataFrame, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = np.arange(len(schedule))
    lw = 1.2

    ax.plot(t, schedule["p_pv_kw"],        color=_COLORS["pv"],      lw=lw, label="PV")
    ax.plot(t, schedule["p_load_kw"],       color=_COLORS["load"],    lw=lw, linestyle="--", label="Load")
    ax.plot(t, schedule["Grid_Import"],     color=_COLORS["import"],  lw=lw, label="Grid Import")
    ax.plot(t, -schedule["Grid_Export"],    color=_COLORS["export"],  lw=lw, label="Grid Export (neg)")
    ax.plot(t, schedule["BESS_Charge"],     color=_COLORS["bess_ch"], lw=lw, label="BESS Charge")
    ax.plot(t, -schedule["BESS_Discharge"], color=_COLORS["bess_dis"],lw=lw, label="BESS Dis (neg)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_ylabel("kW", fontsize=8)
    _ts_axis(ax, schedule)

    ax2 = ax.twinx()
    ax2.fill_between(t, schedule["BESS_SoC"], alpha=0.15, color=_COLORS["soc"])
    ax2.plot(t, schedule["BESS_SoC"], color=_COLORS["soc"], lw=lw, linestyle=":")
    ax2.set_ylabel("SoC (kWh)", fontsize=8, color=_COLORS["soc"])
    ax2.tick_params(axis="y", labelcolor=_COLORS["soc"], labelsize=7)
    ax2.set_ylim(0, 1050)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[annealer] plot -> {out_path}")


def _sa(bqm, num_reads: int, sweeps: int, seed: int | None, polish: bool):
    from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSampler

    t0 = time.perf_counter()
    ss = SimulatedAnnealingSampler().sample(
        bqm, num_reads=num_reads, num_sweeps=sweeps,
        seed=seed, answer_mode="raw")
    t_sa = time.perf_counter() - t0

    t_polish = 0.0
    if polish:
        t1 = time.perf_counter()
        ss = SteepestDescentSampler().sample(bqm, initial_states=ss)
        t_polish = time.perf_counter() - t1

    return ss, t_sa, t_polish


def _leap_hybrid(bqm, time_limit: float):
    try:
        from dwave.system import LeapHybridSampler
    except ImportError as e:
        raise ImportError("uv add dwave-system") from e

    t0 = time.perf_counter()
    ss = LeapHybridSampler().sample(bqm, time_limit=time_limit)
    return ss, time.perf_counter() - t0, 0.0


def _qpu(bqm, num_reads: int):
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
    except ImportError as e:
        raise ImportError("uv add dwave-system") from e

    t0 = time.perf_counter()
    ss = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=num_reads)
    return ss, time.perf_counter() - t0, 0.0


def run_rolling(
    df: pd.DataFrame,
    *,
    bits: int = 4,
    num_reads: int = 1000,
    sweeps: int = 10_000,
    seed: int = 42,
    polish: bool = True,
    use_leap: bool = False,
    use_qpu: bool = False,
    leap_time_limit: float = 3.0,
    window: int = 288,
    step: int = 96,
    lam_balance: float = 1.0,
    lam_soc: float = 1.0,
    lam_peak: float = 1.0,
    lam_xor: float = 0.5,
) -> pd.DataFrame:
    """Rolling-window MPC: solve horizon=window slots, commit=step slots per iteration."""
    if use_qpu:
        sampler = lambda bqm: _qpu(bqm, num_reads)
        backend = "D-Wave QPU"
    elif use_leap:
        sampler = lambda bqm: _leap_hybrid(bqm, leap_time_limit)
        backend = "Leap Hybrid"
    else:
        sampler = lambda bqm: _sa(bqm, num_reads, sweeps, seed, polish)
        backend = "SimulatedAnnealing"

    total_slots = len(df)
    soc_carry = SOC_INIT
    committed: list[pd.DataFrame] = []
    offset = 0
    win_idx = 0
    t_wall_total = 0.0

    while offset < total_slots:
        remaining = total_slots - offset
        horizon = min(window, remaining)
        commit = min(step, remaining)

        chunk = df.iloc[offset: offset + horizon].reset_index(drop=True)
        bqm, ctx = build_qubo(
            chunk,
            bits_grid=bits, bits_bess=bits, bits_soc=bits,
            soc_init=soc_carry,
            lam_balance=lam_balance, lam_soc=lam_soc,
            lam_peak=lam_peak, lam_xor=lam_xor,
        )
        n, nq = len(bqm.variables), bqm.num_interactions
        print(f"[win {win_idx:03d}] offset={offset:4d}  horizon={horizon:3d}  "
              f"commit={commit:3d}  soc_init={soc_carry:6.1f}kWh  "
              f"vars={n}  interactions={nq}")

        ss, t_solve, t_polish_t = sampler(bqm)
        t_wall = t_solve + t_polish_t
        t_wall_total += t_wall

        best = ss.first
        schedule = decode(dict(best.sample), ctx)
        metrics = evaluate(schedule, dict(best.sample), ctx)

        print(f"           energy={best.energy:.2f}  "
              f"true_cost=${metrics['true_cost']:.2f}  "
              f"time={t_wall:.2f}s  backend={backend}")

        committed.append(schedule.iloc[:commit].copy())
        soc_carry = float(schedule.iloc[commit - 1]["BESS_SoC"])

        offset += commit
        win_idx += 1

    print(f"\n[rolling] {win_idx} windows  total_wall={t_wall_total:.1f}s")
    return pd.concat(committed, ignore_index=True)


# ---------- CLI ----------

def main() -> int:
    p = argparse.ArgumentParser(description="D-Wave SA rolling MPC for microgrid dispatch")
    p.add_argument("--data",             default="all_data.csv")
    p.add_argument("--bits",             type=int,   default=4)
    p.add_argument("--window",           type=int,   default=288,
                   help="forecast horizon in slots (default 288 = 3 days)")
    p.add_argument("--step",             type=int,   default=96,
                   help="slots committed per window (default 96 = 1 day)")
    p.add_argument("--total-slots",      type=int,   default=None,
                   help="total slots from dataset (default: all)")
    p.add_argument("--num-reads",        type=int,   default=1000)
    p.add_argument("--sweeps",           type=int,   default=10_000)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no-polish",        action="store_true")
    p.add_argument("--use-leap",         action="store_true",
                   help="Leap hybrid solver (requires DWAVE_API_TOKEN)")
    p.add_argument("--use-qpu",          action="store_true",
                   help="Real D-Wave QPU (requires DWAVE_API_TOKEN)")
    p.add_argument("--leap-time-limit",  type=float, default=3.0)
    p.add_argument("--lam-balance",      type=float, default=1.0)
    p.add_argument("--lam-soc",          type=float, default=1.0)
    p.add_argument("--lam-peak",         type=float, default=1.0)
    p.add_argument("--lam-xor",          type=float, default=0.5)
    p.add_argument("--out-schedule",     default="schedule_annealer.csv")
    p.add_argument("--out-plot",         default="schedule_annealer.png")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    total = args.total_slots or len(df)
    df = df.iloc[:total].reset_index(drop=True)
    print(f"[rolling] dataset={len(df)} slots  window={args.window}  step={args.step}")

    schedule = run_rolling(
        df,
        bits=args.bits,
        num_reads=args.num_reads,
        sweeps=args.sweeps,
        seed=args.seed,
        polish=not args.no_polish,
        use_leap=args.use_leap,
        use_qpu=args.use_qpu,
        leap_time_limit=args.leap_time_limit,
        window=args.window,
        step=args.step,
        lam_balance=args.lam_balance,
        lam_soc=args.lam_soc,
        lam_peak=args.lam_peak,
        lam_xor=args.lam_xor,
    )
    schedule.to_csv(args.out_schedule, index=False)

    from qubo_model import DEMAND_CHARGE, EXPORT_RATE
    tou      = df["tou_usd_kwh"].to_numpy()
    energy   = float((tou * schedule["Grid_Import"].to_numpy() * 0.25).sum())
    export   = float((EXPORT_RATE * schedule["Grid_Export"].to_numpy() * 0.25).sum())
    peak     = float(schedule["Grid_Import"].max())
    cost     = energy + DEMAND_CHARGE * peak - export
    print(f"[result] total_cost=${cost:.2f}  energy=${energy:.2f}  "
          f"demand=${DEMAND_CHARGE * peak:.2f}  export=${export:.2f}  peak={peak:.1f}kW")
    print(f"[output] {args.out_schedule}")

    plot_schedule(schedule, f"D-Wave SA rolling MPC — ${cost:.2f}", args.out_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
