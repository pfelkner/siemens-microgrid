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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qubo_model import SOC_INIT, build_qubo, decode, evaluate


_COLORS = {
    "pv":       "#f9c74f",
    "load":     "#1a1a5e",
    "import":   "#457b9d",
    "export":   "#2a9d8f",
    "bess_ch":  "#e76f51",
    "bess_dis": "#52b788",
    "tou":      "#e63946",
}


def plot_schedule(
    schedule: pd.DataFrame,
    df_input: pd.DataFrame,
    title: str,
    out_path: str,
) -> None:
    """Stacked-area dispatch plot with dual y-axis ToU price."""
    n = len(schedule)
    has_ts = "timestamp" in df_input.columns
    if has_ts:
        t = pd.to_datetime(df_input["timestamp"].iloc[:n].values)
    else:
        t = np.arange(n)

    tou = df_input["tou_usd_kwh"].iloc[:n].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.stackplot(
        t,
        schedule["Grid_Import"].to_numpy(),
        schedule["BESS_Discharge"].to_numpy(),
        schedule["p_pv_kw"].to_numpy(),
        schedule["Grid_Export"].to_numpy(),
        labels=["Grid Import", "BESS Discharge", "PV", "Grid Export"],
        colors=[_COLORS["import"], _COLORS["bess_dis"], _COLORS["pv"], _COLORS["export"]],
        alpha=0.85,
    )

    ax.fill_between(
        t, -schedule["BESS_Charge"].to_numpy(), 0,
        label="BESS Charge", color=_COLORS["bess_ch"], alpha=0.80,
    )

    ax.plot(t, schedule["p_load_kw"].to_numpy(),
            color=_COLORS["load"], lw=1.8, label="Load")

    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylabel("Power (kW)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax2 = ax.twinx()
    ax2.step(t, tou, where="post",
             color=_COLORS["tou"], lw=1.2, linestyle="--", alpha=0.8)
    ax2.set_ylabel("ToU price ($/kWh)", fontsize=9, color=_COLORS["tou"])
    ax2.tick_params(axis="y", labelcolor=_COLORS["tou"])

    # align zero of both y-axes so negative prices map correctly onto the power axis
    y1_lo, y1_hi = ax.get_ylim()
    y2_lo, y2_hi = ax2.get_ylim()
    span1, span2 = y1_hi - y1_lo, y2_hi - y2_lo
    frac1 = -y1_lo / span1   # fraction of zero from bottom on power axis
    frac2 = -y2_lo / span2
    if frac1 > frac2:
        ax2.set_ylim(bottom=-y2_hi * frac1 / (1.0 - frac1))
    else:
        ax.set_ylim(bottom=-y1_hi * frac2 / (1.0 - frac2))

    if has_ts:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        fig.autofmt_xdate(rotation=0, ha="center")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", ncol=3, fontsize=8, framealpha=0.85)

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
    bits: int = 5,
    num_reads: int = 50,
    seed: int = 42,
    polish: bool = True,
    use_leap: bool = False,
    use_qpu: bool = False,
    leap_time_limit: float = 3.0,
    window: int = 96,
    step: int = 96,
) -> pd.DataFrame:
    """Rolling-window MPC: solve horizon=window slots, commit=step slots per iteration."""
    if use_qpu:
        sampler = lambda bqm: _qpu(bqm, num_reads)
        backend = "D-Wave QPU"
    elif use_leap:
        sampler = lambda bqm: _leap_hybrid(bqm, leap_time_limit)
        backend = "Leap Hybrid"
    else:
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
        )
        n, nq = len(bqm.variables), bqm.num_interactions
        # auto-scale: at least 20 sweeps per variable (SA convergence heuristic)
        effective_sweeps = n * 20
        if not (use_leap or use_qpu):
            sampler = lambda bqm: _sa(bqm, num_reads, effective_sweeps, seed, polish)
        print(f"[win {win_idx:03d}] offset={offset:4d}  horizon={horizon:3d}  "
              f"commit={commit:3d}  soc_init={soc_carry:6.1f}kWh  "
              f"vars={n}  interactions={nq}  sweeps={effective_sweeps}")

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
    p.add_argument("--data",             default="artifacts/data/all_data.csv")
    p.add_argument("--bits",             type=int,   default=5)
    p.add_argument("--window",           type=int,   default=96,
                   help="forecast horizon in slots (default 96 = 1 day)")
    p.add_argument("--step",             type=int,   default=96,
                   help="slots committed per window (default 96 = 1 day)")
    p.add_argument("--total-slots",      type=int,   default=None,
                   help="total slots from dataset (default: all)")
    p.add_argument("--num-reads",        type=int,   default=1000)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no-polish",        action="store_true")
    p.add_argument("--use-leap",         action="store_true",
                   help="Leap hybrid solver (requires DWAVE_API_TOKEN)")
    p.add_argument("--use-qpu",          action="store_true",
                   help="Real D-Wave QPU (requires DWAVE_API_TOKEN)")
    p.add_argument("--leap-time-limit",  type=float, default=3.0)
    args = p.parse_args()

    out_schedule = Path("artifacts/results/schedule_annealer.csv")
    out_plot     = Path("artifacts/results/schedule_annealer.png")
    out_schedule.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.iloc[96:].reset_index(drop=True)   # skip zero-PV first day (TMY tz offset artifact)
    total = args.total_slots or len(df)
    df = df.iloc[:total].reset_index(drop=True)
    print(f"[rolling] dataset={len(df)} slots  window={args.window}  step={args.step}")

    schedule = run_rolling(
        df,
        bits=args.bits,
        num_reads=args.num_reads,
        seed=args.seed,
        polish=not args.no_polish,
        use_leap=args.use_leap,
        use_qpu=args.use_qpu,
        leap_time_limit=args.leap_time_limit,
        window=args.window,
        step=args.step,
    )
    schedule.to_csv(out_schedule, index=False)

    from qubo_model import DEMAND_CHARGE, EXPORT_RATE
    tou      = df["tou_usd_kwh"].to_numpy()
    energy   = float((tou * schedule["Grid_Import"].to_numpy() * 0.25).sum())
    export   = float((EXPORT_RATE * schedule["Grid_Export"].to_numpy() * 0.25).sum())
    peak     = float(schedule["Grid_Import"].max())
    cost     = energy + DEMAND_CHARGE * peak - export
    print(f"[result] total_cost=${cost:.2f}  energy=${energy:.2f}  "
          f"demand=${DEMAND_CHARGE * peak:.2f}  export=${export:.2f}  peak={peak:.1f}kW")
    print(f"[output] {out_schedule}")

    plot_schedule(schedule, df, f"D-Wave SA rolling MPC — ${cost:.2f}", out_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
