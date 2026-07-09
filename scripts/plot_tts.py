"""TTS (Time-to-Solution) analysis from existing rounds-sweep and benchmark data.

Three panels:
  1. Convergence probability vs. Benders-rounds budget  (T=4 and T=5)
  2. TTS in query complexity vs. rounds budget
  3. Scaling comparison with dual y-axis:
       left  — Gurobi wall-clock TTS in seconds (T=1..5)
       right — QAOA query-complexity TTS in oracle calls (T=4,5, optimal rounds)

TTS definition (standard quantum computing formulation, p_target=0.99):

    TTS_query = (rounds_mean × shots) × ceil(log(1-p_target) / log(1-p_success))

where p_success = converged_frac (fraction of trials that hit the gap tolerance).
Query complexity = oracle calls = total QAOA measurements until first success.
shots = 1024 (fixed in all runs).

Uses:
    artifacts/results/qc_rounds_sweep.csv   (rounds-sweep, already exists)
    artifacts/results/qc_benchmark.csv      (Gurobi runtimes, already exists)

Run:
    ./.venv/bin/python -m scripts.plot_tts
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SWEEP_CSV = "artifacts/results/qc_rounds_sweep.csv"
BENCH_CSV = "artifacts/results/qc_benchmark.csv"
CLASSICAL_CSV = "results_classical.csv"
OUT = "artifacts/plots/tts_analysis.png"
SHOTS = 1024
P_TARGET = 0.99

COLOR_GUROBI = "#2E7D32"
COLOR_T4 = "#1565C0"
COLOR_T5 = "#E53935"
MARKER = {4: "o", 5: "s"}


def _tts_query(p_success: float, rounds_mean: float, shots: int) -> float:
    """Expected oracle calls to reach P_TARGET success probability."""
    if p_success <= 0 or np.isnan(p_success):
        return float("inf")
    n_trials = 1 if p_success >= 1.0 else math.ceil(
        math.log(1 - P_TARGET) / math.log(1 - p_success)
    )
    return n_trials * rounds_mean * shots


def _tts_wall(p_success: float, runtime_s: float) -> float:
    if p_success <= 0 or np.isnan(p_success):
        return float("inf")
    n_trials = 1 if p_success >= 1.0 else math.ceil(
        math.log(1 - P_TARGET) / math.log(1 - p_success)
    )
    return n_trials * runtime_s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=SWEEP_CSV)
    ap.add_argument("--bench", default=BENCH_CSV)
    ap.add_argument("--classical", default=CLASSICAL_CSV)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--shots", type=int, default=SHOTS)
    args = ap.parse_args(argv)

    # ── Load and prepare rounds-sweep data ────────────────────────────────────
    df = pd.read_csv(args.sweep)
    for col in ("converged_frac", "rounds_mean", "runtime_trial_mean_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tts_query"] = df.apply(
        lambda r: _tts_query(r["converged_frac"], r["rounds_mean"], args.shots), axis=1
    )
    df["tts_wall_s"] = df.apply(
        lambda r: _tts_wall(r["converged_frac"], r["runtime_trial_mean_s"]), axis=1
    )

    # ── Load and merge Gurobi data from both sources ───────────────────────────
    # Small T (1..5): from qc_benchmark.csv
    bench = pd.read_csv(args.bench)
    bench["runtime_gurobi_s"] = pd.to_numeric(bench["runtime_gurobi_s"], errors="coerce")
    gurobi_small = pd.DataFrame({"t": bench["t"], "runtime_s": bench["runtime_gurobi_s"]})

    # Larger T (96..2880): from results_classical.csv (deterministic, M=1)
    classical = pd.read_csv(args.classical)
    det = classical[(classical["mode"] == "deterministic") & (classical["n_scenarios"] == 1)]
    gurobi_large = pd.DataFrame({"t": det["slots"].values, "runtime_s": det["runtime_s"].values})  # type: ignore[union-attr]

    gurobi_all = (pd.concat([gurobi_small, gurobi_large], ignore_index=True)
                  .sort_values("t").reset_index(drop=True))

    # For panel 3: use T-sweep data (T=1..5) — all trials converged (p_success=1.0),
    # so TTS = rounds_mean × shots × 1 trial. These are real measurements, not extrapolation.
    bench_qaoa = bench.copy()
    bench_qaoa["rounds_mean"] = pd.to_numeric(bench_qaoa["rounds_mean"], errors="coerce")
    bench_qaoa["converged_frac"] = pd.to_numeric(bench_qaoa["converged_frac"], errors="coerce")
    bench_qaoa["tts_query"] = bench_qaoa.apply(
        lambda r: _tts_query(r["converged_frac"], r["rounds_mean"], args.shots), axis=1
    )
    qaoa_scaling: pd.DataFrame = bench_qaoa[bench_qaoa["tts_query"] < float("inf")].sort_values("t")  # type: ignore[assignment]

    t_values = sorted(df["t"].unique())
    colors = {4: COLOR_T4, 5: COLOR_T5}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: Convergence probability vs. rounds ────────────────────────────
    for t in t_values:
        sub: pd.DataFrame = df[df["t"] == t].sort_values("max_rounds")  # type: ignore[assignment]
        valid = sub[sub["converged_frac"].notna()]
        ax1.plot(valid["max_rounds"], valid["converged_frac"],
                 color=colors[int(t)], marker=MARKER[int(t)], markersize=7,
                 linewidth=1.8, label=f"T = {int(t)}  (|F| = {int(sub['n_feasible'].iloc[0]):,})")

    ax1.axhline(P_TARGET, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax1.text(float(df["max_rounds"].max()) * 0.98, P_TARGET + 0.01,  # type: ignore[arg-type]
             f"p = {P_TARGET}", ha="right", va="bottom", fontsize=8, color="gray")
    ax1.set_xlabel("Max Benders rounds (budget)", fontsize=10)
    ax1.set_ylabel("Convergence probability  $p_{\\mathrm{success}}$", fontsize=10)
    ax1.set_title("Convergence vs. rounds budget", fontsize=10)
    ax1.set_ylim(-0.05, 1.12)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Query-complexity TTS vs. rounds ───────────────────────────────
    for t in t_values:
        sub: pd.DataFrame = df[df["t"] == t].sort_values("max_rounds")  # type: ignore[assignment]
        finite: pd.DataFrame = sub[sub["tts_query"] < float("inf")]  # type: ignore[assignment]
        if finite.empty:
            continue
        ax2.plot(finite["max_rounds"], finite["tts_query"],
                 color=colors[int(t)], marker=MARKER[int(t)], markersize=7,
                 linewidth=1.8, label=f"T = {int(t)}")
        achieved: pd.DataFrame = finite[finite["converged_frac"] >= P_TARGET]  # type: ignore[assignment]
        if not achieved.empty:
            best = achieved.iloc[0]
            ax2.annotate(
                f" {int(best['tts_query']):,}\n @ {int(best['max_rounds'])} rounds",  # type: ignore[arg-type]
                xy=(float(best["max_rounds"]), float(best["tts_query"])),
                fontsize=8, color=colors[int(t)], va="center",
            )

    ax2.set_xlabel("Max Benders rounds (budget)", fontsize=10)
    ax2.set_ylabel(f"TTS (oracle calls, $p_{{\\mathrm{{target}}}}={P_TARGET}$)", fontsize=10)
    ax2.set_title(f"Query-complexity TTS\n(rounds × {args.shots} shots × expected trials)",
                  fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}"
    ))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Scaling comparison — Gurobi (s) vs. QAOA (oracle calls) ──────
    # x-axis limited to T=1..5: the only range with measured data for both solvers.
    # Gurobi data beyond T=5 is noted as an annotation rather than plotted
    # (next Gurobi point is T=96 — the gap makes the axis misleading otherwise).
    gurobi_shared = gurobi_all[gurobi_all["t"] <= 5]
    t_gurobi = gurobi_shared["t"].to_numpy(dtype=float)  # type: ignore[union-attr]
    rt_gurobi = gurobi_shared["runtime_s"].to_numpy(dtype=float)  # type: ignore[union-attr]

    ax3_left = ax3
    ax3_right = ax3.twinx()

    # Left axis: Gurobi wall-clock TTS (seconds), T=1..5
    l1, = ax3_left.plot(t_gurobi, rt_gurobi, color=COLOR_GUROBI, marker="D",
                        markersize=7, linewidth=1.8, label="Gurobi (wall-clock s)")
    ax3_left.set_ylabel("Gurobi TTS (seconds)", fontsize=10, color=COLOR_GUROBI)
    ax3_left.tick_params(axis="y", labelcolor=COLOR_GUROBI)
    ax3_left.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v*1000:.0f} ms" if v < 0.1 else f"{v:.3f} s"
    ))
    ax3_left.set_ylim(bottom=0)

    # Note: at T≤5 Gurobi times (3–14 ms) are dominated by Python/Gurobi setup
    # overhead — the non-monotonicity (T=4 > T=5) is pure measurement noise.
    ax3_left.text(0.98, 0.97,
                  "Note: T ≤ 5 runtimes (3–14 ms)\ndominated by setup overhead\n→ non-monotonicity is noise",
                  transform=ax3_left.transAxes, fontsize=7, color=COLOR_GUROBI,
                  va="top", ha="right",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_GUROBI, alpha=0.8))

    # Annotation: Gurobi continues well beyond T=5
    gurobi_t2880 = float(gurobi_all[gurobi_all["t"] == 2880]["runtime_s"].values[0])  # type: ignore[index]
    ax3_left.annotate(
        f"Gurobi continues:\nT=96 → 48 ms\nT=2880 → {gurobi_t2880:.1f} s",
        xy=(5, float(rt_gurobi[-1])), xytext=(4.3, float(rt_gurobi.max()) * 0.55),
        fontsize=7, color=COLOR_GUROBI,
        arrowprops=dict(arrowstyle="->", color=COLOR_GUROBI, lw=0.8),
        ha="right",
    )

    # Right axis: QAOA query-complexity TTS — T=1..5, all measured (p_success=1.0)
    qaoa_t = qaoa_scaling["t"].to_numpy(dtype=float)
    qaoa_tts = qaoa_scaling["tts_query"].to_numpy(dtype=float)
    l2, = ax3_right.plot(qaoa_t, qaoa_tts, color=COLOR_T5, marker="o",
                         markersize=8, linewidth=1.8, linestyle="--",
                         label="GM-QAOA (oracle calls, T ≤ 5)")
    ax3_right.set_ylabel("QAOA TTS (oracle calls)", fontsize=10, color=COLOR_T5)
    ax3_right.tick_params(axis="y", labelcolor=COLOR_T5)
    ax3_right.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}"
    ))
    ax3_right.set_ylim(bottom=0)

    ax3_left.set_xlabel("Window size T (slots)", fontsize=10)
    ax3_left.set_xticks([1, 2, 3, 4, 5])
    ax3_left.set_xlim(0.5, 5.5)
    ax3_left.set_title("Scaling comparison: Gurobi wall-clock  vs.  GM-QAOA oracle calls\n"
                        "T = 1..5  (measured, no extrapolation)", fontsize=10)
    ax3_left.legend(handles=[l1, l2], fontsize=8, loc="upper left")
    ax3_left.grid(alpha=0.3)

    fig.suptitle(f"TTS analysis — GM-QAOA Benders  |  $p_{{\\mathrm{{target}}}} = {P_TARGET}$,"
                 f"  shots = {args.shots}", fontsize=12, y=1.02)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\nTTS summary  (p_target={P_TARGET}, shots={args.shots})")
    print(f"{'T':>3}  {'max_rounds':>10}  {'p_success':>10}  {'rounds_mean':>12}  "
          f"{'TTS_query':>12}  {'TTS_wall_s':>12}")
    for _, row in df.sort_values(["t", "max_rounds"]).iterrows():
        tts_q = f"{row['tts_query']:>12,.0f}" if row["tts_query"] < float("inf") else f"{'∞':>12}"
        tts_w = f"{row['tts_wall_s']:>12.1f}" if row["tts_wall_s"] < float("inf") else f"{'∞':>12}"
        print(f"{int(row['t']):>3}  {int(row['max_rounds']):>10}  "  # type: ignore[arg-type]
              f"{row['converged_frac']:>10.2f}  {row['rounds_mean']:>12.1f}  {tts_q}  {tts_w}")

    print(f"\nGurobi TTS (wall-clock, single solve):")
    for _, row in gurobi_all.iterrows():
        print(f"  T={int(row['t'])}: {row['runtime_s']*1000:.1f} ms")  # type: ignore[arg-type]

    print(f"\nQAOA TTS from T-sweep (p_success=1.0 for all, measured rounds_mean):")
    for _, row in qaoa_scaling.iterrows():
        print(f"  T={int(row['t'])}: rounds_mean={float(row['rounds_mean']):.1f}"  # type: ignore[arg-type]
              f"  → {int(row['tts_query']):,} oracle calls")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
