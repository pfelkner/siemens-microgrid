"""Plot normalized approximation ratios from approx_ratio_sweep results.

Usage:
    ./.venv/bin/python -m scripts.plot_approx_ratio
    ./.venv/bin/python -m scripts.plot_approx_ratio --out artifacts/plots/approx_ratio.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

RESULTS = "artifacts/results/approx_ratio_results.csv"
OUT = "artifacts/plots/approx_ratio.png"

WINDOW_LABELS = {
    "peak_eve_d1":   "Peak\nDay 1",
    "peak_eve_d30":  "Peak\nDay 30",
    "pv_surplus_a":  "PV\nSurplus A",
    "pv_surplus_b":  "PV\nSurplus B",
    "high_load_a":   "High\nLoad A",
    "high_load_b":   "High\nLoad B",
    "outage_nat":    "Outage\n(nat.)",
    "outage_forced": "Outage\n(forced)",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.results)
    df["r_greedy"] = pd.to_numeric(df["r_greedy"], errors="coerce")
    df["r_hybrid_mean"] = pd.to_numeric(df["r_hybrid_mean"], errors="coerce")
    df["r_hybrid_std"] = pd.to_numeric(df["r_hybrid_std"], errors="coerce")

    n = len(df)
    x = np.arange(n)
    labels = [WINDOW_LABELS.get(lbl, lbl) for lbl in df["label"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Reference lines: passive = 0, MILP = 1
    ax.axhline(0.0, color="gray", linewidth=1.2, linestyle="--", label="Passive (r = 0)")
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="MILP optimum (r = 1)")

    # Greedy per window
    ax.scatter(x, df["r_greedy"], marker="s", s=70, color="#2196F3",
               zorder=3, label="Greedy heuristic")

    # Hybrid per window + std error bars
    ax.errorbar(x, df["r_hybrid_mean"], yerr=df["r_hybrid_std"],
                fmt="o", markersize=7, color="#E53935",
                capsize=4, capthick=1.5, linewidth=1.5,
                zorder=4, label="GM-QAOA hybrid (mean ± std)")

    # Mean lines across windows
    r_greedy_mean = df["r_greedy"].mean()
    r_hybrid_mean = df["r_hybrid_mean"].mean()
    ax.axhline(r_greedy_mean, color="#2196F3", linewidth=1, linestyle=":",
               alpha=0.8)
    ax.axhline(r_hybrid_mean, color="#E53935", linewidth=1, linestyle=":",
               alpha=0.8)

    # Annotate means on the right margin
    ax.annotate(f"μ={r_greedy_mean:.3f}", xy=(n - 0.5, r_greedy_mean),
                xycoords="data", fontsize=8, color="#2196F3",
                va="center", ha="left")
    ax.annotate(f"μ={r_hybrid_mean:.3f}", xy=(n - 0.5, r_hybrid_mean),
                xycoords="data", fontsize=8, color="#E53935",
                va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.6, n - 0.1)
    ax.set_ylim(-0.05, 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Normalized approximation ratio  r", fontsize=11)
    ax.set_xlabel("Window", fontsize=11)
    ax.set_title("Approximation ratio across T=5 windows\n"
                 r"$r = (C_\mathrm{ref} - C_\mathrm{method})\;/\;(C_\mathrm{ref} - C_\mathrm{opt})$",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
