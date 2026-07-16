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

MILP_COLOR = "black"
GREEDY_COLOR = "#1E88E5"
HYBRID_COLOR = "#E53935"
MEAN_LINE_COLOR = "#26C6DA"
JITTER = 0.09


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

    fig, ax = plt.subplots(figsize=(9, 4.0))

    # Reference lines
    ax.axhline(0.0, color="gray", linewidth=1.1, linestyle="--", zorder=1)
    ax.axhline(1.0, color="black", linewidth=1.1, zorder=1)

    # MILP optimum sits at r = 1 by construction; drawn explicitly per window
    ax.scatter(x - JITTER, np.ones(n), s=55, color=MILP_COLOR, zorder=4)

    # Greedy heuristic
    ax.scatter(x, df["r_greedy"], s=55, color=GREEDY_COLOR, zorder=4)

    # GM-QAOA/Benders hybrid
    ax.errorbar(x + JITTER, df["r_hybrid_mean"], yerr=df["r_hybrid_std"],
                fmt="o", markersize=7.4, color=HYBRID_COLOR,
                ecolor=HYBRID_COLOR, capsize=2.5, capthick=1, linewidth=1,
                alpha=0.9, zorder=3)

    # Greedy mean across windows, annotated in-plot
    r_greedy_mean = df["r_greedy"].mean()
    ax.axhline(r_greedy_mean, color=MEAN_LINE_COLOR, linewidth=1.3,
               linestyle="--", zorder=2)
    ax.annotate(f"greedy mean r = {r_greedy_mean:.3f}",
                xy=(n - 1.05, r_greedy_mean), xycoords="data",
                fontsize=9, color=MEAN_LINE_COLOR, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Normalized approximation ratio  r", fontsize=10.5)

    fig.suptitle("Approximation ratio across T=5 dispatch windows",
                 fontsize=13, fontweight="bold", y=0.99)
    ax.set_title(
        r"$r = (C_{\mathrm{ref}} - C_{\mathrm{method}})\,/\,(C_{\mathrm{ref}} - C^{\star})$",
        fontsize=10, color="dimgray", pad=8)

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=7.4,
                   color=MILP_COLOR, label=r"MILP optimum ($C^{\star}$)"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=7.4,
                   color=GREEDY_COLOR, label="Greedy heuristic"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=7.4,
                   color=HYBRID_COLOR, label="GM-QAOA/Benders hybrid"),
        plt.Line2D([], [], color="gray", linewidth=1.1, linestyle="--",
                   label="Passive baseline (r = 0)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
              framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
