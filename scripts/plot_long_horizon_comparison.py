"""Approximation ratio of the greedy baseline over the long horizon.

Source: artifacts/results/long_horizon_comparison.csv (from
scripts.long_horizon_comparison, T=2880 / 30 days). r_greedy = (C_ref -
C_greedy) / (C_ref - C_opt): passive controller = 0, MILP optimum = 1.

Run:  .venv/bin/python -m scripts.plot_long_horizon_comparison
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt

CSV_PATH = "artifacts/results/long_horizon_comparison.csv"
OUT_PNG = "artifacts/results/long_horizon_comparison.png"
OUT_PDF = "artifacts/results/long_horizon_comparison.pdf"
OUT_CSV = "artifacts/results/long_horizon_comparison_plot.csv"

plt.rcParams.update({"font.size": 12})


def read_row(path: str) -> dict:
    with open(path, newline="") as f:
        return next(csv.DictReader(f))


def main() -> int:
    row = read_row(CSV_PATH)
    r_greedy = float(row["r_greedy"])

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.bar(["greedy"], [r_greedy], color="tab:orange", width=0.5)
    ax.annotate(f"{r_greedy:.3f}", xy=(0, r_greedy), xytext=(0, 4),
               textcoords="offset points", ha="center", fontsize=10)

    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(-0.55, 0.0, "passive", va="bottom", ha="left", fontsize=9, color="gray")
    ax.text(-0.55, 1.0, "MILP optimum", va="bottom", ha="left", fontsize=9, color="gray")

    ax.set_ylabel("r_greedy (normalized approximation ratio)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(-0.6, 0.6)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "r_greedy"])
        writer.writeheader()
        writer.writerow({"label": "long horizon (T=2880, 30 days)", "r_greedy": r_greedy})
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
