"""Relative approximation ratio vs. Benders round budget, for T=4 and T=5.

Source: artifacts/qc_rounds_sweep.csv (one row per (t, max_rounds), aggregated
over n_trials trials). The raw approx_ratio_mean/std is misleading at low
round counts: non-converged trials produce wildly negative cost_quantum
values (sign flips vs. cost_classical), which blow up the std to O(400-600)
and dominate the mean. converged_frac -- the fraction of trials that reached
the gap-tolerance convergence criterion -- is bounded in [0, 1], monotone in
rounds, and free of that outlier problem, so it is used here as the relative
approximation ratio instead of the raw absolute one.

Run:  uv run python -m scripts.plot_rounds_sweep
"""

from __future__ import annotations

import csv
import math

import matplotlib.pyplot as plt

CSV_PATH = "artifacts/qc_rounds_sweep.csv"
OUT_PNG = "artifacts/results/rounds_sweep.png"
OUT_PDF = "artifacts/results/rounds_sweep.pdf"
OUT_CSV = "artifacts/results/rounds_sweep.csv"

COLORS = {1: "tab:green", 2: "tab:purple", 3: "tab:red", 4: "tab:blue", 5: "tab:orange"}

plt.rcParams.update({"font.size": 12})


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows = read_csv(CSV_PATH)

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = []

    for t in sorted({int(r["t"]) for r in rows}):
        pts = [r for r in rows if int(r["t"]) == t and r["converged_frac"] != "nan"
               and not math.isnan(float(r["converged_frac"]))]
        pts.sort(key=lambda r: int(r["max_rounds"]))
        rounds = [int(r["max_rounds"]) for r in pts]
        pct_converged = [float(r["converged_frac"]) * 100 for r in pts]
        n_trials = [int(r["n_trials"]) for r in pts]

        ax.plot(rounds, pct_converged, "o-", color=COLORS.get(t), label=f"T={t}")
        plotted += [{"t": t, "max_rounds": r, "pct_converged": p, "n_trials": n}
                   for r, p, n in zip(rounds, pct_converged, n_trials)]

    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("max rounds")
    ax.set_ylabel("% trials converged")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "max_rounds", "pct_converged", "n_trials"])
        writer.writeheader()
        writer.writerows(plotted)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
