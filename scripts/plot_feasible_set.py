"""|F| per Benders round -- feasibility cuts clear whole state groups, optimality cuts don't.

Runs the hybrid Benders loop once and plots the feasible-set size after each
round's filtering. Bars are colored: red for a round with a feasibility cut
(annotated with the number of states removed), blue otherwise.

Run:  uv run python -m scripts.plot_feasible_set
      uv run python -m scripts.plot_feasible_set --start 0 --slots 2 --max-rounds 15
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt

from qc.benders import LoopRound, benders_loop
from qc.instance import load_instance

OUT_PNG = "artifacts/results/feasible_set.png"
OUT_CSV = "artifacts/results/feasible_set.csv"

# same palette as scripts/plot_tts.py (C_CLASSICAL / C_FIT)
C_CUT = "#0072B2"
C_NORMAL = "#000000"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_feasible_set(rounds: list[LoopRound]) -> plt.Figure:
    rounds_x = [r.round for r in rounds]
    n_states = [r.n_states for r in rounds]
    colors = [C_CUT if r.n_removed else C_NORMAL for r in rounds]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars = ax.bar(rounds_x, n_states, color=colors, width=0.7, zorder=3)
    for bar, r in zip(bars, rounds):
        if r.n_removed:
            ax.annotate(f"-{r.n_removed}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    ax.set_xlabel("$r$")
    ax.set_ylabel("$|F|$")
    ax.set_xticks(rounds_x)
    ax.margins(y=0.12)
    ax.grid(True, axis="y", which="major", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_CUT), plt.Rectangle((0, 0), 1, 1, color=C_NORMAL)]
    ax.legend(handles, ["Feasibility Cut", "Optimality Cut"], loc="upper right")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot |F| per Benders round")
    ap.add_argument("--data", default="artifacts/data/all_data.csv")
    ap.add_argument("--start", type=int, default=645)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--force-outage", type=int, default=None, metavar="T")
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--gap-tol", type=float, default=1e-4)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    inst = load_instance(args.data, start=args.start, T=args.slots,
                         force_outage=args.force_outage)
    result = benders_loop(inst, max_rounds=args.max_rounds, gap_tol=args.gap_tol,
                          shots=args.shots, seed=args.seed)

    fig = plot_feasible_set(result.rounds)
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PNG.replace(".png", ".pdf"))
    print(f"wrote {OUT_PNG} (+.pdf)")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "n_states", "n_removed"])
        writer.writeheader()
        writer.writerows({"round": r.round, "n_states": r.n_states, "n_removed": r.n_removed}
                         for r in result.rounds)
    print(f"wrote {OUT_CSV}")

    print(f"termination: {result.termination} after {len(result.rounds)} rounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
