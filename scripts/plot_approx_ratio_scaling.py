"""Approximation ratio of the greedy baseline vs. horizon length T.

Source: artifacts/classical_vs_greedy_scaling.csv (one row per horizon:
1_month/6_months/1_year). r_greedy = (C_ref_passive - C_greedy) /
(C_ref_passive - C_opt): passive controller = 0, MILP optimum = 1.

Run:  uv run python -m scripts.plot_approx_ratio_scaling
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt

CSV_PATH = "artifacts/classical_vs_greedy_scaling.csv"
OUT_PNG = "artifacts/results/approx_ratio_scaling.png"
OUT_CSV = "artifacts/results/approx_ratio_scaling.csv"

# same palette as scripts/plot_tts.py / plot_feasible_set.py (Okabe-Ito, colorblind-safe)
C_GREEDY = "#0072B2"
C_BOUND = "#000000"

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


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows = read_rows(CSV_PATH)
    t_vals = [int(r["T"]) for r in rows]
    r_vals = [float(r["r_greedy"]) for r in rows]
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t_vals, r_vals, "o-", color=C_GREEDY, label="greedy baseline", zorder=3)
    for i, (t, r, label) in enumerate(zip(t_vals, r_vals, labels)):
        ha = "left" if i == 0 else "right" if i == len(t_vals) - 1 else "center"
        xoff = 10 if i == 0 else -10 if i == len(t_vals) - 1 else 0
        above = i % 2 == 1
        va = "bottom" if above else "top"
        yoff = 10 if above else -10
        ax.annotate(f"{label.replace('_', ' ')}\nr={r:.3f}", xy=(t, r), xytext=(xoff, yoff),
                    textcoords="offset points", ha=ha, va=va, fontsize=8)

    ax.margins(x=0.06)

    ax.axhline(0.0, color=C_BOUND, linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color=C_BOUND, linestyle="--", linewidth=0.8)
    ax.text(t_vals[0], 0.0, "passive", va="bottom", ha="left", fontsize=9, color=C_BOUND)
    ax.text(t_vals[-1], 1.0, "MILP optimum", va="bottom", ha="right", fontsize=9, color=C_BOUND)

    ax.set_xscale("log")
    ax.set_xticks(t_vals)
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.set_xticklabels([f"{t:,}" for t in t_vals], rotation=40, ha="right")
    ax.set_xlabel("$T$")
    ax.set_ylabel("$r_\\mathrm{greedy}$")
    ax.set_ylim(-0.2, 1.15)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PNG.replace(".png", ".pdf"))
    print(f"wrote {OUT_PNG} (+.pdf)")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "T", "r_greedy"])
        writer.writeheader()
        writer.writerows({"label": label, "T": t, "r_greedy": r}
                          for label, t, r in zip(labels, t_vals, r_vals))
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
