"""TTS scaling: fit log S(T) from measured runs, compose with analytic depth.

Model (see scripts/tts_experiment.py):

    TTS(T) = S(T) x t_circuit(T)        [depth units: t_gate = 1]

S(T) is the measured shot budget (sum of per-round shots_99 along the exact
Benders trajectory: sampler="exact", the shots->infinity limit of best-of-
shots, so the round trajectory is a deterministic instance/cut property).
t_circuit is NEVER fitted — qc/depth.py computes it analytically for any T
(ancilla-assisted reflection, the default in qc.depth.circuit_depth).

Online and outage windows are two distinct regimes (27 vs. 18 structurally
feasible patterns per slot, plus outage carries a direct-cost signal that
concentrates the QAOA distribution) and are fitted SEPARATELY — pooling them
lowers the online R^2 without adding information (see the prior all-window
fit in git history). The online fit is the one compared against theory,
since it isolates the pure-enumeration regime: S ~ b^T against the Grover
bound sqrt(27) per slot (the per-shot success probability that must be
overcome is 1/|F| ~ 27^-T, so the number of shots for constant success
probability scales as sqrt(27)^T).

Extrapolation is capped at T=16 (4 hours of 15-min slots): far enough to
show the growth-rate contrast against Gurobi without the fit exponent
overflowing into physically meaningless magnitudes at T=96+.

Twin-axis convention follows scripts/plot_scaling.py: seconds and depth
units are not commensurable, so both axes get the SAME number of log
decades, anchored at their own minimum -- equal visual slope = equal
multiplicative growth. Crossings still mean nothing; the message is the
growth-rate contrast, not an absolute time comparison.

Run:  uv run python -m scripts.plot_tts
      -> artifacts/results/tts_comparison.png(.pdf)   TTS vs Gurobi, twin axis
         artifacts/results/tts_shot_budget.png(.pdf)  S(T) fit + Grover bound
      + fit report (with 95% CI) on stdout
"""

from __future__ import annotations

import csv
from math import ceil, log10

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from qc.depth import PREP_DEPTH_ONLINE, circuit_depth

TTS_CSV = "artifacts/tts_scaling.csv"
CLASSICAL_CSV = "artifacts/classical_scaling.csv"
OUT_PNG = "artifacts/results/tts_comparison.png"
OUT_S_PNG = "artifacts/results/tts_shot_budget.png"

P_LAYERS = 6
T_EXTRAPOLATE_MAX = 16      # 4 hours of 15-min slots
PAD = 1.5
CONFIDENCE = 0.95

# Okabe-Ito colorblind-safe palette
C_ONLINE = "#0072B2"
C_OUTAGE = "#D55E00"
C_CLASSICAL = "#000000"
C_FIT = "#0072B2"
C_GROVER = "#009E73"

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


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fit_log_s(ts: np.ndarray, s: np.ndarray) -> dict:
    """Least squares on log10 S = c0 + c1*T, with 95% CI band ingredients.

    Returns a dict so predict_band() can evaluate the confidence band (of
    the mean regression line, not a prediction interval) at arbitrary T.
    """
    y = np.log10(s)
    n = len(ts)
    c1, c0 = np.polyfit(ts, y, 1)
    resid = y - (c0 + c1 * ts)
    dof = n - 2
    s_resid = float(np.sqrt((resid ** 2).sum() / dof)) if dof > 0 else 0.0
    xbar = float(ts.mean())
    sxx = float(((ts - xbar) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    tcrit = float(stats.t.ppf(0.5 + CONFIDENCE / 2, dof)) if dof > 0 else 0.0
    return dict(c0=float(c0), c1=float(c1), r2=r2, n=n, dof=dof,
               s_resid=s_resid, xbar=xbar, sxx=sxx, tcrit=tcrit)


def predict_band(fit: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yhat, lower, upper) in log10 space at the given T points."""
    yhat = fit["c0"] + fit["c1"] * t
    if fit["dof"] <= 0 or fit["sxx"] == 0:
        return yhat, yhat, yhat
    se = fit["s_resid"] * np.sqrt(1.0 / fit["n"] + (t - fit["xbar"]) ** 2 / fit["sxx"])
    band = fit["tcrit"] * se
    return yhat, yhat - band, yhat + band


def log10_depth(t: np.ndarray) -> np.ndarray:
    depth = np.array([circuit_depth(8 * int(ti), P_LAYERS, PREP_DEPTH_ONLINE)
                      for ti in t], dtype=float)
    return np.log10(depth)


def matched_decade_ylim(data_min: float, data_max: float, decades: float) -> tuple[float, float]:
    low = data_min / PAD
    high = low * 10 ** decades
    needed = log10((data_max * PAD) / low)
    if needed > decades:
        high = low * 10 ** needed
    return low, high


def main() -> int:
    runs = read_csv(TTS_CSV)
    not_converged = [r for r in runs if r["termination"] != "gap"]
    if not_converged:
        print(f"WARNING: {len(not_converged)} runs did not converge; "
              "their S underestimates the true budget")

    regimes = {}
    for kind in ("online", "outage"):
        ts = np.array([int(r["t"]) for r in runs if r["window"] == kind], dtype=float)
        s = np.array([float(r["S"]) for r in runs if r["window"] == kind])
        regimes[kind] = dict(ts=ts, s=s, fit=fit_log_s(ts, s))

    for kind, color in (("online", C_ONLINE), ("outage", C_OUTAGE)):
        f = regimes[kind]["fit"]
        base = 10 ** f["c1"]
        ci_lo, ci_hi = 10 ** (f["c1"] - f["tcrit"] * f["s_resid"] / np.sqrt(f["sxx"])), \
                       10 ** (f["c1"] + f["tcrit"] * f["s_resid"] / np.sqrt(f["sxx"]))
        print(f"{kind:>7}: log10 S(T) = {f['c0']:.3f} + {f['c1']:.3f} T   "
              f"(R^2={f['r2']:.3f}, n={f['n']})")
        print(f"         growth base = {base:.2f}/slot, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    grover = float(np.sqrt(27.0))
    print(f"  Grover-bound reference: sqrt(27) = {grover:.2f}/slot "
          "(sqrt(|F|) scaling, |F|=27^T online)")

    # ---------------- Figure A: shot-budget evidence ----------------
    fig_s, ax_s = plt.subplots(figsize=(4.5, 3.6))
    t_fit = np.linspace(1, 5, 50)
    for kind, color in (("online", C_ONLINE), ("outage", C_OUTAGE)):
        d = regimes[kind]
        ax_s.plot(d["ts"], d["s"], "o", color=color, alpha=0.75,
                 label=f"{kind} windows", zorder=3)
        yhat, lo, hi = predict_band(d["fit"], t_fit)
        ax_s.fill_between(t_fit, 10 ** lo, 10 ** hi, color=color, alpha=0.15, linewidth=0)
        ax_s.plot(t_fit, 10 ** yhat, "-", color=color, linewidth=1.3,
                 label=f"fit: ${10**d['fit']['c1']:.2f}^T$")

    anchor = 10 ** regimes["online"]["fit"]["c0"] * 10 ** regimes["online"]["fit"]["c1"]
    ax_s.plot(t_fit, anchor * grover ** (t_fit - 1.0), ":", color=C_GROVER,
              label=r"Grover bound $\sqrt{27}^{\,T}$")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("$T$ (15-min slots)")
    ax_s.set_ylabel(r"$S(T)$ (shots for 99% success)")
    ax_s.set_xticks([1, 2, 3, 4, 5])
    ax_s.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_s.grid(True, which="minor", alpha=0.1, linewidth=0.3)
    ax_s.legend(loc="upper left", ncol=1)
    fig_s.tight_layout()
    fig_s.savefig(OUT_S_PNG)
    fig_s.savefig(OUT_S_PNG.replace(".png", ".pdf"))
    print(f"wrote {OUT_S_PNG} (+.pdf)")

    # ---------------- Figure B: TTS vs. Gurobi, twin axis ----------------
    f_on = regimes["online"]["fit"]
    ts_meas, s_meas = regimes["online"]["ts"], regimes["online"]["s"]
    tts_meas_log = np.log10(s_meas) + log10_depth(ts_meas)

    t_line = np.linspace(1, T_EXTRAPOLATE_MAX, 200)
    s_line_log, s_lo_log, s_hi_log = predict_band(f_on, t_line)
    depth_log = log10_depth(t_line)
    tts_line_log = s_line_log + depth_log
    tts_lo_log = s_lo_log + depth_log
    tts_hi_log = s_hi_log + depth_log

    classical = [r for r in read_csv(CLASSICAL_CSV) if int(r["t"]) <= T_EXTRAPOLATE_MAX]
    t_c = np.array([int(r["t"]) for r in classical], dtype=float)
    runtime = np.array([float(r["runtime_s"]) for r in classical])

    tts_all = 10 ** np.concatenate([tts_meas_log, tts_line_log])
    runtime_decades = log10((runtime.max() * PAD) / (runtime.min() / PAD))
    tts_decades = log10((tts_all.max() * PAD) / (tts_all.min() / PAD))
    decades = ceil(max(runtime_decades, tts_decades))
    runtime_lo, runtime_hi = matched_decade_ylim(runtime.min(), runtime.max(), decades)
    tts_lo, tts_hi = matched_decade_ylim(tts_all.min(), tts_all.max(), decades)

    fig, ax1 = plt.subplots(figsize=(6.0, 4.2))
    ax2 = ax1.twinx()

    l1, = ax1.semilogy(t_c, runtime, "o-", color=C_CLASSICAL,
                       label="Gurobi MILP runtime (measured)")
    ax2.fill_between(t_line, 10 ** tts_lo_log, 10 ** tts_hi_log,
                     color=C_FIT, alpha=0.15, linewidth=0)
    l2, = ax2.semilogy(ts_meas, 10 ** tts_meas_log, "o", color=C_FIT,
                       label="hybrid TTS (measured)")
    l3, = ax2.semilogy(t_line, 10 ** tts_line_log, "--", color=C_FIT,
                       label="hybrid TTS (fit, 95% CI)")

    for ax in (ax1, ax2):
        ax.set_yscale("log")
    ax1.set_ylim(runtime_lo, runtime_hi)
    ax2.set_ylim(tts_lo, tts_hi)
    ax1.set_xlabel("$T$ (15-min slots)")
    ax1.set_ylabel("Gurobi MILP runtime (s)", color=C_CLASSICAL)
    ax2.set_ylabel("hybrid TTS (gate-layer units)", color=C_FIT)
    ax1.tick_params(axis="y", labelcolor=C_CLASSICAL)
    ax2.tick_params(axis="y", labelcolor=C_FIT)
    ax1.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax1.legend(handles=[l1, l2, l3], loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PNG.replace(".png", ".pdf"))
    print(f"wrote {OUT_PNG} (+.pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
