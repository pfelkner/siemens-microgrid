"""Scenario generator for the stochastic two-stage microgrid extension.

Default (regime-switching) mode: per-day regime draws set the AR(1) innovation
standard deviation for PV (3 regimes: clear/partly_cloudy/overcast) and load
(2 regimes: normal/event). This produces fat-tailed, heteroskedastic scenarios
that concentrate uncertainty on cloudy / high-load-event days.

Homogeneous back-compat mode (homogeneous=True): constant sigma AR(1) identical
to the original Step-3 generator; same seed reproduces bit-for-bit results.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96  # 15-min slots in 24 h

PV_REGIME_NAMES  = ("clear", "partly_cloudy", "overcast")
LOAD_REGIME_NAMES = ("normal", "event")


def _draw_daily_regimes(
    T: int,
    rng: np.random.Generator,
    probs: tuple[float, ...],
) -> np.ndarray:
    """Return a length-T integer array giving the regime index per slot."""
    n_days = int(np.ceil(T / SLOTS_PER_DAY))
    day_regime = rng.choice(len(probs), size=n_days, p=np.asarray(probs))
    return np.repeat(day_regime, SLOTS_PER_DAY)[:T]


def _ar1_with_regime_sigma(
    T: int,
    rng: np.random.Generator,
    sigma_t: np.ndarray,
    phi: float,
) -> np.ndarray:
    """AR(1) process where innovation std is slot-dependent.

    eps[t] = phi * eps[t-1] + N(0, sqrt(1 - phi^2) * sigma_t[t])
    The stationary std of each slot equals sigma_t[t] when phi→0,
    and the within-day temporal correlation is preserved via phi.
    """
    innov_scale = np.sqrt(1.0 - phi ** 2) * sigma_t
    eps = np.empty(T)
    eps[0] = rng.normal(0.0, sigma_t[0])
    for t in range(1, T):
        eps[t] = phi * eps[t - 1] + rng.normal(0.0, innov_scale[t])
    return eps


def generate_scenarios(
    df: pd.DataFrame,
    n_scenarios: int,
    pv_regime_probs: tuple[float, ...] = (0.55, 0.30, 0.15),
    pv_regime_sigmas: tuple[float, ...] = (0.03, 0.15, 0.40),
    load_event_prob: float = 0.05,
    load_sigmas: tuple[float, float] = (0.02, 0.30),
    ar1_phi: float = 0.85,
    seed: int | None = None,
    homogeneous: bool = False,
    pv_noise_sigma: float = 0.15,    # used when homogeneous=True
    load_noise_sigma: float = 0.05,  # used when homogeneous=True
) -> list[pd.DataFrame]:
    """Generate n_scenarios noisy realizations of (p_kw, load_kw).

    Regime-switching mode (default, homogeneous=False):
      - PV sigma is drawn per calendar day from pv_regime_probs/pv_regime_sigmas.
      - Load sigma is drawn per day from a Bernoulli with load_event_prob.
      - AR(1) with phi shapes within-day temporal correlation; the regime sets
        the innovation amplitude for that day.

    Homogeneous mode (homogeneous=True):
      - Constant pv_noise_sigma / load_noise_sigma, identical to Step-3 behaviour.
      - Same seed reproduces bit-for-bit results from that version.

    tou_usd_kwh and grid_available are never randomized.
    """
    rng = np.random.default_rng(seed)
    T = len(df)
    p_pv_base   = df["p_kw"].to_numpy(dtype=float)
    p_load_base = df["load_kw"].to_numpy(dtype=float)

    if homogeneous:
        # Original Step-3 behaviour
        innov_std = np.sqrt(1.0 - ar1_phi ** 2)
        scenarios: list[pd.DataFrame] = []
        for _ in range(n_scenarios):
            eps_pv   = np.zeros(T)
            eps_load = np.zeros(T)
            eps_pv[0]   = rng.normal(0.0, pv_noise_sigma)
            eps_load[0] = rng.normal(0.0, load_noise_sigma)
            for t in range(1, T):
                eps_pv[t]   = ar1_phi * eps_pv[t - 1]   + rng.normal(0.0, innov_std * pv_noise_sigma)
                eps_load[t] = ar1_phi * eps_load[t - 1]  + rng.normal(0.0, innov_std * load_noise_sigma)
            p_pv_s   = np.maximum(0.0, p_pv_base   * (1.0 + eps_pv))
            p_load_s = np.maximum(0.0, p_load_base * (1.0 + eps_load))
            df_s = df.copy()
            df_s["p_kw"]    = p_pv_s
            df_s["load_kw"] = p_load_s
            scenarios.append(df_s)
        return scenarios

    # ---- Regime-switching mode ----
    load_regime_probs = (1.0 - load_event_prob, load_event_prob)
    pv_sigma_map   = np.array(pv_regime_sigmas,  dtype=float)
    load_sigma_map = np.array(load_sigmas,        dtype=float)

    scenarios = []
    regime_records: list[tuple[np.ndarray, np.ndarray]] = []  # (pv_regime, load_regime) per scenario

    for _ in range(n_scenarios):
        slot_regime_pv   = _draw_daily_regimes(T, rng, pv_regime_probs)
        slot_regime_load = _draw_daily_regimes(T, rng, load_regime_probs)

        sigma_pv_t   = pv_sigma_map[slot_regime_pv]
        sigma_load_t = load_sigma_map[slot_regime_load]

        eps_pv   = _ar1_with_regime_sigma(T, rng, sigma_pv_t,   ar1_phi)
        eps_load = _ar1_with_regime_sigma(T, rng, sigma_load_t, ar1_phi)

        p_pv_s   = np.maximum(0.0, p_pv_base   * (1.0 + eps_pv))
        p_load_s = np.maximum(0.0, p_load_base * (1.0 + eps_load))

        df_s = df.copy()
        df_s["p_kw"]    = p_pv_s
        df_s["load_kw"] = p_load_s
        scenarios.append(df_s)
        regime_records.append((slot_regime_pv, slot_regime_load))

    # Attach regime arrays as metadata for diagnostics / plotting
    for sc, (rp, rl) in zip(scenarios, regime_records):
        sc.attrs["slot_regime_pv"]   = rp
        sc.attrs["slot_regime_load"] = rl

    return scenarios


def print_regime_table(
    scenarios: list[pd.DataFrame],
    T: int,
) -> None:
    """Print per-day regime summary for the first scenario."""
    if not scenarios or "slot_regime_pv" not in scenarios[0].attrs:
        return
    slot_regime_pv   = scenarios[0].attrs["slot_regime_pv"]
    slot_regime_load = scenarios[0].attrs["slot_regime_load"]
    n_days = int(np.ceil(T / SLOTS_PER_DAY))
    print(f"\n{'Day':>4}  {'PV regime':<15}  {'σ_PV':>6}   {'Load regime':<12}  {'σ_load':>7}")
    print("-" * 56)
    pv_sigmas   = (0.03, 0.15, 0.40)
    load_sigmas = (0.02, 0.30)
    for d in range(n_days):
        slot = d * SLOTS_PER_DAY
        rp = slot_regime_pv[slot]
        rl = slot_regime_load[slot]
        print(
            f"{d:>4}  {PV_REGIME_NAMES[rp]:<15}  {pv_sigmas[rp]:>6.2f}"
            f"   {LOAD_REGIME_NAMES[rl]:<12}  {load_sigmas[rl]:>7.2f}"
        )
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Scenario generator for stochastic microgrid solver")
    p.add_argument("--data",           default="artifacts/data/all_data.csv")
    p.add_argument("--slots",          type=int,   default=96)
    p.add_argument("--scenarios",      type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=0)
    # Regime-switching knobs
    p.add_argument("--pv-regime-probs",  type=float, nargs=3, default=[0.55, 0.30, 0.15],
                   metavar=("P_CLEAR", "P_PARTLY", "P_OVERCAST"),
                   help="PV regime probabilities (must sum to 1)")
    p.add_argument("--pv-regime-sigmas", type=float, nargs=3, default=[0.03, 0.15, 0.40],
                   metavar=("S_CLEAR", "S_PARTLY", "S_OVERCAST"),
                   help="PV AR(1) noise std per regime")
    p.add_argument("--load-event-prob",  type=float, default=0.05,
                   help="Daily probability of a load-event day")
    p.add_argument("--load-sigmas",      type=float, nargs=2, default=[0.02, 0.30],
                   metavar=("S_NORMAL", "S_EVENT"),
                   help="Load AR(1) noise std for normal / event days")
    # Back-compat homogeneous mode
    p.add_argument("--homogeneous",  action="store_true",
                   help="Use constant-sigma AR(1) (Step-3 behaviour)")
    p.add_argument("--pv-sigma",     type=float, default=0.15,
                   help="PV sigma for homogeneous mode")
    p.add_argument("--load-sigma",   type=float, default=0.05,
                   help="Load sigma for homogeneous mode")
    p.add_argument("--plot",         action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    n = min(args.slots, len(df))
    df = df.iloc[:n].reset_index(drop=True)

    sc = generate_scenarios(
        df, args.scenarios,
        pv_regime_probs=tuple(args.pv_regime_probs),
        pv_regime_sigmas=tuple(args.pv_regime_sigmas),
        load_event_prob=args.load_event_prob,
        load_sigmas=tuple(args.load_sigmas),
        seed=args.seed,
        homogeneous=args.homogeneous,
        pv_noise_sigma=args.pv_sigma,
        load_noise_sigma=args.load_sigma,
    )

    mode = "homogeneous" if args.homogeneous else "regime-switching"
    print(f"Generated {args.scenarios} scenarios, T={n} slots [{mode}]")

    if not args.homogeneous:
        print_regime_table(sc, n)

    pv_mean   = df["p_kw"].mean()
    load_mean = df["load_kw"].mean()
    print(f"{'Scenario':>10}  {'PV ratio':>10}  {'Load ratio':>10}")
    for i, df_s in enumerate(sc):
        pv_r   = df_s["p_kw"].mean()   / max(pv_mean,   1e-9)
        load_r = df_s["load_kw"].mean() / max(load_mean, 1e-9)
        print(f"{i:>10}  {pv_r:>10.4f}  {load_r:>10.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        pv_regime_colors = {0: "gold", 1: "steelblue", 2: "dimgray"}  # clear / partly / overcast

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(df["p_kw"].values,    "k-", lw=2, label="deterministic", zorder=10)
        ax2.plot(df["load_kw"].values, "k-", lw=2, label="deterministic", zorder=10)

        for i, df_s in enumerate(sc):
            if not args.homogeneous and "slot_regime_pv" in df_s.attrs:
                # Colour each day's PV trace by its regime
                slot_regime_pv = df_s.attrs["slot_regime_pv"]
                pv_vals = df_s["p_kw"].values
                xs = np.arange(n)
                n_days = int(np.ceil(n / SLOTS_PER_DAY))
                for d in range(n_days):
                    sl = slice(d * SLOTS_PER_DAY, min((d + 1) * SLOTS_PER_DAY, n))
                    regime = slot_regime_pv[d * SLOTS_PER_DAY]
                    color = pv_regime_colors[regime]
                    ax1.plot(xs[sl], pv_vals[sl], color=color, alpha=0.4, lw=0.8)
                ax2.plot(df_s["load_kw"].values, alpha=0.35, lw=0.8, label=f"s{i}")
            else:
                ax1.plot(df_s["p_kw"].values,    alpha=0.5, label=f"s{i}")
                ax2.plot(df_s["load_kw"].values, alpha=0.5, label=f"s{i}")

        if not args.homogeneous:
            from matplotlib.lines import Line2D
            legend_els = [
                Line2D([0], [0], color="gold",     lw=2, label="clear (σ=0.03)"),
                Line2D([0], [0], color="steelblue", lw=2, label="partly cloudy (σ=0.15)"),
                Line2D([0], [0], color="dimgray",   lw=2, label="overcast (σ=0.40)"),
                Line2D([0], [0], color="k",         lw=2, label="deterministic"),
            ]
            ax1.legend(handles=legend_els, loc="upper right", fontsize=8)

        ax1.set_ylabel("PV power (kW)")
        ax2.set_ylabel("Load (kW)")
        ax2.set_xlabel("Slot")
        title_mode = "homogeneous" if args.homogeneous else "regime-switching"
        ax1.set_title(f"Scenarios (T={n}, M={args.scenarios}, {title_mode})")
        plt.tight_layout()
        out = "scenarios_preview.png"
        plt.savefig(out, dpi=120)
        print(f"Plot saved to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
