"""Scenario generator for the stochastic two-stage microgrid extension."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def generate_scenarios(
    df: pd.DataFrame,
    n_scenarios: int,
    pv_noise_sigma: float = 0.15,
    load_noise_sigma: float = 0.05,
    ar1_phi: float = 0.85,
    seed: int | None = None,
) -> list[pd.DataFrame]:
    """Generate n_scenarios noisy realizations of (p_kw, load_kw).

    PV and load each get an AR(1) noise process:
        eps_t = phi * eps_{t-1} + N(0, sqrt(1 - phi^2) * sigma)
    Noise is multiplicative; PV is clipped at 0 (negative PV is unphysical).
    tou_usd_kwh and grid_available are not randomized.
    """
    rng = np.random.default_rng(seed)
    T = len(df)
    p_pv_base   = df["p_kw"].to_numpy(dtype=float)
    p_load_base = df["load_kw"].to_numpy(dtype=float)
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


def main() -> int:
    p = argparse.ArgumentParser(description="Scenario generator for stochastic microgrid solver")
    p.add_argument("--data",       default="all_data.csv")
    p.add_argument("--slots",      type=int,   default=96)
    p.add_argument("--scenarios",  type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--pv-sigma",   type=float, default=0.15)
    p.add_argument("--load-sigma", type=float, default=0.05)
    p.add_argument("--plot",       action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    n = min(args.slots, len(df))
    df = df.iloc[:n].reset_index(drop=True)

    sc = generate_scenarios(
        df, args.scenarios,
        pv_noise_sigma=args.pv_sigma,
        load_noise_sigma=args.load_sigma,
        seed=args.seed,
    )

    print(f"Generated {args.scenarios} scenarios, T={n} slots")
    pv_mean = df["p_kw"].mean()
    load_mean = df["load_kw"].mean()
    print(f"{'Scenario':>10}  {'PV ratio':>10}  {'Load ratio':>10}")
    for i, df_s in enumerate(sc):
        pv_r   = df_s["p_kw"].mean()   / max(pv_mean,   1e-9)
        load_r = df_s["load_kw"].mean() / max(load_mean, 1e-9)
        print(f"{i:>10}  {pv_r:>10.4f}  {load_r:>10.4f}")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(df["p_kw"].values,    "k-", lw=2, label="deterministic", zorder=10)
        ax2.plot(df["load_kw"].values, "k-", lw=2, label="deterministic", zorder=10)
        for i, df_s in enumerate(sc):
            ax1.plot(df_s["p_kw"].values,    alpha=0.5, label=f"s{i}")
            ax2.plot(df_s["load_kw"].values, alpha=0.5, label=f"s{i}")
        ax1.set_ylabel("PV power (kW)")
        ax2.set_ylabel("Load (kW)")
        ax2.set_xlabel("Slot")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title(
            f"Scenarios (T={n}, M={args.scenarios}, "
            f"σ_PV={args.pv_sigma:.0%}, σ_load={args.load_sigma:.0%})"
        )
        plt.tight_layout()
        out = "scenarios_preview.png"
        plt.savefig(out, dpi=120)
        print(f"Plot saved to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
