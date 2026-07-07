"""Produce a 4-panel overview plot of the first 3 days of all_data.csv."""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SLOTS = 288  # 3 days × 96 slots/day


def main(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.iloc[:SLOTS].copy()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Microgrid Input Data — First 3 Days", fontsize=13, fontweight="bold")

    t = df["timestamp"]

    # 1. PV generation
    ax = axes[0]
    ax.plot(t, df["p_kw"], color="#f5a623", linewidth=1.2)
    ax.set_ylabel("PV Output\n[kW]")
    ax.set_ylim(bottom=0)
    ax.fill_between(t, df["p_kw"], alpha=0.25, color="#f5a623")

    # 2. Load
    ax = axes[1]
    ax.plot(t, df["load_kw"], color="#4a90d9", linewidth=1.2)
    ax.set_ylabel("Load\n[kW]")
    ax.set_ylim(bottom=0)

    # 3. ToU tariff — step plot
    ax = axes[2]
    ax.step(t, df["tou_usd_kwh"], where="post", color="#7b2d8b", linewidth=1.5)
    ax.set_ylabel("ToU Tariff\n[USD/kWh]")
    ax.set_yticks([0.05, 0.15, 0.40])
    ax.set_ylim(0, 0.50)

    # 4. Grid availability — shade outages in red
    ax = axes[3]
    ax.fill_between(t, 1, where=df["grid_available"] == 1, step="post",
                    color="#5cb85c", alpha=0.6, label="Available")
    ax.fill_between(t, 1, where=df["grid_available"] == 0, step="post",
                    color="#d9534f", alpha=0.8, label="Outage")
    ax.set_ylabel("Grid\nAvailability")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Off", "On"])
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8)

    # x-axis formatting
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    fig.autofmt_xdate(rotation=0, ha="center")

    for ax in axes:
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3-day data overview")
    parser.add_argument("--csv", default="artifacts/data/all_data.csv")
    args = parser.parse_args()
    out = Path("artifacts/results/data_overview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    main(args.csv, str(out))
