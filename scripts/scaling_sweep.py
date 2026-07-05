"""Run classical_solver.py across a 2D grid of (T, M) and print the summary table."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SIZES     = [96, 288, 672, 1440, 2880]
DEFAULT_SCENARIOS = [1]


def main() -> int:
    p = argparse.ArgumentParser(description="2D scaling sweep wrapper for classical_solver.py")
    p.add_argument("--sizes",     type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--scenarios", type=int, nargs="+", default=DEFAULT_SCENARIOS,
                   help="Scenario counts to sweep (e.g. --scenarios 1 5 10 25)")
    p.add_argument("--data",         default="artifacts/data/all_data.csv")
    p.add_argument("--time-limit",   type=float, default=None)
    p.add_argument("--mip-gap",      type=float, default=1e-4)
    p.add_argument("--reset",        action="store_true",
                   help="Delete the summary CSV before sweeping")
    args = p.parse_args()

    summary_path = Path("artifacts/results/results_classical.csv")
    if args.reset and summary_path.exists():
        summary_path.unlink()
        print(f"[reset] removed {summary_path}")

    for m_sc in args.scenarios:
        for n in args.sizes:
            cmd = [
                sys.executable, "-m", "classical.classical_solver",
                "--data",         args.data,
                "--slots",        str(n),
                "--scenarios",    str(m_sc),
                "--mip-gap",      str(args.mip_gap),
                "--quiet",
            ]
            if args.time_limit is not None:
                cmd += ["--time-limit", str(args.time_limit)]
            print(f"[sweep] T={n:>4}  M={m_sc:>2}")
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[sweep] WARN: solver returned non-zero exit ({rc}) for T={n} M={m_sc}")

    if not summary_path.exists():
        print("[sweep] no summary file produced")
        return 1

    df = pd.read_csv(summary_path)
    print("\n=== results_classical.csv ===")
    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.float_format", lambda x: f"{x:.4g}"):
        print(df.to_string(index=False))

    # 2D runtime heatmap: rows = M, columns = T
    if "n_scenarios" in df.columns and df["n_scenarios"].nunique() > 1:
        print("\n=== Runtime (s) — rows: M (scenarios), columns: T (slots) ===")
        pivot = df.pivot_table(index="n_scenarios", columns="slots",
                               values="runtime_s", aggfunc="mean")
        with pd.option_context("display.float_format", lambda x: f"{x:7.2f}"):
            print(pivot.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
