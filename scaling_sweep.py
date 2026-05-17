"""Run classical_solver.py across multiple horizon sizes and print the summary table."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SIZES = [96, 288, 672, 1440, 2880]


def main() -> int:
    p = argparse.ArgumentParser(description="Scaling sweep wrapper for classical_solver.py")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--data", default="all_data.csv")
    p.add_argument("--out-summary", default="results_classical.csv")
    p.add_argument("--out-schedule", default="schedule_classical.csv")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--reset", action="store_true",
                   help="Delete the summary CSV before sweeping")
    args = p.parse_args()

    summary_path = Path(args.out_summary)
    if args.reset and summary_path.exists():
        summary_path.unlink()
        print(f"[reset] removed {summary_path}")

    solver = Path(__file__).parent / "classical_solver.py"

    for n in args.sizes:
        log = f"gurobi_T{n}.log"
        cmd = [
            sys.executable, str(solver),
            "--data", args.data,
            "--slots", str(n),
            "--mip-gap", str(args.mip_gap),
            "--out-summary", args.out_summary,
            "--out-schedule", args.out_schedule,
            "--gurobi-log", log,
            "--quiet",
        ]
        if args.time_limit is not None:
            cmd += ["--time-limit", str(args.time_limit)]
        print(f"[sweep] T={n} -> {log}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[sweep] WARN: solver returned non-zero exit ({rc}) for T={n}")

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print("\n=== results_classical.csv ===")
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", lambda x: f"{x:.4g}"):
            print(df.to_string(index=False))
    else:
        print("[sweep] no summary file produced")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
