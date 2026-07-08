"""Benchmark classical MILP solver across a range of T (time slots).

Produces artifacts/results/classical_scaling.csv with columns:
  t, n_qubits, runtime_s, mip_gap, status

n_qubits = 8 * t reflects the QAOA qubit count for the same instance
(8 binary decisions per slot: ch, dis, imp, exp, b_low, b_mid, b_high, y).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

BITS_PER_SLOT = 8  # mirrors qc/instance.py

DEFAULT_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 288, 672, 1440, 2880]

SUMMARY_CSV = Path("artifacts/results/results_classical.csv")
OUT_CSV = Path("artifacts/results/classical_scaling.csv")


def _parse_last_row(t: int) -> dict | None:
    """Read the last row written for slot-count t from the running summary CSV."""
    if not SUMMARY_CSV.exists():
        return None
    with open(SUMMARY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if int(r["slots"]) == t and r.get("n_scenarios", "1") == "1"]
    return matches[-1] if matches else None


def main() -> int:
    p = argparse.ArgumentParser(description="Classical MILP scaling benchmark")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--data", default="artifacts/data/all_data.csv")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--output", default=str(OUT_CSV))
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for t in sorted(args.sizes):
        print(f"[benchmark] T={t:>4}  n_qubits={BITS_PER_SLOT * t:>5}", flush=True)
        cmd = [
            sys.executable, "-m", "classical.deterministic_solver",
            "--data", args.data,
            "--slots", str(t),
            "--mip-gap", str(args.mip_gap),
            "--quiet",
        ]
        if args.time_limit is not None:
            cmd += ["--time-limit", str(args.time_limit)]

        rc = subprocess.call(cmd)
        if rc not in (0, 1):
            print(f"[benchmark] WARN: solver exited with code {rc} for T={t}")

        row = _parse_last_row(t)
        if row is None:
            print(f"[benchmark] WARN: no summary row found for T={t}, skipping")
            continue

        results.append({
            "t": t,
            "n_qubits": BITS_PER_SLOT * t,
            "runtime_s": float(row["runtime_s"]),
            "mip_gap": float(row["mip_gap"]),
            "status": row["status"],
        })
        print(f"           -> {row['status']}  {float(row['runtime_s']):.3f}s  gap={float(row['mip_gap']):.2e}")

    if not results:
        print("[benchmark] no results collected")
        return 1

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "n_qubits", "runtime_s", "mip_gap", "status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[benchmark] wrote {len(results)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
