"""Emit gate-depth scaling over ascending T, split by component.

One p-layer GM-QAOA master circuit per row (initial |F> preparation + p layers).
The mixer's reflection is reported both ancilla-assisted (O(log n) AND-tree,
Qiskit-validated) and ancilla-free (O(n) linear cascade, analytic model), so the
two totals bracket the depth/width tradeoff. All-online window assumed
(c_prep = PREP_DEPTH_ONLINE; parallel per-slot prep -> depth = max block depth).

Run:  uv run python -m scripts.depth_table            # -> artifacts/gate_depth_scaling.csv
      uv run python -m scripts.depth_table --p 12 --out foo.csv
"""

from __future__ import annotations

import argparse
import csv

from qc.depth import (
    COST_DEPTH_AFFINE, PREP_DEPTH_ONLINE,
    reflection_depth, reflection_depth_no_ancilla, reflection_ancillas,
    circuit_depth,
)

FIELDS = [
    "t", "n_bits", "p",
    "prep_init",                # initial |F> preparation, once (constant in T)
    "cost_total",               # cost oracle over all p layers
    "prep_in_mixers",           # A + A_dagger over all p layers
    "reflection_anc",           # AND-tree reflection over all p layers
    "reflection_noanc",         # linear-cascade reflection over all p layers
    "total_depth_anc",          # full circuit, ancilla-assisted mixer
    "total_depth_noanc",        # full circuit, ancilla-free mixer
    "n_ancillas_anc",           # ancilla qubits used (0 for the no-anc variant)
]


def row(t: int, p: int, c_prep: int) -> dict:
    n = 8 * t
    return {
        "t": t,
        "n_bits": n,
        "p": p,
        "prep_init": c_prep,
        "cost_total": p * COST_DEPTH_AFFINE,
        "prep_in_mixers": p * 2 * c_prep,
        "reflection_anc": p * reflection_depth(n),
        "reflection_noanc": p * reflection_depth_no_ancilla(n),
        "total_depth_anc": circuit_depth(n, p, c_prep, ancilla=True),
        "total_depth_noanc": circuit_depth(n, p, c_prep, ancilla=False),
        "n_ancillas_anc": reflection_ancillas(n),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, default=6, help="QAOA layers (default 6)")
    ap.add_argument("--t-max", type=int, default=2880,
                    help="largest T (default 2880 = one 30-day month of 15-min slots)")
    ap.add_argument("--out", default="artifacts/gate_depth_scaling.csv")
    args = ap.parse_args(argv)

    ts = list(range(1, args.t_max + 1))     # every slot count; filter downstream
    rows = [row(t, args.p, PREP_DEPTH_ONLINE) for t in ts]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {args.out}  (p={args.p}, c_prep={PREP_DEPTH_ONLINE})")
    landmarks = {1: "min", 96: "day", 672: "week", 2880: "month"}
    for r in rows:
        if r["t"] in landmarks:
            print(f"  t={r['t']:>4} ({landmarks[r['t']]:>5}) n={r['n_bits']:>5}  "
                  f"refl_anc={r['reflection_anc']:>4} refl_noanc={r['reflection_noanc']:>6}  "
                  f"total_anc={r['total_depth_anc']:>6} total_noanc={r['total_depth_noanc']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
