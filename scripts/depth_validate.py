"""Cross-check qc/depth.py against explicitly built Qiskit circuits.

Two claims are validated:

1. reflection_depth(n) = 2*ceil(log2 n) + 3 — build R0(beta) as X-layer +
   balanced Toffoli AND-tree (n-1 ancillas) + phase + uncompute + X-layer,
   and compare QuantumCircuit.depth() to the formula across n.
2. The per-slot preparation A has T-independent depth — build A for T=1,2,3
   (tiled per-slot state prep on disjoint 8-qubit blocks) and show depth stays
   constant. The single-slot value is the c_prep the formula consumes.

Run:  uv run python -m scripts.depth_validate
"""

from __future__ import annotations

from math import ceil, log2

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation

from qc.depth import reflection_depth
from qc.grover_mixer import _slot_states

BASIS = ["u", "cx"]


def build_reflection(n: int, beta: float = 0.7) -> QuantumCircuit:
    """R0(beta) = exp(-i beta |0><0|) on n system qubits via ancilla AND-tree.

    Balanced binary tree: each level ANDs disjoint pairs of the current active
    wires into fresh ancillas (parallel -> depth 1/level), giving ceil(log2 n)
    levels; a 1-qubit phase on the tree root, then the uncompute mirror.
    """
    sys = QuantumRegister(n, "q")
    anc = QuantumRegister(max(n - 1, 0), "a")
    qc = QuantumCircuit(sys, anc)

    qc.x(sys)                       # map |0..0> -> |1..1> so the phase lands on all-zero

    if n == 1:
        qc.p(beta, sys[0])
        qc.x(sys)
        return qc

    # compute: reduce `active` wires pairwise into ancillas, record the tree
    active = list(sys)
    next_anc = 0
    levels: list[list[tuple]] = []
    while len(active) > 1:
        level, nxt = [], []
        for i in range(0, len(active) - 1, 2):
            target = anc[next_anc]; next_anc += 1
            qc.ccx(active[i], active[i + 1], target)
            level.append((active[i], active[i + 1], target))
            nxt.append(target)
        if len(active) % 2 == 1:
            nxt.append(active[-1])  # odd wire carries over
        levels.append(level)
        active = nxt

    qc.p(beta, active[0])           # phase on the AND of all n bits

    for level in reversed(levels):  # uncompute
        for c1, c2, target in reversed(level):
            qc.ccx(c1, c2, target)

    qc.x(sys)
    return qc


def build_prep(T: int, g: int = 1) -> QuantumCircuit:
    """Tiled per-slot uniform prep over the feasible patterns, T slots.

    g=1 online (27 patterns), g=0 outage (18 patterns).
    """
    patterns = _slot_states(g)
    vec = np.zeros(256)
    vec[patterns] = 1.0 / np.sqrt(len(patterns))
    slot = StatePreparation(vec)
    qc = QuantumCircuit(8 * T)
    for t in range(T):
        qc.append(slot, range(8 * t, 8 * t + 8))
    return qc


def main() -> int:
    print("== reflection_depth(n) vs Qiskit ==")
    print(f"{'n':>4} {'formula':>8} {'qiskit':>8} {'2ceil(log2n)+3':>15}  match")
    ok = True
    for n in [1, 2, 3, 4, 8, 16, 24, 32, 48, 64]:
        qc = build_reflection(n)
        d_q = qc.depth()
        d_f = reflection_depth(n)
        m = d_q == d_f
        ok &= m
        print(f"{n:>4} {d_f:>8} {d_q:>8} {2*ceil(log2(n))+3:>15}  "
              f"{'OK' if m else 'MISMATCH'}")

    print("\n== per-slot prep depth constant in T (transpiled to u,cx) ==")
    print(f"{'kind':>7} {'T':>3} {'n_qubits':>9} {'depth':>7}")
    for g, kind in [(1, "online"), (0, "outage")]:
        d_prev = None
        for T in [1, 2, 3]:
            qc = transpile(build_prep(T, g), basis_gates=BASIS, optimization_level=1)
            d = qc.depth()
            print(f"{kind:>7} {T:>3} {8*T:>9} {d:>7}")
            if d_prev is not None and d != d_prev:
                ok = False
            d_prev = d
        print(f"-> c_prep ({kind}, single slot) = {d_prev}\n")
    print(f"{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
