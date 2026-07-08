"""Analytic gate-depth accounting for the GM-QAOA master circuit.

The NumPy simulation (qc/qaoa.py) applies the Grover mixer as a rank-1
statevector update — it has NO gates, so depth cannot be read out of it.
Depth is a property of a fixed circuit decomposition; this module counts it
exactly under one committed spec, parameterized by the same (n_bits, p, rounds,
cuts) the Benders loop already tracks. See scripts/depth_validate.py for the
Qiskit cross-check that the formulas below match an explicitly built circuit.

Decomposition spec (ancilla-assisted, unit depth per parallel layer)
--------------------------------------------------------------------
Gate set: {X, RZ, phase, CX, CCX}. A layer of gates on disjoint qubits is
depth 1 (ASAP scheduling — what Qiskit's QuantumCircuit.depth() reports).

One GM-QAOA layer = cost oracle U_C(gamma) then Grover mixer U_M(beta):

* Cost oracle, affine objective (weights on single bits, as in
  instance.direct_costs): one parallel layer of RZ over the involved qubits.
  All T slots are disjoint -> depth 1, constant in T.  [COST_DEPTH_AFFINE]
  (The Benders max-over-cuts objective is piecewise-linear and would need
  comparator arithmetic; that overhead is out of scope here — we count the
  affine oracle and treat depth as cut-independent.)

* Grover mixer U_M(beta) = A . R0(beta) . A_dagger, with |F> = A|0..0>:
  - A: per-slot state prep over the S feasible 8-bit patterns (S=27 online,
    18 outage). Slots act on disjoint 8-qubit blocks -> parallel -> depth is a
    constant c_prep independent of T (measured once via Qiskit, see
    PREP_DEPTH_* below). A_dagger has the same depth.
  - R0(beta) = exp(-i beta |0><0|): phase beta on the all-zero state, built as
        X^{on n}  .  MCPhase(beta over n qubits)  .  X^{on n}
    with the n-way controlled phase realized by a balanced Toffoli AND-tree
    into n-1 ancillas (compute), a 1-qubit phase on the tree root, then
    uncompute. Depth:
        R0_depth(n) = 1 (X layer) + ceil(log2 n) (compute)
                      + 1 (phase) + ceil(log2 n) (uncompute) + 1 (X layer)
                    = 2*ceil(log2 n) + 3
    This is the ONLY term that grows with the problem — O(log T) thanks to the
    ancillas (without them the multi-controlled phase is a depth-O(n) chain).
"""

from __future__ import annotations

from math import ceil, log2

# Constant-in-T layer pieces (see module docstring).
COST_DEPTH_AFFINE = 1

# Per-slot preparation depth, measured once with Qiskit (transpiled to
# basis {u, cx}); see scripts/depth_validate.py. Constant in T because slots
# occupy disjoint qubit blocks and prepare in parallel. Filled from the
# validation run; kept as named constants so the formula is self-contained.
PREP_DEPTH_ONLINE = 495   # 8-qubit uniform over 27 patterns, basis {u,cx}
PREP_DEPTH_OUTAGE = 191   # 8-qubit uniform over 18 patterns, basis {u,cx}


def reflection_depth(n: int) -> int:
    """Depth of R0(beta) = exp(-i beta |0><0|) over n qubits, ancilla AND-tree.

    2*ceil(log2 n) + 3 in the committed gate set (two X layers, a compute and
    an uncompute Toffoli tree of ceil(log2 n) levels each, one phase layer).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2 * ceil(log2(n)) + 3        # n=1: ceil(log2 1)=0 -> 3 (X, phase, X)


def reflection_depth_no_ancilla(n: int) -> int:
    """Depth of R0(beta) over n qubits with ZERO ancillas: linear cascade model.

    Without scratch qubits the n-way AND cannot be parallelized into a tree; it
    degrades to a sequential chain, so the compute/uncompute stages each take
    n-1 steps instead of ceil(log2 n):
        2*(n-1) + 3   (X layer, n-1 compute, phase, n-1 uncompute, X layer)
    This is the O(n) analytic model — the contrast term to the ancilla-assisted
    O(log n) reflection_depth, NOT Qiskit-validated (the ancilla version is).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2 * (n - 1) + 3


def reflection_ancillas(n: int) -> int:
    """Ancilla qubits used by the balanced AND-tree reflection: n - 1."""
    return max(n - 1, 0)


def prep_depth(c_prep: int) -> int:
    """A-preparation depth (constant in T); caller passes the measured c_prep."""
    return c_prep


def mixer_depth(n: int, c_prep: int, ancilla: bool = True) -> int:
    """U_M = A . R0 . A_dagger  ->  2*c_prep + reflection.

    ancilla toggles the reflection term only (A is unaffected): True uses the
    O(log n) AND-tree, False the O(n) linear cascade.
    """
    refl = reflection_depth(n) if ancilla else reflection_depth_no_ancilla(n)
    return 2 * prep_depth(c_prep) + refl


def layer_depth(n: int, c_prep: int, cost_depth: int = COST_DEPTH_AFFINE,
                ancilla: bool = True) -> int:
    """One GM-QAOA layer: cost oracle then Grover mixer."""
    return cost_depth + mixer_depth(n, c_prep, ancilla)


def circuit_depth(n: int, p: int, c_prep: int,
                  cost_depth: int = COST_DEPTH_AFFINE, ancilla: bool = True) -> int:
    """Depth of ONE p-layer GM-QAOA master circuit.

    GM-QAOA starts in |F> = A|0..0>, so every circuit pays one initial
    A-preparation (depth c_prep) before the first layer.
    """
    return prep_depth(c_prep) + p * layer_depth(n, c_prep, cost_depth, ancilla)


def total_depth(n: int, p: int, rounds: int, c_prep: int,
                cost_depth: int = COST_DEPTH_AFFINE, ancilla: bool = True) -> int:
    """Total circuit depth over the whole Benders loop.

    rounds master solves, each a p-layer GM-QAOA circuit on the same n qubits
    including its initial |F> preparation. The Grover-regime scaling claim
    lives in reflection_depth's O(log n=8T), multiplied by the (T-independent)
    layer constants and the p*rounds factor.
    """
    return rounds * circuit_depth(n, p, c_prep, cost_depth, ancilla)
