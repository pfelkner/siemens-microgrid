"""Dense-matrix materialization of the GM-QAOA operators.

Test/inspection path ONLY — production evolution (qc/qaoa.py) uses the
rank-1 identity U_M(beta) = I - (1 - e^{-i beta}) |F><F| and never builds
these matrices. This module covers the literal matrix formulation from
QC_Ansatz.md and backs the unitarity / equivalence tests.
"""

from __future__ import annotations

import numpy as np


def grover_mixer_subspace(beta: float, dim: int) -> np.ndarray:
    """U_M(beta) on the feasible subspace, where |F> is the uniform vector."""
    proj = np.full((dim, dim), 1.0 / dim, dtype=complex)
    return np.eye(dim, dtype=complex) - (1.0 - np.exp(-1j * beta)) * proj


def grover_mixer_full(beta: float, n_bits: int, feasible_states: np.ndarray) -> np.ndarray:
    """The same operator embedded in the full 2^n space (identity outside F)."""
    dim = 2 ** n_bits
    f = np.zeros(dim, dtype=complex)
    f[feasible_states] = 1.0 / np.sqrt(len(feasible_states))
    return np.eye(dim, dtype=complex) - (1.0 - np.exp(-1j * beta)) * np.outer(f, f.conj())


def cost_phase(gamma: float, costs: np.ndarray) -> np.ndarray:
    """Diagonal e^{-i gamma H_C}."""
    return np.diag(np.exp(-1j * gamma * np.asarray(costs, dtype=float)))
