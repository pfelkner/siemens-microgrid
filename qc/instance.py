"""PoC instance for the GM-QAOA microgrid dispatch master problem.

Defines the discrete decision bitstring z (uniform 8-bit-per-slot layout),
the structural (x-independent) feasibility predicate that the Grover mixer
encodes, and the z-only "direct" cost part of the diagonal cost Hamiltonian.

Bit convention: bit b of a state int is (state >> b) & 1, with
b = slot * BITS_PER_SLOT + role_offset (LSB-first, slot-major).
Outage vs. online never changes the layout — it only adds pinning rules
in the predicate (imp = exp = 0 during outage, y = 0 while online).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESILIENCY_PER_SLOT = 225.0  # $ revenue per served outage slot (mirrors qubo_model.py)

ROLES = ("ch", "dis", "imp", "exp", "b_low", "b_mid", "b_high", "y")
BITS_PER_SLOT = len(ROLES)


@dataclass(frozen=True)
class Instance:
    p_pv: np.ndarray
    p_load: np.ndarray
    tou: np.ndarray
    g_avail: np.ndarray

    @property
    def T(self) -> int:
        return len(self.g_avail)

    @property
    def n_bits(self) -> int:
        return BITS_PER_SLOT * self.T


def bit_index(t: int, role: str) -> int:
    return t * BITS_PER_SLOT + ROLES.index(role)


def int_to_bits(states: np.ndarray, n_bits: int) -> np.ndarray:
    """(N,) state ints -> (N, n_bits) 0/1 matrix; column b is bit b (LSB-first)."""
    states = np.asarray(states, dtype=np.int64)
    return ((states[:, None] >> np.arange(n_bits)) & 1).astype(np.int64)


def structurally_feasible(bits: np.ndarray, inst: Instance) -> np.ndarray:
    """Vectorized structural feasibility over an (N, n_bits) bit matrix -> (N,) bool.

    Encodes exactly what the Grover mixer spans: charge/discharge XOR,
    import/export XOR, SoC-band one-hot, and the data-driven pinning
    (no grid flow during outage, no served bit while online).
    """
    ok = np.ones(len(bits), dtype=bool)
    for t in range(inst.T):
        s = bits[:, t * BITS_PER_SLOT:(t + 1) * BITS_PER_SLOT]
        ch, dis, imp, exp_, b_lo, b_mid, b_hi, y = (s[:, i] for i in range(BITS_PER_SLOT))
        ok &= ch + dis <= 1
        ok &= imp + exp_ <= 1
        ok &= b_lo + b_mid + b_hi == 1
        if inst.g_avail[t] == 1:
            ok &= y == 0
        else:
            ok &= (imp == 0) & (exp_ == 0)
    return ok
