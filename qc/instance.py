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

    Encodes exactly what the Grover mixer spans: charge/discharge mutual
    exclusion (at most one), import/export mutual exclusion (at most one),
    SoC-band one-hot, and the data-driven pinning (no grid flow during
    outage, no served bit while online).
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


def direct_costs(bits: np.ndarray, inst: Instance) -> np.ndarray:
    """z-only objective part: -RESILIENCY_PER_SLOT per served outage slot.

    Everything else in the objective (ToU energy, demand charge, export)
    depends on the continuous x and enters later via Benders cuts.
    """
    cost = np.zeros(len(bits), dtype=float)
    for t in range(inst.T):
        if inst.g_avail[t] == 0:
            cost -= RESILIENCY_PER_SLOT * bits[:, bit_index(t, "y")]
    return cost


def decode(state: int, inst: Instance) -> list[dict[str, int]]:
    """State int -> one {role: bit} dict per slot (input for the later Gurobi subproblem)."""
    bits = int_to_bits(np.array([state]), inst.n_bits)[0]
    return [
        {role: int(bits[bit_index(t, role)]) for role in ROLES}
        for t in range(inst.T)
    ]


def load_instance(path, start: int = 0, T: int = 2,
                  force_outage: int | None = None) -> Instance:
    """Load a T-slot window from the standard data CSV.

    force_outage: optionally pin grid_available[t] = 0 for one window slot —
    reproducible served-bit for tests/demo when the window has no natural outage.
    """
    import pandas as pd

    df = pd.read_csv(path).iloc[start:start + T].reset_index(drop=True)
    if len(df) < T:
        raise ValueError(f"window [{start}, {start + T}) exceeds data length {len(df) + start}")
    g = df["grid_available"].to_numpy(dtype=int)
    if force_outage is not None:
        g = g.copy()
        g[force_outage] = 0
    return Instance(
        p_pv=df["p_kw"].to_numpy(dtype=float),
        p_load=df["load_kw"].to_numpy(dtype=float),
        tou=df["tou_usd_kwh"].to_numpy(dtype=float),
        g_avail=g,
    )
