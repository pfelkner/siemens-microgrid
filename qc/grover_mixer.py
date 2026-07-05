"""Per-slot cartesian product construction of the Grover-mixer feasible set.

Each slot type (online: g=1, outage: g=0) has its own set of feasible 8-bit
patterns (27 patterns for online, 18 for outage). These per-slot sets are built
once from only 2^8 = 256 candidates using structurally_feasible on a T=1 dummy
instance as the oracle — the slot rules live in exactly one place (qc/instance.py).

The full feasible set F is the cartesian product along g_avail, combined via bit
shifts (slot bit ranges are disjoint, so addition == OR). Construction cost is
O(|F|) instead of O(2^(8T)), making T >= 4 (32 bits) cheap; brute force over
2^(8T) bitstrings survives only as a test reference for T=2.

The mixer matrix itself is never materialized in the production path
(rank-1 update in qc/qaoa.py); dense matrices live in qc/dense.py for tests.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from qc.instance import BITS_PER_SLOT, Instance, int_to_bits, structurally_feasible


def expected_feasible_count(inst: Instance) -> int:
    """Product formula: 27 per online slot (3 ch/dis x 3 imp/exp x 3 bands),
    18 per outage slot (3 ch/dis x 3 bands x 2 served)."""
    return int(np.prod([27 if g == 1 else 18 for g in inst.g_avail]))


@lru_cache(maxsize=None)
def _slot_states(g: int) -> np.ndarray:
    """Feasible 8-bit patterns for one slot type (g=1 online: 27, g=0 outage: 18).

    Computed once per slot type from only 2^8 candidates, using
    structurally_feasible on a T=1 dummy instance as the oracle — the
    slot rules live in exactly one place (qc/instance.py).
    """
    dummy = Instance(p_pv=np.zeros(1), p_load=np.zeros(1),
                     tou=np.zeros(1), g_avail=np.array([g]))
    states = np.arange(2 ** BITS_PER_SLOT, dtype=np.int64)
    feasible = structurally_feasible(int_to_bits(states, BITS_PER_SLOT), dummy)
    result = states[feasible]
    result.setflags(write=False)   # cached — guard against accidental mutation
    return result


def enumerate_feasible(inst: Instance) -> np.ndarray:
    """Sorted int64 array of all structurally feasible bitstring values.

    Cartesian product of the per-slot feasible sets along g_avail, combined
    via bit shifts (slot bit ranges are disjoint, so addition == OR).
    Construction cost is O(|F|) instead of O(2^(8T)) — T >= 4 stays cheap.
    Sorted by construction: the higher slot's bits dominate the integer
    value, and both factors of each outer sum are sorted.
    """
    feas = np.zeros(1, dtype=np.int64)
    for t, g in enumerate(inst.g_avail):
        slot = _slot_states(int(g)) << (BITS_PER_SLOT * t)
        feas = (slot[:, None] + feas[None, :]).ravel()
    return feas
