"""Brute-force construction of the Grover-mixer feasible set.

Enumerates all 2^n bitstrings (vectorized, no per-state Python loop) and
keeps the structurally feasible ones. Built ONCE per instance and reused
across all QAOA layers and all later Benders rounds; a feasibility cut
later just filters this array — no re-enumeration.

The mixer matrix itself is never materialized in the production path
(rank-1 update in qc/qaoa.py); dense matrices live in qc/dense.py for tests.
"""

from __future__ import annotations

import numpy as np

from qc.instance import Instance, int_to_bits, structurally_feasible


def expected_feasible_count(inst: Instance) -> int:
    """Product formula: 27 per online slot (3 ch/dis x 3 imp/exp x 3 bands),
    18 per outage slot (3 ch/dis x 3 bands x 2 served)."""
    return int(np.prod([27 if g == 1 else 18 for g in inst.g_avail]))


def enumerate_feasible(inst: Instance) -> np.ndarray:
    """Sorted int64 array of all structurally feasible bitstring values."""
    states = np.arange(2 ** inst.n_bits, dtype=np.int64)
    bits = int_to_bits(states, inst.n_bits)
    return states[structurally_feasible(bits, inst)]
