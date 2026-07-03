import numpy as np
import pytest

from qc.instance import (
    BITS_PER_SLOT,
    ROLES,
    Instance,
    bit_index,
    int_to_bits,
    structurally_feasible,
)


def make_instance(g_avail):
    """Synthetic instance; pv/load/tou values are irrelevant for the structural predicate."""
    T = len(g_avail)
    return Instance(
        p_pv=np.zeros(T),
        p_load=np.full(T, 100.0),
        tou=np.full(T, 0.10),
        g_avail=np.asarray(g_avail, dtype=int),
    )


def state_from_roles(slot_roles):
    """Build a state int from a list (one dict per slot) of role -> bit."""
    state = 0
    for t, roles in enumerate(slot_roles):
        for role, val in roles.items():
            if val:
                state |= 1 << bit_index(t, role)
    return state


def test_layout_dimensions():
    inst = make_instance([1, 0])
    assert BITS_PER_SLOT == 8
    assert len(ROLES) == 8
    assert inst.T == 2
    assert inst.n_bits == 16


def test_bit_index_is_slot_major_lsb_first():
    assert bit_index(0, "ch") == 0
    assert bit_index(0, "y") == 7
    assert bit_index(1, "ch") == 8
    assert bit_index(1, "b_high") == 14


def test_int_to_bits_roundtrip():
    n = 16
    states = np.array([0, 1, 2 ** 15, 0b1010_0000_0000_0101])
    bits = int_to_bits(states, n)
    assert bits.shape == (4, n)
    reconstructed = (bits * (1 << np.arange(n))).sum(axis=1)
    np.testing.assert_array_equal(reconstructed, states)


def test_feasible_state_passes():
    inst = make_instance([1, 0])
    # slot 0 online: charging, importing, mid band; slot 1 outage: discharging, low band, served
    z = state_from_roles([
        {"ch": 1, "imp": 1, "b_mid": 1},
        {"dis": 1, "b_low": 1, "y": 1},
    ])
    bits = int_to_bits(np.array([z]), inst.n_bits)
    assert structurally_feasible(bits, inst).all()


@pytest.mark.parametrize("slot_roles", [
    # ch+dis simultaneously (slot 0)
    [{"ch": 1, "dis": 1, "b_mid": 1}, {"b_mid": 1, "y": 1}],
    # imp+exp simultaneously (slot 0)
    [{"imp": 1, "exp": 1, "b_mid": 1}, {"b_mid": 1}],
    # band not one-hot: none set (slot 1)
    [{"b_mid": 1}, {"ch": 1}],
    # band not one-hot: two set (slot 0)
    [{"b_low": 1, "b_mid": 1}, {"b_mid": 1}],
    # import during outage (slot 1)
    [{"b_mid": 1}, {"imp": 1, "b_mid": 1}],
    # served bit set on online slot (slot 0)
    [{"b_mid": 1, "y": 1}, {"b_mid": 1}],
])
def test_infeasible_states_rejected(slot_roles):
    inst = make_instance([1, 0])
    z = state_from_roles(slot_roles)
    bits = int_to_bits(np.array([z]), inst.n_bits)
    assert not structurally_feasible(bits, inst).any()


def test_y_free_on_outage_slot():
    inst = make_instance([0])
    z0 = state_from_roles([{"b_mid": 1, "y": 0}])
    z1 = state_from_roles([{"b_mid": 1, "y": 1}])
    bits = int_to_bits(np.array([z0, z1]), inst.n_bits)
    assert structurally_feasible(bits, inst).all()
