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


from qc.instance import RESILIENCY_PER_SLOT, decode, direct_costs, load_instance


def test_direct_costs_only_served_outage_counts():
    inst = make_instance([1, 0])
    z_unserved = state_from_roles([{"b_mid": 1}, {"b_mid": 1, "y": 0}])
    z_served = state_from_roles([{"b_mid": 1}, {"b_mid": 1, "y": 1}])
    bits = int_to_bits(np.array([z_unserved, z_served]), inst.n_bits)
    costs = direct_costs(bits, inst)
    np.testing.assert_allclose(costs, [0.0, -RESILIENCY_PER_SLOT])


def test_decode_roundtrip():
    inst = make_instance([1, 0])
    slot_roles = [
        {"ch": 1, "imp": 1, "b_mid": 1},
        {"dis": 1, "b_low": 1, "y": 1},
    ]
    z = state_from_roles(slot_roles)
    decoded = decode(z, inst)
    assert len(decoded) == 2
    for t, want in enumerate(slot_roles):
        for role in ROLES:
            assert decoded[t][role] == want.get(role, 0)


def test_load_instance_window_and_force_outage(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "timestamp,p_kw,load_kw,tou_usd_kwh,grid_available\n"
        "2025-06-02 00:00:00,0.0,201.3,-0.25,1\n"
        "2025-06-02 00:15:00,5.0,198.7,0.10,1\n"
        "2025-06-02 00:30:00,10.0,190.0,0.30,1\n"
    )
    inst = load_instance(csv, start=1, T=2, force_outage=1)
    assert inst.T == 2
    np.testing.assert_allclose(inst.p_pv, [5.0, 10.0])
    np.testing.assert_allclose(inst.tou, [0.10, 0.30])
    np.testing.assert_array_equal(inst.g_avail, [1, 0])  # forced


def test_load_instance_window_too_long_raises(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "timestamp,p_kw,load_kw,tou_usd_kwh,grid_available\n"
        "2025-06-02 00:00:00,0.0,201.3,-0.25,1\n"
    )
    with pytest.raises(ValueError):
        load_instance(csv, start=0, T=2)
