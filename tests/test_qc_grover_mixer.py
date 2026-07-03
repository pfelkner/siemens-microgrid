import numpy as np

from qc.grover_mixer import enumerate_feasible, expected_feasible_count
from qc.instance import int_to_bits, structurally_feasible

from tests.test_qc_instance import make_instance


def test_expected_count_formula():
    assert expected_feasible_count(make_instance([1, 0])) == 27 * 18   # 486
    assert expected_feasible_count(make_instance([1, 1])) == 27 * 27   # 729
    assert expected_feasible_count(make_instance([0])) == 18


def test_enumeration_matches_formula_and_predicate():
    inst = make_instance([1, 0])
    feas = enumerate_feasible(inst)
    assert feas.dtype == np.int64
    assert len(feas) == expected_feasible_count(inst)
    assert (np.diff(feas) > 0).all()  # sorted, unique
    bits = int_to_bits(feas, inst.n_bits)
    assert structurally_feasible(bits, inst).all()


def test_excluded_states_violate_predicate():
    inst = make_instance([1, 0])
    feas = enumerate_feasible(inst)
    all_states = np.arange(2 ** inst.n_bits, dtype=np.int64)
    excluded = np.setdiff1d(all_states, feas)
    rng = np.random.default_rng(0)
    sample = rng.choice(excluded, size=100, replace=False)
    bits = int_to_bits(sample, inst.n_bits)
    assert not structurally_feasible(bits, inst).any()


def test_product_construction_matches_bruteforce_reference():
    """The product construction must yield exactly the brute-force set."""
    inst = make_instance([1, 0])
    feas = enumerate_feasible(inst)
    states = np.arange(2 ** inst.n_bits, dtype=np.int64)
    ref = states[structurally_feasible(int_to_bits(states, inst.n_bits), inst)]
    np.testing.assert_array_equal(feas, ref)


def test_enumeration_scales_beyond_bruteforce():
    inst3 = make_instance([1, 0, 1])   # 24 bits
    feas3 = enumerate_feasible(inst3)
    assert len(feas3) == expected_feasible_count(inst3) == 27 * 18 * 27
    assert (np.diff(feas3) > 0).all()
    assert structurally_feasible(int_to_bits(feas3, inst3.n_bits), inst3).all()

    inst4 = make_instance([1, 1, 0, 1])  # 32 bits — brute force unmöglich (2^32)
    feas4 = enumerate_feasible(inst4)
    assert len(feas4) == expected_feasible_count(inst4)
    assert (np.diff(feas4) > 0).all()
