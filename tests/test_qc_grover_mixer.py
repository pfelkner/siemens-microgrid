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
