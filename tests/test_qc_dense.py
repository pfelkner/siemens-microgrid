import numpy as np

from qc.dense import cost_phase, grover_mixer_full, grover_mixer_subspace
from qc.grover_mixer import enumerate_feasible

from tests.test_qc_instance import make_instance


def test_subspace_mixer_unitary():
    for beta in (0.3, 1.0, np.pi):
        u = grover_mixer_subspace(beta, dim=27)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(27), atol=1e-12)


def test_mixer_identity_at_beta_zero():
    np.testing.assert_allclose(grover_mixer_subspace(0.0, dim=18), np.eye(18), atol=1e-12)


def test_uniform_state_is_eigenvector():
    dim, beta = 27, 0.7
    f = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    u = grover_mixer_subspace(beta, dim)
    np.testing.assert_allclose(u @ f, np.exp(-1j * beta) * f, atol=1e-12)


def test_full_space_mixer_unitary_and_blockdiag():
    inst = make_instance([0])  # 8 bits -> 256 x 256 dense is cheap
    feas = enumerate_feasible(inst)
    u = grover_mixer_full(0.9, inst.n_bits, feas)
    dim = 2 ** inst.n_bits
    np.testing.assert_allclose(u @ u.conj().T, np.eye(dim), atol=1e-12)
    # acts as identity on infeasible basis states
    infeas = np.setdiff1d(np.arange(dim), feas)
    np.testing.assert_allclose(u[np.ix_(infeas, infeas)], np.eye(len(infeas)), atol=1e-12)
    np.testing.assert_allclose(u[np.ix_(infeas, feas)], 0.0, atol=1e-12)


def test_cost_phase_diagonal_unit_modulus():
    costs = np.array([0.0, 0.5, 1.0])
    u = cost_phase(0.8, costs)
    np.testing.assert_allclose(np.abs(np.diag(u)), 1.0)
    np.testing.assert_allclose(u, np.diag(np.diag(u)))
