"""Subspace simulation (production path) must equal the full 2^n statevector
simulation built from the dense operators — validates the rank-1 shortcut."""

import numpy as np

from qc.dense import grover_mixer_full
from qc.grover_mixer import enumerate_feasible
from qc.instance import direct_costs, int_to_bits
from qc.qaoa import gm_qaoa, normalize_costs, ramp_angles

from tests.test_qc_instance import make_instance


def full_statevector_qaoa(inst, feas, costs, p, gamma_max, beta_max):
    """Reference implementation: dense ops on the full 2^n space."""
    dim = 2 ** inst.n_bits
    energies_full = np.zeros(dim)
    energies_full[feas] = normalize_costs(costs)
    psi = np.zeros(dim, dtype=complex)
    psi[feas] = 1.0 / np.sqrt(len(feas))
    gammas, betas = ramp_angles(p, gamma_max, beta_max)
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * energies_full) * psi
        psi = grover_mixer_full(beta, inst.n_bits, feas) @ psi
    return np.abs(psi) ** 2


def test_subspace_equals_full_statevector():
    inst = make_instance([0])  # T=1 outage slot: 8 bits, 256 full dim, |F| = 18
    feas = enumerate_feasible(inst)
    rng = np.random.default_rng(11)
    costs = direct_costs(int_to_bits(feas, inst.n_bits), inst) + rng.uniform(0, 50, len(feas))

    p, gamma_max, beta_max = 5, np.pi, np.pi
    probs_sub = gm_qaoa(costs, p=p, gamma_max=gamma_max, beta_max=beta_max)
    probs_full = full_statevector_qaoa(inst, feas, costs, p, gamma_max, beta_max)

    np.testing.assert_allclose(probs_full[feas], probs_sub, atol=1e-10)
    infeas = np.setdiff1d(np.arange(2 ** inst.n_bits), feas)
    np.testing.assert_allclose(probs_full[infeas], 0.0, atol=1e-12)
