import numpy as np

from qc.qaoa import gm_qaoa, normalize_costs, ramp_angles, sample_best


def test_normalize_costs_affine_and_flat():
    np.testing.assert_allclose(normalize_costs(np.array([-225.0, 0.0, 225.0])), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(normalize_costs(np.array([3.0, 3.0, 3.0])), [0.0, 0.0, 0.0])


def test_ramp_angles_shape_and_direction():
    gammas, betas = ramp_angles(6, np.pi, np.pi)
    assert len(gammas) == len(betas) == 6
    assert (np.diff(gammas) > 0).all()      # gamma ramps up
    assert (np.diff(betas) < 0).all()       # beta ramps down
    assert (gammas > 0).all() and (betas > 0).all()  # no dead layer


def test_flat_costs_stay_uniform():
    dim = 486
    probs = gm_qaoa(np.zeros(dim), p=6)
    np.testing.assert_allclose(probs, np.full(dim, 1.0 / dim), atol=1e-12)


def test_probabilities_sum_to_one():
    rng = np.random.default_rng(42)
    probs = gm_qaoa(rng.uniform(0.0, 100.0, size=486), p=6)
    assert probs.shape == (486,)
    assert (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)


def test_argmin_amplified_over_uniform_random_costs():
    rng = np.random.default_rng(7)
    costs = rng.uniform(0.0, 100.0, size=486)
    probs = gm_qaoa(costs, p=6)
    assert probs[np.argmin(costs)] > 2.0 / len(costs)  # clearly above uniform


def test_min_cost_set_amplified_two_level_costs():
    # Real round-1 shape: direct costs are two-valued (served vs. not)
    costs = np.where(np.arange(486) < 243, -225.0, 0.0)
    probs = gm_qaoa(costs, p=6)
    p_min_set = probs[costs == -225.0].sum()
    assert p_min_set > 0.6  # uniform baseline would be 0.5


def test_more_layers_do_not_hurt_much():
    rng = np.random.default_rng(3)
    costs = rng.uniform(0.0, 100.0, size=200)
    p1 = gm_qaoa(costs, p=1)[np.argmin(costs)]
    p8 = gm_qaoa(costs, p=8)[np.argmin(costs)]
    assert p8 > p1  # ramp with more layers should amplify more on this instance


def test_sample_best_returns_state_int_of_cheapest_sampled():
    feasible_states = np.array([10, 20, 30], dtype=np.int64)
    costs = np.array([5.0, -1.0, 3.0])
    probs = np.array([0.2, 0.5, 0.3])
    rng = np.random.default_rng(0)
    best = sample_best(probs, feasible_states, costs, rng, shots=256)
    assert best == 20  # cheapest state is sampled with p=0.5 -> certain in 256 shots


def test_sample_best_finds_optimum_end_to_end():
    rng = np.random.default_rng(1)
    dim = 486
    costs = rng.uniform(0.0, 100.0, size=dim)
    feasible_states = np.arange(dim, dtype=np.int64) * 7  # arbitrary distinct ints
    probs = gm_qaoa(costs, p=6)
    best = sample_best(probs, feasible_states, costs, rng, shots=2048)
    assert best == feasible_states[np.argmin(costs)]
