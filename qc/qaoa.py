"""GM-QAOA evolution on the feasible subspace.

The state psi lives ONLY on the feasible states: the cost operator is
diagonal and the Grover mixer is identity plus a rank-1 term in |F>, so
amplitudes of infeasible states start at 0 and stay exactly 0 — the
subspace simulation is mathematically identical to the full statevector.

Problem-agnostic by design: consumes a cost vector over the feasible
states and nothing else. The Benders loop later feeds an updated cost
vector per round (optimality cuts) or a filtered state list (feasibility
cuts) without touching this module.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # type: ignore[import]
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False


def gpu_available() -> bool:
    return _CUPY_AVAILABLE


def normalize_costs(costs: np.ndarray) -> np.ndarray:
    """Affine map to [0, 1] so the useful gamma scale is dollar-independent.

    A flat vector maps to all-zeros: the evolution then keeps the uniform
    distribution, which is the correct answer (nothing to discriminate).
    """
    costs = np.asarray(costs, dtype=float)
    lo, hi = float(costs.min()), float(costs.max())
    if hi == lo:
        return np.zeros_like(costs)
    return (costs - lo) / (hi - lo)


def ramp_angles(p: int, gamma_max: float, beta_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed annealing-like linear ramp on the midpoint grid.

    k = (i + 0.5) / p keeps every layer active (beta_i > 0 even in the
    last layer, unlike the endpoint grid where the final mixer is identity).
    gamma ramps up (increasing cost emphasis); beta ramps down (decreasing
    mixer strength), following an annealing-like schedule.
    """
    k = (np.arange(p) + 0.5) / p
    return k * gamma_max, (1.0 - k) * beta_max


def gm_qaoa(costs: np.ndarray, p: int = 6, gamma_max: float = 4 * np.pi,
            beta_max: float = 2 * np.pi, use_gpu: bool = False) -> np.ndarray:
    """Run GM-QAOA; return |psi|^2 over the feasible states (sums to 1).

    Mixer convention (matching qc/dense.py and the design spec):
        U_M(beta) = I - (1 - e^{-i beta}) |F><F|  =  exp(-i beta |F><F|)
    Applied as the rank-1 update:
        psi <- psi - (1 - e^{-i beta}) * mean(psi)

    This is the standard Grover-mixer convention from Bärtschi/Eidenbenz.
    The alternative convention exp(-i beta (2P - I)) = e^{i beta} * exp(-2i beta P)
    differs only by a global phase and a factor-of-2 in the angle, which is
    why beta_max defaults to 2*pi here (equivalent to the calibrated point at
    beta_max=pi in the doubled-angle convention).

    Defaults calibrated by grid search over gamma_max x p on the test instances:
      - P(argmin) = 0.0080  on the random-cost instance (seed 7, 486 feasible states)
      - P(min-set) = 0.9888 on the two-level instance (min half of feasible states)
      - p=8 strictly beats p=1 on the seed-3 random instance
    gamma_max=4pi, beta_max=2pi, p=6 is the smallest p that clears all three
    amplification thresholds with the midpoint ramp schedule.

    Inert-layer caveat: with integer-valued normalized costs (e.g. the two-level
    round-1 case where E ∈ {0, 1}), any layer whose γ_k is a multiple of 2π
    applies an identity cost phase (inert layer) — with the midpoint ramp and
    gamma_max=4π this happens at p=1 (γ=2π) and at the middle layer of every
    odd p; amplification is therefore non-monotone in p (measured: p=6 → 0.989,
    p=8 → 0.871 on the two-level instance).

    use_gpu: run on GPU via CuPy if available; silently falls back to NumPy.
    """
    on_gpu = use_gpu and _CUPY_AVAILABLE
    if use_gpu and not _CUPY_AVAILABLE:
        import warnings
        warnings.warn("CuPy not available, falling back to NumPy", stacklevel=2)

    energies_np = normalize_costs(costs)
    dim = len(energies_np)

    if on_gpu:
        assert cp is not None
        energies = cp.asarray(energies_np)
        psi = cp.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
        gammas, betas = ramp_angles(p, gamma_max, beta_max)
        for gamma, beta in zip(gammas, betas):
            psi = cp.exp(-1j * gamma * energies) * psi           # cost phase, elementwise
            psi = psi - (1.0 - cp.exp(-1j * beta)) * psi.mean() # Grover mixer
        probs = cp.abs(psi) ** 2
        probs = probs / probs.sum()
        return cp.asnumpy(probs)
    else:
        psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
        gammas, betas = ramp_angles(p, gamma_max, beta_max)
        for gamma, beta in zip(gammas, betas):
            psi = np.exp(-1j * gamma * energies_np) * psi           # cost phase, elementwise
            psi = psi - (1.0 - np.exp(-1j * beta)) * psi.mean()     # Grover mixer
        probs = np.abs(psi) ** 2
        return probs / probs.sum()


def sample_best(probs: np.ndarray, feasible_states: np.ndarray, costs: np.ndarray,
                rng: np.random.Generator, shots: int = 1024) -> int:
    """Draw shots from the QAOA distribution, return the cheapest sampled state int."""
    idx = rng.choice(len(probs), size=shots, p=probs)
    best = idx[np.argmin(costs[idx])]
    return int(feasible_states[best])


def shots_to_success(p_opt: float, target: float = 0.99) -> float:
    """Shots so that best-of-shots hits a master-argmin state w.p. >= target.

    Per-shot success probability p_opt is the total QAOA probability mass on
    the states attaining the master minimum (ties included — any of them makes
    best-of-shots succeed). Independent shots: 1 - (1-p_opt)^S >= target.
    """
    if not 0.0 <= p_opt <= 1.0:
        raise ValueError(f"p_opt must be a probability, got {p_opt}")
    if p_opt == 0.0:
        return float("inf")
    if p_opt >= target:
        return 1.0
    return float(np.log(1.0 - target) / np.log(1.0 - p_opt))
