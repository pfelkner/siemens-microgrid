"""Benders master side (Tasks 8+9): cuts from subproblem duals/Farkas + the hybrid loop.

The subproblem LP takes z only through constraint RHSs, exported per solve as
`rhs_affine`: name -> (const, {(t, role): coef}). Both cut types are therefore
affine functionals over the master's bit layout and evaluate vectorized against
the (N, n_bits) bit matrix of all feasible states:

* optimality cut (feasible LP, duals pi), anchored at the solved z̄:
      q(z) >= q̄ + w·(z − z̄),   w_b = Σ_i pi_i · a_{i,b}
  The anchoring cancels every z-independent RHS term and the variable-bound
  duals, so `duals` + `rhs_affine` is all we need (valid by LP duality: the
  dual point stays dual-feasible when only the RHS moves).
* feasibility cut (infeasible LP, Farkas ray lam): v(z) = Σ_i lam_i · h_i(z)
  is sign-normalized so that v(z) < -FEAS_TOL proves z has no continuous
  continuation (the ray's proof depends on z only through h). Excluded states
  are removed from the feasible-state array — the Grover mixer over the rest
  is implicit (uniform over whatever array the loop passes on).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from feasible_x.feasible_start_x import Instance as SubInstance, Params, SlotConfig
from feasible_x.subproblem import SubproblemResult, solve_subproblem
from qc.grover_mixer import enumerate_feasible
from qc.instance import Instance, bit_index, decode, direct_costs, int_to_bits
from qc.qaoa import gm_qaoa, sample_best

FEAS_TOL = 1e-6


@dataclass(frozen=True)
class Cut:
    const: float
    coef: np.ndarray            # (n_bits,) weights over the master bit layout
    kind: str                   # "optimality" | "feasibility"

    def evaluate(self, bits: np.ndarray) -> np.ndarray:
        """Affine value per state for an (N, n_bits) bit matrix -> (N,)."""
        return self.const + bits @ self.coef


def to_slot_configs(state: int, inst: Instance) -> list[SlotConfig]:
    """Master bitstring -> per-slot SlotConfig (the subproblem's fixed z)."""
    out = []
    for slot in decode(state, inst):
        out.append(SlotConfig(
            batt="charge" if slot["ch"] else ("discharge" if slot["dis"] else "idle"),
            grid="import" if slot["imp"] else ("export" if slot["exp"] else "idle"),
            band="low" if slot["b_low"] else ("mid" if slot["b_mid"] else "high"),
            served=bool(slot["y"]),
        ))
    return out


def build_sub_instance(inst: Instance, state: int,
                       params: Params | None = None) -> SubInstance:
    """qc instance + fixed z -> the feasible_x instance the Gurobi subproblem takes."""
    return SubInstance(
        pv=inst.p_pv, load=inst.p_load, grid_available=inst.g_avail,
        config=to_slot_configs(state, inst),
        params=params if params is not None else Params(),
        tou=inst.tou,
    )


def _weights(multipliers: dict[str, float],
             rhs_affine: dict[str, tuple[float, dict[tuple[int, str], float]]],
             n_bits: int) -> np.ndarray:
    """w_b = sum_i multiplier_i * a_{i,b}, mapped onto the master bit layout."""
    w = np.zeros(n_bits)
    for name, lam in multipliers.items():
        for (t, role), a in rhs_affine[name][1].items():
            w[bit_index(t, role)] += lam * a
    return w


def optimality_cut(res: SubproblemResult, z_bits: np.ndarray, n_bits: int) -> Cut:
    """Anchored optimality cut: q(z) >= q̄ + w·(z − z̄)."""
    w = _weights(res.duals, res.rhs_affine, n_bits)
    return Cut(const=float(res.q_value - w @ z_bits), coef=w, kind="optimality")


def feasibility_cut(res: SubproblemResult, z_bits: np.ndarray,
                    n_bits: int) -> Cut | None:
    """Farkas feasibility cut; None if the certificate cannot separate in z.

    Sign convention is normalized empirically: whatever sign Gurobi's FarkasDual
    gives the functional at the (provably infeasible) anchor z̄, we flip so that
    "excluded" uniformly means evaluate(bits) < -FEAS_TOL. States on the same
    strict side as z̄ share its infeasibility proof (the ray is independent of z).
    """
    if not res.farkas or all(abs(v) <= FEAS_TOL for v in res.farkas.values()):
        return None
    w = _weights(res.farkas, res.rhs_affine, n_bits)
    const = sum(lam * res.rhs_affine[name][0] for name, lam in res.farkas.items())
    v_bar = float(const + w @ z_bits)
    if abs(v_bar) <= FEAS_TOL:
        return None             # numerically degenerate certificate
    if v_bar > 0:
        w, const = -w, -const   # normalize: anchor lands on the excluded side
    return Cut(const=float(const), coef=w, kind="feasibility")
