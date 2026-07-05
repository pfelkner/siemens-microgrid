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

from subproblem.feasible_start_x import Instance as SubInstance, Params, SlotConfig
from subproblem.subproblem import SubproblemResult, solve_subproblem
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
    """qc instance + fixed z -> the subproblem instance the Gurobi subproblem takes."""
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


@dataclass
class LoopRound:
    """One Benders round, with master snapshots for visualization."""
    round: int
    z: int                      # sampled master state this round
    status: str                 # subproblem outcome: "optimal" | "infeasible"
    q: float | None             # Q(z) if feasible
    direct: float               # direct cost g(z)
    ub: float                   # best known total after this round
    lb: float                   # master bound used this round (-inf before first opt cut)
    gap: float                  # ub - lb at record time
    n_states: int               # |F| after this round's filtering
    n_removed: int              # states removed by this round's feasibility cut
    costs: np.ndarray           # master cost vector this round (over `states`)
    states: np.ndarray          # feasible states the costs/probs refer to
    probs: np.ndarray           # QAOA distribution this round


@dataclass
class LoopResult:
    rounds: list[LoopRound]
    best_z: int | None
    best_x: dict | None
    best_value: float           # UB = direct(best_z) + Q(best_z)
    lb: float
    gap: float
    termination: str            # "gap" | "max_rounds" | "infeasible"
    cuts: list[Cut]


def benders_loop(inst: Instance, params: Params | None = None,
                 max_rounds: int = 25, gap_tol: float = 1e-4,
                 shots: int = 1024, seed: int | None = None, p: int = 6) -> LoopResult:
    """The hybrid loop (Task 9): GM-QAOA master <-> Gurobi subproblem via cuts.

    Master cost per round: direct z-costs + pointwise max over all optimality
    cuts (the eta-free encoding from QC_Ansatz.md). Feasibility cuts filter the
    state/bit/direct arrays; the Grover mixer over the survivors is implicit.
    LB is the exact master minimum over the (remaining) enumeration — a valid
    Benders bound, free at PoC scale; the QAOA stays the (heuristic) sampler.
    """
    params = params if params is not None else Params()
    rng = np.random.default_rng(seed)
    states = enumerate_feasible(inst)
    bits = int_to_bits(states, inst.n_bits)
    direct = direct_costs(bits, inst)

    opt_cuts: list[Cut] = []
    cuts: list[Cut] = []
    rounds: list[LoopRound] = []
    ub, best_z, best_x = np.inf, None, None
    lb = -np.inf
    termination = "max_rounds"

    for r in range(1, max_rounds + 1):
        if len(states) == 0:
            termination = "infeasible"
            break
        if opt_cuts:
            q_model = np.max(np.stack([c.evaluate(bits) for c in opt_cuts]), axis=0)
            costs = direct + q_model
            lb = float(costs.min())
        else:
            costs = direct.copy()
            lb = -np.inf
        if ub - lb <= gap_tol:
            termination = "gap"
            break

        probs = gm_qaoa(costs, p=p)
        z = sample_best(probs, states, costs, rng, shots=shots)
        idx = int(np.where(states == z)[0][0])

        res = solve_subproblem(build_sub_instance(inst, z, params))
        n_removed = 0
        if res.feasible:
            cut = optimality_cut(res, bits[idx], inst.n_bits)
            opt_cuts.append(cut)
            cuts.append(cut)
            total = float(direct[idx] + res.q_value)
            if total < ub:
                ub, best_z, best_x = total, z, res.x
        else:
            cut = feasibility_cut(res, bits[idx], inst.n_bits)
            if cut is None:
                warnings.warn(f"degenerate Farkas certificate at z={z}; "
                              "dropping only this state")
                keep = states != z
            else:
                cuts.append(cut)
                keep = cut.evaluate(bits) >= -FEAS_TOL
                assert not keep[idx], "feasibility cut failed to exclude its anchor"
            n_removed = int((~keep).sum())
            rounds.append(LoopRound(r, z, res.status, res.q_value, float(direct[idx]),
                                    ub, lb, ub - lb, int(keep.sum()), n_removed,
                                    costs, states, probs))
            states, bits, direct = states[keep], bits[keep], direct[keep]
            continue

        rounds.append(LoopRound(r, z, res.status, res.q_value, float(direct[idx]),
                                ub, lb, ub - lb, len(states), 0,
                                costs, states, probs))
    else:
        termination = "max_rounds"

    # final bound with all cuts, over the surviving states
    if len(states) and opt_cuts:
        q_model = np.max(np.stack([c.evaluate(bits) for c in opt_cuts]), axis=0)
        lb = float((direct + q_model).min())
    return LoopResult(rounds, best_z, best_x, float(ub), lb, float(ub - lb),
                      termination, cuts)


def brute_force_optimum(inst: Instance,
                        params: Params | None = None) -> tuple[int | None, float, dict | None]:
    """Exact reference: solve the subproblem for every structurally feasible z."""
    params = params if params is not None else Params()
    states = enumerate_feasible(inst)
    direct = direct_costs(int_to_bits(states, inst.n_bits), inst)
    best_z, best_v, best_x = None, np.inf, None
    for s, d in zip(states, direct):
        res = solve_subproblem(build_sub_instance(inst, int(s), params))
        if res.feasible and d + res.q_value < best_v:
            best_z, best_v, best_x = int(s), float(d + res.q_value), res.x
    return best_z, best_v, best_x
