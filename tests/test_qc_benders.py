"""Tests for the Benders cut machinery (Tasks 8+9): rhs_affine, cuts, loop."""
import numpy as np
import pytest

from feasible_x.feasible_start_x import Instance as SubInstance, Params, SlotConfig
from feasible_x.subproblem import solve_subproblem
from qc.benders import (FEAS_TOL, Cut, build_sub_instance, feasibility_cut,
                        optimality_cut, to_slot_configs)
from qc.benders import LoopResult, benders_loop, brute_force_optimum
from qc.grover_mixer import enumerate_feasible
from qc.instance import Instance, bit_index, int_to_bits, load_instance

DATA = "artifacts/data/all_data.csv"


def _qc_online_t1() -> Instance:
    # Deficit slot: load > pv, grid online. 27 structurally feasible states.
    return Instance(p_pv=np.array([100.0]), p_load=np.array([300.0]),
                    tou=np.array([0.20]), g_avail=np.array([1]))


def _state_from(roles_per_slot: list[list[str]]) -> int:
    state = 0
    for t, roles in enumerate(roles_per_slot):
        for role in roles:
            state |= 1 << bit_index(t, role)
    return state


def _t3_instance() -> SubInstance:
    # The reference T=3 online instance from subproblem._demo (conversation.md).
    return SubInstance(
        pv=[100, 400, 150],
        load=[300, 200, 350],
        grid_available=[1, 1, 1],
        config=[
            SlotConfig(batt="discharge", grid="import", band="mid"),
            SlotConfig(batt="charge", grid="export", band="mid"),
            SlotConfig(batt="discharge", grid="import", band="mid"),
        ],
        params=Params(eta=0.9, soc_init=120.0),
        tou=[0.20, 0.08, 0.30],
    )


class TestRhsAffine:
    def test_map_present_and_complete(self):
        res = solve_subproblem(_t3_instance())
        assert res.feasible
        assert set(res.rhs_affine.keys()) == set(res.duals.keys())

    def test_known_coefficients(self):
        p = Params(eta=0.9, soc_init=120.0)
        res = solve_subproblem(_t3_instance())
        const, coefs = res.rhs_affine["gate_ch_0"]
        assert const == 0.0 and coefs == {(0, "ch"): p.p_bess_nom}
        const, coefs = res.rhs_affine["soc_ub_1"]
        assert const == 0.0
        assert coefs == {(1, "b_low"): p.soc_low_th, (1, "b_mid"): p.soc_high_th,
                         (1, "b_high"): p.e_max}
        const, coefs = res.rhs_affine["soc_dyn_0"]
        assert const == p.soc_init and coefs == {}
        const, coefs = res.rhs_affine["bal_on_2"]
        assert const == 350 - 150 and coefs == {}

    def test_map_present_on_infeasible_branch(self):
        bad = SubInstance(
            pv=[0.0], load=[300.0], grid_available=[0],
            config=[SlotConfig(batt="idle", grid="idle", band="mid", served=True)],
            params=Params(eta=0.9, soc_init=500.0),
            tou=[0.20],
        )
        res = solve_subproblem(bad)
        assert not res.feasible
        assert set(res.rhs_affine.keys()) == set(res.farkas.keys())
        const, coefs = res.rhs_affine["served_up_0"]
        m_big = max(300.0, Params().p_bess_nom + 0.0) + 1.0
        assert const == pytest.approx(m_big - 0.0 + 300.0)
        assert coefs == {(0, "y"): pytest.approx(-m_big)}


class TestAdapter:
    def test_roundtrip_online_and_outage(self):
        inst = Instance(p_pv=np.array([100.0, 50.0]), p_load=np.array([300.0, 200.0]),
                        tou=np.array([0.2, 0.3]), g_avail=np.array([1, 0]))
        state = _state_from([["dis", "imp", "b_mid"], ["dis", "b_high", "y"]])
        cfgs = to_slot_configs(state, inst)
        assert cfgs[0] == SlotConfig(batt="discharge", grid="import", band="mid", served=False)
        assert cfgs[1] == SlotConfig(batt="discharge", grid="idle", band="high", served=True)

    def test_build_sub_instance_carries_data(self):
        inst = _qc_online_t1()
        sub = build_sub_instance(inst, _state_from([["dis", "imp", "b_mid"]]))
        assert sub.pv[0] == 100.0 and sub.load[0] == 300.0
        assert sub.tou[0] == 0.20 and sub.grid_available[0] == 1


class TestOptimalityCut:
    def test_tight_at_anchor_and_valid_everywhere(self):
        inst = _qc_online_t1()
        states = enumerate_feasible(inst)
        bits = int_to_bits(states, inst.n_bits)
        z_bar = _state_from([["dis", "imp", "b_mid"]])
        i_bar = int(np.where(states == z_bar)[0][0])
        res = solve_subproblem(build_sub_instance(inst, z_bar))
        assert res.feasible
        cut = optimality_cut(res, bits[i_bar], inst.n_bits)
        values = cut.evaluate(bits)
        # tight at the anchor z̄ …
        assert values[i_bar] == pytest.approx(res.q_value, abs=1e-6)
        # … and a valid lower bound on Q(z) for every z with a feasible subproblem
        for i, s in enumerate(states):
            r = solve_subproblem(build_sub_instance(inst, int(s)))
            if r.feasible:
                assert values[i] <= r.q_value + 1e-5 * max(1.0, abs(r.q_value)), \
                    f"cut above Q at state {int(s)}"


class TestFeasibilityCut:
    def test_excludes_anchor_and_only_infeasible_states(self):
        inst = _qc_online_t1()
        states = enumerate_feasible(inst)
        bits = int_to_bits(states, inst.n_bits)
        # all-idle in a deficit slot: hard balance pv + 0 = load is impossible
        z_bar = _state_from([["b_mid"]])
        i_bar = int(np.where(states == z_bar)[0][0])
        res = solve_subproblem(build_sub_instance(inst, z_bar))
        assert not res.feasible
        cut = feasibility_cut(res, bits[i_bar], inst.n_bits)
        assert cut is not None
        excluded = cut.evaluate(bits) < -FEAS_TOL
        assert excluded[i_bar], "cut must exclude its own anchor"
        # conservative: every excluded state really is infeasible
        for s in states[excluded]:
            r = solve_subproblem(build_sub_instance(inst, int(s)))
            assert not r.feasible, f"cut wrongly excluded feasible state {int(s)}"


class TestLoop:
    def test_converges_to_brute_force_optimum(self):
        # T=2 window containing the first natural outage slot (646): 486 states,
        # feasibility cuts WILL occur (served/band bits force things).
        inst = load_instance(DATA, start=645, T=2)
        result = benders_loop(inst, max_rounds=40, gap_tol=1e-4, shots=4096, seed=0)
        assert result.termination == "gap", \
            f"no convergence: {result.termination}, gap {result.gap}"
        z_star, v_star, _ = brute_force_optimum(inst)
        assert z_star is not None
        assert result.best_value == pytest.approx(v_star, abs=1e-3)
        assert result.best_x is not None
        # UB never below the exact optimum at any point (cut validity end-to-end)
        assert all(r.ub >= v_star - 1e-6 for r in result.rounds if np.isfinite(r.ub))

    def test_no_outage_window_converges(self):
        # flat direct costs, online-only: pure optimality-cut regime
        inst = load_instance(DATA, start=0, T=2)
        result = benders_loop(inst, max_rounds=40, gap_tol=1e-4, shots=4096, seed=0)
        assert result.termination == "gap"
        _, v_star, _ = brute_force_optimum(inst)
        assert result.best_value == pytest.approx(v_star, abs=1e-3)

    def test_history_is_recorded(self):
        inst = load_instance(DATA, start=645, T=2)
        result = benders_loop(inst, max_rounds=40, gap_tol=1e-4, shots=4096, seed=0)
        assert len(result.rounds) >= 1
        r1 = result.rounds[0]
        assert r1.lb == -np.inf                       # no q-model before the first optimality cut
        assert len(r1.costs) == len(r1.states) == len(r1.probs) == r1.n_states + r1.n_removed
        assert result.gap <= 1e-4
