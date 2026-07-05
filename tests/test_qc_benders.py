"""Tests for the Benders cut machinery (Tasks 8+9): rhs_affine, cuts, loop."""
import numpy as np
import pytest

from feasible_x.feasible_start_x import Instance as SubInstance, Params, SlotConfig
from feasible_x.subproblem import solve_subproblem


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
