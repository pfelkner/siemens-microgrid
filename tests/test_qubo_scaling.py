import pytest
import pandas as pd
from qubo import qubo_model
from qubo.qubo_model import build_qubo, DT, GRID_PMAX, DEMAND_CHARGE, EXCEEDANCE_PENALTY, EXPORT_RATE, RESILIENCY_PER_SLOT, BILLING_SLOTS


def _single_slot_df(grid_available: int = 1, tou: float = 0.30) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": ["2024-01-01 00:00"],
        "p_kw": [100.0],
        "load_kw": [100.0],
        "tou_usd_kwh": [tou],
        "grid_available": [grid_available],
    })


class TestMaxObjCoeff:
    def test_exceedance_penalty_dominates_at_default_params(self):
        bits = 4
        step = GRID_PMAX / (2 ** bits - 1)
        msb = step * (2 ** (bits - 1))
        expected = EXCEEDANCE_PENALTY * msb
        result = qubo_model._max_obj_coeff(
            bits, EXCEEDANCE_PENALTY, 0.30, EXPORT_RATE, RESILIENCY_PER_SLOT
        )
        assert abs(result - expected) < 1e-9

    def test_resiliency_dominates_when_large(self):
        bits = 4
        step = GRID_PMAX / (2 ** bits - 1)
        msb = step * (2 ** (bits - 1))
        big_r = EXCEEDANCE_PENALTY * msb * 2
        result = qubo_model._max_obj_coeff(
            bits, EXCEEDANCE_PENALTY, 0.30, EXPORT_RATE, big_r
        )
        assert abs(result - big_r) < 1e-9

    def test_tou_dominates_when_extreme(self):
        bits = 4
        step = GRID_PMAX / (2 ** bits - 1)
        msb = step * (2 ** (bits - 1))
        big_tou = (EXCEEDANCE_PENALTY / DT) * 2
        result = qubo_model._max_obj_coeff(
            bits, EXCEEDANCE_PENALTY, big_tou, EXPORT_RATE, RESILIENCY_PER_SLOT
        )
        assert result == pytest.approx(big_tou * DT * msb)


class TestBuildQuboSignature:
    def test_lam_override_raises_type_error(self):
        df = _single_slot_df()
        with pytest.raises(TypeError):
            build_qubo(df, lam_balance=1.0)  # type: ignore[call-arg]

    def test_obj_scale_raises_type_error(self):
        df = _single_slot_df()
        with pytest.raises(TypeError):
            build_qubo(df, obj_scale=100.0)  # type: ignore[call-arg]

    def test_builds_without_error(self):
        df = _single_slot_df()
        bqm, ctx = build_qubo(df, bits_grid=4, bits_bess=4, bits_soc=4)
        assert len(bqm.variables) > 0

    def test_obj_scale_not_in_ctx(self):
        df = _single_slot_df()
        _, ctx = build_qubo(df)
        assert "obj_scale" not in ctx


class TestNormalization:
    def test_excess_msb_is_one_in_outage(self):
        # During outage (g_avail=0), excess bits appear only in exceedance penalty objective.
        # No B.4 penalty adds to excess. billing_slots=1 forces window_share=1 so
        # ep = EXCEEDANCE_PENALTY dominates; after bqm.scale(1/_moc): excess[bits-1] = 1.0.
        bits = 4
        df = _single_slot_df(grid_available=0)
        bqm, _ = build_qubo(df, bits_grid=bits, bits_bess=bits, bits_soc=bits, billing_slots=1)
        msb_key = f"excess[{bits - 1}]"
        assert abs(bqm.linear[msb_key] - 1.0) < 1e-9
