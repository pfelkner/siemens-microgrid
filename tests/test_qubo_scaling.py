import pytest
import pandas as pd
import qubo_model
from qubo_model import build_qubo, DT, GRID_PMAX, DEMAND_CHARGE, EXPORT_RATE, RESILIENCY_PER_SLOT


def _single_slot_df(grid_available: int = 1, tou: float = 0.30) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": ["2024-01-01 00:00"],
        "p_kw": [100.0],
        "load_kw": [100.0],
        "tou_usd_kwh": [tou],
        "grid_available": [grid_available],
    })


class TestMaxObjCoeff:
    def test_demand_charge_dominates_at_default_params(self):
        bits = 4
        step = GRID_PMAX / (2 ** bits - 1)
        msb = step * (2 ** (bits - 1))
        expected = DEMAND_CHARGE * msb
        result = qubo_model._max_obj_coeff(
            bits, DEMAND_CHARGE, 0.30, EXPORT_RATE, RESILIENCY_PER_SLOT
        )
        assert abs(result - expected) < 1e-9

    def test_resiliency_dominates_when_large(self):
        bits = 4
        step = GRID_PMAX / (2 ** bits - 1)
        msb = step * (2 ** (bits - 1))
        big_r = DEMAND_CHARGE * msb * 2
        result = qubo_model._max_obj_coeff(
            bits, DEMAND_CHARGE, 0.30, EXPORT_RATE, big_r
        )
        assert abs(result - big_r) < 1e-9

    def test_tou_dominates_when_extreme(self):
        bits = 4
        step = GRID_PMAX / (2 ** bits - 1)
        msb = step * (2 ** (bits - 1))
        big_tou = (DEMAND_CHARGE / DT) * 2
        result = qubo_model._max_obj_coeff(
            bits, DEMAND_CHARGE, big_tou, EXPORT_RATE, RESILIENCY_PER_SLOT
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
    def test_peak_msb_is_one_in_outage(self):
        # During outage (g_avail=0), peak bits appear only in demand charge objective.
        # No B.4 penalty adds to peak. After bqm.scale(1/_moc):
        # peak[bits-1] linear = demand_charge * msb / _moc = 1.0 exactly.
        bits = 4
        df = _single_slot_df(grid_available=0)
        bqm, _ = build_qubo(df, bits_grid=bits, bits_bess=bits, bits_soc=bits)
        msb_key = f"peak[{bits - 1}]"
        assert abs(bqm.linear[msb_key] - 1.0) < 1e-9
