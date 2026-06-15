"""QUBO formulation for the Siemens microgrid dispatch problem.

Builds a `dimod.BinaryQuadraticModel` (BQM) consumed by dwave_annealer.py,
which solves it via D-Wave simulated annealing in a rolling-window MPC loop.

Core model (vs. classical_solver.py): power balance (B.1), SoC dynamics (B.2),
box bounds (B.3), peak coupling + demand charge (B.4), SoC-band derating (B.5),
charge/discharge XOR + import/export XOR (B.6), outage resiliency (B.7), ToU
energy cost, and export revenue.

MILP -> QUBO transformation:
  1. Continuous vars binary-encoded: v = (vmax / (2^k - 1)) * sum_i 2^i x_i.
     Box bounds (B.3) satisfied by construction.
  2. Equalities (balance, SoC dynamics, SoC-band one-hot, derating, outage
     balance) -> quadratic penalty lam * (expr)^2.
  3. Inequalities -> (B.4, B.5) Unbalanced Penalization: lam1*h + lam2*h² without
     slack vars; (B.7) equality with binary-encoded slack + squared penalty.
  4. XOR constraints -> native quadratic product penalty
     lam * P_ch(x) * P_dis(x) (no indicator binaries, no Big-M).
  5. Linear objective terms sit on the QUBO diagonal; resiliency reward -r*y_t
     likewise (B.7).

Entry point: dwave_annealer.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import dimod

# ---------- Static parameters (mirrors classical_solver.py; not imported to
# ---------- avoid the hard gurobipy dependency) ----------
DT = 0.25                       # h per slot (15 min)
BESS_CAP = 1000.0               # kWh
BESS_PMAX = 250.0               # kW nominal
ETA_RT = 0.90                   # round-trip efficiency
ETA = math.sqrt(ETA_RT)         # per-direction efficiency
SOC_INIT = 500.0                # kWh
DEMAND_CHARGE = 15.0            # $/kW over billing period
EXPORT_RATE = 0.05              # $/kWh paid for grid export
GRID_PMAX = 1000.0              # kW
SOC_LOW_TH = 100.0              # kWh low-band SoC threshold (B.5)
SOC_HIGH_TH = 900.0             # kWh high-band SoC threshold (B.5)
RESILIENCY_PER_SLOT = 225.0     # $/slot resiliency revenue during outages (B.7)


# ---------- Binary encoding helper ----------
class Encoding:
    """Binary encoding of a continuous variable v in [0, vmax] with k bits.

    v = step * sum_i 2^i x_i,  step = vmax / (2^k - 1).
    Exposes the encoding as {bit_name: coefficient} for use in penalty terms.
    """

    def __init__(self, name: str, vmax: float, bits: int):
        self.name = name
        self.vmax = vmax
        self.bits = bits
        self.step = vmax / (2**bits - 1)
        self.coeffs: dict[str, float] = {
            f"{name}[{i}]": self.step * (2**i) for i in range(bits)
        }

    def decode(self, sample: dict) -> float:
        return sum(c * sample[v] for v, c in self.coeffs.items())


def add_squared_penalty(
    bqm: dimod.BinaryQuadraticModel,
    terms: dict[str, float],
    const: float,
    lam: float,
) -> None:
    """Add lam * (sum_i a_i x_i + const)^2 to the BQM.

    Uses x^2 = x for binaries:
      (sum a_i x_i + c)^2 = sum (a_i^2 + 2 c a_i) x_i
                            + 2 sum_{i<j} a_i a_j x_i x_j + c^2
    """
    names = list(terms)
    for i, u in enumerate(names):
        a = terms[u]
        bqm.add_linear(u, lam * (a * a + 2.0 * const * a))
        for v in names[i + 1:]:
            bqm.add_quadratic(u, v, lam * 2.0 * a * terms[v])
    bqm.offset += lam * const * const


def add_up_penalty(
    bqm: dimod.BinaryQuadraticModel,
    terms: dict[str, float],
    const: float,
    lam1: float,
    lam2: float,
) -> None:
    """Unbalanced Penalization for h(x) ≤ 0, h = sum_i a_i x_i + const.

    P(h) = lam1*h + lam2*h^2.  Minimum at h* = -lam1/(2*lam2) < 0 — inside
    feasible region.  Condition: lam1 < 2*lam2*M (M = max |violation|).
    lam1 is typically set to lam2 * encoding_step so the minimum sits ~1
    quantization level inside the feasible region.
    """
    for v, a in terms.items():
        bqm.add_linear(v, lam1 * a)
    bqm.offset += lam1 * const
    add_squared_penalty(bqm, terms, const, lam2)


def add_product_penalty(
    bqm: dimod.BinaryQuadraticModel,
    terms_a: dict[str, float],
    terms_b: dict[str, float],
    lam: float,
) -> None:
    """Add lam * (sum a_i x_i) * (sum b_j y_j) — the XOR penalty (disjoint vars)."""
    for u, a in terms_a.items():
        for v, b in terms_b.items():
            bqm.add_quadratic(u, v, lam * a * b)


# ---------- Model builder (backend-neutral) ----------
def _max_obj_coeff(
    bits_grid: int,
    demand_charge: float,
    tou_max: float,
    export_rate: float,
    resiliency_rate: float,
) -> float:
    """Exact maximum objective coefficient across all QUBO objective terms.

    Used to set lambda = 2 * _max_obj_coeff (Bakó condition) and as the
    normalization divisor for bqm.scale(), putting all objective coefficients
    in [-1, 1] with lambda_eff = 2.0.
    """
    step_grid = GRID_PMAX / (2 ** bits_grid - 1)
    msb_grid  = step_grid * (2 ** (bits_grid - 1))
    return max(
        demand_charge * msb_grid,
        tou_max * DT * msb_grid,
        export_rate * DT * msb_grid,
        resiliency_rate,
    )


def build_qubo(
    df: pd.DataFrame,
    bits_grid: int = 4,
    bits_bess: int = 4,
    bits_soc: int = 4,
    soc_init: float = SOC_INIT,
    demand_charge: float = DEMAND_CHARGE,
    export_rate: float = EXPORT_RATE,
    resiliency_rate: float = RESILIENCY_PER_SLOT,
) -> tuple[dimod.BinaryQuadraticModel, dict]:
    """Build the core-model QUBO as a dimod BQM.

    All objective coefficients are normalized to [-1, 1] via bqm.scale(1/_moc)
    where _moc = _max_obj_coeff(...). Lambda = 2.0 * _moc satisfies the Bakó
    condition (constraint violation always costlier than max objective gain).

    Returns (bqm, ctx) where ctx holds the encodings needed to decode samples.
    """
    T = len(df)
    p_pv = df["p_kw"].to_numpy(dtype=float)
    p_load = df["load_kw"].to_numpy(dtype=float)
    tou = df["tou_usd_kwh"].to_numpy(dtype=float)
    g_avail = df["grid_available"].to_numpy(dtype=int)

    tou_max = float(np.abs(tou).max()) if np.abs(tou).max() > 0 else 1.0
    _moc = _max_obj_coeff(bits_grid, demand_charge, tou_max, export_rate, resiliency_rate)
    _lam = 2.0 * _moc

    lam_balance   = _lam
    lam_soc       = _lam
    lam_peak      = _lam
    lam_xor       = 4.0 * _lam
    lam_derating  = _lam
    lam_resilience = _lam

    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

    grid_in, grid_out, bess_ch, bess_dis, soc = {}, {}, {}, {}, {}
    b7_up, b7_lo = {}, {}
    _vmax_b7 = 2.0 * (BESS_PMAX + GRID_PMAX)   # max B.7 slack value
    for t in range(T):
        if g_avail[t] == 1:
            grid_in[t] = Encoding(f"grid_in_{t}", GRID_PMAX, bits_grid)
            grid_out[t] = Encoding(f"grid_out_{t}", GRID_PMAX, bits_grid)
        bess_ch[t] = Encoding(f"bess_ch_{t}", BESS_PMAX, bits_bess)
        bess_dis[t] = Encoding(f"bess_dis_{t}", BESS_PMAX, bits_bess)
        soc[t] = Encoding(f"soc_{t}", BESS_CAP, bits_soc)
        if g_avail[t] == 0:
            b7_up[t] = Encoding(f"b7_up_{t}", _vmax_b7, bits_grid)
            b7_lo[t] = Encoding(f"b7_lo_{t}", _vmax_b7, bits_grid)
    peak = Encoding("peak", GRID_PMAX, bits_grid)

    # Snap soc_init onto the SoC encoding grid — otherwise the SoC-dynamics
    # penalty has an unavoidable residual at t=0 that biases the battery
    # into spurious charge/discharge action.
    soc_step = soc[0].step
    soc_init = round(soc_init / soc_step) * soc_step

    # ----- Objective (linear -> QUBO diagonal) -----
    for t in range(T):
        if g_avail[t] == 1:
            for v, c in grid_in[t].coeffs.items():
                bqm.add_linear(v, tou[t] * DT * c)                    # energy cost
            for v, c in grid_out[t].coeffs.items():
                bqm.add_linear(v, -export_rate * DT * c)              # export revenue
    for v, c in peak.coeffs.items():
        bqm.add_linear(v, demand_charge * c)                          # demand charge

    # ----- Constraints as penalties -----
    for t in range(T):
        # B.1 power balance: pv + grid_in - grid_out + dis - ch = load
        if g_avail[t] == 1:
            terms: dict[str, float] = dict(grid_in[t].coeffs)
            terms.update({v: -c for v, c in grid_out[t].coeffs.items()})
            terms.update(bess_dis[t].coeffs)
            terms.update({v: -c for v, c in bess_ch[t].coeffs.items()})
            add_squared_penalty(bqm, terms, p_pv[t] - p_load[t], lam_balance)
        else:
            # B.7 outage balance: |dis - ch + pv - load| ≤ M*(1-y_t), reward -r*y_t
            yt = f"y_{t}"
            bqm.add_linear(yt, -resiliency_rate)
            m_bal = float(BESS_PMAX + max(p_pv[t], 0.0) + max(p_load[t], 0.0) + 1.0)
            net = p_pv[t] - p_load[t]
            # upper:  m_bal*(1-y_t) - (dis - ch + net) - s_up = 0
            terms_up: dict[str, float] = {yt: -m_bal}
            terms_up.update({v: -c for v, c in bess_dis[t].coeffs.items()})
            terms_up.update(bess_ch[t].coeffs)
            terms_up.update({v: -c for v, c in b7_up[t].coeffs.items()})
            add_squared_penalty(bqm, terms_up, m_bal - net, lam_resilience)
            # lower:  (dis - ch + net) + m_bal*(1-y_t) - s_lo = 0
            terms_lo: dict[str, float] = {yt: -m_bal}
            terms_lo.update(bess_dis[t].coeffs)
            terms_lo.update({v: -c for v, c in bess_ch[t].coeffs.items()})
            terms_lo.update({v: -c for v, c in b7_lo[t].coeffs.items()})
            add_squared_penalty(bqm, terms_lo, net + m_bal, lam_resilience)

        # B.2 SoC dynamics: soc_t - soc_{t-1} - eta*ch*DT + dis*DT/eta = 0
        terms = dict(soc[t].coeffs)
        const = 0.0
        if t == 0:
            const = -soc_init
        else:
            terms.update({v: -c for v, c in soc[t - 1].coeffs.items()})
        terms.update({v: -ETA * DT * c for v, c in bess_ch[t].coeffs.items()})
        terms.update({v: (DT / ETA) * c for v, c in bess_dis[t].coeffs.items()})
        add_squared_penalty(bqm, terms, const, lam_soc)

        # B.6 XOR via product penalty (no indicator binaries needed)
        add_product_penalty(bqm, bess_ch[t].coeffs, bess_dis[t].coeffs, lam_xor)
        if g_avail[t] == 1:
            add_product_penalty(bqm, grid_in[t].coeffs, grid_out[t].coeffs, lam_xor)

            # B.4 peak coupling: grid_in[t] ≤ peak  (UP, no slack)
            h_pk: dict[str, float] = dict(grid_in[t].coeffs)
            h_pk.update({v: -c for v, c in peak.coeffs.items()})
            add_up_penalty(bqm, h_pk, 0.0, lam_peak * grid_in[t].step, lam_peak)

        # B.5 SoC-band assignment + power derating:
        #   b_low+b_mid+b_high=1,  P_ch/dis ≤ 0.5*P_nom*(1+b_mid)
        bl, bm, bh = f"b_low_{t}", f"b_mid_{t}", f"b_high_{t}"
        _lam_soc_link = lam_derating * 0.05
        add_squared_penalty(bqm, {bl: 1.0, bm: 1.0, bh: 1.0}, -1.0, lam_derating)
        for v, c in soc[t].coeffs.items():
            bqm.add_quadratic(bl, v, _lam_soc_link * c)
        bqm.add_linear(bl, -_lam_soc_link * SOC_LOW_TH)
        for v, c in soc[t].coeffs.items():
            bqm.add_quadratic(bh, v, -_lam_soc_link * c)
        bqm.add_linear(bh, _lam_soc_link * SOC_HIGH_TH)
        # B.5 derating: P_ch ≤ 0.5·P_nom·(1+b_mid)  (UP, no slack)
        lam1_der = lam_derating * bess_ch[t].step
        h_ch: dict[str, float] = dict(bess_ch[t].coeffs)
        h_ch[bm] = -0.5 * BESS_PMAX
        add_up_penalty(bqm, h_ch, -0.5 * BESS_PMAX, lam1_der, lam_derating)
        h_dis: dict[str, float] = dict(bess_dis[t].coeffs)
        h_dis[bm] = -0.5 * BESS_PMAX
        add_up_penalty(bqm, h_dis, -0.5 * BESS_PMAX, lam1_der, lam_derating)

    # Normalize: objective coefficients → [-1, 1], lambda_eff = 2.0
    bqm.scale(1.0 / _moc)

    ctx = {
        "T": T, "df": df, "grid_in": grid_in, "grid_out": grid_out,
        "bess_ch": bess_ch, "bess_dis": bess_dis, "soc": soc, "peak": peak,
        "g_avail": g_avail, "soc_init": soc_init,
        "demand_charge": demand_charge, "export_rate": export_rate,
        "resiliency_rate": resiliency_rate,
    }
    return bqm, ctx


# ---------- Decoding & validation ----------
def decode(sample: dict, ctx: dict) -> pd.DataFrame:
    T, df = ctx["T"], ctx["df"]
    g = ctx["g_avail"]
    result = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "p_pv_kw": df["p_kw"].values,
        "p_load_kw": df["load_kw"].values,
        "Grid_Import": [ctx["grid_in"][t].decode(sample) if g[t] else 0.0 for t in range(T)],
        "Grid_Export": [ctx["grid_out"][t].decode(sample) if g[t] else 0.0 for t in range(T)],
        "BESS_Charge": [ctx["bess_ch"][t].decode(sample) for t in range(T)],
        "BESS_Discharge": [ctx["bess_dis"][t].decode(sample) for t in range(T)],
        "BESS_SoC": [ctx["soc"][t].decode(sample) for t in range(T)],
        "grid_available": g,
        "SoC_Band": [
            "low" if sample.get(f"b_low_{t}", 0) else
            ("high" if sample.get(f"b_high_{t}", 0) else "mid")
            for t in range(T)
        ],
        "Outage_Served": [
            int(sample.get(f"y_{t}", 0)) if g[t] == 0 else None
            for t in range(T)
        ],
    })
    return result


def evaluate(schedule: pd.DataFrame, sample: dict, ctx: dict,
             repair_peak: bool = True) -> dict:
    """True cost + constraint residuals of a decoded solution.

    repair_peak: peak is analytically determined by the imports
    (smallest encodable value >= max grid_in), so we repair it in
    post-processing — standard QUBO practice for auxiliary/derived variables.
    """
    df = ctx["df"]
    tou = df["tou_usd_kwh"].to_numpy(dtype=float)
    peak_raw = ctx["peak"].decode(sample)
    peak_v = peak_raw
    if repair_peak:
        step = ctx["peak"].step
        needed = math.ceil(schedule["Grid_Import"].max() / step - 1e-9) * step
        peak_v = min(peak_raw, max(needed, 0.0)) if peak_raw > needed else needed

    energy = float((tou * schedule["Grid_Import"].to_numpy() * DT).sum())
    export = float((ctx["export_rate"] * schedule["Grid_Export"].to_numpy() * DT).sum())
    demand = ctx["demand_charge"] * peak_v
    r_rate = ctx.get("resiliency_rate", RESILIENCY_PER_SLOT)
    g = ctx["g_avail"]
    resiliency = float(sum(
        r_rate * int(sample.get(f"y_{t}", 0))
        for t in range(ctx["T"]) if g[t] == 0
    ))

    bal = (schedule["p_pv_kw"] + schedule["Grid_Import"] - schedule["Grid_Export"]
           + schedule["BESS_Discharge"] - schedule["BESS_Charge"]
           - schedule["p_load_kw"]).to_numpy()
    soc_arr = schedule["BESS_SoC"].to_numpy()
    soc_prev = np.concatenate([[ctx["soc_init"]], soc_arr[:-1]])
    soc_resid = soc_arr - soc_prev - ETA * schedule["BESS_Charge"].to_numpy() * DT \
        + schedule["BESS_Discharge"].to_numpy() * DT / ETA
    both = ((schedule["BESS_Charge"] > 1e-6) & (schedule["BESS_Discharge"] > 1e-6)).sum()
    both_grid = ((schedule["Grid_Import"] > 1e-6) & (schedule["Grid_Export"] > 1e-6)).sum()
    peak_viol = float(max(0.0, schedule["Grid_Import"].max() - peak_v))

    return {
        "true_cost": energy + demand - export - resiliency,
        "energy_cost": energy,
        "demand_cost": demand,
        "export_revenue": export,
        "resiliency_revenue": resiliency,
        "peak_import_kw": peak_v,
        "peak_import_raw_kw": peak_raw,
        "max_balance_resid_kw": float(np.abs(bal).max()),
        "max_soc_resid_kwh": float(np.abs(soc_resid).max()),
        "simultaneous_ch_dis_slots": int(both),
        "simultaneous_imp_exp_slots": int(both_grid),
        "peak_undercover_kw": peak_viol,
    }
