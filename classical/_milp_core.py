"""Shared Gurobi MILP core for deterministic and stochastic classical solvers."""

from __future__ import annotations

import math
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

# ---------- Static parameters ----------
DT = 0.25                       # h per slot (15 min)
BESS_CAP = 1000.0               # kWh
BESS_PMAX = 250.0               # kW nominal
ETA_RT = 0.90                   # round-trip efficiency
ETA = math.sqrt(ETA_RT)         # per-direction efficiency
SOC_INIT = 500.0                # kWh
DEMAND_CHARGE = 15.0            # $/kW over billing period
RESILIENCY_PER_MIN = 15.0                       # $/min (band 10-20)
RESILIENCY_PER_SLOT = RESILIENCY_PER_MIN * 15.0 # $ per served 15-min outage slot = 225.0
EXPORT_RATE = 0.05                              # $/kWh paid for grid export
GRID_PMAX = 1000.0              # kW (sanity cap)
SOC_LOW_TH = 100.0              # kWh (10% of 1000)
SOC_HIGH_TH = 900.0             # kWh (90% of 1000)
FRAC_EDGE = 0.5
FRAC_MID = 1.0


def build_microgrid_milp(
    df_list: list[pd.DataFrame],
    scenario_probs: list[float] | None,
    time_limit: float | None,
    mip_gap: float,
    log_file: str,
    quiet: bool,
    resiliency_per_slot: float = RESILIENCY_PER_SLOT,
    export_rate: float = EXPORT_RATE,
    soc_init: float = SOC_INIT,
    peak_floor: float = 0.0,
    peak_mode: str = "demand_charge",
    penalty_rate: float = 0.0,
) -> tuple[gp.Model, dict, list[pd.DataFrame]]:
    """Build and solve the scenario-loop MILP shared by both public solvers.

    Deterministic callers pass a one-item `df_list`. Stochastic callers pass one
    dataframe per scenario. In both cases `peak_import` is a single scalar that
    covers every grid-import value represented in the model.
    """
    if not df_list:
        raise ValueError("df_list must contain at least one dataframe")

    M = len(df_list)
    T = len(df_list[0])
    if any(len(df) != T for df in df_list):
        raise ValueError("all scenario dataframes must have the same length")
    if scenario_probs is not None and len(scenario_probs) != M:
        raise ValueError("scenario_probs length must match df_list length")
    probs: list[float] = scenario_probs if scenario_probs is not None else [1.0 / M] * M

    p_pv = [df["p_kw"].to_numpy(dtype=float) for df in df_list]
    p_load = [df["load_kw"].to_numpy(dtype=float) for df in df_list]
    tou = [df["tou_usd_kwh"].to_numpy(dtype=float) for df in df_list]
    g_avail = [df["grid_available"].to_numpy(dtype=int) for df in df_list]
    outages = [[t for t in range(T) if g_avail[s][t] == 0] for s in range(M)]

    m = gp.Model("microgrid_milp")
    m.Params.OutputFlag = 0 if quiet else 1
    m.Params.LogFile = log_file
    m.Params.MIPGap = mip_gap
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = time_limit

    peak_import = m.addVar(lb=peak_floor, ub=GRID_PMAX, name="peak_import")

    def _n(base: str, s: int) -> str:
        return f"s{s}_{base}" if M > 1 else base

    grid_in = [m.addVars(T, lb=0.0, ub=GRID_PMAX, name=_n("grid_in", s)) for s in range(M)]
    grid_out = [m.addVars(T, lb=0.0, ub=GRID_PMAX, name=_n("grid_out", s)) for s in range(M)]
    bess_ch = [m.addVars(T, lb=0.0, ub=BESS_PMAX, name=_n("bess_ch", s)) for s in range(M)]
    bess_dis = [m.addVars(T, lb=0.0, ub=BESS_PMAX, name=_n("bess_dis", s)) for s in range(M)]
    soc_v = [m.addVars(T, lb=0.0, ub=BESS_CAP, name=_n("soc", s)) for s in range(M)]
    soc_low = [m.addVars(T, vtype=GRB.BINARY, name=_n("soc_low", s)) for s in range(M)]
    soc_mid = [m.addVars(T, vtype=GRB.BINARY, name=_n("soc_mid", s)) for s in range(M)]
    soc_high = [m.addVars(T, vtype=GRB.BINARY, name=_n("soc_high", s)) for s in range(M)]
    ch_active = [m.addVars(T, vtype=GRB.BINARY, name=_n("ch_active", s)) for s in range(M)]
    dis_active = [m.addVars(T, vtype=GRB.BINARY, name=_n("dis_active", s)) for s in range(M)]
    import_active = [m.addVars(T, vtype=GRB.BINARY, name=_n("import_active", s)) for s in range(M)]
    export_active = [m.addVars(T, vtype=GRB.BINARY, name=_n("export_active", s)) for s in range(M)]
    served = [m.addVars(outages[s], vtype=GRB.BINARY, name=_n("served", s)) for s in range(M)]

    for s in range(M):
        q = f"s{s}_" if M > 1 else ""
        for t in range(T):
            if g_avail[s][t] == 1:
                m.addConstr(
                    p_pv[s][t] + grid_in[s][t] - grid_out[s][t]
                    + bess_dis[s][t] - bess_ch[s][t] == p_load[s][t],
                    name=f"{q}bal_on_{t}",
                )
            else:
                m.addConstr(grid_in[s][t] == 0.0, name=f"{q}no_imp_{t}")
                m.addConstr(grid_out[s][t] == 0.0, name=f"{q}no_exp_{t}")
                m_big = max(p_load[s][t], BESS_PMAX + p_pv[s][t]) + 1.0
                resid = p_pv[s][t] + bess_dis[s][t] - bess_ch[s][t] - p_load[s][t]
                m.addConstr(resid <= m_big * (1 - served[s][t]), name=f"{q}out_bal_up_{t}")
                m.addConstr(-resid <= m_big * (1 - served[s][t]), name=f"{q}out_bal_lo_{t}")

            soc_prev = soc_init if t == 0 else soc_v[s][t - 1]
            m.addConstr(
                soc_v[s][t] == soc_prev + ETA * bess_ch[s][t] * DT
                              - (bess_dis[s][t] / ETA) * DT,
                name=f"{q}soc_dyn_{t}",
            )

            m.addConstr(soc_low[s][t] + soc_mid[s][t] + soc_high[s][t] == 1,
                        name=f"{q}band_sum_{t}")
            m.addGenConstrIndicator(soc_low[s][t], True, soc_v[s][t] <= SOC_LOW_TH,
                                    name=f"{q}ind_low_{t}")
            m.addGenConstrIndicator(soc_mid[s][t], True, soc_v[s][t] >= SOC_LOW_TH,
                                    name=f"{q}ind_mid_lo_{t}")
            m.addGenConstrIndicator(soc_mid[s][t], True, soc_v[s][t] <= SOC_HIGH_TH,
                                    name=f"{q}ind_mid_hi_{t}")
            m.addGenConstrIndicator(soc_high[s][t], True, soc_v[s][t] >= SOC_HIGH_TH,
                                    name=f"{q}ind_high_{t}")

            max_power_t = BESS_PMAX * (FRAC_EDGE * (soc_low[s][t] + soc_high[s][t])
                                       + FRAC_MID * soc_mid[s][t])
            m.addConstr(bess_ch[s][t] <= max_power_t, name=f"{q}ch_band_{t}")
            m.addConstr(bess_dis[s][t] <= max_power_t, name=f"{q}dis_band_{t}")

            m.addConstr(bess_ch[s][t] <= BESS_PMAX * ch_active[s][t], name=f"{q}ch_link_{t}")
            m.addConstr(bess_dis[s][t] <= BESS_PMAX * dis_active[s][t], name=f"{q}dis_link_{t}")
            m.addConstr(ch_active[s][t] + dis_active[s][t] <= 1, name=f"{q}xor_bess_{t}")

            m.addConstr(grid_in[s][t] <= GRID_PMAX * import_active[s][t], name=f"{q}imp_link_{t}")
            m.addConstr(grid_out[s][t] <= GRID_PMAX * export_active[s][t], name=f"{q}exp_link_{t}")
            m.addConstr(import_active[s][t] + export_active[s][t] <= 1, name=f"{q}xor_grid_{t}")

            m.addConstr(peak_import >= grid_in[s][t], name=f"{q}peak_{t}")

    energy_cost = gp.quicksum(
        probs[s] * tou[s][t] * grid_in[s][t] * DT
        for s in range(M) for t in range(T)
    )
    if peak_mode == "commit_penalty":
        demand_cost = penalty_rate * (peak_import - peak_floor)
    else:
        demand_cost = DEMAND_CHARGE * peak_import
    resiliency_revenue = gp.quicksum(
        probs[s] * resiliency_per_slot * served[s][t]
        for s in range(M) for t in outages[s]
    )
    export_revenue = gp.quicksum(
        probs[s] * export_rate * grid_out[s][t] * DT
        for s in range(M) for t in range(T)
    )
    m.setObjective(
        energy_cost + demand_cost - resiliency_revenue - export_revenue,
        GRB.MINIMIZE,
    )

    m.update()
    n_vars = m.NumVars
    n_bin = m.NumBinVars
    n_constr = m.NumConstrs + m.NumGenConstrs

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        raise RuntimeError(f"No feasible solution. Gurobi status = {status}")

    schedules: list[pd.DataFrame] = []
    for s in range(M):
        schedules.append(pd.DataFrame({
            "timestamp": df_list[s]["timestamp"].values,
            "p_pv_kw": p_pv[s],
            "p_load_kw": p_load[s],
            "Grid_Import": [grid_in[s][t].X for t in range(T)],
            "Grid_Export": [grid_out[s][t].X for t in range(T)],
            "BESS_Charge": [bess_ch[s][t].X for t in range(T)],
            "BESS_Discharge": [bess_dis[s][t].X for t in range(T)],
            "BESS_SoC": [soc_v[s][t].X for t in range(T)],
            "grid_available": g_avail[s],
        }))

    energy_cost_v = float(sum(
        probs[s] * tou[s][t] * grid_in[s][t].X * DT
        for s in range(M) for t in range(T)
    ))
    if peak_mode == "commit_penalty":
        demand_cost_v = float(penalty_rate * (peak_import.X - peak_floor))
    else:
        demand_cost_v = float(DEMAND_CHARGE * peak_import.X)
    resiliency_v = float(sum(
        probs[s] * resiliency_per_slot * served[s][t].X
        for s in range(M) for t in outages[s]
    ))
    export_revenue_v = float(sum(
        probs[s] * export_rate * grid_out[s][t].X * DT
        for s in range(M) for t in range(T)
    ))
    served_count = int(round(sum(
        sum(served[s][t].X for t in outages[s]) for s in range(M)
    ) / M))

    info = {
        "T": T,
        "M": M,
        "n_vars": n_vars,
        "n_binary_vars": n_bin,
        "n_constraints": n_constr,
        "runtime_s": float(m.Runtime),
        "status": int(status),
        "status_str": status_str(status),
        "mip_gap": float(m.MIPGap) if m.IsMIP else 0.0,
        "objective_bound": float(m.ObjBound) if m.IsMIP else float(m.ObjVal),
        "total_cost": float(m.ObjVal),
        "energy_cost": energy_cost_v,
        "demand_cost": demand_cost_v,
        "resiliency_revenue": resiliency_v,
        "export_revenue": export_revenue_v,
        "peak_import_kw": float(peak_import.X),
        "peak_floor_kw": float(peak_floor),
        "peak_mode": peak_mode,
        "served_count": served_count,
        "outage_slots": len(outages[0]),
    }
    return m, info, schedules


def status_str(status: int) -> str:
    table = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return table.get(status, f"STATUS_{status}")


def run_sanity_checks(schedule: pd.DataFrame, tol: float = 1e-4) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors are hard violations; warnings are non-fatal."""
    errors: list[str] = []
    warnings: list[str] = []

    online = schedule["grid_available"] == 1
    resid = (
        schedule["p_pv_kw"]
        + schedule["Grid_Import"]
        - schedule["Grid_Export"]
        + schedule["BESS_Discharge"]
        - schedule["BESS_Charge"]
        - schedule["p_load_kw"]
    )
    bad_online = online & (resid.abs() > tol)
    if bad_online.any():
        errors.append(
            f"Power balance violated on {int(bad_online.sum())} online slots "
            f"(max |residual| = {resid[bad_online].abs().max():.3e})"
        )

    if schedule["BESS_SoC"].min() < -tol:
        errors.append(f"SoC below 0: min = {schedule['BESS_SoC'].min():.3f}")
    if schedule["BESS_SoC"].max() > BESS_CAP + tol:
        errors.append(f"SoC above capacity: max = {schedule['BESS_SoC'].max():.3f}")

    outage = schedule["grid_available"] == 0
    if (outage & (schedule["Grid_Import"].abs() > tol)).any():
        errors.append("Grid_Import nonzero during outage")
    if (outage & (schedule["Grid_Export"].abs() > tol)).any():
        errors.append("Grid_Export nonzero during outage")

    both_nonzero = (
        (schedule["BESS_Charge"] > tol) & (schedule["BESS_Discharge"] > tol)
    ).sum()
    if both_nonzero > 0:
        errors.append(
            f"VIOLATION: {both_nonzero} slot(s) have simultaneous "
            f"charge+discharge - XOR constraint not enforced correctly"
        )

    return errors, warnings


def check_peak_cover(info: dict, schedules: list[pd.DataFrame], label: str = "peak_import") -> list[str]:
    """Return an error if the reported peak fails to cover all schedule imports."""
    max_grid_in = max(sc["Grid_Import"].max() for sc in schedules)
    if info["peak_import_kw"] < max_grid_in - 1e-3:
        return [
            f"{label} ({info['peak_import_kw']:.2f}) < max grid_in "
            f"({max_grid_in:.2f}) across schedules - coupling constraint broken"
        ]
    return []


def append_summary(out_path: Path, row: dict) -> None:
    M = row.get("M", 1)
    cols = [
        "slots", "n_scenarios", "mode",
        "n_vars", "n_binary_vars", "n_constraints",
        "runtime_s", "status", "mip_gap", "objective_bound",
        "total_cost", "energy_cost", "demand_cost", "resiliency_revenue", "export_revenue",
    ]
    df_row = pd.DataFrame([{
        "slots": row["T"],
        "n_scenarios": M,
        "mode": "stochastic" if M > 1 else "deterministic",
        "n_vars": row["n_vars"],
        "n_binary_vars": row["n_binary_vars"],
        "n_constraints": row["n_constraints"],
        "runtime_s": row["runtime_s"],
        "status": row["status_str"],
        "mip_gap": row["mip_gap"],
        "objective_bound": row["objective_bound"],
        "total_cost": row["total_cost"],
        "energy_cost": row["energy_cost"],
        "demand_cost": row["demand_cost"],
        "resiliency_revenue": row["resiliency_revenue"],
        "export_revenue": row.get("export_revenue", 0.0),
    }])[cols]
    header = not out_path.exists()
    df_row.to_csv(out_path, mode="a", header=header, index=False)
