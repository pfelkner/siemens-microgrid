"""Convenience front-end for the feasible-config sampler and the Benders subproblem.

Pick a scenario three ways and run it, no code editing:

  * prebuilt registry :  uv run python -m feasible_x.scenario_runner --scenario reference-t3
  * JSON file         :  uv run python -m feasible_x.scenario_runner --file my_scenario.json
  * CSV slice         :  uv run python -m feasible_x.scenario_runner --csv all_data.csv --slots 4

Common invocations:

  uv run python -m feasible_x.scenario_runner --list                          # show prebuilt scenarios
  uv run python -m feasible_x.scenario_runner --scenario reference-t3 --solve
  uv run python -m feasible_x.scenario_runner --file my_scenario.json --n 50
  uv run python -m feasible_x.scenario_runner --csv all_data.csv --slots 4 --solve
  uv run python -m feasible_x.scenario_runner --scenario night-deficit --save tpl.json   # dump editable template
  uv run python -m feasible_x.scenario_runner --scenario reference-t3 --n 50 --dump samples.json  # all feasible x

(Run via -m from the repo root so the package imports resolve.)

Add `--solve` to also run the Gurobi subproblem (Task 7) and print Q(z) + duals.
Use `--list` to see prebuilt names, `--save out.json` to dump the chosen scenario as
an editable template, `--n N` to set the sample count, `--selftest` to run checks.
T (number of slots) and Δt (slot length) are both free — T is just the number of slots
you provide; Δt is `params.dt`.

JSON schema (every field optional except it must have `slots`):

    {
      "name": "my-scenario",
      "params": {"eta": 0.9, "soc_init": 120, "dt": 0.25},
      "slots": [
        {"pv": 100, "load": 300, "tou": 0.20, "batt": "discharge", "grid": "import"},
        {"pv": 400, "load": 200, "batt": "charge", "grid": "export"},
        {"pv": 150, "load": 350, "online": false, "served": true, "batt": "discharge"}
      ]
    }

Per-slot defaults: pv=0, load=0, tou=0.05, grid="idle", batt="idle", band="mid",
online=true, served=false — so you only write what matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from feasible_x.feasible_start_x import Instance, SlotConfig, Params, feasible_configs, verify, Infeasible
from feasible_x.subproblem import solve_subproblem

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Heuristic discrete config `z` for data that carries no config of its own
# --------------------------------------------------------------------------- #
def auto_config(pv, load, grid_available) -> list[SlotConfig]:
    """A plausible (not guaranteed feasible) `z`: cover deficits, bank surpluses.

    deficit (load>pv) → discharge + import; surplus → charge + export; mid band.
    Outage slots get grid="idle" and served=False (safe default — flipping served on
    is exactly where feasibility cuts appear; see why_feasibility_cuts.md).
    The sampler doubles as a feasibility oracle, so it will tell you if this `z` is dead.
    """
    cfg = []
    for pv_t, load_t, online in zip(pv, load, grid_available):
        deficit = load_t > pv_t + EPS
        batt = "discharge" if deficit else "charge"
        grid = ("import" if deficit else "export") if online else "idle"
        cfg.append(SlotConfig(batt=batt, grid=grid, band="mid", served=False))
    return cfg


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def from_json(path: str | Path) -> Instance:
    spec = json.loads(Path(path).read_text())
    slots = spec["slots"]
    pv     = [s.get("pv", 0.0) for s in slots]
    load   = [s.get("load", 0.0) for s in slots]
    tou    = [s.get("tou", 0.05) for s in slots]
    online = [1 if s.get("online", True) else 0 for s in slots]
    config = [SlotConfig(batt=s.get("batt", "idle"), grid=s.get("grid", "idle"),
                         band=s.get("band", "mid"), served=bool(s.get("served", False)))
              for s in slots]
    return Instance(pv, load, online, config, Params(**spec.get("params", {})), tou=tou)


def from_csv(path: str | Path, slots: int, params: Params | None = None) -> Instance:
    """First `slots` rows of the dispatch CSV; discrete `z` filled by auto_config."""
    import pandas as pd
    df = pd.read_csv(path).iloc[:slots]
    pv, load = df["p_kw"].to_numpy(), df["load_kw"].to_numpy()
    tou = df["tou_usd_kwh"].to_numpy()
    ga = df["grid_available"].to_numpy()
    return Instance(pv, load, ga, auto_config(pv, load, ga), params or Params(), tou=tou)


def to_json(inst: Instance, name: str = "scenario") -> str:
    """Dump an instance as an editable JSON template (round-trips with from_json)."""
    p = inst.params
    slots = [{"pv": float(inst.pv[t]), "load": float(inst.load[t]), "tou": float(inst.tou[t]),
              "online": bool(inst.grid_available[t]), "batt": c.batt, "grid": c.grid,
              "band": c.band, "served": c.served}
             for t, c in enumerate(inst.config)]
    params = {"dt": p.dt, "eta": p.eta, "soc_init": p.soc_init, "e_max": p.e_max,
              "p_bess_nom": p.p_bess_nom, "p_grid_max": p.p_grid_max}
    return json.dumps({"name": name, "params": params, "slots": slots}, indent=2)


# --------------------------------------------------------------------------- #
# Prebuilt registry
# --------------------------------------------------------------------------- #
def _reference_t3() -> Instance:
    """Canonical verification case: 1 battery, T=3, no outage, SoC starts near the
    100 kWh floor so the temporal SoC coupling ("chain") visibly binds — discharge at
    t0 is capped at 72 kW by the SoC floor, not the 250 kW battery limit. Its reduced
    feasible polytope was worked out by hand (../doc/conversation.md), so the sampler
    and solver are checked against a known answer (LP optimum at battery powers
    72/200/162, Q(z)=1929.25). Same instance as sample_t3.json; default with no --scenario.
    """
    return Instance(
        pv=[100, 400, 150], load=[300, 200, 350], grid_available=[1, 1, 1],
        config=[SlotConfig("discharge", "import", "mid"),
                SlotConfig("charge", "export", "mid"),
                SlotConfig("discharge", "import", "mid")],
        params=Params(eta=0.9, soc_init=120.0), tou=[0.20, 0.08, 0.30])


def _night_deficit() -> Instance:
    # No PV, steady load → discharge + import each slot.
    return Instance(
        pv=[0, 0], load=[200, 250], grid_available=[1, 1],
        config=[SlotConfig("discharge", "import", "mid"),
                SlotConfig("discharge", "import", "mid")],
        params=Params(soc_init=500.0), tou=[0.30, 0.30])


def _outage_served() -> Instance:
    # One online surplus slot, then a deficit outage the battery CAN serve (feasible).
    return Instance(
        pv=[200, 100], load=[150, 250], grid_available=[1, 0],
        config=[SlotConfig("charge", "export", "mid"),
                SlotConfig("discharge", "idle", "mid", served=True)],
        params=Params(soc_init=500.0), tou=[0.15, 0.15])


def _outage_infeasible() -> Instance:
    # Idle battery must serve a deficit outage → no continuation → feasibility cut.
    return Instance(
        pv=[0.0], load=[300.0], grid_available=[0],
        config=[SlotConfig("idle", "idle", "mid", served=True)],
        params=Params(soc_init=500.0), tou=[0.20])


SCENARIOS = {
    "reference-t3": _reference_t3,            # known-answer verification case (SoC chain binds)
    "night-deficit": _night_deficit,          # T=2, no PV, discharge+import
    "outage-served": _outage_served,          # feasible served outage
    "outage-infeasible": _outage_infeasible,  # demonstrates a feasibility cut
}


# --------------------------------------------------------------------------- #
# Runner + CLI
# --------------------------------------------------------------------------- #
def _cost(inst: Instance, x: dict) -> float:
    p = inst.params
    e = sum(inst.tou[t] * x["p_imp"][t] * p.dt for t in range(inst.T))
    ex = sum(p.export_rate * x["p_exp"][t] * p.dt for t in range(inst.T))
    return e + p.demand_charge * x["p_peak"] - ex


def dump_samples(xs: list[dict], path: str | Path, meta: dict) -> None:
    """Write all feasible samples to .json (structured) or .csv (long form)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample", "t", "p_imp", "p_exp", "p_ch", "p_dis", "soc", "p_peak"])
            for i, x in enumerate(xs):
                for t in range(len(x["p_imp"])):
                    w.writerow([i, t, x["p_imp"][t], x["p_exp"][t], x["p_ch"][t],
                                x["p_dis"][t], x["soc"][t], x["p_peak"]])
    else:
        payload = {**meta, "n_samples": len(xs),
                   "samples": [{k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in x.items()} for x in xs]}
        path.write_text(json.dumps(payload, indent=2))
    print(f"[dump] wrote {len(xs)} feasible samples → {path}")


def run(inst: Instance, n: int, seed: int, solve: bool,
        dump: str | None = None, label: str = "scenario") -> int:
    print(f"[instance] T={inst.T}  Δt={inst.params.dt}h  "
          f"online={int(inst.grid_available.sum())}/{inst.T}  SoC_init={inst.params.soc_init}")
    try:
        xs = feasible_configs(inst, n=n, seed=seed)
    except Infeasible as ex:
        print(f"[sampler] INFEASIBLE — no continuous continuation for this z.")
        print(f"          reason: {ex}")
        print(f"          → in the loop this z triggers a Benders FEASIBILITY cut.")
        # still run the solver if asked, to show the Farkas branch
        if solve:
            r = solve_subproblem(inst)
            print(f"[solver ] status={r.status}; Farkas rows: "
                  f"{sum(abs(v) > 1e-9 for v in (r.farkas or {}).values())}")
        return 1

    costs = [_cost(inst, x) for x in xs]
    best = int(np.argmin(costs))
    print(f"[sampler] {len(xs)} feasible samples; cost range "
          f"[{min(costs):.2f}, {max(costs):.2f}]")
    x = xs[best]
    print(f"          cheapest sample: P^ch={np.round(x['p_ch'],1)} "
          f"P^dis={np.round(x['p_dis'],1)} SoC={np.round(x['soc'],1)} peak={x['p_peak']:.1f}")

    if dump:
        # These are feasible CONTINUOUS x (Start-x, Schritt 1) — seeds for the loop's
        # initialization / H_C's initial rest-cost. The Grover mixer encodes discrete z,
        # not these. See ../QC_Ansatz_07-02.md and why_feasibility_cuts.md.
        meta = {"scenario": label, "T": inst.T,
                "variables": ["p_imp", "p_exp", "p_ch", "p_dis", "soc", "p_peak"],
                "note": "feasible continuous x for fixed z (Start-x); not the discrete "
                        "Grover-mixer states",
                "params": {"dt": inst.params.dt, "eta": inst.params.eta,
                           "soc_init": inst.params.soc_init},
                "seed": seed}
        dump_samples(xs, dump, meta)

    if solve:
        r = solve_subproblem(inst)
        if r.feasible:
            verify(inst, r.x)
            nz = {k: round(v, 3) for k, v in r.duals.items() if abs(v) > 1e-6}
            print(f"[solver ] OPTIMAL  Q(z)={r.q_value:.4f}  (best sample {min(costs):.4f})")
            print(f"          P^dis={np.round(r.x['p_dis'],1)} P^ch={np.round(r.x['p_ch'],1)} "
                  f"peak={r.x['p_peak']:.1f}")
            print(f"          nonzero duals: {nz}")
        else:
            print(f"[solver ] INFEASIBLE — Farkas certificate → feasibility cut")
    return 0


def _selftest() -> int:
    # every prebuilt runs; the infeasible one raises; JSON round-trips exactly.
    for name, fn in SCENARIOS.items():
        inst = fn()
        if name == "outage-infeasible":
            try:
                feasible_configs(inst, n=1); assert False, "should be infeasible"
            except Infeasible:
                pass
        else:
            xs = feasible_configs(inst, n=5, seed=0)
            assert xs and all(verify(inst, x) is None for x in xs)
    # JSON round-trip on reference-t3
    inst = _reference_t3()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(to_json(inst)); path = f.name
    r = from_json(path)
    assert r.T == inst.T and np.allclose(r.pv, inst.pv) and np.allclose(r.tou, inst.tou)
    assert [c.batt for c in r.config] == [c.batt for c in inst.config]
    print("[selftest] OK — all prebuilt scenarios run, JSON round-trips.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--scenario", choices=list(SCENARIOS), help="run a prebuilt scenario")
    src.add_argument("--file", help="run a scenario from a JSON file")
    src.add_argument("--csv", help="build a scenario from the first --slots rows of a CSV")
    ap.add_argument("--slots", type=int, default=4, help="rows to take when using --csv")
    ap.add_argument("--list", action="store_true", help="list prebuilt scenarios and exit")
    ap.add_argument("--n", type=int, default=20, help="number of feasible samples to draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solve", action="store_true", help="also solve the LP subproblem (duals)")
    ap.add_argument("--dump", help="write all feasible samples to this .json or .csv path")
    ap.add_argument("--save", help="write the chosen scenario to this JSON path and exit")
    ap.add_argument("--selftest", action="store_true", help="run internal checks and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.list:
        for name, fn in SCENARIOS.items():
            print(f"  {name:20s} T={fn().T}")
        return 0

    if args.file:
        inst = from_json(args.file)
    elif args.csv:
        inst = from_csv(args.csv, args.slots)
    else:
        inst = SCENARIOS[args.scenario or "reference-t3"]()

    if args.save:
        Path(args.save).write_text(to_json(inst, name=args.scenario or "scenario"))
        print(f"[saved] {args.save}")
        return 0

    label = args.scenario or (Path(args.file).stem if args.file else
                              Path(args.csv).stem if args.csv else "scenario")
    return run(inst, n=args.n, seed=args.seed, solve=args.solve, dump=args.dump, label=label)


if __name__ == "__main__":
    sys.exit(main())
