# feasible_x — continuous side of the Benders loop

The classical half of the hybrid microgrid dispatch PoC (`../QC_Ansatz_07-02.md`):
given a fixed discrete config `z`, produce / solve the continuous variables `x`.

## Files

| File | What it is |
|---|---|
| `feasible_start_x.py` | **Task 3** — samples feasible continuous configs for a fixed `z` (the loop's Start-`x`); also a feasibility oracle. Defines the shared `Instance` / `SlotConfig` / `Params` / `verify`. |
| `subproblem.py` | **Task 7** — Gurobi LP at fixed `z`; returns `x*` + duals (optimality) or a Farkas certificate (feasibility cut). |
| `scenario_runner.py` | CLI front-end: run prebuilt / JSON / CSV scenarios through both. |
| `why_feasibility_cuts.md` | Plain-language writeup of why this model hits feasibility cuts. |
| `sample_t3.json` | Example scenario (the T=3 case from `../doc/conversation.md`) for JSON mode. |

Requires the project venv (numpy, scipy, gurobipy). Run scripts by path so the
sibling modules resolve, e.g. from the repo root (`siemens-microgrid/`):

```bash
python feasible_x/scenario_runner.py --list
python feasible_x/scenario_runner.py --scenario reference-t3 --solve
python feasible_x/scenario_runner.py --file feasible_x/sample_t3.json --solve
python feasible_x/scenario_runner.py --csv all_data.csv --slots 4 --solve
python feasible_x/scenario_runner.py --scenario night-deficit --save my.json   # editable template
python feasible_x/scenario_runner.py --scenario reference-t3 --n 50 --dump samples.json  # export all feasible x (.json or .csv)
python feasible_x/scenario_runner.py --selftest
```

Each module also self-checks when run directly:

```bash
python feasible_x/feasible_start_x.py   # sampler demo (conversation T=3)
python feasible_x/subproblem.py         # solver demo (feasible + infeasible branch)
```

## Scenario input

T (number of slots) = the number of slots you provide; Δt (slot length) = `params.dt`.
Neither is hardcoded. Three ways in: a prebuilt (`--scenario`), a JSON file
(`--file`), or a CSV slice (`--csv --slots N`, discrete `z` filled heuristically).

JSON schema — only `slots` is required; per-slot fields default to
`pv=0, load=0, tou=0.05, grid="idle", batt="idle", band="mid", online=true, served=false`:

```json
{
  "name": "my-scenario",
  "params": {"eta": 0.9, "soc_init": 120, "dt": 0.25},
  "slots": [
    {"pv": 100, "load": 300, "tou": 0.20, "batt": "discharge", "grid": "import"},
    {"pv": 400, "load": 200, "batt": "charge", "grid": "export"},
    {"pv": 150, "load": 350, "online": false, "served": true, "batt": "discharge"}
  ]
}
```
