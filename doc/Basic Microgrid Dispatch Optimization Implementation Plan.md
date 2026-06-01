
This document outlines the foundational steps for building a Mixed-Integer Linear Programming (MILP) model to optimize the dispatch of a utility grid connection, Photovoltaic (PV) system, and Battery Energy Storage System (BESS).

---

## Point 1: Data Preparation for the New Mexico Microgrid

Before writing optimization constraints, all time-series inputs must be available at a consistent 15-minute resolution for a **30-day billing period**. The use case is a commercial microgrid near Albuquerque, New Mexico (lat 35.08, lon -106.65, timezone `America/Denver`).

---

### Phase 1.1: Synthesize Data

The following datasets must be prepared. Where real measurements are unavailable, synthetic profiles are generated as described below.

**Time Series Data (15-min resolution, 30 days = 2880 rows):**

- **PV Generation (`p_kw`):** Physics-based simulation using `pvlib` and a Typical Meteorological Year (TMY) dataset (see Phase 1.2). New Mexico has among the highest irradiance in the US; expect peak AC output around 1:00 PM MST.

- **Electrical Load (`load_kw`):** Synthetic commercial building profile. Baseline 200 kW with Gaussian peaks: morning (+100 kW centred at 08:00) and evening (+80 kW centred at 18:00). Add ±5% white noise to break the symmetry.

- **ToU Energy Cost (`tou_usd_kwh`):** Three-tier PNM-inspired schedule (Mountain Time):
  | Period | Hours | Rate |
  |--------|-------|------|
  | Off-peak | 22:00 – 06:00 | $0.05/kWh |
  | Mid-peak | 06:00 – 16:00 | $0.15/kWh |
  | On-peak | 16:00 – 22:00 | $0.40/kWh |

- **Grid Availability (`grid_available`):** Bernoulli draw per slot with outage probability $p = 0.005$ (≈ 99.5% uptime), seeded for reproducibility. Value is 1 (available) or 0 (outage).

**Static Parameters (scalars passed directly to the optimizer):**

| Parameter | Value |
|-----------|-------|
| Demand charge rate | $15.00/kW |
| BESS capacity | 1 000 kWh |
| BESS max charge/discharge power | 250 kW |
| BESS round-trip efficiency | 0.90 (√0.90 per direction) |
| Initial SoC | 500 kWh |
| Resiliency revenue rate | $15/min of islanded operation = $225/slot (band: $10–20/min) |
| Export tariff | $0.05/kWh paid for grid export |

All time series are merged into a single `all_data.csv` keyed on `timestamp` (tz-naive UTC offset removed, local Mountain Time assumed throughout).

---

### Phase 1.2: PV Data Generation — Implementation

See **`pv_data.py`** for the full implementation. It follows the same structure as `template.py` (PVGIS ERA5 TMY → pvlib PVWatts chain → 15-min interpolation → tiled to the billing window), adapted for the NM context:

- Location: Albuquerque, NM (`lat=35.08, lon=-106.65`, `America/Denver`)
- Tilt reduced to 20° (closer to NM latitude for annual yield optimum)
- No curtailment input — replaces the German feed-in/market-price CSVs with synthetic load, ToU tariff, and grid-availability builders
- Outputs a single `all_data.csv` containing `p_kw`, `load_kw`, `tou_usd_kwh`, and `grid_available`

Run with: `python pv_data.py --start-date 2025-06-01 --end-date 2025-06-30`

---

## Phase 2: Step-by-Step Implementation

We recommend using the **PuLP** library in Python for this deterministic baseline. It is readable and comes with the CBC solver.

### Step 1: Initialize the Model

- Create a PuLP linear programming problem object.
    
- Set the objective sense to `Minimize`.
    

### Step 2: Define the Decision Variables

Create dictionaries of continuous variables for every time step $t$:

- `Grid_Import[t]` (Lower bound: 0)
    
- `Grid_Export[t]` (Lower bound: 0)
    
- `BESS_Charge[t]` (Lower bound: 0, Upper bound: BESS Max Power)
    
- `BESS_Discharge[t]` (Lower bound: 0, Upper bound: BESS Max Power)
    
- `BESS_SoC[t]` (Lower bound: 0, Upper bound: BESS Capacity)
    

Create a single continuous variable for the entire period to track peak demand:

- `Max_Grid_Import` (Lower bound: 0)
    

### Step 3: Build the Core Constraints

**1. Power Balance Constraint**

At every time step $t$, the electrical load must be met by the sum of all generation, storage, and grid interactions.

$$P_{\text{load}}(t) = P_{\text{PV}}(t) + P_{\text{grid\_import}}(t) - P_{\text{grid\_export}}(t) + P_{\text{BESS\_discharge}}(t) - P_{\text{BESS\_charge}}(t)$$

**2. BESS Dynamics Constraint**

Track the State of Charge (SoC) based on charging/discharging efficiency.

_For the initial time step ($t=0$):_

$$SoC(0) = SoC_{\text{init}} + (P_{\text{BESS\_charge}}(0) \cdot \eta_{\text{charge}}) - \frac{P_{\text{BESS\_discharge}}(0)}{\eta_{\text{discharge}}}$$

_For all subsequent time steps ($t>0$):_

$$SoC(t) = SoC(t-1) + (P_{\text{BESS\_charge}}(t) \cdot \eta_{\text{charge}}) - \frac{P_{\text{BESS\_discharge}}(t)}{\eta_{\text{discharge}}}$$

**3. Demand Charge Tracking**

Ensure `Max_Grid_Import` captures the highest import peak.

- For all $t$: `Max_Grid_Import >= Grid_Import[t]`
    

### Step 4: Formulate the Objective Function

The goal is to minimize total costs, which include Time of Use (ToU) energy costs and fixed demand charges, minus any potential resiliency revenue.

$$\text{Minimize} \sum_{t} (\text{Cost}_{\text{ToU}}(t) \cdot P_{\text{grid\_import}}(t)) + (\text{Cost}_{\text{Demand}} \cdot P_{\text{grid\_max}}) - \sum_{t} (\text{Revenue}_{\text{Resiliency}}(t))$$

_(Note: For the most basic baseline, you can set the Resiliency Revenue term to 0)._

### Step 5: Solve and Extract

- Call `model.solve()`.
    
- Write a loop to extract the optimized values of your variables (`Grid_Import`, `BESS_Charge`, `BESS_Discharge`, `BESS_SoC`) for each time step into a Pandas DataFrame.
    

### Step 6: Visualize the Dispatch Strategy

- Use `matplotlib` or `plotly` to plot your results.
    
- Create a stacked bar chart showing how the load is met against the ToU price curve. Ensure the battery is charging during cheap hours and discharging during peak price hours.