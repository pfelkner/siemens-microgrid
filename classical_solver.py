import docplex.mp.model as cplex_model
import pandas as pd

def run_microgrid_analysis():
    # csv created by pv_data.py
    df = pd.read_csv("all_data.csv")
    timestamps = df["timestamp"].tolist()

    tou_prices = df["tou_usd_kwh"].tolist()
    pv_forecast = df["p_kw"].tolist()
    load_forecast = df["load_kw"].tolist()

    T = len(tou_prices)
    dt = 0.25 # data in csv has 0.25 hour steps

    mdl = cplex_model.Model(name="Microgrid_Analysis")

    # battery charges and discharges with 95% efficiency
    eta_charge    = 0.95
    eta_discharge = 0.95

    # maxmimum power with which battery can charge or discharge in kW
    max_battery_power = 150.0
    # maximum battery in kW
    battery_capacity = 300.0

    # amount of energy bought and sold into power grid in kW
    p_buy = mdl.continuous_var_list(T, lb=0, name="grid_buy")
    p_sell = mdl.continuous_var_list(T, lb=0, name="grid_sell")
    # amount of power battery charges or discharges in kW
    p_ch = mdl.continuous_var_list(T, lb=0, ub=max_battery_power, name="bess_charge")
    p_dis = mdl.continuous_var_list(
        T, lb=0, ub=max_battery_power, name="bess_discharge"
    )
    # State of charge of battery in kWh
    soc = mdl.continuous_var_list(T + 1, lb=5, ub=battery_capacity, name="soc")
    # is battery charging/buying
    is_charging = mdl.binary_var_list(T, name="is_charging")
    is_buying = mdl.binary_var_list(T, name="is_buying")
    # Big M - stops selling and buying at the same time
    M_grid = 300.0
    p_max = mdl.continuous_var(lb=0, name="peak_demand")

    # Inital state of charge
    mdl.add_constraint(soc[0] == 20)
    # SoC needs to be the same in the end as initial state
    mdl.add_constraint(soc[T] >= 20)

    for t in range(T):
        # energy balance ingoing/outgoing
        mdl.add_constraint(
            p_buy[t] - p_sell[t] + pv_forecast[t] + p_dis[t]
            == load_forecast[t] + p_ch[t])

        # SoC for next hour needs to be the same as current SoC + charge + discharge
        mdl.add_constraint(
    soc[t+1] == soc[t]
              + p_ch[t]  * eta_charge    * dt
              - p_dis[t] / eta_discharge * dt)

        # Battery cannot charge and discharge at the same time
        mdl.add_constraint(p_ch[t] <= is_charging[t] * max_battery_power)
        mdl.add_constraint(p_dis[t] <= (1 - is_charging[t]) * max_battery_power)
        # Peak-Demand charges
        mdl.add_constraint(p_max >= p_buy[t])
        # no buying and selling energy at the same time
        mdl.add_constraint(p_buy[t] <= is_buying[t] * M_grid)
        mdl.add_constraint(p_sell[t] <= (1 - is_buying[t]) * M_grid)

    sell_prices = [price * 1.2 for price in tou_prices] 

    energy_cost = mdl.sum(p_buy[t] * tou_prices[t] * dt for t in range(T))
    energy_revenue = mdl.sum(p_sell[t] * sell_prices[t] * dt for t in range(T))
    demand_charge = p_max * 15.0 * (1/30) # with CPLEX community, we can only optimize one day so peak charge should also only be for one day of month
    resiliency_revenue = soc[T] * 0.45

    mdl.minimize(energy_cost - energy_revenue + demand_charge - resiliency_revenue)
    solution = mdl.solve()

    # --- generating terminal output ---
    if solution:
        header = f"{'Time':>11} | {'Price [USD/kWh]':>14} | {'PV':>5} | {'Load':>5} | {'Grid Buy':>8} | {'Grid sell':>8} | {'SoC':>6} | {'Operation':<15}"
        print(header)
        print("-" * len(header))

        for t in range(T):
            buy_val = solution.get_value(p_buy[t])
            sell_val = solution.get_value(p_sell[t])
            soc_val = solution.get_value(soc[t])
            ch_val = solution.get_value(p_ch[t])
            dis_val = solution.get_value(p_dis[t])

            if ch_val > 0.1:
                action = f"CHARGE (+{ch_val:.1f})"
            elif dis_val > 0.1:
                action = f"DISCHARGE (-{dis_val:.1f})"
            else:
                action = "IDLE"

            time_str = str(timestamps[t])[11:16]

            print(
                f" {time_str:>10} | {tou_prices[t]:>15.2f} | {pv_forecast[t]:5.1f} | {load_forecast[t]:5.1f} | {buy_val:>8.1f} | {sell_val:>9.1f} | {soc_val:6.1f} | {action}"
            )

        total_buy = solution.get_value(energy_cost)
        total_sell = solution.get_value(energy_revenue)
        total_peak = solution.get_value(demand_charge)
        total_res = solution.get_value(resiliency_revenue)

        print("-" * len(header))
        print(f"COST BREAKDOWN:")
        print(f"  (+) Energy purchase (ToU):      {total_buy:>8.2f} USD")
        print(f"  (+) Demand Charge (Peak):  {total_peak:>8.2f} USD")
        print(f"  (-) Energy sales:         {total_sell:>8.2f} USD")
        print(f"  (-) Resilience revenue:      {total_res:>8.2f} USD")
        print(f"  " + "=" * 35)
        print(f"  TOTAL COST:           {solution.objective_value:>8.2f} USD")
        print(f"  (Negative = profit, Positive = cost)")

    else:
        print("No solution could be found")


if __name__ == "__main__":
    run_microgrid_analysis()
