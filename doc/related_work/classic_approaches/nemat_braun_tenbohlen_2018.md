## M. Nemati, M. Braun, and S. Tenbohlen, Optimization of unit commitment and economic dispatch in microgrids based on genetic algorithm and mixed integer linear programming. In Applied energy, pages 944-963,  Elsevier Science, Amsterdam, 2018.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_crossref_citationtrail_10_1016_j_apenergy_2017_07_007)  
[DOI](https://doi.org/10.1016/j.apenergy.2017.07.007)

## Overview
Almost all techniques used for solving the classic Unit Commitment problem have been at least partially adapted to also work with microgrids, 
for example Lagrangian relaxation, **mixed integer linear programming (MILP)**, meta-heuristic methods such as particle swarm optimization 
and the **genetic algorithm (GA)**. 

The paper mainly compares the performance of GA and MILP in regard to unit commitment and economic dispatch. 
- It takes aging of the battery (BESS) into account for cost minimization
- It gives several mathematical formulas which might be useful in paper, for example calculation of active power output of PV generator

The paper models BESS with the following parameters which might be useful for us as well:
- min SOC is fixed at 10% of total BESS capacity, max SOC at 90% 
- battery state should be fixed to only move between min SOC and max SOC 
- initial SOC of BESS and BESS for next optimization step (next day) should always be fixed at min SOC

---

## MILP
- In order to optimize unit commitment and economic dispatch problem with MILP, you need to linearize the non-linear cost functions in a piecewise manner into segments with linear slopes
- The number of segments influences optimization: The more segments, the higher the computational cost

---

## Notes from the Paper
- Figure 4.2: Shows how number of segments influences computational cost
- Figure 5.3: Overview over optimized unit commitment across several forms of energy: diesel generator, fuel cell, micro gas turbine, wind turbine, and photovoltaic generator
