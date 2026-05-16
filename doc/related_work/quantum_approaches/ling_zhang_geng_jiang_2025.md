## J. Ling, Q. Zhang, G. Geng, and Q. Jiang, Hybrid quantum annealing decomposition framework for unit commitment. In Electric Power Systems Research, article 111121, Elsevier Science, Amsterdam, 2025.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_crossref_citationtrail_10_1016_j_epsr_2024_111121)  
[DOI](https://doi.org/10.1016/j.epsr.2024.111121)

## Overview
The paper proposes hybrid quantum–classical framework to solve unit commitment problem using adiabatic quantum computers and quantum annealing

It splits the large unit commitment problem into smaller, quantum-compatible sub-problems and transforms these into QUBOs for quantum optimization
- It uses a quadratic unconstrained binary optimization modeling method to transform the sub-problems into the form of quantum computing 
- It introduces variable reduction to handle limited quantum resources

---

## UC Decomposition
To discretize the unit commitment problem, Lagrangian decomposition and Benders decomposition techniques are used. This way, continuous variables do not need to be discretized which is the 
other way to transform the unit commitment problem into sub-problems manageable by adiabatic quantum computers 
(by transforming it into a pure integer programming problem)

In the paper, Benders decomposition is used to split unit commitment problem into two sub-problems: the integer programming master problem and the linear programming sub-problem
- These two sub-problems are solved iteratively to find the optimal solution of the unit commitment problem
- The  integer programming master problem is the one that presents most challenge to classical computers and accounts for approximately 90% of the solution time 

Slack variables are used to transform inequality constraints into equality constraints
- However, too large slack variables leads to too many auxiliary qubits which affects QUBO efficiency, so they need to be carefully reduced
- To determine the correct slack variable’s size, the paper analyzes the maximum reachability limit of the inequality constraints

---

## Validation
The paper validates the approach through quantum simulations on classical computers and experiments on real quantum annealing hardware, e.g. DWaveSampler with number of physical qubits ranging from 46 to 47