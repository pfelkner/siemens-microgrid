## D. Bucher, D. Porawski et al., Efficient QAOA Architecture for Solving Multi-Constrained Optimization Problems, 2025 IEEE International Conference on Quantum Computing and Engineering, pages 356–367, IEEE, Broomfield, 2025.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_arxiv_primary_2506_03115)  
[DOI](https://doi.org/10.1109/QCE65121.2025.00048)

## Overview
The paper looks at improving quantum approaches for combinatorial optimization problems (such as the unit commitment problem)
- usually, with these problems are handled via QUBO where slack variables are used and constraints are added as penalties to the objective function
- These additional variables and penalties add a lot of complexity which leads to deteriorating solution quality

---

## Constraint encoding in this approach

The paper uses the QAOA which it describes as “digitized simulation of the adiabatic process”
- Constraints are handled in two different ways: via XY -mixers and the oracle-based IF method
-  One-hot constraints are not added via penalty terms but are handled via XY -mixers limiting the search-space to only feasible solutions
-  Inequality constraints are handled via the oracle-based IF method
- QAOA switches between two  Hamiltonians: a  cost Hamiltonian  and a  mixing Hamiltonian
- the paper also omits the classical solver which can often be found together with QAOA
 

