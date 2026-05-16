## P. Halffmann, P. Holzer, K. Plociennik, and M. Trebing, A quantum computing approach for the unit commitment problem. In International conference on operations research, pages 113-120, Springer, Cham, 2022.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_springer_books_10_1007_978_3_031_24907_5_14)  
[DOI](https://doi.org/10.1007/978-3-031-24907-5_14)

## Overview
The paper models unit commitment problem as a quadratic unconstrained optimization problem to solve it on quantum computer
- It is a proof-of-concept where unit commitment problem is formulated with minimal variable and starting costs that demands satisfaction as well as minimum running and idle times
- this is supposed to reduce number of and the connectivity between qubits and should ensure that all constraints are satisfied even when they are transformed in penalty terms
- It transforms unit commitment problem to a QUBO matrix and then to Ising Hamiltonians, measuring the total energy of a physical system

---

## Comparison to MILP
Other papers shows that microgrid with three to twelve power units shows that while D-Wave returns near-optimal solutions, the computation via Gurobi is 30000 times faster for the largest instance