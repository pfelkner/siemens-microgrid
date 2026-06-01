## F. Phillipson, S. Muller, Quantum Approaches for the Unit Commitment Problem-a Literature Survey. In International Conference on Quantum Engineering Sciences and Technologies for Industry and Services, pages 3-13, Springer Nature Switzerland, Cham, 2025.

[DOI](https://doi.org/10.1007/978-3-032-13855-2_1)

## Definition UC and classic approaches
Unit commitment problem in general aims to minimize operational cost while honoring to technical constraints such as minimum up and downtimes or generator ramp up rates
- It is especially challenging in systems with high integration of renewable energy sources

Classical techniques, especially Mixed-Integer Programming using commercial solvers such as CPLEX and Gurobi, have evolved a lot in recent years (in the United States, for example, such models save approximately $5 billion annually in costs for power system operators)

---

## Overview quantum approaches

There are various quantum approaches for this problem in literature:
1. Quantum Approximate optimization Algorithm (QAOA)
   - promising for binary decision variables
   - often uses decomposition in sub-problems that can be solved with quantum or classical approaches
   - often integrates problem with meta heuristics to improve solution quality
   - problems: scalability, does not handle noisy data well
2. Quantum Annealing (QA)
   - effective for smaller problem instances where solutions close to the global optimum can be found
   - larger problems need to be decomposed in sub-problems which negatively impacts  solution quality
   - problem: scalability
3. Quantum Machine Learning
   - uses Neural Networks enhanced with quantum models 
4. Hybrid quantum-classical methods on Benders Decomposition
   - combine classic and quantum methodologies
   - enhance scalability and computational efficiency of classical approaches
   - good for large problems

---

## Notes from the Paper
- Table 1 clusters the main papers for the approaches according to their methodology