## N. Nikmehr, P. Zhang, and M.A. Bragin, Quantum distributed unit commitment: An application in microgrids. In IEEE transactions on power systems, pages 3592-3603, IEEE, New York, 2022.

[DOI](https://doi.org/10.1109/TPWRS.2022.3141794)

## Overview
Within the unit commitment problem, binary commitment decision variables are the main reason for  combinatorial complexity and thus, for computational cost

Currently, Quantum Approximate Optimization Algorithm (QAOA) is used for unit commitment optimization where a multi-level parameterized quantum circuit optimizes the expected value of the QUBO objective function

---
## Decomposition of UC

The paper mainly focuses on the decomposition framework for quantum optimization of the unit commitment problem

In order to solve large unit commitment problems, it uses decomposition and coordination methods, especially the alternate direction method of multipliers (ADMM)
- advantage of this method to split problem: the sub-problems are smaller and easier to solve than when you split them up via Augmented Lagrangian Relaxation (ALR)
- this way, continuous variables can be used and not only  binary and integer variables as in other approaches
- this approach also allows for handling of distributed optimization of the sub-problems

To prepare the unit commitment problem, the objective function is first converted to a QUBO problem
- It is important that the  quadratic objective function should include the existing constraints
- Then the objective function can be mapped to the Hamiltonian Ising model

