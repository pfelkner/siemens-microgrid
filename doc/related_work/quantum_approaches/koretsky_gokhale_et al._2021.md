## S. Koretsky, P. Gokhale,et al., Adapting quantum approximation optimization algorithm (QAOA) for unit commitment. In 2021 IEEE International Conference on Quantum Computing and Engineering, pages 181-187, IEEE, Broomfield, 2021.
[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_arxiv_primary_2110_12624)  
[DOI](https://doi.org/10.48550/arXiv.2110.12624)

## Overview
As current quantum machines only have a limited amount of qubits, hybrid algorithms combining classical and quantum approaches which minimize the number of qubits needed are currently appealing 
- As the Unit commitment problem discrete binary and continuous variables, it is a good candidate to solve with a hybrid approach 

In classic approaches with MILP, CPLEX uses a branch-and-bound approach to solve problems like the unit commitment problem 

---

## Hybrid Approach

The paper does a  hybrid quantum-classical approach, using the Quantum Approximation Optimization Algorithm (QAOA) and a classical optimizer 
- QAOA also has the advantage that it can run on  gate-model quantum computers and does not need adiabatic ones as with annealing
- QAOA is used for the binary variables, while the classic optimizer is used for the continuous variables
- The  outer loop of the algorithm is done with the classic optimizer, the inner one with the QAOA
- This way you do not need to discretize continuous variables
- The solution reduces the number of gates while still leveraging quantum advantages

