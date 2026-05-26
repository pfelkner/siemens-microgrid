## R. Mahroo, and A. Kargarian, Hybrid quantum-classical unit commitment. In 2022 IEEE Texas Power and Energy Conference (TPEC), pages 1-5, IEEE, Texas, 2022.
[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_arxiv_primary_2201_03701)  
[DOI](https://doi.org/10.1109/TPEC54980.2022.9750763)

## Hybrid Approach
The challenge of using classical annealing for the unit commitment problem is when you discretize continuous variables for a unit commitment problem with N instances, you need N(h+1)  qubits

The paper proposes an iterative hybrid quantum-classical algorithm where the unit commitment problem is split up into a quadratic subproblem, a quadratic unconstrained binary optimization (QUBO) subproblem, and an unconstrained quadratic subproblem
- the non-QUBO problems are solved by a classical solver, the QUBO problem with the quantum approximate optimization algorithm (QAOA)
- QAOA approximates QUBO problems and finds minima/maxima for an objective function with a discrete domain with a large configuration space
- the coordination of the three sub-problems happens via a three-block alternating direction method of multipliers algorithm (ADMM)
