## J. Zhou, Z. Zhu, L. Zhu, and S. Bu, Problem-Structure-Informed Quantum Approximate Optimization Algorithm for Large-Scale Unit Commitment with Limited Qubits. arXiv preprint arXiv:2503.20509, 2025.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_arxiv_primary_2503_20509)  
[DOI](https://doi.org/10.48550/arXiv.2503.20509)

## Classic Approaches
Classic approaches include MILP, heuristic methods and reinforcement learning
- MILP provides an optimal solution, but is not suitable for large-scale problems as it becomes computational expensive
- Heuristic methods such as the  Genetic Algorithm or Particle Swarm Optimization scale better and are more flexible but often do not achieve a global optimim
- Reinforcement learning needs a lot of (re-)training, hyper-parameter tuning and often introduces computational overhead

---

## Hybrid Approach

The paper proposes using the Quantum Approximate Optimization Algorithm (QAOA) to solve the unit commitment problem
- It splits up the large problem into smaller sub-problems
- The decomposition is done with regard to the  structure of power system topology