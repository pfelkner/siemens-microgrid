## Y. Prasad, S. Chatrati, and S. Masthan, Constraint-Preserving QAOA for Real-Time Optimal Power Flow in Renewable-Rich Distribution Networks, Authorea Preprints, 2025.
[DOI](https://doi.org/10.36227/techrxiv.175977400.07371879/v1)

## Overview
Quantum Approximate Optimization Algorithm (QAOA) seems to be the most promising approach and is thus used in this paper

The paper uses QAOA with a Power-Flow Preserving Mixer (PF-Mixer) which eliminates the need for penalty terms and ensures feasibility of the solution

---

## Quantum challenges

Direct quantum applications to unit commitment problem have several challenges:
- Constraint Handling via penalties: you need large penalty weights to prevent constraint violations, but the larger the penalty weight is, the more numerically unstable your system is which leads to unrealistic solutions
- problem encoding: unit commitment problems have many continuous variables, but if you do not encode them well, you require many qubits which introduces massive overhead
- Mixer Design: your mixers should be designed to represent the actual physical solution space and exclude infeasible regions

Quantum annealing performs well for small problem instances but needs extensive problem reformulation

