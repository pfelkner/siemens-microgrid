## B. Salgado, A. Sequeira, and L.P. Santos, In A hybrid classical-quantum approach to highly constrained unit commitment problems. arXiv preprint arXiv:2412.11312, 2024.
[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_proquest_journals_3145907200)  
[DOI](https://doi.org/10.48550/arXiv.2412.11312)

## Overview
The paper presents an iterative, hybrid quantum-classical algorithm that solves unit commitment problem in polynomial time
- The problem is split into two sub-problems: a QUBO with the QAOA algorithm and a quadratic optimization problem with a classic solver
- QAOA finds approximate solutions to combinatorial optimization problems
- When you can transform a combinatorial problem as a QUBO problem, you can solve it  approximately via a QAOA

---

## QUBO
In its original form, QUBOs do not include constraints, but only binary variables 
- You can either add constraints to the QUBO via continuous slack variables and indicator variables
- However, these need to be discretized which increases the number qubits needed 
- Or you can add constraints by splitting the problem into a QUBO and a quadric problem containing the constraints and use a classic solver on it

The first paper to use a QUBO did the first approach and discretized the continuous variables 
- This is very effective for small problems, but with larger problems (and thus, more continuous variables) its performance decreases significantly 

---

## Hybrid Approach

The paper favors an iterative way with  decomposition and uses a QUBO with QAOA algorithm for the binary variables and a classis solver for the non-binary ones
Before transforming the problem into a QUBO, the problem needs to be minimized by removing some constraints which transformed into penalty terms in the objective function
- It might be a good idea to make the penalty terms adaptive, starting at low penalties which increase with each iteration
-  In the QUBO problem, all non-binary variables are treated as constants
- QAOA then searched for approximate solutions  in polynomial time
- In the quadratic problem, all binary variables are treated as constants