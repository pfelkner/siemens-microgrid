## L. Montero, A. Bello, and J. Reneses, A review on the unit commitment problem: Approaches, techniques, and resolution methods. In Energies, page 1296. MDPI, Basel, 2022.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_proquest_journals_2632717824)  
[DOI](https://doi.org/10.3390/en15041296)

## Overview of Classical Approaches
There are several non-quantum methodologies used to address the unit commitment problem:
- Exhaustive Enumeration: Tries all feasible solutions (brute force approach)
- Expert System: Encodes human expertise and rules into algorithms (often sub-optimal results)
- Priority List: Ranks decisions based on their impact on objective function (e.g. minimizing cost)
- Fuzzy Logic: Uses if–then rules, generalized input data and approximate reasoning to handle uncertainty (lacks accuracy)
- Neural Networks: Learns patterns from data to predict good solutions (requires training data)
- Optimization Problem: Formulates problem as a mathematical optimization, typically solved with methods like MILP (standard approach today)
- Hybrid Methodologies: Combine multiple approaches to improve performance

All techniques have several advantages and disadvantages, for example:

| Technique                | Advantages                                                                    | Disadvantages                                                         |
|------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Exhaustive Enumeration | Guarantees optimal solution                                                   | Computationally expensive, infeasible for real problems               |
| Expert System          | Fast, handles large knowledge bases, combines theoretical–practical knowledge | Non-optimal results, hard to implement                                |
| Priority List          | Fast, elatively easy to implement, mathematically grounded                    | Can be far from optimal                                               |
| Fuzzy Logic            | Handles uncertainty, flexible with data, handles diverse data types           | Non-optimal, difficult rule design and implementation                 |
| Neural Networks        | Handles complex, nonlinear systems, detects hidden patterns, robust to noise  | Non-optimal, complex design, retraining needed, scaling expensive     |
| Optimization Problem   | Optimal or near-optimal solutions, strong solver support                      | Scaling issues, often requires simplification, weak under uncertainty |

## Notes from the Paper
- Figure 1: Overview of techniques and optimization variants
- Table 3: Overview over constraints that can be introduced, optimality of the solution, and their computational cost  

---

## MILP (Mixed Integer Linear Programming)

### General
- Most Widespread approach today

- In basic MILP inputs are treated as **known and fixed**, so uncertainty is not introduced
- Quadratic problems are often **linearized**
- However, scaling issues remain which leads to trade-off: Model accuracy vs computational tractability
- Simplification and tightening of the formula is often necessary, so you need to model as detailed as necessary, but as simple and tight as possible

## Solvers
- **Gurobi** and **CPLEX**
- Advantages: often find optimal solutions, strong industry adoption
- Disadvantages: may struggle with large-scale or highly nonlinear problems

---

## Handling Uncertainty

Real power systems are uncertain due to renewable energy variability, varying energy demand etc.
To handle this uncertainty, you can use:

- Stochastic MILP: Uses probability distributions and models expected behavior and risk explicitly
- Robust MILP: No probabilities, instead optimizes against the worst-case scenario (finds safe solution under worst case)
