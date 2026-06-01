## D. Bucher, J. Stein, S. Feld, C. Linnhoff-Popien, IF-QAOA: A Penalty-Free Approach to Accelerating Constrained Quantum Optimization, In Physical Review, American Physical Society, Maryland, 2025.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_proquest_journals_3303156428)  
[DOI](https://doi.org/10.1103/fb5m-cl9m)

## Overview

Traditionally, QUBO has no constraints but, as most real-world optimization problems involve constraints, these constraints are added as penalty terms to the objective function
- inequality constraints are handled by adding auxiliary binary variables (= slack variables ) that limit the solution space 

The paper uses a QAOA approach with an Indicator Function (“an oracle-based subroutine that evaluates constraint satisfaction in an additional register”)
- this way, you do not need penalties or additional variables
- in most tests, the proposed approach was faster than a QUBO implementation

