## A. Ajagekar, F. You, Quantum computing for energy systems optimization: Challenges and opportunities. Energy, pages 76–89, Elsevier Science, Amsterdam, 2019.

[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_proquest_journals_2370257533)  
[DOI](https://doi.org/10.1016/j.energy.2019.04.186)

## Overview
The paper explains the basics of quantum computing and the difference between the gate model an annealing
- superposition as the ability of a qubit to exist in an infinite number of states and  entanglement as co-relation between “individually random behaviors of two qubits”
- With the gate model, qubits are influenced by interaction with the environment which can lead to decoherence
- With annealing, number of qubits is higher than in circuit models

---

## Annealing Approach

With annealing, only problems that can be mapped to a QUBO or an Ising model can be solved
- For unit commitment this means discretizing the problem space (which is characterized by continuous variables) into equally spaced intervals
- The problem is that the more intervals you need, the more extra qubits are introduced and the more likely that the solution quality deteriorates