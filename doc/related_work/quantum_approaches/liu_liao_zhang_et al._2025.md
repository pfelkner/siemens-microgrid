## M. Liu, M. Liao, R. Zhang, et al., Quantum computing as a catalyst for microgrid management: Enhancing decentralized energy systems through innovative computational techniques. Sustainability, MDPI, Basel, page 3662, 2025.
[LMU Link](https://opac.ub.lmu.de/PrimoRecord/cdi_gale_onefilemisc_PPES_A838640518)  
[DOI](https://doi.org/10.3390/su17083662)

## Classic Approaches
Classical approaches rely heavily on  deterministic approaches, such as MILP and mixed-integer nonlinear programming (MINLP),  stochastic optimizations, metaheuristics such as the Genetic Algorithm, and deep learning/reinforcement learning
- MILP and MINLP do not scale well and also ignore uncertainty because to function the need precise forecasts
-  Stochastic optimization allow for uncertainty, but introduce a lot of  computational complexity (as the number of possible scenarios to considers drows exponentially with the problem size)
-  Metaheuristic approaches allow for a lot of flexibility, however they often fail to converge and need a lot of parameter tuning
- Reinforcement learning and deep-learning provide real-time scheduling, but the required training data and re-training issues remain

---

## Hybrid Approach
The paper presents a hybrid quantum–classical computational framework which formulates the unit commitment problem as a QUBO and then uses the Quantum Approximate Optimization Algorithm (QAOA)
- QAOA uses quantum superposition and entanglement to solve  problems with discrete variables
- the framework  is implemented on a D-Wave Advantage 5000 quantum annealer using 5000  qubits for encoding state and constraints
