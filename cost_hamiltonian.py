import numpy as np
import itertools

# Binary Variables
B_LOW  = 0 # low SoC band active
B_MID  = 1 # mid SoC band active
B_HIGH = 2 # high SoC band active
B_CH   = 3 # charge active
B_DIS  = 4 # discharge active
B_IMP  = 5 # import active
B_EXP  = 6 # export active
B_Y    = 7 # outage slot fully served

# other parameters
T = 2 # number of timeslots
num_discrete_vars_per_t = 8
total_qubits = T * num_discrete_vars_per_t

# Penalty assigned to qubit states violating physical constraints (Feasibility Cuts)
INFEASIBLE_PENALTY = 1e9

r_res = [225.0, 225.0] # resiliency bonus per timeslot

def get_slot_bits(bitstring, t):
    """Slices global bitstring to extract only bits relevant for each timeslot."""
    base = t * num_discrete_vars_per_t
    return bitstring[base : base + num_discrete_vars_per_t]

def calculate_direct_cost(bitstring):
    """ Calculates the base objective function values strictly dependent on z (the discrete variables)."""
    direct_cost = 0.0
    for t in range(T):
        bits = get_slot_bits(bitstring, t)
        direct_cost -= r_res[t] * bits[B_Y]
    return direct_cost

def evaluate_benders_cuts(bitstring, benders_cuts, initial_continuous_cost):
    """ Evaluates classical continuous surrogate model (q(z)) for a given discrete state."""
    if not benders_cuts:
        return initial_continuous_cost, True

    cut_values = []
    for cut_func in benders_cuts:
        val = cut_func(bitstring)
        if val is None:
            return INFEASIBLE_PENALTY, False
        cut_values.append(val)

    if not cut_values:
        return initial_continuous_cost, True
    return max(cut_values), True


def build_cost_hamiltonian(benders_cuts, initial_continuous_cost):
    """
    Builds the diagonal of the Cost Hamiltonian (H_C) for the QAOA."""
    dim = 2 ** total_qubits
    H_C_diag = np.zeros(dim)
    penalty_mask = np.zeros(dim, dtype=bool)

    for i, bitstring in enumerate(itertools.product([0, 1], repeat=total_qubits)):
        surrogate_cost, feasible = evaluate_benders_cuts(
            bitstring, benders_cuts, initial_continuous_cost
        )
        if not feasible:
            H_C_diag[i] = INFEASIBLE_PENALTY
            penalty_mask[i] = True
        else:
            H_C_diag[i] = calculate_direct_cost(bitstring) + surrogate_cost

    return H_C_diag, penalty_mask