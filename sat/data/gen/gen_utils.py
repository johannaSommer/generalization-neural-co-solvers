import random
import torch
import numpy as np
from pysat.solvers import Minisat22


def clause_to_nsat_adj(iclauses, n_vars):
    n_cells = sum([len(iclause) for iclause in iclauses])
    # construct adjacency for neurosat model
    nsat_indices = np.zeros([n_cells, 2], dtype=np.int64)
    cell = 0
    for clause_idx, iclause in enumerate(iclauses):
        vlits = [ilit_to_vlit(x, n_vars) for x in iclause]
        for vlit in vlits:
            nsat_indices[cell, :] = [vlit, clause_idx]
            cell += 1
    assert(cell == n_cells)
    adj_nsat = torch.sparse.FloatTensor(torch.Tensor(nsat_indices).T.long(), torch.ones(n_cells), 
                                       torch.Size([n_vars*2, len(iclauses)]))
    return adj_nsat

def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

def solve_sat(n_vars, iclauses):
    solver = Minisat22()
    for iclause in iclauses: solver.add_clause(iclause)
    return solver.solve(), solver.get_model()

def gen_iclause_pair(n, p_k_2, p_geo):
    solver = Minisat22()
    iclauses = []
    while True:
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        iclause = generate_k_iclause(n, k)
        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)
        else:
            break
    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat
    