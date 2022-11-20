import torch
from torch_sparse import SparseTensor
# from pysat.solvers import Glucose3


def check_block_adj(batch):
    components = []
    for i, nc in enumerate(batch['n_clauses']):
        components.append(torch.ones((batch['n_vars'][i], nc)))
    components = torch.block_diag(*components)
    bdiag = torch.cat([components, components]).to(batch['adj'].device)
    return (torch.all(torch.eq(bdiag * batch['adj'], batch['adj'])))


def adj_to_clause(batch):
    problems = []
    adj = batch['adj']
    cut = int(adj.shape[0]/2)

    counter_vars, counter_clauses = 0, 0

    for i in range(len(batch['is_sat'])):
        subm = adj[counter_vars : (counter_vars+batch['n_vars'][i]), 
                    counter_clauses : (counter_clauses + batch['n_clauses'][i])].T
        subm2 = adj[counter_vars+cut : (cut+counter_vars+batch['n_vars'][i]), 
                    counter_clauses : (counter_clauses + batch['n_clauses'][i])].T
        subm = torch.cat((subm, subm2), dim=1)
        clauses = []
        for j in range(batch['n_clauses'][i]):
            new_subm = torch.reshape(subm[j], (2,-1))
            pos_ind = torch.flatten(torch.nonzero(new_subm[0])+1)
            neg_ind = -torch.flatten(torch.nonzero(new_subm[1])+1)
            assert len(torch.cat((pos_ind, neg_ind))) > 0
            clauses.append(torch.cat((pos_ind, neg_ind)).tolist())
        problems.append(clauses)
        
        counter_vars += batch['n_vars'][i]
        counter_clauses += batch['n_clauses'][i]

    assert adj.shape[0] == counter_vars*2
    assert adj.shape[1] == counter_clauses
    return problems


def solve_sat(problems):
    solutions = []
    for problem in problems:
        solver = Glucose3(with_proof=True)
        for clause in problem:
            solver.add_clause(clause)
        solutions.append(int(solver.solve()))
    solver.delete()
    return solutions


def get_nsat_blocks(b):
    cvars, cclauses = 0, 0
    cut = int(b['adj'].shape[0]/2)
    adj = b['adj']
    pos, neg = [], []
    for i in range(len(b['is_sat'])):
        sub_pos = adj[cvars : (cvars+b['n_vars'][i]), 
                                    cclauses : (cclauses + b['n_clauses'][i])]
        pos.append(sub_pos)
        sub_neg = adj[cvars+cut : (cut+cvars+b['n_vars'][i]), 
                                    cclauses : (cclauses + b['n_clauses'][i])]
        neg.append(sub_neg)
        cvars += b['n_vars'][i]
        cclauses += b['n_clauses'][i]
    return pos, neg


def sparse_blockdiag(single_adjs, square=True):
    single_adjs = [(sa.coo(), sa.size(0), sa.size(1)) for sa in single_adjs]
    values = torch.cat([sa[0][2] for sa in single_adjs])
    indices = []
    counter = 0
    sec_counter = 0
    for sa in single_adjs:
        if square:
            indices.append(torch.stack((sa[0][0], sa[0][1])) + counter)
        else:
            indices.append(torch.stack((sa[0][0] + counter, sa[0][1] + sec_counter)))
        counter += sa[1]
        sec_counter += sa[2]
    indices = torch.cat(indices, dim=1)
    if square: assert counter == sec_counter
    adj = torch.sparse_coo_tensor(indices, values, [counter, sec_counter])
    adj = SparseTensor.from_torch_sparse_coo_tensor(adj)
    return adj

