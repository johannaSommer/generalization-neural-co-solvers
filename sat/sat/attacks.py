import math
import copy
import torch
import numpy as np
from sat.solvers.circuitsat import evaluate_circuit, custom_csat_loss


def attack_random(a, model, batch, delta=0.01, deltaadc=1/4, counter=100, **args):
    """
    Executes perturbations proposed in Proposition 1, random

    Keyword arguments:
    a -- name of the attack ["sat", "del", "adc"]
    model -- SAT model
    batch -- samples to be perturbed
    delta -- budget used by sat & del, delta*sum(adj) edges will be flipped max
    deltaadc -- clauses added by adc, deltaadc*n_clauses will be added
    counter -- how many times to sample pert that does not delete a clause during del,
               will throw ValueError if none is found
    """
    repr = model.get_representation(batch)
    nnodes = copy.deepcopy(batch['n_clauses'])

    if a == "sat":
        if model.name == "NSAT":
            misc_mask = torch.sigmoid(model(batch)) < 0.5
        else:
            misc_mask = torch.ones(len(batch['n_clauses'])).bool()
            misc_mask = misc_mask.to(batch['adj'].device())
    elif a == "del":
        misc_mask = torch.sigmoid(model(batch)) > 0.5
    else:
        misc_mask = torch.sigmoid(model(batch)) > 0.5
        
    no_delete = False
    adj_mask, _, budget, M = get_inits(a, batch, delta, deltaadc, repr, mask=misc_mask)
        
    M_before = M
    while not no_delete:
        # randomly choose indices, allow only perturbation within a sample
        M = copy.deepcopy(M_before)
        ind = torch.randperm(M.shape[0])
        if a == "del":
            repr_temp = repr * adj_mask
            ind = ind[repr_temp.view(-1)[ind].bool()]
        else:
            ind = ind[adj_mask.view(-1)[ind].bool()]
        ind = ind[:budget]

        # flip randomly selected edges in original problem
        M[ind] = 1
        M = torch.reshape(M, adj_mask.shape)
        no_delete = (a in ["sat", "adc"]) or torch.all(repr.sum(dim=0) - M.sum(dim=0) > 0)
        counter -= 1
        if counter == 0: raise ValueError()
        
    # ensure satisfiability
    if a == "sat":
        sat_mask = get_sat_mask(repr, batch) * adj_mask
        M = torch.logical_and(sat_mask.bool(), M).float()

    repr = apply(a, repr, M, nnodes, batch, rem_zerocl=True)
    batch = model.reconstruct(repr, batch)
    batch['adj'] = batch['adj'].detach()
    return batch


def attack_opt(a, model, batch, steps=500, lr=0.1, delta=0.05, numsamples=20, temp=5, deltaadc=1/4, return_pert=False, **args):
    """
    Executes perturbations proposed in Proposition 1, optimized

    Keyword arguments:
    a -- name of the attack ["sat", "del", "adc"]
    model -- SAT model
    batch -- samples to be perturbed
    steps -- number of attack optimization steps
    lr -- learning rate
    delta -- budget used by sat & del, delta*sum(adj) edges will be flipped max
    numsamples -- number of samples taken from continuous perturbation
    temp -- temperature parameter applied on model predictions (to avoid zero gradients)
    deltaadc -- clauses added by adc, deltaadc*n_clauses will be added
    """
    assert a in ["sat", "del", "adc"] 
    nnodes = copy.deepcopy(batch['n_clauses'].int())

    # budget per sample and perturbation matrix init
    repr = model.get_representation(batch)
    repr_before = copy.deepcopy(repr)
    adj_mask, iterator, budget, M = get_inits(a, batch, delta, deltaadc, repr)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none') if model.name == "NSAT" else csat_loss_wrapper

    # get edges that ensure satisfiability
    sat_mask = get_sat_mask(repr, batch)

    # inits for early stopping
    best_loss = torch.full(batch['is_sat'].shape, float('inf'))
    best_M = M.detach().cpu()
    optimizer = torch.optim.Adam([M], lr=lr)

    for _ in range(steps):
        # apply perturbation to input
        M.requires_grad = True
        repr = apply(a, repr_before, M, nnodes, batch)

        # reconstruct original structure and get predictions
        batch = model.reconstruct(repr, batch)
        outputs = model(batch)
        outputs = outputs / temp
        loss = -loss_fn(outputs, batch['is_sat'] if model.name == "NSAT" else batch)

        # early stopping
        best_loss, best_M = update_best_pert(loss, best_loss, best_M, M, iterator)

        # update perturbation
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projections
        M = apply_projections(a, M, repr_before, sat_mask, adj_mask, iterator, budget)
        batch['adj'] = batch['adj'].detach()
        
    # sample perturbation and apply
    with torch.no_grad():
        M = best_M.to(M.device)
        best_loss = torch.full_like(best_loss, float('inf'), device='cpu')
        best_M = torch.full_like(M, float('inf'), device='cpu')

        for i in range(numsamples):
            if (i == 0) and (a != "adc"):
                M_sampled = topk(M, batch['n_clauses'], budget)
            else:
                M_sampled = torch.bernoulli(M.detach())

            # a clause CAN NOT be deleted, sample until constraint is fulfilled
            if a == "del": M_sampled = sample_delete(M, M_sampled, repr_before)
            
            # keep only best sample
            repr = apply(a, repr_before, M_sampled, nnodes, batch, rem_zerocl=True)
            batch = model.reconstruct(repr, batch)
            outputs = model(batch)
            loss = -loss_fn(outputs, batch['is_sat'] if model.name == "NSAT" else batch)
            best_loss, best_M = update_best_pert(loss, best_loss, best_M, M_sampled, batch['n_clauses'])
        
    # apply best perturbation and prepare output
    best_M = best_M.to(repr.device)
    repr = apply(a, repr_before, best_M, nnodes, batch, rem_zerocl=True)
    batch = model.reconstruct(repr, batch)

    assert torch.all(torch.eq(best_M, best_M * adj_mask))
    blocks = torch.block_diag(*[torch.ones(x, y) for (x, y) in zip(batch['n_vars'], batch['n_clauses'])])
    
    if model.name == "NSAT":
        adj_mask = torch.cat([blocks, blocks]).to(batch['adj'].device)
        assert torch.all(torch.eq(batch['adj'], batch['adj'] * adj_mask))   
        assert torch.all(batch['adj'] >= 0)
        assert torch.all(batch['adj'] <= 1)
        assert torch.all(torch.eq(batch['adj'], batch['adj'].int()))
        assert len(torch.unique(batch['adj'])) == 2

    if return_pert:
        return batch, best_M
    else:
        return batch


def csat_loss_wrapper(outputs, batch, k=20):
    outputs = evaluate_circuit(batch, torch.sigmoid(outputs), k)
    loss = custom_csat_loss(outputs, mean=False)
    return loss


######################
# PROJECTIONS
######################

def project_no_clause_delete(M, adj_before):
    """
    Projection such that the expected value of deletions in a clause is not 
    larger than the number of variables in a clause
    """
    # ensure that no clause is completely deleted
    mask = M.sum(dim=0) > (adj_before.sum(dim=0) - 1)
    for k, indicator in enumerate(mask):
        if indicator:
            diff = M[:, k].sum() - (adj_before[:, k].sum() - 1)
            nonzeros = (M[:, k] > 0).int()
            nonzeros = nonzeros * (diff / nonzeros.sum())
            M[:, k] = M[:, k] - nonzeros
            M[:, k] = torch.clamp(M[:, k], 0, 1)
    return M


def project_edge_budget(M, sections, edge_budget):
    """
    Projection on the expected number of flips 
    """
    prev = 0
    sections = torch.cumsum(sections, dim=0)
    for _, n in enumerate(sections):
        sec_counter = 0
        while M[:, prev:n].sum() > edge_budget:
            diff = M[:, prev:n].sum() - edge_budget
            nonzeros = (M[:, prev:n] > 0).int()
            nonzeros = nonzeros * (diff / nonzeros.sum())
            M[:, prev:n] = M[:, prev:n] - nonzeros
            M[:, prev:n] = torch.clamp(M[:, prev:n], 0, 1)
            sec_counter += 1
            if sec_counter > 100:
                raise ValueError()
        assert M[:, prev:n].sum() <= edge_budget
        prev = n
    return M


def get_sat_mask(repr, batch):
    """
    Returns indices of the graph edges that make the problem satisfiable,
    only one edge per clause
    """
    if any([s is None for s in batch['solution']]):
        return None
    vc = 0
    all_idx = torch.Tensor([]).to(repr.device)
    sat_mask = torch.ones(repr.shape).to(repr.device)
    for i in range(len(batch['n_clauses'])):
        indices = get_solution_indices(batch['clauses'][i], batch['solution'][i], batch['n_vars'][i].item())
        assert len(indices) == batch['n_clauses'][i]
        assert torch.all(torch.eq(torch.sort(indices[:, 1])[0], indices[:, 1]))
        all_idx = torch.cat((all_idx, indices[:, 0].to(repr.device) + vc))
        vc += batch['n_vars'][i]*2
    all_idx = all_idx.unsqueeze(0)
    sat_mask = sat_mask.scatter_(0, all_idx.long(), torch.zeros(all_idx.shape).cuda()) 
    assert torch.all((~sat_mask.bool()).int().sum(0) == 1)
    return sat_mask.to(repr.device)


######################
# UTILS
######################

def get_solution_indices(clauses, solution, numvars):
    """
    Parse edge indices from raw solution
    """
    indices = []
    solution = np.array(solution)
    for i, c in enumerate(clauses):
        res_vars = np.intersect1d(np.array(c), solution)[0]
        if res_vars < 0: res_vars = abs(res_vars) + numvars
        res_vars -= 1
        indices.append([res_vars, i])
    return torch.Tensor(indices)


def apply(a, repr_before, M, nnodes, batch, rem_zerocl=False):
    """
    Apply perturbation by either XOR or appending clauses
    """
    if a == "sat":
        repr = repr_before + torch.where(repr_before.bool(), -M, M)
    elif a == "del":
        repr = repr_before - M
    else:
        M_chunk = list(torch.chunk(M, len(nnodes), dim=1))
        if rem_zerocl:
            batch['n_clauses'] = torch.Tensor([n + (M_chunk[i].sum(0) != 0).sum() for i, n in enumerate(nnodes)]).int()
        cc = 0
        repr = copy.deepcopy(repr_before)
        for k in range(len(nnodes)):

            if rem_zerocl:
                M_chunk[k] = M_chunk[k][:, M_chunk[k].sum(0) > 0]
            if M_chunk[k].size(1) > 0:
                repr = torch.cat([repr[:, :(cc + nnodes[k])], M_chunk[k], repr[:, (cc + nnodes[k]):]], dim=1)
            cc += nnodes[k] + M_chunk[k].size(1)
    return repr


def topk(M, nc, budget):
    """
    Get top-k edge flips
    """
    M_sampled = torch.zeros_like(M)
    prev = 0
    for _, n in enumerate(torch.cumsum(nc, dim=0)):
        indices = np.unravel_index(
            torch.topk(M[:, prev:n].flatten(), 
            min((M[:, prev:n] > 0).sum().item(), budget)).indices.cpu().numpy(), M[:, prev:n].shape)
        M_sampled[indices[0], prev+indices[1]] = 1
        prev = n
    return M_sampled


def update_best_pert(loss, best_loss, best_M, M, iterator):
    """
    Early stopping functionality, keeps track of best loss and perturbation
    """
    with torch.no_grad():
        improved_mask = best_loss > loss.cpu()
        if improved_mask.any():
            best_loss[improved_mask] = loss[improved_mask].cpu()
            prev = 0
            for k, n in enumerate(np.cumsum(iterator)):
                if improved_mask[k]:
                    best_M[..., prev:n] = M[..., prev:n].cpu()
                prev = n
        return best_loss, best_M


def apply_projections(a, M, repr_before, sat_mask, adj_mask, sections, budget):
    """
    Wrapper to apply all projections during optimized attack
    """
    with torch.no_grad():
        M.data = torch.clamp(M, 0, 1).detach()
        # enforce block diag structure also in M matrix
        M.data = M * adj_mask
        if a == "sat":
            # enforce satisfiability
            M.data = sat_mask * M
        elif a == "del":
            # enforce only delete
            M.data = M * repr_before
        # enforce edge budget
        M = project_edge_budget(M, sections, budget)
        if a == "del":
            # enforce no complete clause deletions
            M = project_no_clause_delete(M, repr_before)
        assert torch.all(torch.eq(M, M*adj_mask))
    return M


def sample_delete(M, M_sampled, repr_before):
    """
    Sample until no-clause-delete constraint is fulfilled
    """
    flag = False
    while not flag:
        mask = (M_sampled.sum(dim=0) <= (repr_before.sum(dim=0)-1))
        M_sampled[:, ~mask] = torch.bernoulli(M)[:, ~mask]
        if torch.all(M_sampled.sum(dim=0) <= (repr_before.sum(dim=0)-1)):
            flag = True 
    assert torch.all(M_sampled.sum(dim=0) <= (repr_before.sum(dim=0)-1))
    M_sampled = M_sampled.detach() 
    return M_sampled


def get_inits(a, batch, delta, deltaadc, repr, mask=None):
    """
    Returns initializations: perturbation matrix, attack budget and adjacency mask
    """
    batch_size = len(batch['is_sat'])
    if a == "adc":
        added = int(deltaadc * max([a.shape[1] for a in batch['single_adjs']]))
        iterator = torch.Tensor([added for _ in range(batch_size)]).int()
        for i in range(batch_size):
            batch['n_clauses'][i] += added
        budget = math.ceil(batch['adj'].sum(dim=0).mean().item())
        budget = budget * added
        M = torch.zeros(batch['adj'].shape[0], added * batch_size).to(batch['adj'].device)
        if mask is not None:
            budget = (budget * (~mask).sum()).int()
            M = M.flatten()
        adj_mask = torch.block_diag(*[torch.ones(x*2, added) for x in batch['n_vars']])
    else:
        if mask is None:
            budget = (repr.sum() * delta / batch_size).int().cuda()
        else:
            budget = torch.ceil(repr.sum() * delta * ((~mask).sum()/mask.size(0))).int()
        M = torch.zeros(repr.shape).to(repr.device)
        adj_mask = [torch.ones(x*2, y) for (x, y) in zip(batch['n_vars'], batch['n_clauses'].int().tolist())]
        if mask is not None: 
            adj_mask = [torch.zeros_like(t) if mask[i] else t for (i, t) in enumerate(adj_mask)]
            M = M.flatten()
        adj_mask = torch.block_diag(*adj_mask)
        iterator = copy.deepcopy(batch['n_clauses'])
    adj_mask = adj_mask.to(repr.device).int()
    return adj_mask, iterator, budget, M
