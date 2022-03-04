import torch
import random
import numpy as np


def check_constraints(adv_coo, coordinates, x, y, subspaces=None, return_difference=False):
    n_points = None
    if isinstance(x, list):
        n_points = [len(x_i) for x_i in x]
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

    detour = (torch.norm(x - adv_coo, dim=-1) + torch.norm(y - adv_coo, dim=-1) - torch.norm(x - y, dim=-1))
    
    counter, ap = 0, []
    for i, c in enumerate(coordinates):
        if subspaces is None:
            all_pairs = torch.tril_indices(len(c), len(c), offset=-1)
        else:
            triu = torch.triu(torch.ones(len(c), len(c)), diagonal=1)

            # Remove X and Y (captured by `detour` variable)
            if subspaces is not None:
                ss_exclude_idx = torch.tensor(subspaces[i][2]).T.sort(dim=0).values
                triu[ss_exclude_idx[0], ss_exclude_idx[1]] = 0

            all_pairs = triu.nonzero().T

        ap.append(all_pairs + counter)
        counter += len(c)
        
    lengths = [len(a[0]) for a in ap]
    if n_points is not None:
        lengths = [l for l, n in zip(lengths, n_points) for _ in range(n)]
        ap = [a for a, n in zip(ap, n_points) for _ in range(n)]
    ap = torch.cat(ap, dim=1)  
    coordinates = torch.cat([torch.Tensor(c) for c in coordinates])
    adv_coo_rep = torch.repeat_interleave(adv_coo, torch.Tensor(lengths).long(), dim=0)
    
    all_detours = (
      torch.norm(coordinates[ap[0]] - adv_coo_rep, dim=1)
      + torch.norm(coordinates[ap[1]] - adv_coo_rep, dim=1)
      - torch.norm(coordinates[ap[0]] - coordinates[ap[1]], dim=1)
    )
    
    all_detours = torch.split(all_detours, lengths)
    constraint_violated = [(all_detours[i] <= detour[i]).any().item() for i in range(len(adv_coo))]

    if n_points is not None:
        chained_range = [i for n in n_points for i in range(n)]
        extra_indices = [j - torch.tril_indices(n+1, n+1, offset=-1) for j, n in enumerate(chained_range)]
        lengths = [idx.shape[1] for idx in extra_indices]
        adv_coo_rep_2 = torch.repeat_interleave(adv_coo, torch.Tensor(lengths).long(), dim=0)
        extra_indices = torch.cat(extra_indices, dim=1)

        extra_detours = (
            torch.norm(x[extra_indices[0]] - adv_coo_rep_2, dim=1)
            + torch.norm(y[extra_indices[1]] - adv_coo_rep_2, dim=1)
            - torch.norm(x[extra_indices[0]] - y[extra_indices[1]], dim=1)
        )
        extra_detours = torch.split(extra_detours, lengths)

        extra_constraint_violated = [(extra_detours[i] <= detour[i]).any().item() for i in range(len(adv_coo))]

        constraint_violated = [c1 or c2 for c1, c2 in zip(constraint_violated, extra_constraint_violated)]

    is_satisfied = [not cv for cv in constraint_violated]
    if not return_difference:
        return is_satisfied
    else:
        if n_points is None or len(extra_detours[i]) == 0:
            return is_satisfied, [detour[i] - all_detours[i].min() for i in range(len(adv_coo))]
        else:
            return is_satisfied, [
                (detour[i] - torch.min(all_detours[i].min(), extra_detours[i].min()) 
                 if len(extra_detours[i])
                 else detour[i] - all_detours[i].min())
                for i in range(len(adv_coo))
            ]
        

def add_to_route(route, pairs):
    found_count = 0
    if not isinstance(pairs[0], list):
        pairs = [pairs]
    route = route[:]
    for pair in pairs:
        for k, elem in enumerate(route):
            if elem == pair[0] and route[k-1] == pair[1]:
                route = route[:k] + [(max(route)+1)] + route[k:]
                found_count += 1
            elif elem == pair[1] and route[k-1] == pair[0]:
                route = route[:k] + [(max(route)+1)] + route[k:]
                found_count += 1
    assert found_count == len(pairs)
    return route


def find_subspace(points, routes, n_subspaces=1):
    possible_solutions = []
    for r in range(0, len(routes)):
        possible_solutions.append([routes[r-1], routes[r]])
    possible_solutions = np.array(possible_solutions)
    solset = possible_solutions.tolist()
    assert len(solset) > 0
    random.shuffle(solset)
    solset = solset[0: min(n_subspaces, len(solset))]
    point_1 = [points[s[0]] for s in solset]
    point_2 = [points[s[1]] for s in solset]
    return point_1, point_2, solset


def scale_coordinates(coos, sqrt=False):
    coos_resh = torch.flatten(coos, start_dim=1, end_dim=2)
    minimum, _ = torch.min(coos_resh, dim=-1)
    maximum, _ = torch.max(coos_resh, dim=-1)
    coos = (coos - minimum.view(-1, 1, 1)) / (maximum.view(-1, 1, 1) - minimum.view(-1, 1, 1))
    assert torch.all(coos <= 1)
    assert torch.all(coos >= 0)
    if sqrt:
        coos = coos * torch.sqrt(torch.Tensor([2]))/2
        assert torch.all(coos <= (torch.sqrt(torch.Tensor([2]))/2)[0].item())
        assert torch.all(coos >= 0)
    return coos