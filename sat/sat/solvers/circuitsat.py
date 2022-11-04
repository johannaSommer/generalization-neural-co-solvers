import torch
import warnings
import torch_sparse
import torch.nn as nn
from torch_sparse import SparseTensor


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.l1(x))
        x = self.l2(x)
        return x

class CircuitSAT(nn.Module):
    def __init__(self, dim=100, dim_agg=50, dim_class=30, n_rounds=20, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.name = "CSAT"
        self.dim = dim
        self.n_rounds = n_rounds
        
        self.init = nn.Linear(4, dim)
        self.forward_msg = MLP(dim, dim_agg, dim)
        self.backward_msg = MLP(dim, dim_agg, dim)
        self.forward_update = nn.GRU(dim, dim)
        self.backward_update = nn.GRU(dim, dim)

        self.classif = MLP(dim, dim_class, 1)

    def forward(self, sample):
        self.forward_update.flatten_parameters()
        self.backward_update.flatten_parameters()
        adj = sample['adj']
        h_state = self.init(sample['features'].cuda()).unsqueeze(0)

        for _ in range(self.n_rounds):
            f_pre_msg = self.forward_msg(h_state.squeeze(0))
            f_msg = torch_sparse.matmul(adj, f_pre_msg)

            _, h_state = self.forward_update(f_msg.unsqueeze(0), h_state)

            b_pre_msg = self.backward_msg(h_state.squeeze(0))
            b_msg = torch_sparse.matmul(adj.t(), b_pre_msg)

            _, h_state = self.backward_update(b_msg.unsqueeze(0), h_state)
            
        return self.classif(h_state.squeeze(0))



def evaluate_circuit(sample, emb, epoch, eps=1.2, hard=False):
    # explore exploit with annealing rate
    t = epoch ** (-eps)
    inds = torch.cat(sample['indicator'], dim=1).view(-1)

    # set to negative to make sure we dont accidentally use nn preds for or/and
    temporary = emb.clone()
    temporary[inds == 1] = -1
    temporary[inds == -1] = -1

    # NOT gate
    temporary[sample['features'][:, 1] == 1] = 1 - emb[sample['features'][:, 0] == 1].clone()
    emb = temporary.clone()

    # OR gate
    idx = torch.arange(inds.size(0))[inds==1]
    or_gates = torch_sparse.index_select(sample['adj'], 1, idx.to(emb.device))
    e_gated = torch_sparse.mul(or_gates, emb)
    row, col, vals = e_gated.coo()
    assert torch.all(vals >= 0)

    if hard:
        e_gated = e_gated.max(dim=0)
    else:
        eps = e_gated + (-e_gated.max(dim=0).unsqueeze(0))
        _, _, vals = eps.coo()
        e_temp = torch.exp(vals / t).clone()
        # no elementwise multiplication in torch_sparse
        e_temp = SparseTensor(row=row, rowptr=None, col=col, value=e_temp, sparse_sizes=e_gated.sizes())
        e_gated = sparse_elem_mul(e_gated, e_temp).sum(dim=0) / e_temp.sum(dim=0)
        assert torch.all(torch.eq(e_gated, e_gated))

    # AND gate
    idx2 = torch.arange(inds.size(0))[inds==-1]
    and_gates = torch_sparse.index_select(sample['adj'], 0, idx.to(emb.device))
    and_gates = torch_sparse.index_select(and_gates, 1, idx2.to(emb.device))
    e_gated = torch_sparse.mul(and_gates, e_gated.unsqueeze(1))
    row, col, vals = e_gated.coo()
    assert torch.all(vals >= 0)
    
    if hard:
        e_gated = e_gated.min(dim=0)
    else:
        eps = e_gated + (-e_gated.min(dim=0).unsqueeze(0))
        _, _, vals = eps.coo()
        e_temp = torch.exp(-vals / t).clone()
        e_temp = SparseTensor(row=row, rowptr=None, col=col, value=e_temp, sparse_sizes=e_gated.sizes())
        e_gated = sparse_elem_mul(e_gated, e_temp).sum(dim=0) / e_temp.sum(dim=0)
        assert torch.all(torch.eq(e_gated, e_gated))

    assert torch.all(e_gated >= 0)
    assert len(e_gated) == len(sample['n_vars'])
    return e_gated


def sparse_elem_mul(s1, s2):
    # until torch_sparse support elementwise multiplication, we have to do this
    s1 = s1.to_torch_sparse_coo_tensor()
    s2 = s2.to_torch_sparse_coo_tensor()
    return SparseTensor.from_torch_sparse_coo_tensor(s1 * s2)
