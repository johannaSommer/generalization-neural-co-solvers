import torch
import torch.nn as nn


class NeuroSAT(nn.Module):
    def __init__(self, dim=128, n_rounds=26, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.name = "NSAT"
        self.dim = dim
        self.n_rounds = n_rounds

        self.init_ts = torch.ones(1)

        self.L_init = nn.Linear(1, dim)
        self.C_init = nn.Linear(1, dim)

        self.L_msg = MLP(dim, dim, dim)
        self.C_msg = MLP(dim, dim, dim)

        self.L_update = nn.LSTM(dim*2, dim)
        self.C_update = nn.LSTM(dim, dim)

        self.L_vote = MLP(dim, dim, 1)
        self.denom = torch.sqrt(torch.Tensor([dim]))

    def forward(self, problem):
        self.last_state = None
        n_vars = problem['n_vars'].sum()
        n_clauses = problem['n_clauses'].sum()
        n_probs = len(problem['n_clauses'])
        adj = problem['adj']

        init_ts = self.init_ts.cuda()
        L_init = self.L_init(init_ts).view(1, 1, -1)
        L_init = L_init.repeat(1, n_vars*2, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)

        L_state = (L_init, torch.zeros(1, n_vars*2, self.dim).cuda())
        C_state = (C_init, torch.zeros(1, n_clauses, self.dim).cuda())

        for _ in range(self.n_rounds):
            L_hidden = L_state[0].squeeze(0)
            L_pre_msg = self.L_msg(L_hidden)
            LC_msg = torch.matmul(adj.T, L_pre_msg)

            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)

            C_hidden = C_state[0].squeeze(0)
            C_pre_msg = self.C_msg(C_hidden)
            CL_msg = torch.matmul(adj, C_pre_msg)

            _, L_state = self.L_update(torch.cat([CL_msg, flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0),
                                       L_state)

        logits = L_state[0].squeeze(0)
        self.last_state = logits
        vote = self.L_vote(logits)
        
        # reshape such that we have a vector of length 2 for every variable (literal & complement)
        vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1)
        # split tensor into votes for every problem, as they do not have the same dimensions
        vote_join = list(torch.split(vote_join, list(problem['n_vars'])))
        vote_mean = torch.stack([torch.mean(x) for x in vote_join]).to(adj.device)
        
        assert len(vote_join) == n_probs
        return vote_mean

    def reconstruct(self, repr, batch):
        nv = torch.cumsum(torch.cat((torch.zeros(1), batch['n_vars'])), dim=0).int()*2
        nc = torch.cumsum(torch.cat((torch.zeros(1), batch['n_clauses'])), dim=0).int()
        pos, neg = [], []
        for i in range(nv.size(0)-1):
            subm = repr[nv[i]:nv[i+1], nc[i]:nc[i+1]].clone()
            pos.append(subm.view(2, -1, subm.size(1))[0])
            neg.append(subm.view(2, -1, subm.size(1))[1])
        batch['adj'] = torch.cat([torch.block_diag(*pos), torch.block_diag(*neg)])
        return batch


    def get_representation(self, batch: dict):
        nv = torch.cumsum(torch.cat((torch.zeros(1), batch['n_vars'])), dim=0).int()
        nc = torch.cumsum(torch.cat((torch.zeros(1), batch['n_clauses'])), dim=0).int()
        adj = batch['adj'].view(2, -1, batch['adj'].size(1))
        chunks = []
        for i in range(nv.size(0)-1):
            subm = adj[:, nv[i]:nv[i+1], nc[i]:nc[i+1]]
            chunks.append(torch.cat((subm[0], subm[1])))
        repr = torch.block_diag(*chunks)
        assert batch['n_vars'].sum()*2 == repr.size(0)
        assert sum(batch['n_clauses']) == repr.size(1)
        block_check = torch.block_diag(*[torch.ones(x*2, y) for (x, y) in zip(batch['n_vars'], batch['n_clauses'])])
        assert torch.all(torch.eq(repr, repr*block_check.to(repr.device)))
        return repr


    def collate_fn(self, problems):
        """
        Collate fn for torch.DataLoader to parse problem attributes into one batch dict
        """
        # a for-loop is currently neccessary because pytorch does not support ragged tensors
        # we have a varying number of clauses & literals
        is_sat, n_clauses, n_vars, clauses = [], [], [], []
        single_adjs, adj_pos, adj_neg, solution = [], [], [], []
        fnames = []
        
        for p in problems:
            adj = p['adj_nsat'].to_dense()
            single_adjs.append(adj)
            # re-sort adjacency such that regular literals come first and then, in the same order, the negated vars
            adj_pos.append(adj[:int(adj.shape[0]/2), :])
            adj_neg.append(adj[int(adj.shape[0]/2):, :])
            
            try:
                solution.append(p['solution'])
                is_sat.append(float(p['label']))
            except:
                solution.append([-1])
                is_sat.append(float(-1))
            n_vars.append(p['n_vars'])
            n_clauses.append(p['n_clauses'])
            clauses.append(p['clauses'])
            try:
                fnames.append(p['filename'])
            except:
                pass

        # create disconnected graphs
        adj_pos = torch.block_diag(*adj_pos)
        adj_neg = torch.block_diag(*adj_neg)

        sample = {
            'batch_size': len(problems),
            'n_vars': torch.Tensor(n_vars).int(),
            'n_clauses': torch.Tensor(n_clauses).int(),
            'is_sat': torch.Tensor(is_sat).float().cuda(),
            'adj': torch.cat([adj_pos, adj_neg]).cuda(),
            'single_adjs': single_adjs,
            'clauses': clauses,
            'solution': solution,
            'fnames': fnames
        }
        return sample


def flip(msg, n_vars):
    return torch.cat([msg[n_vars:2*n_vars, :], msg[:n_vars, :]], dim=0)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.l1(x))
        x = self.r(self.l2(x))
        x = self.l3(x)
        return x
