import os
import random
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from sat.data import SATDataset
from torch.utils.data.dataloader import DataLoader
from sat.attacks import attack_opt, attack_random
from sklearn.metrics import accuracy_score
from itertools import compress
import pickle as pkl

from sat.solvers.neurosat import NeuroSAT
from sat.data import get_SAT_training_data
import argparse


def train_nsat(model, train, val, dataset_name, epochs=60, batch_size=32, lr=0.00002, 
               weight_decay=1e-10, augment="none", grad_clip=0.65, aug_rate=0.2,
               model_path='/trained_models/', adv_path=""):
    """
    Executes training and validation on the NeuroSAT model
    Keyword arguments:
    model -- SAT model
    train, val, test -- SAT data
    name -- dataset name, e.g. SAT-10-40
    epochs, batch_size, lr, weight_decay, grad_clip -- training parameters
    augment -- clean training ("none") or adversarial training ("adv")
    aug_rate -- percentage of training samples to be adversarially perturbed
    path -- where to save best trained model
    """

    if isinstance(train, SATDataset) and isinstance(val, SATDataset):
        dl_train = DataLoader(dataset=train, collate_fn=model.collate_fn,
                              shuffle=True, batch_size=batch_size)
    else:
        raise ValueError('Data is not provided as SAT Dataset.')

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.cuda()

    best_val = 0.0
    val_accs, train_losses = [], []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    simulation_name = model.name + '_' + dataset_name + '_' + datetime.now().strftime("%d:%m-%H:%M:%S.%f")
    print("Experiment identifier: ", simulation_name)

    for epoch in range(0, epochs):
        loss_temp = []
        model.train()

        if augment == "adv": update_adv_ex(simulation_name, adv_path)

        for _, batch in enumerate(dl_train):
            aug_now = (random.random() <= aug_rate)
            if augment == "adv" and aug_now:
                new_batch = perturb_batch(batch, model, optim, adv_path+simulation_name)
            elif augment == "adv" and (epoch > 0):
                new_batch = load_pert(batch, adv_path+simulation_name, model)
            else:
                new_batch = batch
            optim.zero_grad()
            outputs = model(new_batch)
            loss = loss_fn(outputs, new_batch['is_sat'])
            loss_temp.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
            optim.step()
            
        train_losses.append(np.mean(np.array(loss_temp)).item())
        print("Train Epoch Loss", epoch, np.mean(np.array(loss_temp)).item(), flush=True)
        val_acc = eval_nsat(model, val, batch_size=batch_size)
        print("Validation Accuracy", val_acc, flush=True)
        val_accs.append(val_acc)
        
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), model_path + simulation_name + '_MAX.pt')
        if epoch == 0: aug_rate *= 0.5

    return_dict = dict()
    return_dict['train_loss'] = train_losses
    return_dict['val_accs'] = val_accs
    return_dict['Name'] = simulation_name
    return return_dict


def eval_nsat(model, val, perturb="clean", batch_size=32, **attackargs):
    """
    Executes training and validation on the NeuroSAT model
    Keyword arguments:
    model -- SAT model
    val -- SAT validation data
    perturb -- perturbation applied on val samples
    batch_size -- batch_size
    attackargs -- attack parameters, e.g. number of steps or budget
    """

    if isinstance(val, SATDataset):
        dl_val = DataLoader(dataset=val, collate_fn=model.collate_fn,
                            shuffle=False, batch_size=batch_size)
    else:
        raise ValueError('Data is not provided as SAT Dataset.')

    sigmoid = nn.Sigmoid()
    ts, ps, outs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    for _, batch in enumerate(dl_val):

        if perturb == "clean":
            sample_new = batch

        elif perturb == "random-sat":
            assert torch.all(batch['is_sat'].bool())
            sample_new = attack_random("sat", model, batch, **attackargs)

        elif perturb == "random-del":
            assert not torch.any(batch['is_sat'].bool())
            sample_new = attack_random("del", model, batch, **attackargs)

        elif perturb == "random-adc":
            assert not torch.any(batch['is_sat'].bool())
            sample_new = attack_random("adc", model, batch, **attackargs)

        elif perturb == "optimized-sat":
            assert torch.all(batch['is_sat'].bool())
            sample_new = attack_opt("sat", model, batch, **attackargs) 

        elif perturb == "optimized-del":
            assert not torch.any(batch['is_sat'].bool())
            sample_new = attack_opt("del", model, batch, **attackargs) 
            
        elif perturb == "optimized-adc":
            assert not torch.any(batch['is_sat'].bool())
            sample_new = attack_opt("adc", model, batch, **attackargs) 

        else:
            raise ValueError("this kind of perturbation has not been implemented")

        outputs = model(sample_new)
        outputs = sigmoid(outputs)
        target = sample_new['is_sat'].cpu()
        ts = torch.cat((ts, target.cpu()))
        outs = torch.cat((outs, outputs.flatten().detach().cpu()))
        preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
        ps = torch.cat((ps, preds.flatten().cpu()))

    val_acc = accuracy_score(ts.numpy(), ps.numpy())
    return val_acc.item()


def update_adv_ex(simname, path):
    if not os.path.exists(path + simname):
        print("Making directory to save adv. examples: ", path + simname)
        os.mkdir(path + simname)
    else:
        for file in os.listdir(path + simname):
            if random.random() > 0.5:
                os.remove(path+simname + "/" + file)


def perturb_batch(batch, model, optim, folder, delta_sat=0.01, steps=10):
    shape_before = batch['adj'].shape
    sum_before = batch['is_sat'].sum()
    bt, bf = split_batch(batch)
    optim.zero_grad()
    if bt['adj'] is not None:
        bt, M = attack_opt("sat", model, bt, return_pert=True, delta=delta_sat, steps=steps)
        save_pert(M, bt, "sat", folder)
    optim.zero_grad()
    r = random.random()
    if bf['adj'] is not None:
        if r < 0.5:
            bf, M = attack_opt("del", model, bf, return_pert=True, steps=steps) 
            save_pert(M, bf, "del", folder)
        else:
            bf, M = attack_opt("adc", model, bf, return_pert=True, steps=steps)
            save_pert(M, bf, "adc", folder)

    if bf['adj'] is None:
        new_batch = bt
    elif bt['adj'] is None:
        new_batch = bf
    else:
        new_batch = combine_batch_nsat([bt, bf], same_label=False, shuffle=True)
    
    assert sum_before == new_batch['is_sat'].sum()
    if r < 0.5:
        assert new_batch['adj'].shape == shape_before
    else:
        assert shape_before[0] == new_batch['adj'].shape[0]
        assert shape_before[1] < new_batch['adj'].shape[1]
    return new_batch


def split_batch(b):
    b_sat = {
        'is_sat': b['is_sat'][b['is_sat'].bool()],
        'n_clauses': b['n_clauses'][b['is_sat'].bool()],
        'n_vars': b['n_vars'][b['is_sat'].bool()],
        'single_adjs': list(compress(b['single_adjs'], b['is_sat'].bool())),
        'clauses': list(compress(b['clauses'], b['is_sat'].bool())),
        'fnames': list(compress(b['fnames'], b['is_sat'].bool())),
        'solution': list(compress(b['solution'], b['is_sat'].bool()))
    }
    if len(b_sat['single_adjs']) > 0:
        pos = torch.block_diag(*[s.view(2, -1, s.shape[-1])[0] for s in b_sat['single_adjs']])
        neg = torch.block_diag(*[s.view(2, -1, s.shape[-1])[1] for s in b_sat['single_adjs']])
        b_sat['adj'] = torch.cat([pos, neg]).cuda()
    else: 
        b_sat['adj'] = None
    
    b_unsat = {
        'is_sat': b['is_sat'][~b['is_sat'].bool()],
        'n_clauses': b['n_clauses'][~b['is_sat'].bool()],
        'n_vars': b['n_vars'][~b['is_sat'].bool()],
        'fnames': list(compress(b['fnames'], ~b['is_sat'].bool())),
        'single_adjs': list(compress(b['single_adjs'], ~b['is_sat'].bool())),
        'solution': list(compress(b['solution'], ~b['is_sat'].bool()))
    }
    if len(b_unsat['single_adjs']) > 0:
        pos = torch.block_diag(*[s.view(2, -1, s.shape[-1])[0] for s in b_unsat['single_adjs']])
        neg = torch.block_diag(*[s.view(2, -1, s.shape[-1])[1] for s in b_unsat['single_adjs']])
        b_unsat['adj'] = torch.cat([pos, neg]).cuda()
    else:
        b_unsat['adj'] = None
    return b_sat, b_unsat


def load_pert(batch, folder, model):
    repr = model.get_representation(batch)
    new_repr = []
    cc, cv = 0, 0
    for i, fname in enumerate(batch['fnames']):
        subm = repr[cv:cv+batch['n_vars'][i]*2, cc:cc+batch['n_clauses'][i]]
        cc += batch['n_clauses'][i]
        cv += batch['n_vars'][i]*2
        if os.path.isfile(folder+"/"+fname):
            with open(folder+"/"+fname, 'rb') as f:
                obj = pkl.load(f)
            m = torch.from_numpy(obj['M'])
            if obj['attack'] in ["sat", "del"]:
                subm = torch.logical_xor(subm, m.cuda()).float()
            elif obj['attack'] == "adc":
                subm = torch.cat((subm, m.cuda()), dim=1)
                batch['n_clauses'][i] += obj['M'].shape[1]
            else:
                raise NotImplementedError()
        
        new_repr.append(subm)

    new_repr = torch.block_diag(*new_repr)
    batch = model.reconstruct(new_repr, batch)
    return batch


def save_pert(M, b, pert, folder):
    if pert in ['sat', 'del']:
        cvars, cclauses = 0, 0
        singles = []
        for i in range(len(b['is_sat'])):
            # get perturbation for individual sample
            M_single = M[cvars : (cvars+b['n_vars'][i]*2), 
                                      cclauses : (cclauses + b['n_clauses'][i])]
            singles.append(M_single)

            # save with sample file name
            obj = {
                'M': M_single.cpu().numpy(),
                'attack': pert
            }
            with open(folder+"/"+b['fnames'][i], 'wb') as f:
                pkl.dump(obj, f)

            cvars += b['n_vars'][i]*2
            cclauses += b['n_clauses'][i]
        assert torch.cat([torch.flatten(p) for p in singles]).sum() == M.sum()
        
    elif pert == "adc":
        cvars = 0
        M = torch.chunk(M, len(b['n_clauses']), dim=1)
        ms = []
        for i, (nv, m) in enumerate(zip(b['n_vars'], M)):
            m = m[cvars:cvars+nv*2, :]
            obj = {
                'M': m.cpu().numpy(),
                'attack': pert
            }
            with open(folder+"/"+b['fnames'][i], 'wb') as f:
                pkl.dump(obj, f)
            ms.append(m)
            cvars += nv*2
        assert torch.cat([torch.flatten(p) for p in ms]).sum() == torch.cat(M, dim=1).sum()
    else:
        raise NotImplementedError()


def combine_batch_nsat(input, same_label=True, shuffle=False):
    pos, neg, n_clauses, n_vars, is_sat = [], [], [], [], []
    n = len(input)
    
    assert n > 0
    if n == 1:
        return input[0]

    for b in input:
        adjs_pos, adjs_neg = get_nsat_blocks(b)
        for i in range(len(b['is_sat'])):
            pos.append(adjs_pos[i])
            neg.append(adjs_neg[i])
            n_clauses.append(b['n_clauses'][i])
            n_vars.append(b['n_vars'][i])
            is_sat.append(b['is_sat'][i])
            
    is_sat = torch.Tensor(is_sat)
    n_vars = np.array(n_vars)
    n_clauses = np.array(n_clauses)

    if same_label:
        assert torch.all(torch.eq(is_sat, is_sat[:int(len(is_sat)/n)].repeat(n)))

    if shuffle: 
        shuffle_idx = torch.randperm(len(pos)).tolist()
        pos = np.array(pos, dtype=object)[shuffle_idx].tolist()
        neg = np.array(neg, dtype=object)[shuffle_idx].tolist()
        is_sat = is_sat[shuffle_idx]
        n_clauses = n_clauses[shuffle_idx]
        n_vars = n_vars[shuffle_idx]
    
    batch = {
    'adj': torch.cat((torch.block_diag(*pos), torch.block_diag(*neg))).cuda(),
    'is_sat': is_sat.cuda(),
    'n_clauses': n_clauses,
    'n_vars': n_vars,
    'batch_size': len(is_sat)
    }  
    assert check_block_adj(batch)
    return batch


def get_nsat_blocks(b):
    cvars, cclauses = 0, 0
    cut = int(b['adj'].shape[0]/2)
    adj = b['adj'].detach().cpu()
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


def check_block_adj(batch):
    components = []
    for i, nc in enumerate(batch['n_clauses']):
        components.append(torch.ones((batch['n_vars'][i], nc)))
    # create disconnected graphs
    components = torch.block_diag(*components)
    bdiag = torch.cat([components, components]).to(batch['adj'].device)
    return (torch.all(torch.eq(bdiag * batch['adj'], batch['adj'])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run training (and adversarial finetuning)')
    parser.add_argument('-advtrain', default=False, action='store_true', help='Execute adversarial finetuning?')
    args = parser.parse_args()

    model = NeuroSAT()
    model.cuda()
    train, val, test, name = get_SAT_training_data("SAT-10-40")

    # REGULAR TRAINING
    results = train_nsat(model, train, val, name, model_path="../trained_models/")
    print(results)

    # ADV. FINETUNING
    if args.advtrain:
        results = train_nsat(model, train, val, name, model_path="../trained_models/", augment="adv", aug_rate=0.05, lr=0.00001)
        print(results)