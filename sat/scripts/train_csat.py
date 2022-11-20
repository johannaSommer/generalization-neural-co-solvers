import torch
import argparse
import torch.nn as nn
import numpy as np
from datetime import datetime
from sat.data import SATDataset, get_SAT_training_data
from sat.solvers.circuitsat import *
from sat.attacks import attack_opt, attack_random
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score


def train_csat(model, train, val, dataset_name, epochs=60, batch_size=32, lr=0.00002, device="cuda",
               weight_decay=1e-10, grad_clip=0.65, model_path="/trained_models/"):
    """
    Executes training and validation on the NeuroSAT model
    """
    if isinstance(train, SATDataset) and isinstance(val, SATDataset):
        dl_train = DataLoader(dataset=train, collate_fn=model.collate_fn, pin_memory=True,
                              shuffle=True, batch_size=batch_size, num_workers=1)
    else:
        raise ValueError('Data is not provided as SAT Dataset.')

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.cuda()

    best_val = 0.0
    val_accs, train_losses = [], []
    return_dict = dict()
    simulation_name = model.name + '_' + dataset_name + '_' + datetime.now().strftime("%d:%m-%H:%M:%S.%f")
    print("Experiment identifier: ", simulation_name)
    
    for epoch in range(0, epochs):
        loss_temp = []
        model.train()

        for _, batch in enumerate(dl_train):
            batch['adj'] = batch['adj'].to(device)
            batch['is_sat'] = batch['is_sat'].to(device)
            optim.zero_grad()
            batch['features'] = batch['features'].to(device)
            model.forward_update.flatten_parameters()
            model.backward_update.flatten_parameters()
            outputs = model(batch)
            outputs = evaluate_circuit(batch, torch.sigmoid(outputs), epoch+1)
            loss = custom_csat_loss(outputs)
            loss_temp.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
            optim.step()
            
        train_losses.append(np.mean(np.array(loss_temp)).item())
        print("Train Epoch Loss", epoch, np.mean(np.array(loss_temp)).item(), flush=True)
        val_acc = eval_csat(model, val, batch_size=batch_size)
        print("Validation Accuracy", val_acc, flush=True)
        val_accs.append(val_acc)
        
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), model_path + simulation_name + '_MAX.pt')


    return_dict = process_dict(return_dict, len(dl_train))
    return_dict['val_accs'] = val_accs
    return_dict['Name'] = simulation_name
    return return_dict


def eval_csat(model, val, batch_size=32, perturb="clean", **attackargs):
    """
    Evaluates model on validation data
    """  
    if isinstance(val, SATDataset):
        dl_val = DataLoader(dataset=val, collate_fn=model.collate_fn, pin_memory=False,
                            shuffle=False, batch_size=batch_size, num_workers=0)
    else:
        raise ValueError('Data is not provided as SAT Dataset.')

    ts, ps, outs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    for _, batch in enumerate(dl_val):

        if perturb == "clean":
            sample_new = batch

        elif perturb == "random-sat":
            assert torch.all(batch['is_sat'].bool())
            sample_new = attack_random("sat", model, batch, **attackargs)

        elif perturb == "optimized-sat":
            assert torch.all(batch['is_sat'].bool())
            sample_new = attack_opt("sat", model, batch, **attackargs) 

        else:
            raise ValueError("this kind of perturbation has not been implemented")

        with torch.no_grad():
            model.forward_update.flatten_parameters()
            model.backward_update.flatten_parameters()
            sample_new['features'] = sample_new['features'].cuda()
            sample_new['adj'] = sample_new['adj'].cuda()
            outputs = model(sample_new)
            outputs = torch.sigmoid(outputs)
            outputs = evaluate_circuit(sample_new, outputs, 1, hard=True)

        ts = torch.cat((ts, sample_new['is_sat']))
        outs = torch.cat((outs, outputs.flatten().detach().cpu()))
        preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
        ps = torch.cat((ps, preds.flatten().cpu()))
        print(accuracy_score(ts.numpy(), ps.numpy()))

    val_acc = accuracy_score(ts.numpy(), ps.numpy())
    return val_acc.item()


def process_dict(d, nbatches):
    # reduction is necessary or writing to db will fail because of size
    for key in d.keys():
        values = torch.Tensor(d[key])
        values = values.view(-1, nbatches).mean(-1)
        d[key] = values.tolist()
    return d


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run training (and adversarial finetuning)')
    parser.add_argument('-advtrain', default=False, action='store_true', help='Execute adversarial finetuning?')
    args = parser.parse_args()

    model = CircuitSAT()
    model.cuda()
    train, val, test, name = get_SAT_training_data("SAT-3-10")

    results = train_csat(model, train, val, name, model_path="../trained_models/")
    print(results)
