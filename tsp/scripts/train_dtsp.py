import torch
import numpy as np
from datetime import datetime
from tsp.data import TSPDataset
from sklearn.metrics import accuracy_score
from tsp.attacks import attack_rand, attack_opt
from torch.utils.data.dataloader import DataLoader
from tsp.utils import *
import argparse
from tsp.data import get_DTSP_training_data
from tsp.solvers.dtsp import DTSP


def train_tsp(model, train, val, dataset_name, epochs=1500, batch_size=16, batches_per_epoch=64,
                   lr=0.0001, weight_decay=1e-10, grad_clip=0.65):
    """
    Executes training and validation on the TSP neural solvers

    Keyword arguments:
    model -- TSP model
    train, val, test -- TSP data
    name -- dataset name, e.g. TSP20
    epochs, batch_size, batches_per_epoch, lr, weight_decay, grad_clip -- training parameters
    """
    if isinstance(train, TSPDataset) and isinstance(val, TSPDataset):
        dl_train = DataLoader(dataset=train, collate_fn=model.collate_fn,
                              shuffle=True, batch_size=batch_size)
    else:
        raise ValueError('Data is not provided as SAT Dataset.')

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0   
    train_losses, val_accs = [], []

    simulation_name = model.name + '_' + dataset_name + '_' + datetime.now().strftime("%d:%m-%H:%M:%S")
    print("Experiment identifier: ", simulation_name)
    
    for epoch in range(1, epochs+1):
        loss_temp = []
        model.train()
        
        for i, batch in enumerate(dl_train):
            optim.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch['target'].to(model.device))
            loss_temp.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
            optim.step()
            if i > batches_per_epoch: break

        train_losses.append(np.mean(np.array(loss_temp)).item())
        print("Train Epoch Loss", epoch, np.mean(np.array(loss_temp)).item(), flush=True)
        val_acc = eval_tsp(model, val, batch_size=batch_size)
        print("Validation Accuracy", val_acc, flush=True)
        val_accs.append(val_acc)
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'../trained_models/{simulation_name}_MAX.pt')

    return_dict = dict()
    return_dict['train_loss'] = train_losses
    return_dict['val_accs'] = val_accs
    return_dict['Name'] = simulation_name
    return return_dict


def eval_tsp(model, val, pert="clean", batch_size=12, **attackargs):
    """
    Executes training and validation on TSP neural solvers

    Keyword arguments:
    model -- TSP model
    val -- TSP validation data
    perturb -- perturbation applied on val samples
    batch_size -- batch_size
    attackargs -- attack parameters, e.g. number of steps or budget
    """
    ps, ts, outs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])    
    model.eval()
    dl_val = DataLoader(dataset=val, collate_fn=model.collate_fn, shuffle=False, batch_size=batch_size)

    for _, batch in enumerate(dl_val):  
            
        logits = model(batch)
        probs = torch.sigmoid(logits)
        preds = torch.where(probs > 0.5, torch.ones(probs.shape).to(model.device), torch.zeros(probs.shape).to(model.device))
            
        ps = torch.cat((ps, preds.cpu()))
        ts = torch.cat((ts, batch['target'].cpu()))
        outs = torch.cat((outs, probs.detach().cpu()))
            
    val_acc = accuracy_score(ts.numpy(), ps.numpy())
    print(val_acc)
    true_acc = accuracy_score(ts.numpy()[ts.bool()], ps.numpy()[ts.bool()])
    print(true_acc)
    false_acc = accuracy_score(ts.numpy()[~ts.bool()], ps.numpy()[~ts.bool()])
    print(false_acc)
    return val_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run training')
    args = parser.parse_args()

    model = DTSP().cuda()
    train, val, test, name = get_DTSP_training_data()
    res = train_tsp(model, train, val, name)
    print(res)


