import torch
import random
import argparse
from tsp.data import get_DTSP_training_data
from tsp.solvers.dtsp import DTSP
from sklearn.metrics import accuracy_score
from tsp.attacks import attack_rand, attack_opt
from torch.utils.data.dataloader import DataLoader
from tsp.utils import *
from tsp.data import get_DTSP_training_data
from tsp.solvers.dtsp import DTSP


def eval_tsp(model, val, pert="clean", batch_size=12, **attackargs):
    ps, ts, outs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])    
    model.eval()
    dl_val = DataLoader(dataset=val, collate_fn=model.collate_fn, shuffle=False, batch_size=batch_size)

    for _, batch in enumerate(dl_val):
        
        if pert == "clean":
            pass                                                                                      
        elif pert == "random":
            batch = attack_rand(batch, model, **attackargs)
        elif pert == "opt":
            batch = attack_opt(batch, model, **attackargs)
        else: 
            raise NotImplementedError()     
            
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

    parser = argparse.ArgumentParser(description='Run Attack')
    parser.add_argument('-seed', default=0, action='store_true', help='Python random seed')
    parser.add_argument('-opt', default=False, action='store_true', help='Optimized or random attack')
    parser.add_argument('-model_path', default="../trained_models/trained_dtsp.pt", action='store_true', help='Model to be attacked')
    args = parser.parse_args()

    random.seed(args.seed)
    model = DTSP().cuda()
    model.load_state_dict(torch.load(args.model_path))
    train, val, test, name = get_DTSP_training_data()
    res = eval_tsp(model, val, pert="opt" if args.opt else "random")
    print(res)