import torch
from train_csat import eval_csat
from sat.solvers.circuitsat import CircuitSAT
from sat.data import get_SAT_val_data

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute attacks')
    parser.add_argument('-opt', default=False, action='store_true', help='Optimized or random attack')
    parser.add_argument('-model_path', default="../trained_models/pretrained_csat.pt", help='Directory of model to be attacked')
    args = parser.parse_args()

    model = CircuitSAT().cuda()
    model.load_state_dict(torch.load(args.model_path))
    val = get_SAT_val_data("SAT-3-10-true")

    attack_type = ("optimized-sat" if args.opt else "random-sat")
    res = eval_csat(model, val, perturb=attack_type)
    print(res)