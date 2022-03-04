import torch
from sat.training import eval_nsat
from sat.solvers.neurosat import NeuroSAT
from sat.data import get_NSAT_val_data

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute attacks')
    parser.add_argument('-type', help='SAT, ADC or DEL attack')
    parser.add_argument('-opt', default=False, action='store_true', help='Optimized or random attack')
    parser.add_argument('-model_path', default="../trained_models/pretrained_nsat.pt", help='Directory of model to be attacked')
    args = parser.parse_args()
    assert args.type in ["sat", "del", "adc"]

    model = NeuroSAT().cuda()
    model.load_state_dict(torch.load(args.model_path))
    if args.type == "sat":
        val = get_NSAT_val_data("SAT-10-40-true")
    else:
        val = get_NSAT_val_data("SAT-10-40-false")

    attack_type = ("optimized-" if args.opt else "random-") + args.type
    res = eval_nsat(model, val, perturb=attack_type)
    print(res)