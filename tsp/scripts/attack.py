import torch
import random
import argparse
from tsp.data import get_DTSP_training_data
from tsp.training import eval_tsp
from tsp.solvers.dtsp import DTSP


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