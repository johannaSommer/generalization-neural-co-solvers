import os
import numpy as np
import random
import pickle
import argparse
import shutil
from gen_utils import make_csat_batch, gen_iclause_pair


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate SR(U(start, end)) data')
    parser.add_argument('-n_pairs', default=10000, help='How many problem pairs to generate')
    parser.add_argument('-min_n', default=3, help='start value for number of variables')
    parser.add_argument('-max_n', default=10, help='end value for number of variables')
    parser.add_argument('-p_k_2', default=0.3)
    parser.add_argument('-p_geo', default=0.4)
    parser.add_argument('-py_seed', default=0)
    parser.add_argument('-np_seed', default=0)
    parser.add_argument('-remove_ss', default=False)
    args = parser.parse_args()

    random.seed(args.py_seed)
    np.random.seed(args.np_seed)

    n_cnt = args.max_n - args.min_n + 1
    problems_per_n = args.n_pairs * 1.0 / n_cnt
    problems = []
    batches = []
    num = 0

    name = 'SAT-' + str(args.min_n) + '-' + str(args.max_n)
    out_dir = '../' + name
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for n_var in range(args.min_n, args.max_n+1):

        lower_bound = int((n_var - args.min_n) * problems_per_n)
        upper_bound = int((n_var - args.min_n + 1) * problems_per_n)

        for problems_idx in range(lower_bound, upper_bound):
            print('Processing Problem ', num)

            n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(n_var, args.p_k_2, args.p_geo)                
            iclauses.append(iclause_unsat)                 
            iclauses[-1] = iclause_sat
    
            out_dict = make_csat_batch(iclauses, n_vars)
            assert out_dict['label']
            
            # save output dict to new directory
            with open(out_dir + "/" + name + "--" + str(num) + '_true.pkl', 'wb') as f_dump:
                pickle.dump(out_dict, f_dump)

            num += 1
