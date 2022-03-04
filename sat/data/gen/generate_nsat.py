import os
import random
import pickle
import argparse
import shutil
import numpy as np
from gen_utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate SAT-x-y data')
    parser.add_argument('-n_pairs', default=30000, help='How many problem pairs to generate')
    parser.add_argument('-min_n', default=10, help='start value for number of variables')
    parser.add_argument('-max_n', default=40, help='end value for number of variables')
    parser.add_argument('-p_k_2', default=0.3)
    parser.add_argument('-p_geo', default=0.4)
    parser.add_argument('-py_seed', default=0)
    parser.add_argument('-np_seed', default=0)
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

            # generate clauses here
            n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(n_var, args.p_k_2, args.p_geo)                
            iclauses.append(iclause_unsat)

            out_dict = dict()
            out_dict['clauses'] = iclauses
            out_dict['n_clauses'] = len(iclauses)
            out_dict['n_vars'] = n_vars
            out_dict['label'], out_dict['solution'] = solve_sat(n_vars, iclauses)
            assert not out_dict['label']
            
            # construct adjacency for nsat model
            out_dict['adj_nsat'] = clause_to_nsat_adj(iclauses, n_vars)
            
            # save output dict to new directory
            with open(out_dir + "/" + name + "--" + str(num) + '_false.pkl', 'wb') as f_dump:
                pickle.dump(out_dict, f_dump) 
                
            iclauses[-1] = iclause_sat
            out_dict = dict()
            out_dict['clauses'] = iclauses
            out_dict['n_clauses'] = len(iclauses)
            out_dict['n_vars'] = n_vars
            out_dict['label'], out_dict['solution'] = solve_sat(n_vars, iclauses)
            assert out_dict['label']
            
            # construct adjacency for nsat model
            out_dict['adj_nsat'] = clause_to_nsat_adj(iclauses, n_vars)

            # save output dict to new directory
            with open(out_dir + "/" + name + "--" + str(num) + '_true.pkl', 'wb') as f_dump:
                pickle.dump(out_dict, f_dump)

            num += 1
