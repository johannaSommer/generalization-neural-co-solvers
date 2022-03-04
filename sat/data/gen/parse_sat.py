import os
import numpy as np
import pickle
import itertools
import shutil
import argparse
from sat.data.gen.gen_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate SATLIB data')
    parser.add_argument('-in_dir', default='../satlib/', help='directory with raw data')
    parser.add_argument('-out_dir', default='../satlib_pp/', help='directory to save processed data in')
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.mkdir(args.out_dir)

    for file in os.listdir(args.in_dir):
        # open file and parse out clause format
        print("Processing file", file)
        file1 = open(args.in_dir + file, 'r')
        lines = file1.read().splitlines()
        clauses = [line[:-1].split() for line in lines[:-3] if not line.startswith('c') and not line.startswith('p')]
        clauses = [list(map(int, clause)) for clause in clauses]

        out = dict()
        out['clauses'] = clauses
        out['n_clauses'] = len(clauses)
        flatten = itertools.chain.from_iterable
        nvars = np.max(np.abs(list(flatten(clauses))))
        out['n_vars'] = nvars
        out['label'], out['solution'] = solve_sat(nvars, clauses)
        assert out['label']

        # construct adjacency for mis model
        out['adj_nsat'] = clause_to_nsat_adj(clauses, nvars)

            # save output dict to new directory
        with open(args.out_dir + file[:-4] + '.pkl', 'wb') as f_dump:
            pickle.dump(out, f_dump)

    print("Successfully preprocessed all provided data.")
