import torch
from sat.utils import check_block_adj, solve_sat, adj_to_clause
from sat.attacks import attack_random, attack_opt
from sat.data import get_NSAT_val_data
from torch.utils.data.dataloader import DataLoader
from sat.solvers.neurosat import NeuroSAT


def test_sat_random(bsize=5):
    dl_val, model = get_model_and_dl("SAT-10-40-true", bsize)
    for sample in dl_val:
        sample = attack_random("sat", model, sample) 
        assert check_block_adj(sample)
        problems = adj_to_clause(sample)
        solutions = solve_sat(problems)
        for (s_new, s_old) in zip(solutions, sample['is_sat']):
            assert s_new == s_old
            print(s_new, s_old.item())


def test_unsat_random(bsize=5):
    dl_val, model = get_model_and_dl("SAT-10-40-false", bsize)
    for attack in ["adc", "del"]:
        for sample in dl_val:
            sample = attack_random(attack, model, sample) 
            assert check_block_adj(sample)
            problems = adj_to_clause(sample)
            solutions = solve_sat(problems)
            for (s_new, s_old) in zip(solutions, sample['is_sat']):
                assert s_new == s_old
                print(s_new, s_old.item())


def test_sat_opt(bsize=32):
    dl_val, model = get_model_and_dl("SAT-10-40-true", bsize)
    for i, sample in enumerate(dl_val):
        sample = attack_opt("sat", model, sample, steps=20) 
        assert check_block_adj(sample)
        problems = adj_to_clause(sample)
        solutions = solve_sat(problems)
        for (s_new, s_old) in zip(solutions, sample['is_sat']):
            assert s_new == s_old
            print(i, s_new, s_old.item())


def test_unsat_opt(bsize=32):
    dl_val, model = get_model_and_dl("SAT-10-40-false", bsize)
    for attack in ["del", "adc"]:
        for i, sample in enumerate(dl_val):
            sample = attack_opt(attack, model, sample, steps=20) 
            assert check_block_adj(sample)
            problems = adj_to_clause(sample)
            solutions = solve_sat(problems)
            for (s_new, s_old) in zip(solutions, sample['is_sat']):
                assert s_new == s_old
                print(i, s_new, s_old.item())


def get_model_and_dl(data, batch_size=1):
    model_path = "../trained_models/pretrained_nsat.pt"
    model = NeuroSAT().cuda()
    model.load_state_dict(torch.load(model_path))
    val = get_NSAT_val_data(data)
    dl_val = DataLoader(dataset=val, collate_fn=model.collate_fn, batch_size=batch_size)
    return dl_val, model
 