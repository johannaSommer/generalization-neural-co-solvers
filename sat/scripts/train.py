from sat.training import train_nsat
from sat.solvers.neurosat import NeuroSAT
from sat.data import get_NSAT_training_data
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run training (and adversarial finetuning)')
    parser.add_argument('-advtrain', default=False, action='store_true', help='Execute adversarial finetuning?')
    args = parser.parse_args()

    model = NeuroSAT()
    model.cuda()
    train, val, test, name = get_NSAT_training_data("SAT-10-40")

    # REGULAR TRAINING
    results = train_nsat(model, train, val, name, model_path="./trained_models/", epochs=args.epochs, lr=args.lr)
    print(results)

    # ADV. FINETUNING
    if args.advtrain:
        results = train_nsat(model, train, val, name, model_path="../trained_models/", augment="adv", aug_rate=0.05, lr=0.00001)
        print(results)