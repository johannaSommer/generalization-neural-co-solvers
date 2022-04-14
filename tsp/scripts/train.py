import random
import argparse
from tsp.data import get_DTSP_training_data
from tsp.training import train_tsp
from tsp.solvers.dtsp import DTSP


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run training')
    args = parser.parse_args()

    model = DTSP().cuda()
    train, val, test, name = get_DTSP_training_data()
    res = train_tsp(model, train, val, name)
    print(res)


