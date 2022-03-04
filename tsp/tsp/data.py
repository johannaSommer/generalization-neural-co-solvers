import os
import numpy as np
from torch.utils.data import Dataset
    
    
def get_DTSP_training_data():
    train = TSPDataset('/TSP20/train')
    val = TSPDataset('/TSP20/val')
    test = TSPDataset('/TSP20/test')
    return train, val, test, "DTSP-GEN"


class TSPDataset(Dataset):
    def __init__(self, path):
        self.data_names = [path + '/' + x for x in os.listdir(path)]
        self.path = path

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        item = read_graph(self.data_names[idx])
        return item


def read_graph(filepath):
    with open(filepath, "r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n, n), dtype=int)
        Mw = np.zeros((n, n), dtype=float)

        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i, j = [int(x) for x in line.split()]
            Ma[i, j] = 1
            line = f.readline()

        # Parse edge weights
        while 'EDGE_WEIGHT_SECTION' not in line: line = f.readline();
        for i in range(n):
            Mw[i, :] = [float(x) for x in f.readline().split()]

        # Parse tour
        while 'TOUR_SECTION' not in line: line = f.readline();
        route = [int(x) for x in f.readline().split()]
        
        # Parse tour
        while 'NODE_POSITION' not in line: line = f.readline();
        nodes = [x for x in f.readline().split()]

    return Ma, Mw, route, nodes, filepath
