import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("/path/to/data/")

def get_SAT_val_data(name, random_state=42):
    if name == "SAT-10-40":
        _, val, _, _ = get_SAT_training_data("SAT-10-40")
        return val
    
    elif name in ["SAT-10-40-true", "SAT-10-40-false"]:
        _, val, _, _ = get_SAT_training_data("SAT-10-40")
        if name == "SAT-10-40-true":
            filtered = [f for f in val.data_names if "true" in f]
        else:
            filtered = [f for f in val.data_names if "false" in f]
        val.data_names = np.array(filtered)
        return val
    
    elif name in ["SAT-50-100-true", "SAT-50-100-false"]:
        val = get_SAT_val_data("SAT-50-100")
        if name == "SAT-50-100-true":
            filtered = [f for f in val.data_names if "true" in f]
        else:
            filtered = [f for f in val.data_names if "false" in f]
        val.data_names = np.array(filtered)
        return val

    elif name in ["SAT-3-10-true", "SAT-3-10-false"]:
        val = get_SAT_val_data("SAT-3-10")
        if name == "SAT-3-10-true":
            filtered = [f for f in val.data_names if "true" in f]
        else:
            filtered = [f for f in val.data_names if "false" in f]
        val.data_names = np.array(filtered)
        return val

    elif name in ["gcol-large", "gcol-small", "random3sat", "uni3sat", "SAT-3-10", "SAT-50-100", "SAT-100-300"]:
        data_names = os.listdir(DATA_PATH) / name
        val_idx, _ = train_test_split(np.arange(len(data_names)), test_size=0.5, 
                                             random_state=random_state)
        val = SATDataset(DATA_PATH, val_idx)
        return val
    else:
        raise NotImplementedError()  


def get_SAT_training_data(name, random_state=42, split_size=0.16666):
    path = DATA_PATH / name
    data_names = os.listdir(path)
    data_names = [n for n in data_names if not n.startswith('lt_')]
    train_idx, val_idx = train_test_split(np.arange(len(data_names)), test_size=split_size,
                                              random_state=random_state)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=random_state)  
    train = SATDataset(path, train_idx)
    val = SATDataset(path, val_idx)
    test = SATDataset(path, test_idx)
    return train, val, test, name



class SATDataset(Dataset):
    """
    Dataset class for SAT data
    
    Keyword arguments:
    path -- path to preprocessed .cnf files
    idx -- indices for train/test/val items
    """
    def __init__(self, path, idx):
        self.data_names = [n for n in os.listdir(path) if not n.startswith('lt_')]
        self.data_names = np.array(self.data_names)[idx]
        self.path = path

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        name = self.data_names[idx]
        item = pickle.load(open(self.path / name, "rb"))
        item['filename'] = self.data_names[idx]
        return item

