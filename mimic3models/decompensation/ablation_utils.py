import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mimic3models import common_utils
import numpy as np

def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    return data

def collate_fn(batch):
    xs, ys, lengths = zip(*batch)
    lengths = torch.tensor(lengths)

    # pad x to [batch, max_len, 76]
    padded_x = nn.utils.rnn.pad_sequence(xs, batch_first=True)

    # build mask
    max_len = padded_x.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]

    # labels to tensor
    ys = torch.tensor(ys)

    return padded_x, ys, mask

class VarLenDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data                # list of np arrays, each shape (seq_len, 76)
        self.labels = labels            # 1D tensor or list of ints/floats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)  # convert to tensor
        y = torch.tensor(self.labels[idx])
        length = x.size(0)
        return x, y, length

def load_data(reader, discretizer, normalizer, chunk_size=None):
    if chunk_size is None:
        N = reader.get_number_of_examples()
    else:
        N = chunk_size
        
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    ys = ret["y"]
    names = ret["name"]
    
    data = preprocess_chunk(data, ts, discretizer, normalizer)
    y = np.array(ys)
    
    dataset = VarLenDataset(data, y)

    return dataset