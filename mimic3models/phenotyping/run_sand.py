import numpy as np
import argparse
import os

import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import common_utils

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from SAnD.core.model import SAnD
from SAnD.utils.trainer import NeuralNetworkClassifier


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/'))
args = parser.parse_args()

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'))

val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ph_ts{}.input_str-previous.start_time-zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())

# Define model parameters
in_feature = 76
seq_len = 2804
n_heads = 8 # Number of heads for multi-head attention layer: Should be fixed at 8
factor = 120 # Dense interpolation factor (M): This depends on the task at hand
num_class = 25 # Number of output class
num_layers = 2 # Number of multi-head attention layers (N): This depends on the task at hand
learning_rate = 0.0005
betas = (0.9, 0.98)
eps = 4e-09
weight_decay = 5e-4
no_of_epochs = 200
batch_size = 32
dropout_rate = 0.4

# Read data
train_raw = utils.load_data(train_reader, discretizer,
                                normalizer, seq_len, args.small_part)
val_raw = utils.load_data(val_reader, discretizer,
                              normalizer, seq_len, args.small_part)

# Build the model
model = NeuralNetworkClassifier(
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers, dropout_rate=dropout_rate),
    nn.CrossEntropyLoss(),
    optim.Adam,
    optimizer_config={"lr": learning_rate, "betas": betas, "eps": eps, "weight_decay": weight_decay},
    experiment=Experiment()
)

# Prepare training
print("==> training")
# Ensure y values are numpy arrays
train_x, train_y = train_raw
val_x, val_y = val_raw
train_y = np.array(train_y)
val_y = np.array(val_y)

train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.long)  # classification labels
val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
val_y_tensor = torch.tensor(val_y, dtype=torch.long)

# Wrap into TensorDataset
train_ds = TensorDataset(train_x_tensor, train_y_tensor)
val_ds = TensorDataset(val_x_tensor, val_y_tensor)

# Wrap into DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# training network
print("==> fitting model")

model.fit(
    {"train": train_loader,
     "val": val_loader},
    epochs=no_of_epochs
)

# ensure that the code uses test_reader
print("==> testing")
del train_reader
del val_reader
del train_raw
del val_raw

test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

ret = utils.load_data(test_reader, discretizer,
                                   normalizer, seq_len, args.small_part)
data = np.array(ret[0], dtype=np.float32)   # (N, seq_len, in_feature)
labels = np.array(ret[1], dtype=np.int64)   # (N,)

# Convert DataFrame to tensor (float)
test_x_tensor = torch.tensor(data, dtype=torch.float32)
test_y_tensor = torch.tensor(labels, dtype=torch.long)

test_ds = TensorDataset(test_x_tensor, test_y_tensor)

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
model.evaluate(test_loader)
model.save_to_file("save_params/phenotyping/")