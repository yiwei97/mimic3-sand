%load_ext autoreload
%autoreload 2

import numpy as np
import argparse
import os
import sys

os.chdir('/Users/evan.millikan/Documents/Personal/OAN6250-BigDataHealth/mimic3-sand')
sys.path.append('/Users/evan.millikan/Documents/Personal/OAN6250-BigDataHealth/mimic3-sand')


import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import common_utils

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from SAnD_Repl.sand_env import SAnDEnv


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args(['--network', 'haha'])
print(args)

if args.small_part:
    args.save_every = 2**30

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
args_dict['header'] = discretizer_header
args_dict['task'] = 'ph'
args_dict['num_classes'] = 25
args_dict['target_repl'] = target_repl

# Define model parameters
in_feature = 76
seq_len = 2804
n_heads = 8 # Number of heads for multi-head attention layer: Should be fixed at 8
factor = 120 # Dense interpolation factor (M): This depends on the task at hand
num_class = 25 # Number of output class
num_layers = 2 # Number of multi-head attention layers (N): This depends on the task at hand
d_model = 128
dropout_rate = 0.4
attn_window=4
mode = 'multiclass'
optimizer = 'adam'
optimizer_config = {
    'lr' : 0.0005,
    'betas' : (0.9, 0.98),
    'eps' : 4e-09,
    'weight_decay' : 5e-4,
}
num_epochs = 10
batch_size = 32


# Read data
train_raw = utils.load_data(train_reader, discretizer,
                                normalizer, seq_len, args.small_part)
val_raw = utils.load_data(val_reader, discretizer,
                              normalizer, seq_len, args.small_part)

sand_env = SAnDEnv(
    input_features=in_feature,
    seq_len=seq_len,
    num_heads=n_heads,
    factor=factor,
    n_layers=num_layers,
    d_model=d_model,
    dropout_rate=dropout_rate,
    n_class=num_class,
    attn_window=attn_window,
    mode=mode,
    optimizer=optimizer,
    optimizer_config=optimizer_config
)


# Prepare training
print("==> training")
# Ensure y values are numpy arrays
train_x, train_y = train_raw
val_x, val_y = val_raw
train_y = np.array(train_y)
val_y = np.array(val_y)

train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)  # classification labels
val_x = torch.tensor(val_x, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.long)

# Wrap into TensorDataset
train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)


# training network
print("==> fitting model")

sand_env.train(
    train_ds,
    eval_dataset=None,
    model_name='phenotype',
    num_epochs=num_epochs,
    batch_size=batch_size,
    save_frequency=10,
    num_workers=0,
    dataset_name='phenotype'
)

# ensure that the code uses test_reader
print("==> testing")
test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

ret = utils.load_data(test_reader, discretizer,
                                   normalizer, seq_len, args.small_part)
data = np.array(ret[0], dtype=np.float32)   # (N, seq_len, in_feature)
labels = np.array(ret[1], dtype=np.int64)   # (N,)

# Convert DataFrame to tensor (float)
test_x = torch.tensor(data, dtype=torch.float32)
test_y = torch.tensor(labels, dtype=torch.long)

test_ds = TensorDataset(test_x, test_y)

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
sand_env.evaluate(
    test_ds,
    model_name='phenotype',
    batch_size=batch_size,
    num_workers=0,
    dataset_name='phenotype'
)
