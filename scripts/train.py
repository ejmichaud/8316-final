#!/usr/bin/env python
# coding: utf-8
"""
Trains a neural network on the Ising model data (see `simulate.py`).
Can choose size of network (width), amount of data, and system size.
"""

import os
from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
# import scipy.stats
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("sparse-parity-v4")
ex.captured_out_filter = apply_backspaces_and_linefeeds

# class FastTensorDataLoader:
#     """
#     A DataLoader-like object for a set of tensors that can be much faster than
#     TensorDataset + DataLoader because dataloader grabs individual indices of
#     the dataset and calls cat (slow).
#     """
#     def __init__(self, *tensors, batch_size=32, shuffle=False):
#         """
#         Initialize a FastTensorDataLoader.

#         :param *tensors: tensors to store. Must have the same length @ dim 0.
#         :param batch_size: batch size to load.
#         :param shuffle: if True, shuffle the data *in-place* whenever an
#             iterator is created out of this object.

#         :returns: A FastTensorDataLoader.
#         """
#         assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
#         self.tensors = tensors

#         self.dataset_len = self.tensors[0].shape[0]
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         # Calculate # batches
#         n_batches, remainder = divmod(self.dataset_len, self.batch_size)
#         if remainder > 0:
#             n_batches += 1
#         self.n_batches = n_batches

#     def __iter__(self):
#         if self.shuffle:
#             self.indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
#         else:
#             self.indices = None
#         self.i = 0
#         return self

#     def __next__(self):
#         if self.i >= self.dataset_len:
#             raise StopIteration
#         if self.indices is not None:
#             indices = self.indices[self.i:self.i+self.batch_size]
#             batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
#         else:
#             batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
#         self.i += self.batch_size
#         return batch

#     def __len__(self):
#         return self.n_batches



def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    width = 50
    depth = 3
    activation = 'Tanh'
    dropout = 0.0

    size = 32
    D = 20000 # training samples
    data_dir = '/om/user/ericjm/results/class/8.316/final/all-0/'
    
    batch_size = 512
    epochs = 500
    lr = 1e-4
    test_samples = 5000
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    verbose=False
    seed = 0

# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(width,
        depth,
        activation,
        dropout,
        size,
        D,
        data_dir,
        batch_size,
        epochs,
        lr,
        test_samples,
        device,
        dtype,
        verbose,
        seed,
        _log
    ):

    sizes = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128]
    temperatures = np.linspace(1, 3.5, 50)
    assert size in sizes, f"Size must be one of {sizes}"

    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if activation == 'ReLU':
        activation_fn = nn.ReLU
    elif activation == 'Tanh':
        activation_fn = nn.Tanh
    elif activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    else:
        assert False, f"Unrecognized activation function identifier: {activation}"

    # create model
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(size*size, width))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
        elif i == depth - 1:
            layers.append(nn.Linear(width, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*layers).to(device)
    _log.debug("Created model.")
    _log.debug(f"Model has {sum(t.numel() for t in mlp.parameters())} parameters") 
    ex.info['P'] = sum(t.numel() for t in mlp.parameters())

    class DataSet(torch.utils.data.Dataset):
        def __init__(self, samples, labels, temps):
            super(DataSet, self).__init__()
            self.labels  = labels
            self.samples = samples
            self.temps   = temps
            if len(samples) != len(labels):
                raise ValueError(
                    f"should have the same number of samples({len(samples)}) as there are labels({len(labels)})")
                
        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            y = self.labels[index]
            x = self.samples[index]
            t = self.temps[index]
            return x, y, t


    #Here is some code to read all the different files and make a dataset
    all_data  = []
    all_temps = []
    for T in temperatures:
        f = torch.load(os.path.join(data_dir, f'{size}_{T}_{2000}.pt'))
        all_data.append(f['data'].float())
        all_temps.append(torch.ones(f['data'].shape[0])*T)

    all_data = torch.cat(all_data).flatten(1)
    all_temps = torch.cat(all_temps).flatten()

    # import code; code.interact(local=locals())
    # all_data    = np.reshape(all_data,(all_data.shape[0],all_data.shape[1]*all_data.shape[2]))
    #build a numpy array that has labels 1 for below phase transition and 0 for above transition
    T_c = 2 / np.log(1 + np.sqrt(2))
    all_labels = (all_temps < T_c).float()

    # convert to torch tensors
    # all_data   = torch.from_numpy(all_data.astype("float32"))
    # all_labels = torch.from_numpy(all_labels)
    # all_temps  = torch.from_numpy(all_temps)

    all_dataset = DataSet(samples=all_data.to(device),
                        labels=all_labels.to(device),
                        temps=all_temps.to(device))

    all_dataset = torch.utils.data.Subset(all_dataset, np.random.choice(len(all_dataset), D+test_samples, replace=False))
    data_train, data_test = torch.utils.data.random_split(all_dataset, [D, test_samples])
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    # import code; code.interact(local=locals())
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr)
    criterion = nn.BCELoss()

    ex.info['train_losses'] = []
    ex.info['test_losses']  = []
    ex.info['train_accuracies'] = []
    ex.info['test_accuracies']  = []
    ex.info['epochs'] = []
    for epoch in tqdm(range(epochs), desc='Epochs', disable=not verbose):
        mlp.train(True)
        running_loss = 0.0; updates=0
        for x, y, t in train_loader:
            # import code; code.interact(local=locals())
            opt.zero_grad()
            y_hat = mlp(x)
            loss  = criterion(y_hat.flatten(),y.float()) 
            loss.backward()
            opt.step()
            running_loss += loss.item()
            updates += 1
            del x,y
        mlp.eval()
        with torch.no_grad():
            y_hat = mlp(data_train[:][0])
            acc = np.mean(y_hat.flatten().round().cpu().numpy() == data_train[:][1].cpu().numpy())
            ex.info['train_accuracies'].append(acc.item())
            train_loss = criterion(y_hat.flatten(),data_train[:][1].float())
            ex.info['train_losses'].append(train_loss.item())
            y_hat = mlp(data_test[:][0])
            acc = np.mean(y_hat.flatten().round().cpu().numpy() == data_test[:][1].cpu().numpy())
            ex.info['test_accuracies'].append(acc.item())
            test_loss = criterion(y_hat.flatten(),data_test[:][1].float())
            ex.info['test_losses'].append(test_loss.item())
            ex.info['epochs'].append(epoch)
        

