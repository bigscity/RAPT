import os
import json

import torch
import logging
import numpy as np
import torch.nn as nn

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


class Batch(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def to(self, device):
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self[key] = self[key].to(device)
        return self


class DiagnosisPrediction(Dataset):

    def __init__(self, path, args):
        super(DiagnosisPrediction, self).__init__()

        self.args = args
        self.records = list()

        self.process(path)

        logging.info('{} {}'.format(path, len(self.records)))

    def process(self, path):
        if path == 'train':
            num = 7000
        elif path == 'val':
            num = 1000
        else:
            num = 2000

        for _ in range(num):

            x = torch.randn(10, 129)
            week = torch.randint(10, 40, (10, ))

            y = torch.randint(0, 2, (1, )).float()
            self.records.append((x, week, y))

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self):
        return len(self.records)


def collate_rp(batch):
    _x, _week, _y = zip(*batch)
    x, week, y = list(_x), list(_week), list(_y)

    mask = [torch.ones(t.size(0)) for t in x]
    x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    week = nn.utils.rnn.pad_sequence(week, batch_first=True, padding_value=0)
    mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
    y = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    return Batch(x=x, week=week, y=y, mask=mask)


class PreTrain(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.records = list()
        self.process()

        logging.info('pre-train data {}'.format(len(self.records)))

    def process(self):

        for _ in range(10000):

            x = torch.randn(10, 129)
            week = torch.randint(10, 40, (10, ))

            self.records.append((x, week))

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self):
        return len(self.records)


def collate_pt(batch):
    _x, _week = zip(*batch)
    x, week = list(_x), list(_week)

    mask = [torch.ones(t.size(0)) for t in x]
    x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    week = nn.utils.rnn.pad_sequence(week, batch_first=True, padding_value=0)
    mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)

    return Batch(x=x, week=week, mask=mask)


def load_data(args, path_list=None):

    if args.model == 'Pretrain':
        dataset = PreTrain(args)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pt, drop_last=True)
        return loader
    else:
        loaders = list()
        for path in path_list:
            dataset = DiagnosisPrediction(path, args)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_rp)
            loaders.append(loader)

        return loaders[0] if len(loaders) == 1 else tuple(loaders)
