import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

# add my benchmark
class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        dir_data = '/data/jjh_backup/1_3/testset'
        self.apath = os.path.join(dir_data, self.name[2:])
        self.dir_hr = os.path.join(self.apath)
        self.dir_lr = None
        self.ext = ('', '.png')

