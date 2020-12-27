# The DoG loss
from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class DoGLoss(nn.Module):
    def __init__(self, kernel_size, basicloss=nn.MSELoss(),rgb_range=1):
        super(DoGLoss, self).__init__()

        self.basicloss=basicloss
        self.gaussianconv2d = common.GaussianConv2d(kernel_size)

    def forward(self, hr, sigma, sr_low, sr_hig, mode='all'):
        hr_low = self.gaussianconv2d(hr, sigma)
        hr_hig = hr - hr_low
        low_loss = self.basicloss(hr_low, sr_low)        
        hig_loss = self.basicloss(hr_hig, sr_hig)
        if mode == 'all':
            return low_loss + hig_loss
        if mode == 'hig':
            return hig_loss
        if mode == 'low':
            return low_loss

        
        

