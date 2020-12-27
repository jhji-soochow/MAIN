import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from deform_conv import DeformConv


if __name__ == '__main__':
    a = torch.rand((2, 1, 30, 30)).cuda() 

    offnet = nn.Conv2d(1, 18, 3, padding=1).cuda()

    dconv = DeformConv(1, 64, 3, padding=1).cuda()

    offset = offnet(a)
    b = dconv(a, offset)

    print(b)
