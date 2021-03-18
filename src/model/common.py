import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, dilation=dilation)


class ChannelZeroPad(nn.Module):
    def __init__(self, prepadding=1, postpadding=0, value=0):
        super(ChannelZeroPad, self).__init__()
        self.prepadding = prepadding
        self.postpadding = postpadding
        self.value = 0

    def forward(self, input):
        return F.pad(input, (0, 0, 0, 0, self.prepadding, self.postpadding))


class MyUpsampler(nn.Module):
    def __init__(self, conv, upscale_factor, n_feats, bias=True):
        super(MyUpsampler, self).__init__()

        self.upscale_factor = upscale_factor
        self.conv1 = conv(n_feats, n_feats // 2, 3, bias)
        self.conv2 = conv(n_feats // 2, self.upscale_factor ** 2 - 1, 3, bias)
        self.ChannelZeroPad = ChannelZeroPad(1, 0, 0)
        self.positionupscale = nn.PixelShuffle(self.upscale_factor)
        self.relu = nn.ReLU(True)

    def forward(self, x, preintp_x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.ChannelZeroPad(x)
        x += preintp_x.repeat(1, self.upscale_factor**2, 1, 1)
        x = self.positionupscale(x)
        return x