import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, dilation=dilation)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, size,  scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, size, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, size, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# 增加了两个自定义的激活函数
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        return x * cdf


class ReLUn(nn.Module):
    def __init__(self, n):
        super(ReLUn, self).__init__()
        self.n = n

    def forward(self, x):
        x[x < -self.n] = -self.n
        x[x > self.n] = self.n
        return x + self.n


class FReLU(nn.Module):
    def __init__(self, in_channels):
        super(FReLU, self).__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.max(x, x1)
        return x

# GaussianConv2d以及其辅助函数
class Gaussian_fun(nn.Module):
    def __init__(self, kernel_size):
        super(Gaussian_fun, self).__init__()
        self.kernel_size = kernel_size
        self.x_grid, self.y_grid = Gaussian_fun._init_grid(kernel_size)
        
    @staticmethod
    def _init_grid(size):
        x = torch.arange(-size // 2 + 1, size // 2 + 1).float()
        y = torch.arange(-size // 2 + 1, size // 2  + 1).float()
        x_grid, y_grid = torch.meshgrid(x, y)
        return torch.nn.Parameter(x_grid), torch.nn.Parameter(y_grid)
    
    def _compute_kernel(self, sigma):
        size = self.kernel_size
        g = torch.exp(-((self.x_grid**2 + self.y_grid**2) / (2.0 * sigma**2)))
        g = g / g.sum()
        g = g.view(-1, 1, size, size)
        return g


class GaussianConv2d(nn.Module):
    def __init__(self, kernel_size):
        super(GaussianConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = np.int((kernel_size - 1) / 2)
        self.gaussian_fun = Gaussian_fun(self.kernel_size)

    def forward(self, x, sigma):
        # import pdb; pdb.set_trace()
        w = x.size(2)
        h = x.size(3)
        for i in range(x.size(0)):
            # kernel=self.gaussian_fun._compute_kernel(sigma[i, 0, 0, 0]) #因为sigma是认为给定的，所以只有一个
            kernel=self.gaussian_fun._compute_kernel(sigma)
            if x.size(1) == 3:
                channel_1 = F.conv2d(x[i, 0, :, :].view(-1, 1, w, h), 
                                    kernel, stride=self.stride, padding=self.padding)
                channel_2 = F.conv2d(x[i, 1, :, :].view(-1, 1, w, h), 
                                    kernel, stride=self.stride, padding=self.padding)
                channel_3 = F.conv2d(x[i, 2, :, :].view(-1, 1, w, h),
                                    kernel, stride=self.stride, padding=self.padding)
                fig = torch.cat([channel_1, channel_2, channel_3], 1)
            if x.size(1) == 1:
                fig = F.conv2d(x[i, 0, :, :].view(-1, 1, w, h),
                               kernel, stride=self.stride, padding=self.padding)
            if i == 0:
                out = fig
            else:
                out = torch.cat([out, fig], 0)
        return out


class ChannelZeroPad(nn.Module):
    def __init__(self, prepadding=1, postpadding=0, value=0):
        super(ChannelZeroPad, self).__init__()
        self.prepadding = prepadding
        self.postpadding = postpadding
        self.value = 0

    def forward(self, input):
        return F.pad(input, (0, 0, 0, 0, self.prepadding, self.postpadding))


class MyUpsampler(nn.Module):
    def __init__(self, conv, upscale_factor, n_feats, act=None, bias=True):
        super(MyUpsampler, self).__init__()

        self.upscale_factor = upscale_factor[0]
        self.conv1 = conv(n_feats, n_feats // 2, 3, bias)
        self.conv2 = conv(n_feats // 2, self.upscale_factor ** 2 - 1, 3, bias)
        self.ChannelZeroPad = ChannelZeroPad(1, 0, 0)
        self.positionupscale = nn.PixelShuffle(self.upscale_factor)
        # self.relu = nn.ReLU(True)
        self.relu = FReLU(n_feats // 2)
        self.act = act
        if self.act is not None:
            if act == 'relu':
                self.act = nn.ReLU(True)
            else:
                raise NotImplementedError

    def forward(self, x, preintp_x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.ChannelZeroPad(x)
        x += preintp_x.repeat(1, self.upscale_factor**2, 1, 1)
        x = self.positionupscale(x)
        if self.act is not None:
            x = self.act(x)
        return x


def mandstd(x):
    mean = torch.mean(x, dim=(1,2,3), keepdim=True)
    std = torch.std(x,dim=(1,2,3), keepdim=True)
    return mean, std



if __name__ == '__main__':
    G = GaussianConv2d(5)
    x = torch.rand((1,3,7,7))

    p = G(x, torch.tensor([[[[1]]]]))

    print(p)