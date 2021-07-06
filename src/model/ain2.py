from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return AIN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Spatial Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self, channel):
        super(SALayer, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, 1, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        atten = self.spatial_attention(x)
        return atten * x


##  Attention-aware fearure enhancement (AFE)
class AFE(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(AFE, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        
        self.front = nn.Sequential(*modules_body)
        self.CA = CALayer(n_feat, reduction)
        self.SA = SALayer(n_feat)
        self.act = act
        self.res_scale = res_scale
        self.fushion = nn.Conv2d(n_feat * 3, n_feat, 1)

    def forward(self, x):

        x_ = self.front(x)
        ca = self.CA(x_)
        sa = self.SA(x_)
        att = torch.cat((x_, ca, sa), 1)
        res = self.fushion(att)
        res += x
        return res


class AIB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, 
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(AIB, self).__init__()
            
        self.scale1_1 = nn.Conv2d(n_feat, n_feat, 3, padding=1)
        self.scale2_1 = nn.Conv2d(n_feat, n_feat, 5, padding=2)
        self.act = act

        self.scale1_2 = nn.Conv2d(n_feat * 2, n_feat * 2, 3, padding=1)
        self.scale2_2 = nn.Conv2d(n_feat * 2, n_feat * 2, 5, padding=2)

        self.csab1 = AFE(conv, n_feat * 2, kernel_size, 
                           reduction, bias=True, bn=False, act=act)
        self.csab2 = AFE(conv, n_feat * 2, kernel_size,
                           reduction, bias=True, bn=False, act=act)

        self.fushion = nn.Conv2d(n_feat * 4, n_feat, 1)

    def forward(self, x):

        br1_1 = self.act(self.scale1_1(x))
        br2_1 = self.act(self.scale2_1(x))

        cat1 = torch.cat((br1_1, br2_1), 1)

        br1_2 = self.csab1(self.scale1_2(cat1))
        br2_2 = self.csab2(self.scale2_2(cat1))
        cat2 = torch.cat((br1_2, br2_2), 1)
        fuse = self.fushion(cat2)
        return x + fuse


class AIN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(AIN, self).__init__()
        
        n_feats = 64
        n_blocks = 7
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)
        reduction = 16

        self.n_blocks = n_blocks
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                AIB(conv, n_feats, kernel_size, reduction))
        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        # define tail module
        self.tail = conv(n_feats, n_feats, kernel_size)
        self.tail2 =  common.MyUpsampler(conv, scale, n_feats)

    def forward(self, x):
        x_ = x

        # extract f_0
        f_0 = self.head(x)
        res = f_0

        # extract f_n
        f_n = self.body(f_0)
        f = f_n + res

        g = self.tail(f)
        x = self.tail2(g, x_)     # to conduct real interpolation
        return x 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AIN')
    parser.add_argument('--scale', type=int, default=2,
                        help='super resolution scale')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='n colors of input')
    # parser.add_argument('--rgb_range', type=float, default=255,
    #                     help='residual scaling')
    args = parser.parse_args()
    
    a = torch.rand((2, 1, 30, 30)).cuda() 
    b = torch.rand((2, 1, 60, 60)).cuda() 

    loss = nn.MSELoss()
    model = make_model(args).cuda()

    result = model(a)
    Loss = loss(result, b)
    Loss.backward()

