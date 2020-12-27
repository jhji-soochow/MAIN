# This AIN use FReLU to replace ReLU
from model import common
# import common
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
                common.FReLU(channel // reduction),
                # nn.ReLU(inplace=True),
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
            common.FReLU(channel // 2),
            # nn.ReLU(inplace=True),
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
        bias=True, bn=False, act=None, res_scale=1):

        super(AFE, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(common.FReLU(n_feat)) # 改成FReLU
        
        self.front = nn.Sequential(*modules_body)
        self.CA = CALayer(n_feat, reduction)
        self.SA = SALayer(n_feat)
        self.act = common.FReLU(n_feat) # 改成FReLU
        self.res_scale = res_scale
        self.fushion = nn.Conv2d(n_feat * 3, n_feat, 1)

    def forward(self, x):
        x_ = self.front(x)
        ca = self.CA(x_)
        sa = self.SA(x_)
        att = torch.cat((x_, ca, sa), 1)
        res = self.fushion(att)
        res += x
        #res = self.act(res)
        return res


class AIB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, 
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(AIB, self).__init__()
            
        self.scale1_1 = nn.Conv2d(n_feat, n_feat, 3, padding=1)
        self.scale2_1 = nn.Conv2d(n_feat, n_feat, 5, padding=2)
        self.act = common.FReLU(n_feat) # 改成FReLU

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


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            AIB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Attention-aware Inception Network (AIN)
class AIN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(AIN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        act = None
        # act = common.FReLU()
        self.act = act
        # RGB mean for DIV2K
        #self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = common.MyUpsampler(conv, scale, n_feats, act=None)

    def forward(self, x):
        #x = self.sub_mean(x)
        # import pdb; pdb.set_trace()
        x_ = x
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res, x_)
        #x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


# if __name__ == '__main__':
#     import argparse
#     import torch
#     parser = argparse.ArgumentParser(description='RCSAN')
#     parser.add_argument('--model', default='RCSAN',
#                         help='model name')
#     parser.add_argument('--chop', action='store_true',
#                         help='enable memory-efficient forward')
#     parser.add_argument('--n_resblocks', type=int, default=16,
#                         help='number of residual blocks')
#     parser.add_argument('--n_feats', type=int, default=64,
#                         help='number of feature maps')
#     parser.add_argument('--n_resgroups', type=int, default=10,
#                         help='number of residual groups')
#     parser.add_argument('--reduction', type=int, default=16,
#                         help='number of feature maps reduction')
#     parser.add_argument('--scale', type=int, default=2,
#                         help='super resolution scale')
#     parser.add_argument('--n_colors', type=int, default=1,
#                         help='n colors of input')
#     parser.add_argument('--res_scale', type=float, default=1,
#                         help='residual scaling')
#     args = parser.parse_args()

#     if args.model == 'RCSAN':
#         args.model = 'RCSAN'
#         args.n_resgroups = 1
#         args.n_resblocks = 8
#         args.n_feats = 64
#         args.chop = True


#     model = make_model(args)

#     a = torch.rand((1, 1, 40, 40))
#     preintp_a = torch.rand((1, 1, 80, 80))
    
#     result = model(a, preintp_a)

