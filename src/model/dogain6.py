# DoGAIN
# from model import common, ain
import common, ain
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return DoGAIN(args)


class ImDecomposition(nn.Module):
    def __init__(self, args):
        super(ImDecomposition, self).__init__()
        self.G = common.GaussianConv2d(kernel_size=5)
    
    def forward(self, x):
        # x1 = self.G(x, torch.tensor(1))
        # x2 = self.G(x, torch.tensor(2))
        # x3 = self.G(x, torch.tensor(4))
        x4 = self.G(x, torch.tensor(8))
        return x-x4, x4


class myBatchNorm2d(nn.BatchNorm2d):
    def __init__(self):
        super(myBatchNorm2d, self).__init__(1)

    def go(self, x):
        
    
    def inverse(self, x):
        x = x / (self.eps + self.running_var) + self.running_mean
        return x


class DoGAIN(nn.Module):
    def __init__(self, args):
        super(DoGAIN, self).__init__()

        self.rgb_range = args.rgb_range
        self.AIN_1 = ain.AIN(args)
        # self.AIN_2 = ain.AIN(args)
        # self.AIN_3 = ain.AIN(args)
        self.decomp = ImDecomposition(args)

        self.bnlayer = myBatchNorm2d()

    def forward(self, x):
        x_h, x_l = self.decomp(x)

        # import pdb; pdb.set_trace()

        x_h = self.bnlayer(x)
        x_h = self.AIN_1(x_h)
        x_h = self.bnlayer.inverse(x)

        # x_h = x_h * h_std + h_mean

        # l_mean, l_std = 127.5, 127.5
        # x_l = (x_l - l_mean) / l_std
        # x_l = self.AIN_3(x_l)
        # x_l = x_l * l_std + l_mean

        return x_h + x_l
    

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


if __name__ == '__main__':
    import argparse
    import torch
    parser = argparse.ArgumentParser(description='AIN')
    parser.add_argument('--model', default='AIN',
                        help='model name')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    parser.add_argument('--n_resblocks', type=int, default=7,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--n_resgroups', type=int, default=1,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    parser.add_argument('--scale', type=int, default=2,
                        help='super resolution scale')
    parser.add_argument('--n_colors', type=int, default=1,
                        help='n colors of input')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
    args = parser.parse_args()

    args.scale = [2]

    model = make_model(args)

    a = torch.rand((3, 1, 40, 40)) * 255

    result = model(a)

    print(result)

