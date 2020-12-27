import torch
import torch.nn as nn
from rcan import make_model

class FusionNet(nn.Module):
    def __init__(self, args):
        super(FusionNet, self).__init__()
        self.models = nn.ModuleList([make_model(args) for i in range(3)])
        modelname = 'RCAN_BIX2_G10R20_1000'
        ckpt_dir = ['../experiment/' + modelname + '/model/model_best.pt',
                    '../experiment/' + modelname + '136/model/model_best.pt',
                    '../experiment/' + modelname + '192/model/model_best.pt']
        self.state_dicts = [torch.load(ckpt_dir[i]), map_location='cpu') for i in range(3)]     
        self.load_model()
        
    def load_model(self):
        for i in range(3):
            self.models[i].load_state_dict(self.state_dicts[i]['state_dict'])
    
    def forward(self, x):
        y = []
        for i in range(3):
            y.append(self.models[i](x))
        y = torch.cat(y, 1)
        y = torch.mean(y, 1, keepdim=True)
        return y