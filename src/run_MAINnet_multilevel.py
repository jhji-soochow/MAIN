from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from os.path import exists, join
from model.ain import make_model
from time import time
from math import log10
from os import listdir, makedirs
from data.common import crop_for_scale, directdownsample
from utility import quantize
import math


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ["bmp", ".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = np.float32(np.array(Image.open(filepath)))
    if len(img.shape) > 2:
        if (img[:,:,0] - img[:,:,1]).sum() == 0 and (img[:,:,0] - img[:,:,2]).sum() == 0 and (img[:,:,1] - img[:,:,2]).sum() == 0:
            img = img[:,:,0]
    return img


def preprocess(img, scale):
    hr = crop_for_scale(img, scale)
    lr = directdownsample(hr, scale)
    return lr, hr


def postprocess(lr, preds, scale=2):
    
    post_pred = []
    for i in range(6):
        pred = preds[i].squeeze(0).squeeze(0).cpu()
        pred = quantize(pred, rgb_range=1).numpy()

        for h in range(pred.shape[0]):
            for w in range(pred.shape[1]):
                if np.abs(lr[h][w] - pred[2*h][2*w]) < 1:
                    pred[2*h][2*2] = lr[h][w]
                else:
                    print("error")
    
        post_pred.append(pred)
    
    return post_pred


def calc_psnr(sr, hr, shave=0, rgb_range=255):
    diff = (hr - sr)  / rgb_range
    if shave == 0:
        valid = diff
    else:
        valid = diff[shave:-shave, shave:-shave]

    mse = np.mean(np.power(valid, 2))
    return -10 * math.log10(mse)


class MAIN(nn.Module):
    def __init__(self, args):
        super(MAIN, self).__init__()

        self.modelnumber = args.modelscale

        model_list = []
        for i in range(self.modelnumber):
            args.n_resblocks = i + 1
            model = make_model(args)
            model_list.append(model)

        self.models = nn.ModuleList(model_list)

        ckpt_dir = '../experiment/AIN'

        # pc_96_MSARN7_epoch400_30.65.pth

        if self.modelnumber == 8:
            self.state_dicts = [
                                torch.load(ckpt_dir + '1_96x2/' + 'model/model_latest.pt', map_location='cpu'),
                                torch.load(ckpt_dir + '2_96x2/' + 'model/model_latest.pt', map_location='cpu'),
                                torch.load(ckpt_dir + '3_96x2/' + 'model/model_latest.pt', map_location='cpu'),
                                torch.load(ckpt_dir + '4_96x2/' + 'model/model_latest.pt', map_location='cpu'),
                                torch.load(ckpt_dir + '5_96x2/' + 'model/model_latest.pt', map_location='cpu'),
                                torch.load(ckpt_dir + '6_96x2/' + 'model/model_latest.pt', map_location='cpu'),
                                torch.load(ckpt_dir + '_96x2/' + 'model/model_latest.pt', map_location='cpu'),
                                torch.load(ckpt_dir + '8_96x2/' + 'model/model_latest.pt', map_location='cpu')]
           
        self.load_model()
        
    def load_model(self):

        for i in range(self.modelnumber):
            self.models[i].load_state_dict(self.state_dicts[i])
    
    def forward(self, x):
        y = []
        for i in range(self.modelnumber):
            out = self.models[i](x)
            y.append(out)
        output = torch.cat(y, 1)
        output = torch.mean(output, 1, keepdim=True)
        y.append(output)
        return y



def test(args, model=None, use_cuda=True):
    testdir = args.testdir    
    modelnumber = args.modelscale

    print("Test MAIN on dataset:", testdir)


    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        image_filenames = [x for x in listdir(testdir) if is_image_file(x)]

        PSNRs = []

        for imname in image_filenames:
            gt = load_img(join(testdir, imname))
            lr, hr = preprocess(gt, scale=2)
            input_img = torch.from_numpy(lr).unsqueeze_(0).unsqueeze_(0)

            if use_cuda:
                input_img = input_img.cuda()

            prediction = model(input_img)

            # pred = postprocess(lr, prediction)
            psnrs = np.zeros(modelnumber + 1)
            for i in range(modelnumber + 1):
                pred = quantize(prediction[i], rgb_range=255).cpu().squeeze(0).squeeze(0).numpy()
                psnrs[i] = calc_psnr(pred, hr, shave=0, rgb_range=255)  # scale = shave

                # save ?
            print(imname, ':',psnrs)
            PSNRs.append(psnrs)

        print("Average PSNRs")
        print(np.mean(PSNRs, 0))

                
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run MAIN')

    # model init
    parser.add_argument('--n_resgroups', type=int, default=1,
                    help='number of residual groups')
    # parser.add_argument('--n_resblocks', type=int, default=7,
    #                 help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
    parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
    parser.add_argument('--modelscale', type=int, default=8,
                    help='scale of models')

    parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
    
    
    parser.add_argument('--testdir', type=str, default='/data/jjh_backup/1_3/testset/Set18',
                    help='dataset directory')
    parser.add_argument('--scale', type=int, default=[2],
                    help='super resolution scale')
    parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
    
    parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
    
    parser.add_argument('--save', action='store_true',
                    help='whether to save result')

    args = parser.parse_args()

    ## hyperparameters
    use_cuda = not args.cpu and torch.cuda.is_available()

    model = MAIN(args)

    test(args, model=model, use_cuda=use_cuda)

    
