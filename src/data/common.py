# import random
import numpy as np
import skimage.color as sc
import pdb
import torch

def crop_for_scale(im, scale=2):
    if len(im.shape) > 2:
        [ih, iw, id] = im.shape
        ih = ih - ih % scale
        iw = iw - iw % scale
        output=np.zeros([ih, iw, id])
        for i in range(id):
            output[:,:,i] = im[0:ih,0:iw,i]
    else:
        [ih, iw] = im.shape
        ih = ih - ih % scale
        iw = iw - iw % scale
        output = im[0:ih,0:iw]
    return output

def directdownsample(im, scale=2):
    if len(im.shape) > 2:
        [ih, iw, id] = im.shape
        r_i = np.arange(0, ih, scale)
        c_i = np.arange(0, iw, scale)
        im = im[r_i, :, :]
        im = im[:, c_i, :]        
    else:
        [ih, iw] = im.shape
        r_i = np.arange(0, ih, scale)
        c_i = np.arange(0, iw, scale)
        im = im[r_i, :]
        im = im[:, c_i]
    return im


def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix= np.random.randint(iw - ip + 1)
    iy = np.random.randint(ih- ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
            img = np.round(img)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and np.random.rand() < 0.5
    vflip = rot and np.random.rand() < 0.5
    rot90 = rot and np.random.rand() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(a) for a in args]
    



