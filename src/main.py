import torch
import pdb
import utility
import data
import model
import loss
import skimage
from option import args

import numpy as np
import sys
from importlib import import_module

# random seed
torch.manual_seed(args.seed) # cpu
torch.cuda.manual_seed(args.seed) #gpu
torch.backends.cudnn.deterministic=True # cudnn

print(args)

module = import_module(args.trainer)
Trainer = getattr(module, 'Trainer')

checkpoint = utility.checkpoint(args)

if args.visdom:  
    # use visdom for visualization
    from visdom import Visdom
    env_name = 'SR_code_' + args.save + '_' + str(args.patch_size) + 'x' + str(args.scale)
    vis = Visdom(port=8097, server="http://localhost", env=env_name)
    if len(checkpoint.log) > 0:
        logger_test =vis.line(X=torch.arange(0, len(checkpoint.log)), Y=checkpoint.log[:,:, 0], opts=dict(title="PSNR"))
    else:
        logger_test = vis.line(np.arange(50), opts=dict(title="PSNR"))
    if args.onlydraw:
        sys.exit() 
else:
    logger_test = None
    vis = None

if checkpoint.ok:
    loader = data.Data(args) # load dataset
    model = model.Model(args, checkpoint) # call the Model function that defined in model.__init__
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint, logger_test, vis)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

