import os
import math
from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import pdb
import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, logger_test, vis):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.logger_test = logger_test
        self.vis = vis

    def train(self):
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')

        if self.args.test_only:# 如果只是测试的话，不需要以前的log  可以设置为0
            self.ckp.log = torch.zeros(1, len(self.loader_test) + 1)
        else:
            self.ckp.add_log(torch.zeros(1, len(self.loader_test)+1)) #最后增加一维用于保存平均值

        if self.args.data_test[0] == 'Demo':
            length_list = [1] # 不算数量
        else:
            length_list = [loader.dataset.count for loader in self.loader_test]
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for lr, hr, filename in tqdm(d, ncols=80):
                lr, hr = self.prepare(lr, hr)
                sr = self.model(lr)
                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                self.ckp.log[-1, idx_data] += utility.calc_psnr(
                    sr, hr, self.scale, self.args.rgb_range, dataset=d
                )
                if self.args.save_gt:
                    save_list.extend([lr, hr])

                if self.args.save_results:
                    self.ckp.save_results(d, filename[0], save_list, self.scale)

            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.name,
                    self.scale,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data]
                )
            )

        # 计算这一轮的平局值
        self.ckp.log[-1, -1] =  sum(self.ckp.log[-1,0:-1] * torch.tensor(length_list).float()) / sum(length_list)
        best = self.ckp.log.max(0)
        
        self.ckp.write_log(
            '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                'Average',
                self.scale,
                self.ckp.log[-1, -1], # 最后一个记录的最后一个数据
                best[0][-1],
                best[1][-1]
            )
        )

        # For visdom
        if self.logger_test:
            Y = self.ckp.log
            X = torch.arange(0, len(Y))
            legends = [d.dataset.name for idx_data, d in enumerate(self.loader_test) ]
            legends.append('Average')
            self.vis.line(X=X, Y=Y, win=self.logger_test, opts=dict(legend=legends, title="PSNR"))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][-1] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs

