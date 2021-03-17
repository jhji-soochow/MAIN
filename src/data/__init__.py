from importlib import import_module
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import pdb

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower()) # 动态导入
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['mySet5', 'mySet15', 'mySet18', 'myBSDS100', 'myUrban12', 'myUrban100', 'myManga109']:
                m = import_module('data.mybenchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)

            self.loader_test.append(DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=args.n_threads
            ))

