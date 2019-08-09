import torch as th
import numpy as np

from tasks.torch.train import Scorer, DatasetOp
from torch_geometric.data import DataLoader

import time


class TestScorer:

    def __init__(self, scorer, data_op):
        self.scorer = scorer
        self.data_op = data_op

    def __call__(self, tools, model, dataset_path):
        start_time = time.time()
        model.eval()

        loader = self.data_op(dataset_path)
        loader = DataLoader(
            loader, batch_size=32,
            shuffle=False, num_workers=6
        )

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        test_scoring = {}

        size = 0

        for batch in loader:
            batch = batch.to(device)
            size += batch.num_graphs
            out = model(batch)

            sc = self.scorer(out, batch.y, classes=len(tools))

            for k, a in sc.items():
                if k not in test_scoring:
                    test_scoring[k] = []
                test_scoring[k].append(a)

        for k in list(test_scoring.keys()):
            ts = test_scoring[k]
            ts = th.cat(ts)
            test_scoring[k] = {
                'mean': ts.mean().item(),
                'std': ts.std().item()
                }

        test_scoring['test_time'] = (time.time() - start_time)
        test_scoring['num_graphs'] = size

        return test_scoring


class CategoryScorer:

    def __init__(self, scorer, data_op):
        self.scorer = scorer
        self.data_op = data_op

    def __call__(self, tools, model, dataset_path):
        start_time = time.time()
        model.eval()

        loader = self.data_op(dataset_path)
        loader = DataLoader(
            loader, batch_size=32,
            shuffle=False, num_workers=6
        )

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        cat_name = ['reachability', 'termination',
                    'memory', 'overflow']

        test_scoring = {}

        size = 0

        for batch in loader:
            batch = batch.to(device)
            size += batch.num_graphs
            out = model(batch)

            cat = batch.category.detach().numpy()
            cat = np.array([np.where(r == 1)[0][0] for r in cat])

            sc = self.scorer(out, batch.y, classes=len(tools))

            for k, a in sc.items():
                if k not in test_scoring:
                    test_scoring[k] = []
                test_scoring[k].append(a)

            for k, a in sc.items():
                for i, c in enumerate(cat_name):
                    key = k + "_" + c
                    if key not in test_scoring:
                        test_scoring[key] = []
                    ci = np.where(cat == i)[0]
                    test_scoring[key].append(a[ci])

        for k in list(test_scoring.keys()):
            ts = test_scoring[k]
            ts = th.cat(ts)
            test_scoring[k] = {
                'mean': ts.mean().item(),
                'std': ts.std().item()
                }

        test_scoring['test_time'] = (time.time() - start_time)
        test_scoring['num_graphs'] = size

        return test_scoring


def build_test(config, data_key='test'):

    if not isinstance(config, dict):
        config = {'type': 'score', 'scores': config}

    type = config['type']
    del config['type']

    data_op = DatasetOp(data_key, False)
    config['data_op'] = data_op

    if type == 'score':
        return TestScorer(
            Scorer(**config)
        )

    if type == 'category':
        return CategoryScorer(
            Scorer(**config)
        )
