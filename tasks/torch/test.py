import torch as th

from tasks.torch.train import Scorer, DatasetOp
from torch_geometric.data import DataLoader


class TestScorer:

    def __init__(self, scorer):
        self.scorer = scorer
        self.data_op = DatasetOp('test', False)

    def __call__(self, tools, model, dataset_path):
        model.eval()

        loader = self.data_op(dataset_path)
        loader = DataLoader(
            loader, batch_size=32,
            shuffle=False, num_workers=6
        )

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        test_scoring = {}

        for batch in loader:
            batch = batch.to(device)
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

        return test_scoring


def build_test(config):

    if not isinstance(config, dict):
        config = {'type': 'score', 'scores': config}

    type = config['type']
    del config['type']

    if type == 'score':
        return TestScorer(
            Scorer(**config)
        )
