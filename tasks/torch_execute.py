import taskflow as tsk
from taskflow import task_definition, backend

from tasks import train_utils, proto_data
from tasks import rank_scores
from tasks.torch_model import build_model_from_config

import torch as th
from torch_geometric.data import DataLoader

import numpy as np
import time


class TrainLogger:

    def __init__(self):
        pass

    def start_epoch(self, epoch_id):
        self._epoch_start = time.time()

    def end_epoch(self, epoch_id):
        print("Epoch %d Time %f" % (epoch_id, (time.time() - self._epoch_start)))

    def iteration(self, it, train_loss, val_loss=None, val_scores={}):
        txt = "Iteration %d Train Loss %f" % (it, train_loss)
        if val_loss is not None:
            txt = txt + (" Val Loss %f" % val_loss)
        for k, v in val_scores.items():
            txt = txt + " Val %s %f" % (k, v)
        print(txt)


class Rank_BCE(th.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x.reshape([-1])
        y = y.reshape([-1])

        filt = th.abs(2 * y - 1)

        neg_abs = - x.abs()
        loss = filt * (x.clamp(min=0) - x * y + (1 + neg_abs.exp()).log())
        return loss.mean()


def select_loss(loss_type):

    if loss_type == 'rank_bce':
        return Rank_BCE(), True

    raise ValueError("Unknown loss function %s." % loss_type)


def select_score(type):
    return rank_scores.select_score(type, {}, {})


def score(scores, pred, target, classes=15):
    labels = ['%d' % i for i in range(classes)]
    avg_score = {k: [] for k in scores}
    score_func = {k: select_score(k) for k in scores}
    for i in range(pred.shape[0]):
        p = train_utils.get_ranking(pred[i, :], labels)
        t = train_utils.get_ranking(target[i, :], labels)
        for k in scores:
            avg_score[k].append(score_func[k](p, t))

    return {k: th.tensor(avg_score[k]) for k in scores}


def tensor_len(classes):
    if isinstance(classes, int):
        n = classes
    else:
        n = len(classes)
    return int(n * (n + 1) / 2)


def setup_optimizer(config, model):
    optim_type = config['type']
    del config['type']

    if optim_type.lower() == 'adam':
        return th.optim.Adam(model.parameters(), **config)

    if optim_type.lower() == 'rmsprop':
        return th.optim.RMSprop(model.parameters(), **config)

    raise ValueError("Unknown optimizer %s" % optim_type)


def setup_sheduler(config, optim, model):
    scheduler = None
    if 'scheduler' in config:
        scheduler = th.optim.lr_scheduler.StepLR(
            optim, **config['scheduler']
        )
    return scheduler


def setup_optim_sheduler(config, model):
    optim = setup_optimizer(config['optimizer'], model)
    scheduler = setup_sheduler(config, optim, model)
    return optim, scheduler


def train_iteration(model, batch, loss_func, optimizer):
    optimizer.zero_grad()
    out = model(batch)
    loss = loss_func(out, batch.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_model(model, validate_loader, loss_func, device,
                   valid_scores=None, logit=False, classes=15):
    model.eval()
    val_loss = []
    val_scores = {}
    if valid_scores:
        val_scores = {k: [] for k in valid_scores}

    for batch in validate_loader:
        batch = batch.to(device)
        out = model(batch)
        loss = loss_func(out, batch.y)
        val_loss.extend(
            [loss.item()]*batch.num_graphs
        )

        if valid_scores:
            if logit:
                out = th.sigmoid(out)
            pred = th.round(out)
            sc = score(valid_scores, pred, batch.y, classes=classes)
            for k, a in sc.items():
                val_scores[k].append(a)

    if valid_scores:
        for k in list(val_scores.keys()):
            ts = val_scores[k]
            ts = th.cat(ts)
            val_scores[k] = ts.mean().item()

    model.train()
    return np.mean(val_loss), val_scores


def train_model(config, model, train_loader, device, valid_loader=None,
                validate_step=10, valid_scores=None):
    model.train()

    logger = TrainLogger()
    loss_func, logit = select_loss(config['loss'])
    optimizer, sheduler = setup_optim_sheduler(config, model)

    for ep in range(config['epoch']):
        logger.start_epoch(ep)

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            train_loss = train_iteration(model, batch, loss_func,
                                         optimizer)
            valid_loss = None
            scores = {}
            if valid_loader is not None and i % validate_step == 0:
                valid_loss, scores =\
                        validate_model(model, valid_loader, loss_func, device,
                                       valid_scores, logit,
                                       len(config['tools']))
                logger.iteration(i, train_loss, valid_loss, scores)

        logger.end_epoch(ep)
        if sheduler is not None:
            sheduler.step()

    return model, logit


def test_model(config, model, test_loader, device):
    model.eval()
    test_scores = config['scores']
    classes = len(config['tools'])
    test_scoring = {k: [] for k in test_scores}

    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)

        if config['logit']:
            out = th.sigmoid(out)
        pred = th.round(out)
        sc = score(test_scores, pred, batch.y, classes=classes)
        for k, a in sc.items():
            test_scoring[k].append(a)

    for k in list(test_scoring.keys()):
        ts = test_scoring[k]
        ts = th.cat(ts)
        test_scoring[k] = {
            'mean': ts.mean().item(),
            'std': ts.std().item()
            }

    return test_scoring


# Configuration:
#  {
#   'training': {'epoch': 10, 'batch': 32, 'shuffle':True}
#   }
#
#
@task_definition()
def execute_model(tools, config, dataset_path, env=None):

    train_dataset = proto_data.BufferedDataset(
                        proto_data.GraphDataset(dataset_path, 'train',
                                                shuffle=True))

    validate_size = int(len(train_dataset) * config['training']['validate'])
    valid_dataset = train_dataset[:validate_size]
    train_dataset = train_dataset[validate_size:]

    train_config = config['training']

    loader_config = {
        'shuffle': train_config['shuffle'],
        'batch_size': train_config['batch'],
        'num_workers': 6
    }

    model, out_channels = build_model_from_config(config)

    final = th.nn.Linear(out_channels, tensor_len(tools))
    model = th.nn.Sequential(model, final)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_loader = DataLoader(train_dataset, **loader_config)
    val_loader = DataLoader(valid_dataset, batch_size=32, num_workers=6)

    train_config['tools'] = tools
    # Train model
    model, logit = train_model(train_config, model, train_loader, device,
                               valid_loader=val_loader, validate_step=100,
                               valid_scores=config['testing']['scores'])

    test_dataset = proto_data.GraphDataset(dataset_path, 'test')
    test_config = config['testing']
    test_config['tools'] = tools
    test_config['logit'] = logit
    eval = test_model(test_config,
                      model,
                      DataLoader(test_dataset, **loader_config),
                      device)
    for k, v in eval.items():
        print("Test %s: %f (std: %f)" % (k, v['mean'], v['std']))


def build_model(config):
    tools, train, test = train_utils.get_svcomp_train_test(
        **config['dataset']
    )
    dataset = proto_data.download_lmdb(
        tools, config['dataset']['competition'],
        train, test, category=config['dataset']['category'],
        ast_bag=True
    )
    del config['dataset']
    return execute_model(tools, config, dataset)


if __name__ == '__main__':
    config = {
        'key': 'test_0',
        'dataset': {
            'key': '2019_reachability_all_10000',
            'competition': '2019',
            'category': 'reachability',
            'test_ratio': 0.2,
            'min_tool_coverage': 0.8
        },
        'training': {
            'epoch': 100,
            'batch': 32,
            'shuffle': True,
            'loss': 'rank_bce',
            'validate': 0.1,
            'optimizer': {
                'type': 'adam', 'lr': 0.001, 'weight_decay': 0.00001
            },
            'scheduler': {
                'step_size': 20,
                'gamma': 0.5
            }
        },
        'testing': {
            'scores': ['spearmann']
        }
    }
    train_test = build_model(config)
    with backend.openLocalSession() as sess:
        sess.run(train_test)
