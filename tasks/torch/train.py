import numpy as np
import torch as th
from torch import optim
from torch.optim import lr_scheduler
from torch import nn
from tasks.torch import loss
from torch_geometric.data import DataLoader
import copy
import random
import math

from tasks.utils import train_utils, rank_scores, element_scores
from tasks.data import proto_data
from tasks.torch.graph import build_graph


class SuperConverge(lr_scheduler.LambdaLR):

    def __init__(self, optimizer, d_model, warmup, last_epoch=-1):
        self.mult = 1/math.sqrt(d_model)
        self.warmup = 1/math.sqrt(warmup**3)

        super().__init__(optimizer, lambda x: self.calc_lr(x), last_epoch)

    def calc_lr(self, num_steps):
        num_steps = num_steps + 1
        return self.mult * min(1/math.sqrt(num_steps), num_steps*self.warmup)


def prepare_inst_dict(D, keys):

    for k in keys:
        if k in D:
            val = D[k]
            if not isinstance(val, dict):
                D[k] = {'type': val}


def instantiate_obj(mod, type, config):

    if type in dir(mod):
        module = getattr(mod, type)
        return module(**config)

    raise ValueError("Unknown obj \"%s\" in module \"%s\"" % (type, str(mod)))


def instantiate_loss(type, config):

    if type.startswith('torch::'):
        type_ = type[7:]
        return instantiate_obj(nn, type_, config)

    if type.startswith('masked::'):
        type_ = type[8:]

        cfg = copy.copy(config)
        cfg['reduction'] = 'none'

        inner_loss = instantiate_obj(loss, type_, cfg)

        sup_cfg = {'loss': inner_loss}

        if 'reduction' in config:
            sup_cfg['reduction'] = config['reduction']

        try:
            return instantiate_obj(
                loss, 'MaskedLoss', sup_cfg
            )
        except ValueError:
            return instantiate_obj(
                nn, 'MaskedLoss', sup_cfg
            )

    if type.startswith('tasks::'):
        type_ = type[7:]
        return instantiate_obj(loss, type_, config)

    raise ValueError("Unknown loss: %s" % type)


def instantiate_scheduler(type, config):
    sched = None
    mode = 'epoch'

    if 'mode' in config:
        mode = config['mode']
        del config['mode']

    if type == 'tasks::SuperConverge':
        sched = SuperConverge(**config)

    if type.startswith('torch::'):
        type_ = type[7:]
        mod = lr_scheduler

        sched = instantiate_obj(mod, type_, config)

    if sched is not None:
        return {'obj': sched, 'mode': mode}

    raise ValueError("Unknown type: %s" % type)


def instantiate_optim(type, config):

    if type.startswith('torch::'):
        type_ = type[7:]
        mod = optim

        return instantiate_obj(mod, type_, config)

    raise ValueError("Unknown type: %s" % type)


def instantiate_config(config, model):

    config = copy.deepcopy(config)

    prepare_inst_dict(config, ['loss', 'optimizer', 'scheduler'])

    for i, l in enumerate(['loss', 'optimizer', 'scheduler']):
        if l in config:
            cfg = config[l]
            type = cfg['type']
            del cfg['type']

            if i == 0:
                config[l] = instantiate_loss(type, cfg)

            if i == 1:
                cfg['params'] = model.parameters()
                config[l] = instantiate_optim(type, cfg)

            if i == 2 and 'optimizer' in config:
                cfg['optimizer'] = config['optimizer']
                config[l] = instantiate_scheduler(type, cfg)

    return config


def inject_tap(S):
    return "\n".join([
        "\t%s" % sp for sp in S.split("\n")
    ])


def to_str(obj, name, keys, alt={}):
    D = {
        k: getattr(obj, k)
        for k in keys if hasattr(obj, k)
    }

    D.update(alt)

    inner = [
        inject_tap(
            "(%s) %s" % (k, str(v))
        ) for k, v in D.items() if v is not None
    ]
    inner = '\n'.join(inner)
    return "%s (\n %s \n)" % (name, inner)


class ModelOptimizer:

    def __init__(self, model, optim,
                 loss, scheduler=None):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.scheduler = scheduler

    def train(self, X, y):
        self.model.train()
        self.optim.zero_grad()

        out = self.model(X)
        loss = self.loss(out, y)
        loss.backward()
        self.optim.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss

    def to(self, device):
        self.model = self.model.to(device)

    def __str__(self):
        return to_str(
            self, 'ModelOptimizer', [
                'model', 'optim', 'loss', 'scheduler'
            ]
        )


def rank_preprocessor(score):

    def inner_score(pred, target, labels):
        p = train_utils.get_ranking(pred, labels)
        t = train_utils.get_ranking(target, labels)
        return score(p, t)

    return inner_score


def element_preprocessor(score, pos):

    def inner_score(pred, target, labels):
        p = pred[pos]
        t = target[pos]
        return score(p, t)

    return inner_score


def select_score(type):

    if ':' in type:
        t, p = type.split(':')

        try:
            score = element_scores.select_score(t)
            pos = int(p)
            return element_preprocessor(score, pos)
        except ValueError:
            pass

    rank_score = rank_scores.select_score(type, {}, {})
    return rank_preprocessor(rank_score)


class Scorer:

    def __init__(self, scores):
        if not isinstance(scores, list):
            scores = [scores]

        self.scores = scores
        self.score_func = {k: select_score(k) for k in scores}

    def __call__(self, pred, target, classes=15):
        pred = th.sigmoid(pred)
        labels = ['%d' % i for i in range(classes)]
        avg_score = {k: [] for k in self.scores}
        score_func = self.score_func
        for i in range(pred.shape[0]):
            p = pred[i, :]
            t = target[i, :]
            for k in self.scores:
                avg_score[k].append(score_func[k](p, t, labels))

        R = {k: th.tensor(avg_score[k], dtype=th.float) for k in self.scores}
        return R


def build_check_pos(step):

    if step <= -1:
        def all(pos):
            return True
        return all

    if not isinstance(step, list):
        step = [step]
    step = set(step)

    def check_pos(pos):
        return pos in step

    return check_pos


class Validator:

    def __init__(self, score, split, checkpoint_step=-1):
        self.split = split
        self.checkpoint_step = checkpoint_step
        self.checkpoint = build_check_pos(checkpoint_step)
        self.score = score
        self.scorer = Scorer(score)

    def reduce_dataset(self, dataset):
        validate_size = int(len(dataset) * self.split)
        self.dataset = dataset[:validate_size]
        self.dataset = DataLoader(self.dataset, batch_size=32, num_workers=6)
        return dataset[validate_size:]

    def __call__(self, model, device, step, classes=15, loss_func=None):
        if self.checkpoint(step):
            model.eval()
            val_loss = []
            val_score = []

            for batch in self.dataset:
                batch = batch.to(device)
                out = model(batch)
                if loss_func is not None:
                    loss = loss_func(out, batch.y)
                    val_loss.extend(
                        [loss.item()]*batch.num_graphs
                    )

                valid_score = self.scorer(out, batch.y, classes=classes)
                valid_score = valid_score[self.score]
                val_score.append(
                    valid_score
                )

            val_score = th.cat(val_score).mean().item()
            if loss_func is not None:
                return val_score, np.mean(val_loss)
            return val_score

    def __str__(self):
        return to_str(self, 'Validator',
                      ['split', 'score', 'checkpoint_step'])


def instantiate_validator(config):
    return Validator(**config)


def graph_augmention(options=[0, 1, 2, 5, 6]):

    def select_filter(option):
        select = {
            1: [0],
            2: [1, 2],
            3: [0, 2],
            4: [0, 1],
            5: [2],
            6: [1]
        }
        return select[option]

    def augment(data):
        option = random.choice(options)

        if option == 0:
            return data

        filter = select_filter(option)
        edge_attr = data.edge_attr.numpy()
        edge_attr_t = edge_attr.transpose()
        edge_index = data.edge_index.numpy()

        pos = []
        for f in filter:
            pos.append(
                np.where(edge_attr_t[f] == 1)[0]
            )
        pos = np.hstack(pos)
        edge_attr = edge_attr[pos, :]
        edge_index = edge_index[:, pos]
        data.edge_index = th.tensor(edge_index)
        data.edge_attr = th.tensor(edge_attr)
        return data

    return augment


def dataset_transform(augment):

    transform = None
    options = [0, 1, 2, 5, 6]
    if isinstance(augment, list):
        options = augment
        augment = True
    if augment:
        transform = graph_augmention(options)

    return transform


class DatasetOp:

    def __init__(self, key, shuffle, augment=False):
        self.key = key
        self.shuffle = shuffle
        self.augment = augment

    def __call__(self, dataset_path):
        return proto_data.GraphDataset(
            dataset_path, self.key, shuffle=self.shuffle,
            transform=dataset_transform(
                self.augment
            )
        )

    def __str__(self):
        return to_str(self, 'DatasetOp', ['key', 'shuffle', 'augment'])


def print_log(epoch, iteration, loss, val_loss=None, val_score=None):

    if val_loss is not None:
        print("Epoch %i:%i Train Loss %f Val Loss %f Val Score %f" %
              (epoch, iteration, loss, val_loss, val_score))
        return
    print("Epoch %i:%i Train Loss %f" % (epoch, iteration, loss))


class ModelTrainer:

    def __init__(self, epoch, batch, optimizer,
                 dataset_op, shuffle=True, validate=None,
                 scheduler=None):
        self.epoch = epoch
        self.batch = batch
        self.optimizer = optimizer
        self.dataset_op = dataset_op
        self.validate = validate
        self.scheduler = scheduler
        self.shuffle = shuffle

    def _prepare(self, dataset_path):
        dataset = self.dataset_op(dataset_path)
        if self.validate is not None:
            dataset = self.validate.reduce_dataset(
                dataset
            )
        return DataLoader(
            dataset, batch_size=self.batch,
            shuffle=self.shuffle, num_workers=6
        )

    def train_iter(self, tools, dataset_path):

        loader = self._prepare(dataset_path)

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.optimizer.to(device)

        for ep in range(self.epoch):

            for i, batch in enumerate(loader):
                batch = batch.to(device)
                train_loss = self.optimizer.train(
                    batch, batch.y
                )

                if self.validate is not None:
                    res = self.validate(
                        self.optimizer.model,
                        device, i,
                        loss_func=self.optimizer.loss,
                        classes=len(tools)
                    )
                    if res is not None:
                        val_score, val_loss = res
                        yield ep, i, train_loss, val_loss, val_score
                        continue
                yield ep, i, train_loss, None, None

            if self.scheduler is not None:
                self.scheduler.step()

    def train(self, tools, dataset_path, log=print_log):

        for L in self.train_iter(tools, dataset_path):
            log(*L)

        return self.optimizer.model

    def __str__(self):
        cfg = {
            'epoch': self.epoch,
            'batch': self.batch,
            'shuffle': self.shuffle
        }
        return to_str(self, 'ModelTrainer',
                      ['optimizer', 'validate', 'dataset_op',
                       'scheduler'],
                      alt={'config': cfg})


def build_training(config, model):

    config = instantiate_config(config, model)

    if 'scheduler' not in config:
        config['scheduler'] = {'mode': 'epoch', 'obj': None}

    step_scheduler = None
    epoch_scheduler = None

    mode = config['scheduler']['mode']
    if mode == 'epoch':
        epoch_scheduler = config['scheduler']['obj']
    else:
        step_scheduler = config['scheduler']['obj']

    optimizer = ModelOptimizer(
        model, config['optimizer'], config['loss'],
        scheduler=step_scheduler
    )
    del config['optimizer'], config['loss'], config['scheduler']

    config['optimizer'] = optimizer

    if 'validate' in config:
        config['validate'] = instantiate_validator(
            config['validate']
        )

    if 'shuffle' not in config:
        config['shuffle'] = True
    if 'augment' not in config:
        config['augment'] = False

    config['dataset_op'] = DatasetOp(
        'train', config['shuffle'], config['augment']
    )
    del config['augment']

    config['scheduler'] = epoch_scheduler

    return ModelTrainer(**config)
