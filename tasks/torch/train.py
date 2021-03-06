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
from tqdm import tqdm


class SuperConverge(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, d_model, warmup, last_epoch=-1):
        self.mult = 1/math.sqrt(d_model)
        self.warmup = 1/math.sqrt(warmup**3)

        super().__init__(optimizer, last_epoch)

    def calc_lr(self, num_steps):
        num_steps = num_steps + 1
        return self.mult * min(1/math.sqrt(num_steps), num_steps*self.warmup)

    def get_lr(self):
        return [self.calc_lr(self.last_epoch)
                for base_lr in self.base_lrs]

# Copyied form Torch 1.2.0
class AdamW(th.optim.Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = th.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = th.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = th.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    th.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


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

    if type == "torch::AdamW":
        return AdamW(**config)

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
                 loss, scheduler=None, clip_grad=False):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.scheduler = scheduler
        self.clip_grad = clip_grad

    def resume(self, model, optim, scheduler):
        self.model.load_state_dict(model)
        self.optim.load_state_dict(optim)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler)

    def checkpoint(self):
        scheduler = None if self.scheduler is None else self.scheduler.state_dict()
        return {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': scheduler
        }

    def train(self, X, y):
        self.model.train()
        self.optim.zero_grad()

        out = self.model(X)
        loss = self.loss(out, y)
        loss.backward()

        clip_grad = self.clip_grad
        if clip_grad and clip_grad > 0:
            th.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

        self.optim.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss

    def to(self, device):
        self.model = self.model.to(device)
        self.loss = self.loss.to(device)

        for state in self.optim.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.to(device)

        if self.scheduler is not None:
            for state in self.scheduler.state.values():
                for k, v in state.items():
                    if isinstance(v, th.Tensor):
                        state[k] = v.to(device)

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

    def __call__(self, model, device, step, classes=15, loss_func=None,
                 norm=None):
        if self.checkpoint(step):
            model.eval()
            val_loss = []
            val_score = []

            if norm is None:
                def id(x):
                    return x
                norm = id

            for batch in self.dataset:
                batch = batch.to(device)
                batch.x = norm(batch.x)
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


def edge_subsample(size):

    def subsample(data):
        edge_index = data.edge_index
        if edge_index.size(1) <= size:
            return data

        sample = th.randint(1, edge_index.size(1), (size,))
        data.edge_index = data.edge_index[:, sample]

        return data

    return subsample


def dataset_transform(augment, subsample):

    transform = None
    options = [0, 1, 2, 5, 6]
    if isinstance(augment, list):
        options = augment
        augment = True
    if augment:
        transform = graph_augmention(options)

    if subsample and subsample > 0:
        sub = edge_subsample(subsample)
        if transform is not None:
            def chain(data):
                return sub(transform(data))
            transform = chain
        else:
            transform = sub

    return transform


class DatasetOp:

    def __init__(self, key, shuffle, augment=False, subsample=False):
        self.key = key
        self.shuffle = shuffle
        self.augment = augment
        self.subsample = subsample

    def __call__(self, dataset_path, buffer=False):
        if buffer:
            return proto_data.InMemGraphDataset(
                dataset_path, self.key, shuffle=self.shuffle,
                transform=dataset_transform(
                    self.augment, self.subsample
                )
            )
        return proto_data.GraphDataset(
            dataset_path, self.key, shuffle=self.shuffle,
            transform=dataset_transform(
                self.augment, self.subsample
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


def isnan(x):
    return th.isnan(x).any().item()


class ModelTrainer:

    def __init__(self, epoch, batch, optimizer,
                 dataset_op, shuffle=True, validate=None,
                 norm=False, buffer=False, scheduler=None):
        self.epoch = epoch
        self.batch = batch
        self.optimizer = optimizer
        self.dataset_op = dataset_op
        self.validate = validate
        self.scheduler = scheduler
        self.shuffle = shuffle
        self.norm = norm
        self.total_batches = -1
        self.last_epoch = 0
        self.buffer = buffer

    def _prepare(self, dataset_path):
        dataset = self.dataset_op(dataset_path, self.buffer)

        if self.validate is not None:
            dataset = self.validate.reduce_dataset(
                dataset
            )
        return DataLoader(
            dataset, batch_size=self.batch,
            shuffle=self.shuffle, num_workers=8
        )

    def _norm(self, loader, device):

        if not self.norm:
            def id(x):
                return x
            return id

        cnt = th.zeros(1, dtype=th.double)
        fst_moment = None
        snd_moment = None

        for i, batch in tqdm(enumerate(loader)):
            self.total_batches = max(self.total_batches, i)
            batch = batch.to(device)

            x = batch.x.double()
            dim = x.size(1)

            if fst_moment is None:
                fst_moment = th.zeros(dim, dtype=th.double)
                snd_moment = th.zeros(dim, dtype=th.double)

            sample = x.size(0)
            sample = (cnt + sample).double()

            while True:
                sum_ = th.sum(x, dim=0, dtype=th.double)
                sum_sq = th.sum(th.pow(x, 2), dim=0, dtype=th.double)
                inter_fst = (cnt * fst_moment + sum_)
                inter_fst = th.div(inter_fst, sample)
                inter_snd = (cnt * snd_moment + sum_sq)
                inter_snd = th.div(inter_snd, sample)
                if not isnan(inter_fst) and not isnan(inter_snd):
                    break
                else:
                    print("NaN")

            fst_moment = inter_fst
            snd_moment = inter_snd

            cnt = sample

        mean = fst_moment.float()
        eps = 1e-09
        std = th.sqrt(snd_moment - fst_moment ** 2 + eps).float()

        def norm(x):
            batch_mean = mean.expand_as(x)
            batch_std = std.expand_as(x)
            prep = (x - batch_mean) / batch_std
            return prep

        return norm

    def resume(self, epoch, model, optim, scheduler):
        self.optimizer.resume(model, optim, scheduler)
        self.last_epoch = epoch

    def train_iter(self, tools, dataset_path):

        loader = self._prepare(dataset_path)

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.optimizer.to(device)

        norm_func = self._norm(loader, device)

        for ep in range(self.last_epoch, self.epoch):

            iterator = enumerate(loader)
            iterator = tqdm(iterator) if self.total_batches == -1 else\
                tqdm(iterator, total=self.total_batches)

            for i, batch in iterator:
                if batch.num_graphs <= 1:
                    print("There exists compatible issues for batches of size 1. Skip.")
                    continue

                batch = batch.to(device)
                batch.x = norm_func(batch.x)
                train_loss = self.optimizer.train(
                    batch, batch.y
                )

                self.total_batches = max(self.total_batches, i)

                if self.validate is not None:
                    res = self.validate(
                        self.optimizer.model,
                        device, i,
                        loss_func=self.optimizer.loss,
                        classes=len(tools), norm=norm_func
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


def build_training(config, model, data_key='train'):

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

    clip_grad = False
    if 'clip_grad' in config:
        clip_grad = config['clip_grad']
        del config['clip_grad']

    optimizer = ModelOptimizer(
        model, config['optimizer'], config['loss'],
        scheduler=step_scheduler, clip_grad=clip_grad
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
    if 'subsample' not in config:
        config['subsample'] = False

    if isinstance(config['shuffle'], int):
        random.seed(config['shuffle'])
        th.manual_seed(config['shuffle'])

    config['dataset_op'] = DatasetOp(
        data_key, config['shuffle'], config['augment'],
        subsample=config['subsample']
    )
    del config['augment']
    del config['subsample']

    config['scheduler'] = epoch_scheduler

    return ModelTrainer(**config)
