import taskflow as tsk
from taskflow import task_definition, backend

from tasks import train_utils
from tasks.rank_scores import spearmann_score
from tasks.download_to_torch import GraphDataset

import torch as th
import torch_geometric as pyg

from torch_geometric.nn import global_max_pool, MessagePassing
from torch.nn import Linear, Sequential, Dropout
import torch.nn.functional as F

import time
import numpy as np

from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='iteration',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')


class EmbedLayer(MessagePassing):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        pyg.nn.inits.reset(self.nn)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        out = self.nn(x)
        out = F.relu(out)
        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class Net(th.nn.Module):

    def __init__(self, intermediate, out_channels):
        super().__init__()

        self.conv1 = EmbedLayer(Linear(148, intermediate))
        self.out = Linear(intermediate, out_channels)

    def forward(self, data):
        pos, batch = data.x, data.batch
        x1 = self.conv1(pos, batch)
        out = global_max_pool(x1, batch)
        out = self.out(out)
        return out


class Custom_BCE(th.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        filt = th.abs(2 * y - 1)

        neg_abs = - x.abs()
        loss = filt * (x.clamp(min=0) - x * y + (1 + neg_abs.exp()).log())
        return loss.mean()


def rank_score(pred, target, classes=15):
    pred = th.sigmoid(pred)
    labels = ['%d' % i for i in range(classes)]
    avg_score = []
    for i in range(pred.shape[0]):
        p = train_utils.get_ranking(pred[i, :], labels)
        t = train_utils.get_ranking(target[i, :], labels)
        avg_score.append(spearmann_score(p, t))
    return th.tensor(avg_score)


@task_definition()
def test_run(env=None):
    dataset = GraphDataset(env.get_cache_dir())
    plotter = VisdomLinePlotter("Test plotter")

    loader = pyg.data.DataLoader(dataset, batch_size=32, num_workers=8,
                                 shuffle=True)

    device = th.device('cpu')
    model = Net(120, 120).to(device)
    optimizer = th.optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.00001)

    start = time.time()
    epoch_start = time.time()
    epoch_time = []

    loss_func = Custom_BCE().to(device)

    model.train()
    offset = 0
    for i in range(9):
        print("Epoch %d:" % i)
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            y = th.reshape(batch.y, [int(batch.y.shape[0]/120), 120])
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_func(
                out, y
            )
            loss.backward()
            acc = th.mean(rank_score(out, y))
            print("It: %d Time: %f Graphs: %d Train_Loss: %f Train_Spear: %f" %
                  (i, (time.time() - start), batch.num_graphs, loss.item(), acc))
            plotter.plot(
                'loss', 'train', 'custom_bce', offset+i, loss.item()
            )
            plotter.plot(
                'acc', 'train', 'spearmann_rank', offset+i, acc
            )
            optimizer.step()
            start = time.time()
        offset += 215
        epoch_time.append(time.time() - epoch_start)
        epoch_start = time.time()

    print("Average epoch time: %f (std: %f)" % (np.mean(epoch_time),
                                                np.std(epoch_time)))


if __name__ == '__main__':
    with backend.openLocalSession() as sess:
        sess.run(test_run())
