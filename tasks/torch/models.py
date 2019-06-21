import torch as th
from torch_geometric.nn import global_add_pool, global_max_pool,\
                               global_mean_pool
from torch_geometric.utils import softmax, scatter_
from torch_geometric.nn import GINConv, MessagePassing
from torch.nn import functional as F


class Embedding(th.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = th.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.lin(x)
        out = F.relu(out)
        return out


class ConditionalGlobalAttention(th.nn.Module):

    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.query_embed = th.nn.Linear(cond_channels, in_channels)
        self.scale = th.sqrt(th.tensor(in_channels, dtype=th.float))

    def attention(self, x, batch, condition, size=None):
        size = batch[-1].item() + 1 if size is None else size

        query = self.query_embed(condition.float())[batch, :]
        attention = (x * query).sum(dim=1)
        attention = attention / self.scale
        attention = softmax(attention, batch, size)
        return attention

    def forward(self, x, batch, condition, size=None):

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        attention = self.attention(x, batch, condition, size)
        attention = attention.unsqueeze(1).repeat(1, x.shape[1])

        out = scatter_('add', attention * x, batch, size)
        return out


class EdgeAttention(th.nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_row, x_col = x.index_select(0, row), x.index_select(0, col)
        out = th.cat([x_row, x_col, edge_attr], dim=1)
        out = self.nn(x)
        out = scatter_('add', x, row, dim_size=x.size(0))

        return out


class EdgeGIN(MessagePassing):

    def __init__(self, gin_nn, edge_nn):
        super().__init__()
        self.nn = gin_nn
        self.edge = EdgeAttention(edge_nn)

    def forward(self, x, edge_index, edge_attr):

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x_e = self.edge(x, edge_index, edge_attr)
        out = self.nn(x + self.propagate(edge_index, x=x_e))
        return out

    def message(self, x_j):
        return x_j


class GINMLP(th.nn.Module):

    def __init__(self, in_channel, hidden, out_channel, dropout,
                 batch_norm):
        super().__init__()

        seq = []
        for i, h in enumerate(hidden):
            drop = dropout[i]
            hid = hidden[i]
            b = batch_norm[i]
            seq.extend(self._build_layer(in_channel, hid, drop, b))
            in_channel = hid
        seq.append(th.nn.Linear(in_channel, out_channel))
        self.sequence = th.nn.Sequential(*seq)

    def _norm_layer(self, type, channel):
        if type is True or type == 'batch':
            return th.nn.BatchNorm1d(channel)

        if type == 'layer':
            return th.nn.LayerNorm(channel)
        raise ValueError("Unknown norm type %s." % type)

    def _build_layer(self, in_channel, out_channel, dropout, batch_norm):
        if batch_norm:
            return [
                th.nn.Linear(in_channel, out_channel),
                self._norm_layer(batch_norm, out_channel),
                th.nn.Dropout(p=dropout),
                th.nn.ReLU()
            ]
        return [
            th.nn.Linear(in_channel, out_channel),
            th.nn.Dropout(p=dropout),
            th.nn.ReLU()
        ]

    def forward(self, x):
        return self.sequence(x)
