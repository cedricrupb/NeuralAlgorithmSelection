import torch as th
from torch_geometric.utils import softmax, scatter_
from torch_geometric.nn import GINConv, MessagePassing
from torch.nn import functional as F


def weight_reset(m):
    if isinstance(m, th.nn.Linear):
        m.reset_parameters()


def gcn_init(m):
    if isinstance(m, th.nn.Linear):
        th.nn.init.xavier_uniform_(m.weight, gain=0.1)
        th.nn.init.zeros_(m.bias)


class Embedding(th.nn.Module):

    def __init__(self, in_channels, out_channels, act='relu'):
        super().__init__()
        self.pre_norm = th.nn.BatchNorm1d(in_channels, affine=False)
        self.lin = th.nn.Linear(in_channels, out_channels)
        self.norm = th.nn.BatchNorm1d(out_channels)
        self.act = _handle_activation(act)

    def reset_parameters(self):
        self.lin.apply(gcn_init)

    def forward(self, x):
        out = x

        # Pre activation
        out = self.pre_norm(out)

        # activation
        out = self.lin(out)
        out = self.norm(out)
        out = self.act(out)

        return out


class ConditionalGlobalAttention(th.nn.Module):

    def __init__(self, in_channels, cond_channels, aggr='mean'):
        super().__init__()
        self.query_embed = th.nn.Linear(cond_channels, in_channels)
        self.scale = th.sqrt(th.tensor(in_channels, dtype=th.float))
        self.aggr = aggr
        self.reset_parameters()

    def attention(self, x, batch, condition, size=None):
        size = batch[-1].item() + 1 if size is None else size

        query = self.query_embed(condition.float())[batch, :]
        attention = (x * query).sum(dim=1)
        attention = attention / self.scale
        # attention = F.elu(attention)
        attention = softmax(attention, batch, size)
        return attention

    def reset_parameters(self):
        self.query_embed.reset_parameters()

    def forward(self, x, batch, condition, size=None):

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        attention = self.attention(x, batch, condition, size)
        attention = attention.unsqueeze(1).expand_as(x)

        out = attention * x

        if self.aggr == 'sum':
            _, count = batch.unique(return_counts=True)
            nodes = count[batch].unsqueeze(1).expand_as(x).float()
            out = nodes * out

        out = scatter_('add', out, batch, size)
        return out


class EdgeAttention(th.nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        self.nn.apply(gcn_init)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_row, x_col = x.index_select(0, row), x.index_select(0, col)
        out = th.cat([x_row, x_col, edge_attr.float()], dim=1)
        out = self.nn(out)
        out = scatter_('add', out, col, dim_size=x.size(0))

        return out


class EdgeGIN(MessagePassing):

    def __init__(self, gin_nn, edge_nn, **kwargs):
        super().__init__(**kwargs)
        self.nn = gin_nn
        self.edge = edge_nn

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.nn(x + self.propagate(edge_index, x=x,
                                          edge_attr=edge_attr))

    def message(self, x_i, x_j, edge_index_j, edge_attr):
        attr = edge_attr[edge_index_j]
        inter = th.cat([x_i, x_j, attr.float()], dim=1)
        return self.edge(inter)


# Simplified edge gin
class SEdgeGIN(MessagePassing):

    def __init__(self, gin_nn, **kwargs):
        super().__init__(**kwargs)
        self.nn = gin_nn

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x_e = th.cat([x,
                      x,
                      th.zeros((x.size(0), edge_attr.size(1)))], dim=1)
        p = self.propagate(edge_index, x=x,
                           edge_attr=edge_attr)
        return self.nn(x_e + p)

    def message(self, x_i, x_j, edge_index_j, edge_attr):
        attr = edge_attr[edge_index_j]
        inter = th.cat([x_i, x_j, attr.float()], dim=1)
        return inter


# attentional edge gin
class AEdgeGIN(MessagePassing):

    def __init__(self, in_channels, edge_channels, gin_nn):
        super().__init__()
        self.query_embed = th.nn.Linear(edge_channels, 2*in_channels)
        self.scale = th.sqrt(th.tensor(2*in_channels, dtype=th.float))
        self.nn = gin_nn

    def edge(self, x, edge_index, edge_attr):
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_sel = self.query_embed(edge_attr.float())

        x_row, x_col = x.index_select(0, row), x.index_select(0, col)
        x_in = th.cat([x_row,
                       x_col], dim=1)

        att = (x_in * edge_sel).sum(dim=1)
        att = att / self.scale

        att = softmax(att, col, None)
        att = att.unsqueeze(1).expand_as(x_col)

        _, inv, num = col.unique(return_inverse=True, return_counts=True)
        num = num[inv].unsqueeze(1).expand_as(x_col).float()

        return scatter_('add', num*att*x_col, col, dim_size=x.size(0))

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x_e = self.edge(x, edge_index, edge_attr)

        return self.nn(x + x_e)


def _min_list(L):
    return [t for t in L if t is not None]


def _handle_dropout(dropout):
    if dropout is None or dropout == 0.0:
        return None
    return th.nn.Dropout(p=dropout)


def _handle_norm(norm, channel):

    if norm is None:
        return None

    if norm is True or norm == 'batch':
        return th.nn.BatchNorm1d(channel)

    if norm == 'layer':
        return th.nn.LayerNorm(channel)


def _handle_activation(act):

    if act == 'relu':
        return th.nn.ReLU()

    if act == 'elu':
        return th.nn.ELU()


def _build_layer(in_channel, out_channel, activation="relu",
                 dropout=None, norm=None):
    return _min_list([
        th.nn.Linear(in_channel, out_channel),
        _handle_norm(norm, out_channel),
        _handle_dropout(dropout),
        _handle_activation(activation)
    ])


def _build_mlp_seq(channels, dropout=None, norm=None):
    if len(channels) < 2:
        raise ValueError("Need at least in and out channel")

    in_channel = channels[0]
    out_channel = channels[-1]
    channels = channels[1:-1]

    if not isinstance(dropout, list):
        dropout = [dropout]*len(channels)

    if not isinstance(norm, list):
        norm = [norm]*len(channels)

    seq = []
    for i, h in enumerate(channels):
        drop = dropout[i]
        nor = norm[i]
        seq.extend(
            _build_layer(in_channel, h, dropout=drop, norm=nor)
        )
        in_channel = h

    seq.append(th.nn.Linear(in_channel, out_channel))

    return seq


def cap_mlp(seq, out_channel, activation='relu', dropout=None, norm=None):
    seq.append(_handle_dropout(dropout))
    seq.append(_handle_norm(norm, out_channel))
    seq.append(_handle_activation(activation))
    return th.nn.Sequential(*_min_list(seq))


def prepare_gin(nn_cfg, in_channel, out_channel):

    hidden = []
    if 'hidden' in nn_cfg:
        hid = nn_cfg['hidden']
        if not isinstance(hid, list):
            hid = [hid]
        hidden = hid

    size = len(hidden)

    dropout = [0.1]*len(hidden)
    if 'dropout' in nn_cfg:
        dr = nn_cfg['dropout']
        if not isinstance(dr, list):
            dr = [dr]*size
        if len(dr) < len(hidden):
            raise ValueError("Dropout has to match hidden layers.\
                              Got %i but expected %i" % (len(dr), len(hidden)))
        dropout = dr

    norm = [False]*len(hidden)
    if 'norm' in nn_cfg:
        n = nn_cfg['norm']
        if not isinstance(n, list):
            n = [n]*size
        if len(n) < len(hidden):
            raise ValueError("Norm has to match hidden layers.\
                              Got %i but expected %i" % (len(n), len(hidden)))
        norm = n

    channels = [in_channel]
    channels.extend(hidden)
    channels.append(out_channel)

    cfg = {
        'channels': channels,
        'dropout': dropout,
        'norm': norm
    }
    return cfg


class CEdgeGIN(EdgeGIN):

    def __init__(self, in_channels, out_channels, edge_channels,
                 gin_nn, edge_nn={}):

        gin = prepare_gin(
            gin_nn, in_channels, out_channels
        )
        gin = cap_mlp(_build_mlp_seq(**gin), out_channels,
                      norm=True)

        edge = prepare_gin(
            edge_nn, 2*in_channels+edge_channels, in_channels
        )
        edge = cap_mlp(_build_mlp_seq(**edge), in_channels,
                       norm=True)
        super().__init__(gin, edge)


class CSEdgeGIN(SEdgeGIN):

    def __init__(self, in_channels, out_channels, edge_channels,
                 gin_nn):

        gin = prepare_gin(
            gin_nn, 2*in_channels + edge_channels, out_channels
        )
        gin = cap_mlp(_build_mlp_seq(**gin), out_channels,
                      norm=True)

        super().__init__(gin)


class CAEdgeGIN(AEdgeGIN):

    def __init__(self, in_channels, out_channels, edge_channels,
                 gin_nn):

        gin = prepare_gin(
            gin_nn, in_channels, out_channels
        )
        gin = cap_mlp(_build_mlp_seq(**gin), out_channels,
                      norm=True)

        super().__init__(in_channels, edge_channels, gin)


class CGIN(GINConv):

    def __init__(self, in_channels, out_channels, edge_channels,
                 gin_nn):

        gin = prepare_gin(
            gin_nn, in_channels, out_channels
        )
        gin = cap_mlp(_build_mlp_seq(**gin), out_channels,
                      norm=True)

        super().__init__(gin)


class ReadoutClf(th.nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.0, norm=False):
        super().__init__()

        seq = []

        if norm:
            seq.append(th.nn.BatchNorm1d(in_channels))

        seq.append(th.nn.Linear(in_channels, out_channels))

        if dropout > 0:
            seq.append(th.nn.Dropout(dropout))

        self.nn = th.nn.Sequential(*seq)

    def forward(self, input):
        return self.nn(input)
