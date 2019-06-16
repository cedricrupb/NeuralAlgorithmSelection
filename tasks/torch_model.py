import torch as th
import torch_geometric as pyg
from torch_geometric.nn import global_add_pool, global_max_pool, GINConv
from torch_geometric.utils import softmax, scatter_
from torch.nn import functional as F


def select_readout(type, in_channel):

    if type is None:
        return None

    if type == 'add':
        return global_add_pool

    if type == 'max':
        return global_max_pool

    if type.startswith('attention:'):
        type, cond = type.split(':')
        cond = int(cond)
        return ConditionalGlobalAttention(
            in_channel, cond
        )

    raise ValueError("Unknown readout type %s" % type)


def select_layer(type, kwargs, sparse=False):

    if type == 'embedding':
        if sparse:
            return SparseEmbeddingLayer(**kwargs)
        else:
            return EmbeddingLayer(**kwargs)


def select_glue(type, kwargs, sparse=False):
    pass


class EmbeddingLayer(th.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = th.nn.Linear(in_channels, out_channels)

    def forward(self, data):
        x = data.x
        out = self.lin(x)
        out = F.relu(out)
        return out


class SparseEmbeddingLayer(th.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embed = th.nn.EmbeddingBag(in_channels, out_channels, mode="sum")
        self.norm = th.nn.LayerNorm(out_channels)

    def forward(self, data):
        out = self.embed(
            data.sparse_index, data.offset, data.weight
        )
        out = self.norm(out)
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


class TestModel(th.nn.Module):

    def __init__(self, embed, att):
        super().__init__()
        self.embed = embed
        self.attention = att

    def forward(self, data, return_attention=False):
        n1 = self.embed(data)
        o1 = self.attention(n1, data.batch, data.category)

        if return_attention:
            att = self.attention.attention(
                n1, data.batch, data.category
            )
            return o1, att

        return o1


class GlobalAddAggregator(th.nn.Module):

    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, data):
        x = self.embed(data)
        x = global_add_pool(x, data.batch)
        return x


def prepare_config(config):
    in_channels = config['input']
    strategy = config['strategy']

    layers = []

    for l in config['layers']:

        out_dim = in_channels
        if 'dim' in l:
            out_dim = l['dim']

        readout = 'add'
        if 'readout' in l:
            readout = l['readout']

        layers.append((l['type'], {
            'in_channels': in_channels,
            'out_channels': out_dim,
            'readout': readout
        }))

        in_channels = out_dim

    return strategy, layers


def build_model_from_config(config):
    in_channel = config['embed'][0]
    out_channel = config['embed'][1]
    sparse = False
    if 'sparse' in config:
        sparse = config['sparse']

    embed = SparseEmbeddingLayer(in_channel, out_channel) if sparse else\
        EmbeddingLayer(in_channel, out_channel)
    att = ConditionalGlobalAttention(out_channel, 4)
    return TestModel(embed, att), out_channel
