import torch as th
import torch_geometric as pyg
from torch_geometric.nn import global_add_pool, MessagePassing
from torch.nn import functional as F


class GlobalSumEmbedder(th.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = th.nn.BatchNorm1d(148)

    def forward(self, data):
        x, batch = data.x, data.batch
        out = global_add_pool(x, batch)
        out = self.norm(out)
        return out


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


def build_model_from_config(config):
    return GlobalSumEmbedder(), 148
