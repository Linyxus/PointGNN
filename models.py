from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_cluster import knn
from layers import DGCNNConv


GetConv = Callable[[int, int], nn.Module]


def get_conv(name: str) -> GetConv:
    all_conv = {
        'GCN': lambda i, o: GCNConv(i, o, cached=False),
        'GAT': lambda i, o: GATConv(i, o // 4, heads=4),
        'GIN': lambda i, o: GINConv(nn.Sequential(nn.Linear(i, o), nn.ELU(), nn.Linear(o, o)))
    }
    return all_conv[name]


def get_activation(name: str) -> nn.Module:
    assert name in ['relu', 'prelu', 'elu', 'leakyrelu'], f'unsupported activation {name}'
    if name == 'relu':
        return nn.ReLU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()


class GNN(nn.Module):
    def __init__(self, conv: str,
                 input_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
                 dropout: float,
                 activation: str,
                 dynamic: bool,
                 k: int):
        super(GNN, self).__init__()

        def act():
            return get_activation(activation)

        self.dynamic = dynamic
        self.k = k

        conv = get_conv(conv)
        self.layers = []
        self.layers.append(conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(conv(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(self.layers)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            act(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def construct_graph(self, x: torch.FloatTensor, batch: torch.LongTensor):
        return knn(x, x, self.k, batch, batch, num_workers=512)

    def forward(self, x: torch.FloatTensor, batch: torch.LongTensor):
        xs = []
        edge_index = self.construct_graph(x, batch)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            xs.append(x)
            if self.dynamic:
                edge_index = self.construct_graph(x, batch)
        x = torch.cat(xs, dim=1)
        x = self.projection(x)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class DynamicGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, k: int, dropout: float):
        super(DynamicGNN, self).__init__()
        self.layers = []
        self.layers.append(DGCNNConv(input_dim, hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(DGCNNConv(hidden_dim, hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(self.layers)
        self.k = k

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.FloatTensor, batch: torch.LongTensor):
        xs = []
        edge_index = knn(x, x, self.k, batch, batch, num_workers=512)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            xs.append(x)
        x = self.projection(torch.cat(xs, dim=1))
        x = global_max_pool(x, batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)
