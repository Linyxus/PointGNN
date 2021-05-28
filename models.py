import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_cluster import knn
from layers import DGCNNConv


class GNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(GNN, self).__init__()


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
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.FloatTensor, batch: torch.LongTensor):
        xs = []
        for layer in self.layers:
            edge_index = knn(x, x, self.k, batch, batch, num_workers=512)
            x = layer(x, edge_index)
            xs.append(x)
        x = self.projection(torch.cat(xs, dim=1))
        x = global_max_pool(x, batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)
