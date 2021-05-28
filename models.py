import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GCNConv, GATConv
from torch_cluster import knn
from layers import DGCNNConv


class GAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int, num_layers: int, dropout: float):
        super(GAT, self).__init__()
        self.layers = []
        self.layers.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        self.layers = nn.ModuleList(self.layers)
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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, batch: torch.LongTensor):
        xs = []
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.projection(x)
        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super(GCN, self).__init__()
        self.layers = []
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))
        self.layers = nn.ModuleList(self.layers)
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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, batch: torch.LongTensor):
        xs = []
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.projection(x)
        x = global_mean_pool(x, batch)
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
        for layer in self.layers:
            edge_index = knn(x, x, self.k, batch, batch, num_workers=512)
            x = layer(x, edge_index)
            x = F.elu(x)
            xs.append(x)
        x = self.projection(torch.cat(xs, dim=1))
        x = global_max_pool(x, batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)
