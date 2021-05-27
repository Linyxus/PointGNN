import torch
import torch.nn as nn
from .layers import DGCNNConv


class DynamicGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, k: int, dropout: float):
        super(DynamicGNN, self).__init__()
        self.layers = []
        self.layers.append(DGCNNConv(input_dim, hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(DGCNNConv(hidden_dim, hidden_dim, hidden_dim))
        self.layers.append(DGCNNConv(hidden_dim, hidden_dim, output_dim))
        self.layers = nn.ModuleList(self.layers)
        self.k = k

    def forward(self, x: torch.FloatTensor):
        pass
