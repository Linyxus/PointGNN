import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing


class DGCNNConv(MessagePassing):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(DGCNNConv, self).__init__(aggr='max', flow='target_to_source')
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, output_dim))

    def __repr__(self):
        return f'EdgeConv()'

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor):
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: torch.FloatTensor, x_j: torch.FloatTensor) -> torch.FloatTensor:
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=1))
