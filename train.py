from time import perf_counter
import argparse

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from models import DynamicGNN


def test():
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x).max()(dim=1).indices
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)


def train():
    model.train()

    total_loss = 0.0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.max(dim=1).indices
        correct += pred.eq(data.y).sum().item()
    return total_loss / len(train_dataset), correct / len(train_dataset)


if __name__ == '__main__':
    params = {
        'hidden_dim': 128,
        'k': 20,
        'dropout': 0.5,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 200
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset_path = osp.expanduser('~/datasets')
    dataset_path = osp.join(dataset_path, 'ModelNet40')
    train_dataset = ModelNet(
        root=dataset_path,
        name='40',
        train=True,
        transform=T.SamplePoints(num=2048), pre_transform=T.NormalizeScale()
    )
    test_dataset = ModelNet(
        root=dataset_path,
        name='40',
        train=False,
        transform=T.SamplePoints(num=2048),
        pre_transform=T.NormalizeScale()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
    )

    model = DynamicGNN(
        train_dataset.num_features, params['hidden_dim'], train_dataset.num_classes,
        num_layers=3, k=params['k'], dropout=params['dropout']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    for epoch in range(1, params['num_epochs'] + 1):
        tic = perf_counter()
        loss, train_acc = train()
        toc = perf_counter()
        print(f'epoch {epoch}: train loss {loss:.6f}, train acc {train_acc:.6f}, time {toc - tic:.4f} sec')
