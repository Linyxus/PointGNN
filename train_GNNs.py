from time import perf_counter
import argparse
from tqdm import tqdm

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_cluster import knn

from models import GCN, GAT


def construct_graph(x: torch.FloatTensor, batch: torch.LongTensor, k: int) -> torch.LongTensor:
    return knn(x, x, k, batch, batch)


def test():
    model.eval()

    correct = 0
    pbar = tqdm(total=len(test_dataset))
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.pos, data.edge_index, data.batch).max(dim=1).indices
        correct += pred.eq(data.y).sum().item()
        pbar.update(data.num_graphs)
    pbar.close()
    return correct / len(test_dataset)


def train():
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct = 0
    pbar = tqdm(total=len(train_dataset))
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.pos, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        pred = out.max(dim=1).indices
        correct += pred.eq(data.y).sum().item()
        pbar.update(data.num_graphs)
        pbar.set_postfix({'loss': total_loss / total_samples, 'acc': correct / total_samples})
    pbar.close()
    return total_loss / len(train_dataset), correct / len(train_dataset)


if __name__ == '__main__':
    params = {
        'hidden_dim': 128,
        'k': 40,
        'dropout': 0.5,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 200
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--nni', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset_path = osp.expanduser('~/datasets')
    dataset_path = osp.join(dataset_path, 'ModelNet40')
    train_dataset = ModelNet(
        root=dataset_path,
        name='40',
        train=True,
        transform=T.SamplePoints(num=2048),
        pre_transform=T.NormalizeScale()
    )
    test_dataset = ModelNet(
        root=dataset_path,
        name='40',
        train=False,
        transform=T.SamplePoints(num=2048),
        pre_transform=T.NormalizeScale()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
    )

    model = GCN(
        3, params['hidden_dim'], train_dataset.num_classes,
        num_layers=2, dropout=params['dropout']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    for epoch in range(1, params['num_epochs'] + 1):
        tic = perf_counter()
        loss, train_acc = train()
        test_acc = test()
        toc = perf_counter()
        print(f'epoch {epoch}: train loss {loss:.6f}, train acc {train_acc:.6f}, test acc {test_acc:.6f} time {toc - tic:.4f} sec')
