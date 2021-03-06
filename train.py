from time import perf_counter
import argparse

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm

from models import DynamicGNN


def test():
    model.eval()

    correct = 0
    pbar = tqdm(total=len(test_dataset))
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.pos, data.batch).max(dim=1).indices
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
        out = model(data.pos, data.batch)
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
        'lr': 0.01,
        'weight_decay': 1e-5,
        'num_epochs': 200
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--graph_cons', type=str, choices=['static', 'dynamic'])
    parser.add_argument('--nni', action='store_true')
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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
    )

    model = DynamicGNN(
        3, params['hidden_dim'], train_dataset.num_classes,
        num_layers=2, k=params['k'], dropout=params['dropout']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'], eta_min=0.001)

    for epoch in range(1, params['num_epochs'] + 1):
        tic = perf_counter()
        loss, train_acc = train()
        test_acc = test()
        # lr_scheduler.step()
        toc = perf_counter()
        print(f'epoch {epoch}: train loss {loss:.6f}, train acc {train_acc:.6f}, test acc {test_acc:.6f} time {toc - tic:.4f} sec')
