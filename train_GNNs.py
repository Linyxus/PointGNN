from time import perf_counter
import argparse
from tqdm import tqdm

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import GNN


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
        'activation': 'leakyrelu',
        'k': 40,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 1e-5,
        'num_epochs': 200,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dynamic', type=lambda s: s == 'yes', default=False)
    parser.add_argument('--nni', action='store_true')
    parser.add_argument('--gnn', type=str, choices=['GCN', 'GAT', 'GIN'], default='GCN')
    parser.add_argument('--tensorboard', nargs='?')
    args = parser.parse_args()

    print(args.__dict__)

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

    model = GNN(
        args.gnn, 3, params['hidden_dim'], train_dataset.num_classes,
        num_layers=2, dropout=params['dropout'],
        activation=params['activation'], dynamic=args.dynamic, k=params['k']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    summary_writer = SummaryWriter(comment=args.tensorboard) if args.tensorboard is not None else None

    for epoch in range(1, params['num_epochs'] + 1):
        tic = perf_counter()
        loss, train_acc = train()
        test_acc = test()
        toc = perf_counter()
        print(f'epoch {epoch}: train loss {loss:.6f}, train acc {train_acc:.6f}, test acc {test_acc:.6f} time {toc - tic:.4f} sec')

        if summary_writer is not None:
            summary_writer.add_scalar('Train/Loss', loss, epoch)
            summary_writer.add_scalar('Train/Accuracy', train_acc, epoch)
            summary_writer.add_scalar('Test/Accuracy', test_acc, epoch)
            summary_writer.flush()
