import os.path as osp
import torch
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T


if __name__ == '__main__':
    dataset_path = osp.expanduser('~/datasets')
    dataset_path = osp.join(dataset_path, 'ModelNet40')
    train_dataset = ModelNet(root=dataset_path, name='40', train=True, transform=T.FaceToEdge(), pre_transform=T.NormalizeScale())
    test_dataset = ModelNet(root=dataset_path, name='40', train=False, transform=T.FaceToEdge(), pre_transform=T.NormalizeScale())
