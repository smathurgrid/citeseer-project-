import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def create_random_split(data, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.num_nodes

    perm = torch.randperm(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


def load_data(device, split_type="default"):

    dataset = Planetoid(
        root='Data/raw',
        name='CiteSeer',
        transform=NormalizeFeatures()
    )

    data = dataset[0]

    # -------- Split Handling --------

    
    if split_type == "random_80_10_10":
        data = create_random_split(data, train_ratio=0.8, val_ratio=0.1)

    elif split_type == "random_60_20_20":
        data = create_random_split(data, train_ratio=0.6, val_ratio=0.2)

    # default → do nothing (Planetoid split)

    data = data.to(device)

    return dataset, data