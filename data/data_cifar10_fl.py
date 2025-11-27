# data/data_cifar10_fl.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class Cifar10DataConfig:
    data_root: str = "./data"
    num_clients: int = 100
    iid: bool = True
    dirichlet_alpha: float = 0.5  # non-iid のときに使う
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 0


def _get_cifar10_datasets(data_root: str):
    """
    train / test の Dataset を返す。

    モデルは ResNet-18 を想定しているので、
    標準的な ToTensor + Normalize だけを入れる。
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    return train_ds, test_ds


def _partition_iid(train_ds: Dataset, num_clients: int, seed: int) -> List[List[int]]:
    """
    IID: ランダムに均等分割する。
    """
    num_samples = len(train_ds)
    all_indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    size_per_client = num_samples // num_clients
    client_indices: List[List[int]] = []
    for i in range(num_clients):
        start = i * size_per_client
        end = (i + 1) * size_per_client if i < num_clients - 1 else num_samples
        client_indices.append(all_indices[start:end])
    return client_indices


def _partition_dirichlet(
    train_ds: datasets.CIFAR10,
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """
    Dirichlet 分割による non-iid 分割。
    ラベルごとに Dirichlet(α) でクライアントへの割り当て比率を決める。
    """
    rng = random.Random(seed)
    num_classes = 10
    labels = torch.tensor(train_ds.targets)  # CIFAR10 は targets を持つ

    # クラスごとのインデックスを集める
    class_indices = [torch.where(labels == c)[0].tolist() for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]

    # 各クラスごとに Dirichlet(α) を引いて割り当てる
    for c in range(num_classes):
        idx_c = class_indices[c]
        num_c = len(idx_c)
        # Dirichlet 抽選
        # torch.distributions を使うか、自前で numpy でもOK
        dist = torch.distributions.Dirichlet(torch.full((num_clients,), alpha))
        props = dist.sample().tolist()
        # 割り当て数に変換
        counts = [int(round(p * num_c)) for p in props]

        # 丸めのせいで合計がズレるので調整
        diff = num_c - sum(counts)
        # diff > 0 なら diff 回だけランダムに +1
        # diff < 0 なら -1 して調整
        while diff != 0:
            j = rng.randrange(num_clients)
            if diff > 0:
                counts[j] += 1
                diff -= 1
            else:
                if counts[j] > 0:
                    counts[j] -= 1
                    diff += 1

        assert sum(counts) == num_c

        # 実際にインデックスを配る
        offset = 0
        for client_id, cnt in enumerate(counts):
            if cnt == 0:
                continue
            client_indices[client_id].extend(idx_c[offset : offset + cnt])
            offset += cnt

    # 最後に shuffle
    for idxs in client_indices:
        rng.shuffle(idxs)

    return client_indices


def build_cifar10_federated_loaders(config: Cifar10DataConfig):
    """
    CIFAR-10 を num_clients 個のクライアントに分割し、
    各クライアントの DataLoader と test_loader を返す。
    """
    train_ds, test_ds = _get_cifar10_datasets(config.data_root)

    if config.iid:
        client_indices = _partition_iid(train_ds, config.num_clients, seed=config.seed)
    else:
        client_indices = _partition_dirichlet(
            train_ds,
            num_clients=config.num_clients,
            alpha=config.dirichlet_alpha,
            seed=config.seed,
        )

    client_loaders: List[DataLoader] = []
    for idxs in client_indices:
        subset = Subset(train_ds, idxs)
        loader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        client_loaders.append(loader)

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return client_loaders, train_eval_loader, test_loader
