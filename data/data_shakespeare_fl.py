# data/data_shakespeare_fl.py

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ShakespeareDataConfig:
    data_root: str = "./data/shakespeare"
    train_json: str = "train.json"
    test_json: str = "test.json"
    seq_len: int = 80
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 0
    num_clients: int = 100  # 使うクライアント数（users から先頭 num_clients を使う）


def _load_leaf_shakespeare_json(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def _build_vocab_from_texts(texts: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    char -> id 辞書と id -> char リストを作る。
    先頭には <pad> を予約して、残りに出現文字を詰める。
    """
    chars = set()
    for t in texts:
        chars.update(list(t))

    # 固定の pad トークンを 0 として入れる
    idx2char = ["<pad>"] + sorted(chars)
    char2idx = {ch: i for i, ch in enumerate(idx2char)}
    return char2idx, idx2char


class ShakespeareClientDataset(Dataset):
    """
    あるクライアントのテキスト群から
    (input_seq, target_seq) のペアを作る Dataset。

    - texts: 文字列のリスト
    - char2idx: 文字 -> id
    - seq_len: シーケンス長

    各テキスト t について、エンコードした列 e を使って
      e[start : start+seq_len] -> inputs
      e[start+1 : start+seq_len+1] -> targets
    を生成する（1 文字シフトの next-char）。
    """

    def __init__(self, texts: List[str], char2idx: Dict[str, int], seq_len: int):
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.seq_len = seq_len

        for t in texts:
            if not t:
                continue
            encoded = [char2idx.get(ch, 0) for ch in t]  # 未知文字は <pad> に落とす
            if len(encoded) <= seq_len:
                continue
            # ストライド 1 でシーケンスを切り出す（必要ならストライドを seq_len にしてもよい）
            for start in range(0, len(encoded) - seq_len):
                inp = encoded[start : start + seq_len]
                tgt = encoded[start + 1 : start + seq_len + 1]
                self.inputs.append(torch.tensor(inp, dtype=torch.long))
                self.targets.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def build_shakespeare_federated_loaders(
    config: ShakespeareDataConfig,
) -> Tuple[List[DataLoader], DataLoader, Dict[str, int], List[str]]:
    """
    LEAF 形式の Shakespeare JSON から

      - client-wise DataLoader のリスト
      - テスト用 DataLoader
      - vocab (char2idx, idx2char)

    を返す。
    """
    random.seed(config.seed)

    train_path = os.path.join(config.data_root, config.train_json)
    test_path = os.path.join(config.data_root, config.test_json)

    train_data = _load_leaf_shakespeare_json(train_path)
    test_data = _load_leaf_shakespeare_json(test_path)

    # vocab 構築のために train + test の全テキストを集める
    all_texts: List[str] = []

    for user in train_data["users"]:
        user_texts = train_data["user_data"][user]["x"]
        all_texts.extend(user_texts)

    for user in test_data["users"]:
        user_texts = test_data["user_data"][user]["x"]
        all_texts.extend(user_texts)

    char2idx, idx2char = _build_vocab_from_texts(all_texts)

    # クライアントごとの DataLoader
    users = list(train_data["users"])
    # 先頭 num_clients を使う（順番はそのままでも、seed で shuffle してもよい）
    if config.num_clients < len(users):
        users = users[: config.num_clients]

    client_loaders: List[DataLoader] = []
    for user in users:
        texts = train_data["user_data"][user]["x"]
        ds = ShakespeareClientDataset(texts, char2idx, config.seq_len)
        if len(ds) == 0:
            # データが少なすぎるユーザはスキップ（必要ならログを出す）
            continue
        loader = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        client_loaders.append(loader)

    # test 用 Dataset: 全ユーザのテキストをまとめて 1 つの Dataset にする
    test_texts: List[str] = []
    for user in test_data["users"]:
        user_texts = test_data["user_data"][user]["x"]
        test_texts.extend(user_texts)

    test_ds = ShakespeareClientDataset(test_texts, char2idx, config.seq_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return client_loaders, test_loader, char2idx, idx2char
