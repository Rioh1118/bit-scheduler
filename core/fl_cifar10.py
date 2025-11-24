# core/fl_cifar10.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from data.data_cifar10_fl import Cifar10DataConfig, build_cifar10_federated_loaders
from quantization.delta_quantization import flatten_deltas, quantize_model_delta
from quantization.qsgd import bit_to_s_levels
from schedule.schedules import make_fixed_schedule


@dataclass
class Cifar10FLConfig:
    num_clients: int = 100
    clients_per_round: int = 10
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 64
    lr_local: float = 0.1
    bit: int = 8  # fixed-bit default
    device: str = "cuda"
    seed: int = 0

    # data
    data_root: str = "./data"
    iid: bool = True
    dirichlet_alpha: float = 0.5

    num_workers: int = 2


@dataclass
class FLLogs:
    test_acc: List[float]
    best_test_acc: List[float]
    bits_cum: List[float]
    bits_cum_normalized: List[float]
    B_FP: float
    d: int


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    CIFAR-10 用に ResNet-18 を組む。
    ひとまず torchvision の resnet18 をそのまま使い、
    入力 32x32 に対しても動くようにする（精度は後で調整）。
    """
    model = models.resnet18(weights=None, num_classes=num_classes)
    return model


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / float(total)


def _compute_d_for_bits(model: nn.Module, device: torch.device) -> int:
    """
    Route1 の d を「flatten した Δ の次元」として求める。
    """
    state = model.state_dict()
    zeros = {
        k: torch.zeros_like(v).to(device)
        for k, v in state.items()
        if torch.is_tensor(v) and v.is_floating_point()
    }
    g_flat, _ = flatten_deltas(zeros, state)
    return int(g_flat.numel())


def _compute_bits_per_round(b_t: int, d: int, m_t: int) -> int:
    """
    QSGD uplink bits:

        B_t^{(i)} = d * (1 + ceil(log2(s_t + 1))) + 32
        B_t = m_t * B_t^{(i)}
    """
    s_t = bit_to_s_levels(b_t)
    per_client = d * (1 + math.ceil(math.log2(s_t + 1))) + 32
    return m_t * per_client


def train_client_local(
    model: nn.Module,
    loader: DataLoader,
    config: Cifar10FLConfig,
    device: torch.device,
) -> None:
    model.train()
    model.to(device)

    opt = torch.optim.SGD(
        model.parameters(), lr=config.lr_local, momentum=0.9, weight_decay=5e-4
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.local_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        # デバッグ時だけ表示
        print(f"local epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}")


def run_cifar10_fl(
    config: Cifar10FLConfig,
    bit_schedule: Optional[List[int]] = None,
) -> FLLogs:
    """
    CIFAR-10 + ResNet-18 + FedAvg + QSGD Δ 量子化の FL ループ。
    """

    _set_global_seed(config.seed)

    device = torch.device(config.device)

    # データ準備
    data_config = Cifar10DataConfig(
        data_root=config.data_root,
        num_clients=config.num_clients,
        iid=config.iid,
        dirichlet_alpha=config.dirichlet_alpha,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    client_loaders, test_loader = build_cifar10_federated_loaders(data_config)

    # グローバルモデル
    global_model = build_resnet18_cifar10(num_classes=10).to(device)

    # ラウンド数と bit スケジュール
    T = config.num_rounds
    if bit_schedule is None:
        bit_schedule = make_fixed_schedule(T, config.bit)
    else:
        if len(bit_schedule) != T:
            raise ValueError(
                f"bit_schedule length {len(bit_schedule)} != num_rounds {T}"
            )

    # d と B_FP
    d = _compute_d_for_bits(global_model, device)
    m_t = config.clients_per_round
    B_FP = float(T * m_t * d * 32)

    # client sampling パターン
    all_client_ids = list(range(config.num_clients))
    selected_ids_per_round: List[List[int]] = []
    rng = random.Random(config.seed + 1)
    for _ in range(T):
        selected_ids = rng.sample(all_client_ids, m_t)
        selected_ids_per_round.append(selected_ids)

    # logging
    test_acc_list: List[float] = []
    best_test_acc_list: List[float] = []
    bits_cum: List[float] = []

    quantizer_cache: Dict[int, "QSGDQuantizer"] = {}

    best_so_far = 0.0
    cum_bits = 0.0

    for t in range(T):
        b_t = bit_schedule[t]
        selected_ids = selected_ids_per_round[t]

        # 更新前の重みをフラットにして保存
        with torch.no_grad():
            params_before = torch.cat(
                [
                    p.detach().view(-1)
                    for p in global_model.parameters()
                    if p.is_floating_point()
                ]
            )

        global_state = {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
            if torch.is_tensor(v) and v.is_floating_point()
        }

        quantized_deltas: List[Dict[str, torch.Tensor]] = []

        for cid in selected_ids:
            client_loader = client_loaders[cid]

            # クライアントモデル作成 & 初期化
            client_model = build_resnet18_cifar10(num_classes=10).to(device)
            client_model.load_state_dict(global_model.state_dict())

            # ローカル学習
            train_client_local(client_model, client_loader, config, device)

            # Δ 量子化
            with torch.no_grad():
                client_state = client_model.state_dict()
                q_delta = quantize_model_delta(
                    global_state,
                    client_state,
                    bit=b_t,
                    quantizer_cache=quantizer_cache,
                )
            quantized_deltas.append(q_delta)

        # サーバ更新
        with torch.no_grad():
            new_state = global_model.state_dict()
            for name, param in new_state.items():
                if not (torch.is_tensor(param) and param.is_floating_point()):
                    continue
                delta_sum = torch.zeros_like(param)
                for q_delta in quantized_deltas:
                    if name in q_delta:
                        delta_sum += q_delta[name]
                delta_avg = delta_sum / float(len(quantized_deltas))
                param.add_(delta_avg)
            global_model.load_state_dict(new_state)

        with torch.no_grad():
            params_after = torch.cat(
                [
                    p.detach().view(-1)
                    for p in global_model.parameters()
                    if p.is_floating_point()
                ]
            )
            delta_norm = torch.norm(params_after - params_before).item()

        # 評価 & bits ログ
        acc = evaluate_model(global_model, test_loader, device)
        best_so_far = max(best_so_far, acc)

        B_t = _compute_bits_per_round(b_t, d, m_t)
        cum_bits += B_t

        test_acc_list.append(acc)
        best_test_acc_list.append(best_so_far)
        bits_cum.append(cum_bits)

        print(
            f"[Round {t+1}/{T}] bit={b_t}, "
            f"test_acc={acc:.4f}, best={best_so_far:.4f}, "
            f"bits_norm={cum_bits / B_FP:.3f}, "
            f"||Δ_global||={delta_norm:.3e}"
        )

    bits_cum_normalized = [b / B_FP for b in bits_cum]

    return FLLogs(
        test_acc=test_acc_list,
        best_test_acc=best_test_acc_list,
        bits_cum=bits_cum,
        bits_cum_normalized=bits_cum_normalized,
        B_FP=B_FP,
        d=d,
    )
