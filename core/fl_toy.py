# core/fl_toy.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantization.delta_quantization import flatten_deltas, quantize_model_delta
from quantization.qsgd import bit_to_s_levels
from schedule.schedules import make_fixed_schedule


@dataclass
class ToyFLConfig:
    num_clients: int = 20
    clients_per_round: int = 5
    num_rounds: int = 10
    input_dim: int = 20
    hidden_dim: int = 64
    num_classes: int = 2
    local_epochs: int = 1
    batch_size: int = 32
    lr_local: float = 0.1
    bit: int = 8  # fixed-bit スケジュール用デフォルト
    device: str = "cpu"
    seed: int = 0


@dataclass
class ToyFLLogs:
    test_acc: List[float]
    best_test_acc: List[float]
    bits_cum: List[float]
    bits_cum_normalized: List[float]
    B_FP: float
    d: int


class ToyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def create_synthetic_federated_dataset(
    num_clients: int,
    samples_per_client: int,
    input_dim: int,
    num_classes: int,
    device: torch.device,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    """
    合成データ (線形可分くらいの簡単な分類問題) を作り、
    クライアントごとに split する。
    """
    total_samples = num_clients * samples_per_client

    # 潜在的な真の線形分類器
    W_true = torch.randn(input_dim, num_classes, device=device)

    X = torch.randn(total_samples, input_dim, device=device)
    logits = X @ W_true  # (N, num_classes)
    y = torch.argmax(logits, dim=1)

    # クライアントごとに等分割
    client_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for _ in range(num_clients):
        end = start + samples_per_client
        X_c = X[start:end]
        y_c = y[start:end]
        client_data.append((X_c, y_c))
        start = end

    # テスト用データ
    X_test = torch.randn(1000, input_dim, device=device)
    logits_test = X_test @ W_true
    y_test = torch.argmax(logits_test, dim=1)

    return client_data, (X_test, y_test)


def train_client(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    config: ToyFLConfig,
    device: torch.device,
) -> None:
    """
    ローカル SGD (local_epochs エポック) を 1 クライアント分実行。
    """
    model.train()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    opt = torch.optim.SGD(model.parameters(), lr=config.lr_local)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(config.local_epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()


def evaluate_accuracy(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device
) -> float:
    """
    シンプルな精度評価。
    """
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y.to(device)).float().sum().item()
        return correct / float(y.numel())


def _compute_d_for_bits(model: nn.Module, device: torch.device) -> int:
    """
    Route1 の「量子化対象の次元 d」を flatten_deltas を通じて求める。
    """
    state = model.state_dict()
    zeros = {
        k: torch.zeros_like(v).to(device)
        for k, v in state.items()
        if torch.is_tensor(v) and v.is_floating_point()
    }
    # state_dict() はすでに device 上の tensor を持っている前提
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


def run_toy_fl(
    config: ToyFLConfig,
    bit_schedule: Optional[List[int]] = None,
) -> ToyFLLogs:
    """
    合成データ + MLP で FedAvg + QSGD Δ 量子化を回す最小ループ。

    Args:
        config: FL 設定
        bit_schedule: 長さが config.num_rounds のビットスケジュール。
            None の場合は fixed-bit (config.bit) を使用。
    """

    _set_global_seed(config.seed)

    device = torch.device(config.device)

    # グローバルモデル
    global_model = ToyNet(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    ).to(device)

    # データ作成
    samples_per_client = 200  # 適当なサイズ
    client_data, (X_test, y_test) = create_synthetic_federated_dataset(
        num_clients=config.num_clients,
        samples_per_client=samples_per_client,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        device=device,
    )

    # ラウンド数
    T = config.num_rounds

    # bit スケジュール
    if bit_schedule is None:
        bit_schedule = make_fixed_schedule(T, config.bit)
    else:
        if len(bit_schedule) != T:
            raise ValueError(
                f"bit_schedule length {len(bit_schedule)} does not match "
                f"config.num_rounds {T}"
            )

    # d と B_FP の計算
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

    # logging 用
    test_acc_list: List[float] = []
    best_test_acc_list: List[float] = []
    bits_cum: List[float] = []

    quantizer_cache: Dict[int, "QSGDQuantizer"] = {}

    best_so_far = 0.0
    cum_bits = 0.0

    # メインループ
    for t in range(T):
        b_t = bit_schedule[t]
        selected_ids = selected_ids_per_round[t]

        global_state = {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
            if torch.is_tensor(v) and v.is_floating_point()
        }

        # 量子化された Δ を集める
        quantized_deltas: List[Dict[str, torch.Tensor]] = []

        for cid in selected_ids:
            X_c, y_c = client_data[cid]

            # クライアントモデルを global からコピー
            client_model = ToyNet(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_classes=config.num_classes,
            ).to(device)
            client_model.load_state_dict(global_model.state_dict())

            # ローカルトレーニング
            train_client(client_model, X_c, y_c, config, device)

            # Δ を量子化
            with torch.no_grad():
                client_state = client_model.state_dict()
                q_delta = quantize_model_delta(
                    global_state,
                    client_state,
                    bit=b_t,
                    quantizer_cache=quantizer_cache,
                )
            quantized_deltas.append(q_delta)

        # サーバ更新: w_{t+1} = w_t + (1/m_t) sum_i Q(Δ_i)
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

        # 評価 & bits ログ
        acc = evaluate_accuracy(global_model, X_test, y_test, device)
        best_so_far = max(best_so_far, acc)

        B_t = _compute_bits_per_round(b_t, d, m_t)
        cum_bits += B_t

        test_acc_list.append(acc)
        best_test_acc_list.append(best_so_far)
        bits_cum.append(cum_bits)

    bits_cum_normalized = [b / B_FP for b in bits_cum]

    return ToyFLLogs(
        test_acc=test_acc_list,
        best_test_acc=best_test_acc_list,
        bits_cum=bits_cum,
        bits_cum_normalized=bits_cum_normalized,
        B_FP=B_FP,
        d=d,
    )
