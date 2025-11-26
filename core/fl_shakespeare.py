# core/fl_shakespeare.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.data_shakespeare_fl import (
    ShakespeareDataConfig,
    build_shakespeare_federated_loaders,
)
from quantization.delta_quantization import flatten_deltas, quantize_model_delta
from quantization.qsgd import bit_to_s_levels
from schedule.schedules import make_fixed_schedule


@dataclass
class ShakespeareFLConfig:
    num_clients: int = 100  # 実際には ~700 くらいあるはずだが、まずは 100 から
    clients_per_round: int = 10
    num_rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 32
    lr_local: float = 0.5
    bit: int = 8
    device: str = "cuda"
    seed: int = 0

    # data
    data_root: str = "./data/shakespeare"
    train_json: str = "train.json"
    test_json: str = "test.json"
    seq_len: int = 80
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


class ShakespeareLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,  # (seq, batch, feat)
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) の整数 id テンソル
        戻り値: (batch * seq_len, vocab_size) のロジット
        """
        # (batch, seq) -> (batch, seq, emb)
        emb = self.emb(x)
        # LSTM は (seq, batch, emb) なので transpose
        emb = emb.transpose(0, 1)  # (seq, batch, emb)
        out, _ = self.lstm(emb)  # (seq, batch, hidden)
        out = out.transpose(0, 1)  # (batch, seq, hidden)
        logits = self.fc(out)  # (batch, seq, vocab)
        return logits.reshape(-1, self.vocab_size)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    next-char の「1 ステップ精度」を accuracy として定義する。
    （単に全トークンの予測が当たった割合）
    """
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)  # (batch, seq_len)
            logits = model(xb)  # (batch * seq_len, vocab)
            preds = torch.argmax(logits, dim=1)  # (batch * seq_len,)
            targets = yb.reshape(-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
    if total_tokens == 0:
        return 0.0
    return total_correct / float(total_tokens)


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
    config: ShakespeareFLConfig,
    device: torch.device,
) -> None:
    model.train()
    model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=config.lr_local)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # pad=0 を無視

    for epoch in range(config.local_epochs):
        total_loss = 0.0
        total_tokens = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)  # (batch, seq)
            opt.zero_grad()
            logits = model(xb)  # (batch * seq, vocab)
            loss = loss_fn(logits, yb.reshape(-1))
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.numel()
            total_tokens += yb.numel()
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = 0.0
        print(f"local epoch {epoch}: token-avg loss={avg_loss:.4f}")


def run_shakespeare_fl(
    config: ShakespeareFLConfig,
    bit_schedule: Optional[List[int]] = None,
) -> FLLogs:
    """
    Shakespeare + LSTM + FedAvg + QSGD Δ 量子化の FL ループ。
    """
    _set_global_seed(config.seed)

    device = torch.device(config.device)

    # データ準備
    data_config = ShakespeareDataConfig(
        data_root=config.data_root,
        train_json=config.train_json,
        test_json=config.test_json,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        num_clients=config.num_clients,
    )
    client_loaders, test_loader, char2idx, idx2char = (
        build_shakespeare_federated_loaders(data_config)
    )

    vocab_size = len(char2idx)

    # 実際に使うクライアント数は DataLoader の数に合わせる
    num_clients_actual = len(client_loaders)
    if num_clients_actual < config.num_clients:
        print(
            f"Warning: requested num_clients={config.num_clients}, "
            f"but only {num_clients_actual} clients had enough data. Using {num_clients_actual}."
        )
    num_clients = num_clients_actual

    # グローバルモデル
    global_model = ShakespeareLSTM(vocab_size=vocab_size).to(device)

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
    all_client_ids = list(range(num_clients))
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

        global_state = {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
            if torch.is_tensor(v) and v.is_floating_point()
        }

        quantized_deltas: List[Dict[str, torch.Tensor]] = []

        for cid in selected_ids:
            client_loader = client_loaders[cid]

            # クライアントモデル作成 & 初期化
            client_model = ShakespeareLSTM(vocab_size=vocab_size).to(device)
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
            f"bits_norm={cum_bits / B_FP:.3f}"
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
