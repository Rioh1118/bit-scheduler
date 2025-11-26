# quantization/rd_profile.py

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from quantization.delta_quantization import quantize_model_delta


@torch.no_grad()
def estimate_phi_emp_from_states(
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    bits_list: Sequence[int],
) -> Dict[int, float]:
    """
    Empirical RD プロファイル φ_emp(b) を推定する。

    φ_emp(b) := E ||Q_b(Δ) - Δ||^2 / E ||Δ||^2

    - global_state: そのラウンドのグローバルモデルの state_dict（FP tensor だけ）
    - client_states: 同じグローバルからローカルトレーニングした後の
      各クライアントの state_dict（FP tensor だけ）リスト
    - bits_list: 測りたいビット幅のリスト（例: [2,4,8,16,32]）

    量子化器には quantize_model_delta を使うので、
    本番と同じ block QSGD ロジックがそのまま使われる。
    """
    if len(client_states) == 0:
        raise ValueError("client_states must be non-empty")

    # Δ のノルム^2 をまず一度だけ計算（bit に依存しない）
    denom = 0.0
    deltas_per_client: List[Dict[str, torch.Tensor]] = []

    for client_state in client_states:
        delta_state: Dict[str, torch.Tensor] = {}
        for name, g_param in client_state.items():
            if name not in global_state:
                continue
            g = g_param
            g0 = global_state[name]
            if not (torch.is_tensor(g) and torch.is_tensor(g0)):
                continue
            if not (g.is_floating_point() and g0.is_floating_point()):
                continue
            delta = (g - g0).detach()
            delta_state[name] = delta
            denom += float(delta.pow(2).sum().item())
        deltas_per_client.append(delta_state)

    if denom == 0.0:
        raise RuntimeError("All deltas are zero; cannot estimate RD profile.")

    # 各ビット幅ごとの分子（ノイズの二乗ノルムの期待値）を計算
    num_per_bit: Dict[int, float] = {int(b): 0.0 for b in bits_list}
    quantizer_cache: Dict[int, "QSGDQuantizer"] = {}

    for bit in bits_list:
        b = int(bit)
        for client_state, delta_state in zip(client_states, deltas_per_client):
            # 量子化された Δ を、本番と同じ quantize_model_delta で計算
            q_delta_state = quantize_model_delta(
                global_state,
                client_state,
                bit=b,
                quantizer_cache=quantizer_cache,
            )
            # Q_b(Δ) - Δ のノルム^2 を足す
            for name, delta in delta_state.items():
                if name not in q_delta_state:
                    continue
                diff = (q_delta_state[name] - delta).detach()
                num_per_bit[b] += float(diff.pow(2).sum().item())

    phi_emp: Dict[int, float] = {}
    for b, num in num_per_bit.items():
        phi_emp[b] = num / denom

    return phi_emp
