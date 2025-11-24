# quantization/delta_quantization.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .qsgd import QSGDQuantizer, bit_to_s_levels

TensorState = Dict[str, torch.Tensor]

# メタ情報: (name, shape, numel, device, dtype)
MetaEntry = Tuple[str, torch.Size, int, torch.device, torch.dtype]

BLOCK_SIZE = 4096


def _iter_float_params(global_state: TensorState, client_state: TensorState):
    """
    state_dict から「量子化対象にする float tensor だけ」を取り出す内部 helper。
    - tensor でないもの
    - 非 float (int/bool 等)
    は無視する。
    """
    for name, g_param in global_state.items():
        if not torch.is_tensor(g_param):
            continue
        if not g_param.is_floating_point():
            continue
        if name not in client_state:
            raise KeyError(f"Key {name} not found in client_state.")
        c_param = client_state[name]
        if c_param.shape != g_param.shape:
            raise ValueError(
                f"Shape mismatch for {name}: global {g_param.shape}, "
                f"client {c_param.shape}"
            )
        yield name, g_param, c_param


def flatten_deltas(
    global_state: TensorState, client_state: TensorState
) -> Tuple[torch.Tensor, List[MetaEntry]]:
    """
    Δ = client_state - global_state を 1 本のベクトルに flatten する。

    Returns:
        g_flat: 1D tensor (sum of all numel)
        meta:   list of (name, shape, numel, device, dtype)
               → unflatten のための情報
    """
    flats: List[torch.Tensor] = []
    meta: List[MetaEntry] = []

    for name, g_param, c_param in _iter_float_params(global_state, client_state):
        # 計算グラフは不要なので detach
        delta = c_param.detach() - g_param.detach()
        delta_flat = delta.reshape(-1)
        flats.append(delta_flat)
        meta.append((name, delta.shape, delta_flat.numel(), delta.device, delta.dtype))

    if not flats:
        raise ValueError("No floating-point tensors found to quantize.")

    # すべて同じ device/dtype を想定（通常のモデルならそうなっている）
    g_flat = torch.cat(flats, dim=0)
    return g_flat, meta


def unflatten_to_state_dict(g_flat: torch.Tensor, meta: List[MetaEntry]) -> TensorState:
    """
    flatten_deltas で作った meta に従って 1D ベクトルを state_dict 形式に戻す。

    Args:
        g_flat: 1D tensor
        meta:   list of (name, shape, numel, device, dtype)

    Returns:
        state_dict 形式の dict[name -> tensor]
    """
    out: TensorState = {}
    offset = 0
    total_numel = g_flat.numel()

    for name, shape, numel, device, dtype in meta:
        if offset + numel > total_numel:
            raise ValueError("g_flat is too short for provided meta information.")
        part = g_flat[offset : offset + numel]
        offset += numel
        out[name] = part.view(shape).to(device=device, dtype=dtype)

    if offset != total_numel:
        # 余りがある場合は meta と g_flat がずれている
        raise ValueError(
            f"g_flat has extra elements: used {offset}, total {total_numel}."
        )

    return out


def quantize_model_delta(
    global_state: TensorState,
    client_state: TensorState,
    bit: int,
    quantizer_cache: Dict[int, QSGDQuantizer] | None = None,
) -> TensorState:
    """
    モデルの state_dict 同士から Δ を計算し、それを QSGD 量子化する。

    Args:
        global_state: server の w_t.state_dict()
        client_state: client の w_i^{(t)}.state_dict()
        bit:          実効ビット幅 b_t
        quantizer_cache:
            {bit: QSGDQuantizer} のキャッシュ。None の場合は内部で dict を作る。

    Returns:
        quantized_delta_state: dict[name -> quantized Δ tensor]
            → これをサーバ側で平均して w_t に足す。
    """
    if quantizer_cache is None:
        quantizer_cache = {}

    s_levels = bit_to_s_levels(bit)
    quantizer = quantizer_cache.get(bit)
    if quantizer is None:
        quantizer = QSGDQuantizer(s_levels=s_levels)
        quantizer_cache[bit] = quantizer

    g_flat, meta = flatten_deltas(global_state, client_state)

    d = g_flat.numel()
    q_flat = torch.empty_like(g_flat)

    for start in range(0, d, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, d)
        block = g_flat[start:end]

        # block は必ず 1D なので QSGDQuantizer の仕様と一致
        q_block = quantizer(block)
        q_flat[start:end] = q_block

    quantized_delta_state = unflatten_to_state_dict(q_flat, meta)
    return quantized_delta_state
