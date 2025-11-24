# tests/test_delta_quantization.py

import torch

from quantization.delta_quantization import (flatten_deltas,
                                             quantize_model_delta,
                                             unflatten_to_state_dict)
from quantization.qsgd import QSGDQuantizer, bit_to_s_levels


def test_flatten_unflatten_roundtrip():
    # シンプルな state_dict で flatten→unflatten が元に戻るか確認
    state = {
        "w1": torch.randn(3, 4),
        "w2": torch.randn(5),
    }
    # global_state = 0, client_state = state という形で Δ = state にする
    zeros = {k: torch.zeros_like(v) for k, v in state.items()}

    g_flat, meta = flatten_deltas(zeros, state)
    assert g_flat.ndim == 1
    assert g_flat.numel() == state["w1"].numel() + state["w2"].numel()

    restored = unflatten_to_state_dict(g_flat, meta)
    assert restored.keys() == state.keys()
    for k in state.keys():
        assert torch.allclose(restored[k], state[k])


def test_quantize_model_delta_shapes():
    # 形だけチェックしておく
    d1, d2 = (3, 4), (5,)
    global_state = {
        "w1": torch.zeros(d1),
        "w2": torch.zeros(d2),
    }
    client_state = {
        "w1": torch.randn(d1),
        "w2": torch.randn(d2),
    }

    cache = {}
    q_delta = quantize_model_delta(global_state, client_state, bit=8, quantizer_cache=cache)

    assert set(q_delta.keys()) == set(global_state.keys())
    for k in global_state.keys():
        assert q_delta[k].shape == global_state[k].shape


def test_quantize_model_delta_unbiased_empirical():
    torch.manual_seed(0)

    d = 1000
    num_samples = 2000
    b = 8

    # 単一パラメータの state_dict
    g = torch.randn(d)
    global_state = {"w": torch.zeros_like(g)}
    client_state = {"w": g.clone()}

    cache = {}
    accum = torch.zeros_like(g)

    for _ in range(num_samples):
        q_delta = quantize_model_delta(global_state, client_state, bit=b, quantizer_cache=cache)
        accum += q_delta["w"]

    g_hat = accum / num_samples

    diff = g_hat - g
    rel_l2 = diff.norm().item() / g.norm().item()
    max_abs = diff.abs().max().item()

    # qsgd.py のテストと同じくらい緩めの基準
    assert rel_l2 < 0.02, f"Relative L2 error too large: {rel_l2}"
    assert max_abs < 0.1, f"Max abs error too large: {max_abs}"


if __name__ == "__main__":
    test_flatten_unflatten_roundtrip()
    test_quantize_model_delta_shapes()
    test_quantize_model_delta_unbiased_empirical()
    print("All delta_quantization tests passed.")

