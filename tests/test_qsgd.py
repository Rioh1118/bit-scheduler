import math

import torch

from quantization.qsgd import QSGDQuantizer, bit_to_s_levels


def test_bit_to_s_levels_basic():
    # b = 2 -> s = 1 (levels: {0, 1})
    assert bit_to_s_levels(2) == 1
    # b = 3 -> s = 3
    assert bit_to_s_levels(3) == 3
    # b = 4 -> s = 7
    assert bit_to_s_levels(4) == 7

    # Check the relationship b = 1 + ceil(log2(s+1))
    for b in [2, 3, 4, 8, 16]:
        s = bit_to_s_levels(b)
        b_recovered = 1 + math.ceil(math.log2(s + 1))
        assert b_recovered == b


def test_qsgd_zero_vector():
    d = 100
    q = QSGDQuantizer(s_levels=bit_to_s_levels(8))

    g = torch.zeros(d)
    gq = q(g)

    assert torch.allclose(gq, g), "Quantized zero vector should be exactly zero."


def test_qsgd_unbiased_empirical():
    torch.manual_seed(0)

    d = 1000
    num_samples = 2000
    b = 8
    s = bit_to_s_levels(b)
    q = QSGDQuantizer(s_levels=s)

    # 固定の g を何度も量子化して平均をとる
    g = torch.randn(d)
    accum = torch.zeros_like(g)

    for _ in range(num_samples):
        gq = q(g)
        accum += gq

    g_hat = accum / num_samples

    # 相対 L2 誤差と最大絶対誤差をチェック
    diff = g_hat - g
    rel_l2 = diff.norm().item() / g.norm().item()
    max_abs = diff.abs().max().item()

    # そこまで厳しくしないが、「十分小さい」ことを確認
    assert rel_l2 < 0.02, f"Relative L2 error too large: {rel_l2}"
    assert max_abs < 0.1, f"Max abs error too large: {max_abs}"


if __name__ == "__main__":
    # 手動実行用の簡易チェック
    test_bit_to_s_levels_basic()
    test_qsgd_zero_vector()
    test_qsgd_unbiased_empirical()
    print("All QSGD tests passed.")

