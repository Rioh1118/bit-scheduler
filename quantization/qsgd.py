from __future__ import annotations

from dataclasses import dataclass

import torch


def bit_to_s_levels(b: int)->int:
    """
    Convert effective bits per coordinate b to QSGD's s (number of levels  - 1)

    We use the  relationship
        b = 1 + ceil(log2(s+1))

    For b >=2, choosing s = 2^(b-1) -  satisfies this exactly.
    """
    if b < 2:
        raise ValueError(f"bit_to_s_levels expects b >= 2, got {b}")
    return (1 << (b - 1)) - 1  # 2^(b-1) - 1

@dataclass
class QSGDQuantizer:
    """
    QSGD-style L2-normalized stochastic uniform quantizer (vector version).

    This implementation assumes a *1D tensor* g of shape (d,).
    The caller（flatten helper）側でモデルを 1 ベクトルにしてから呼ぶ想定。

    Quantization steps:
        1. r = ||g||_2
        2. if r == 0: return 0
        3. y = g / r
        4. u = |y| in [0, 1]
        5. scaled = u * s, where s is the number of quantization levels - 1
        6. stochastic rounding between floor(scaled)/s and ceil(scaled)/s
        7. reattach sign and scale back by r

    This quantizer is unbiased in expectation: E[Q(g) | g] = g.
    """

    s_levels: int                      # s in {1, 2, ...}
    eps: float = 1e-12                 # for ||g|| ≈ 0 判定

    def __post_init__(self) -> None:
        if self.s_levels < 1:
            raise ValueError(f"s_levels must be >= 1, got {self.s_levels}")

    def __call__(self, g: torch.Tensor) -> torch.Tensor:
        return self.quantize(g)

    def quantize(self, g: torch.Tensor) -> torch.Tensor:
        """
        Quantize a 1D tensor g using QSGD.

        Args:
            g: torch.Tensor of shape (d,) and float dtype.

        Returns:
            Quantized tensor of shape (d,) on the same device/dtype as g.
        """
        if g.dim() != 1:
            raise ValueError(
                f"QSGDQuantizer expects a 1D tensor, got shape {tuple(g.shape)}"
            )

        # L2 norm
        r = torch.linalg.norm(g, ord=2)
        if r.item() < self.eps:
            # If the vector is (almost) zero, return an exact zero vector
            return torch.zeros_like(g)

        # Normalize
        y = g / r
        u = y.abs()  # in [0, 1]

        s = float(self.s_levels)

        # Map [0,1] -> [0, s] and do stochastic rounding on the integer grid
        scaled = u * s
        lower = torch.floor(scaled)
        upper = torch.clamp(lower + 1.0, max=s)  # ensure <= s

        prob_upper = scaled - lower  # ∈ [0, 1)
        # Stochastic decision
        rnd = torch.rand_like(prob_upper)
        is_upper = (rnd < prob_upper).to(g.dtype)

        # Quantized magnitude in [0, 1]
        q_u = (lower * (1.0 - is_upper) + upper * is_upper) / s

        # Reattach sign and L2 norm
        q_y = q_u * torch.sign(y)
        q_g = q_y * r

        return q_g
