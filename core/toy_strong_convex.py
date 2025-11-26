# core/toy_strong_convex.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ToyQuadraticConfig:
    """
    1次元の強凸二次損失 f(w) = (mu/2) (w - w*)^2 に対する
    ノイズ付き SGD モデルの設定。
    """

    T: int = 50  # ラウンド数
    mu: float = 1.0  # 強凸パラメータ
    lr: float = 0.2  # 学習率 eta
    sigma2_base: float = 1.0  # ノイズのスケール sigma_0^2
    d_eff: float = 1.0  # RD モデル上の「有効次元」
    w0: float = 1.0  # 初期点
    w_star: float = 0.0  # 最適解


def rd_phi(b: np.ndarray, d_eff: float = 1.0) -> np.ndarray:
    """
    RD 型モデル: phi(b) = 2^{-gamma b}, gamma = 2 / d_eff.
    b: shape (T,)
    """
    b = np.asarray(b, dtype=float)
    gamma = 2.0 / float(d_eff)
    return 2.0 ** (-gamma * b)


def alpha_weights(cfg: ToyQuadraticConfig) -> np.ndarray:
    """
    P(φ) の重み alpha_t = rho^{T-1-t} を返す。
    rho = (1 - eta * mu)^2.
    """
    rho = (1.0 - cfg.lr * cfg.mu) ** 2
    T = cfg.T
    # exponents: T-1, T-2, ..., 0
    exponents = np.arange(T - 1, -1, -1, dtype=float)
    alpha = rho**exponents
    return alpha


def compute_excess_risk(cfg: ToyQuadraticConfig, bits: Sequence[float]) -> float:
    """
    与えられた bit スケジュール b_t に対して、
    1次元強凸二次の期待 excess risk E[f(w_T) - f*] を解析的に計算する。

    bits: 長さ T の配列（float も可）
    """
    b = np.asarray(bits, dtype=float)
    if b.shape[0] != cfg.T:
        raise ValueError(f"bits length {b.shape[0]} != T={cfg.T}")

    alpha = alpha_weights(cfg)
    phi = rd_phi(b, d_eff=cfg.d_eff)

    rho = (1.0 - cfg.lr * cfg.mu) ** 2
    e0 = cfg.w0 - cfg.w_star

    # E[e_T^2] = rho^T e0^2 + eta^2 sigma0^2 sum_{t} alpha_t phi(b_t)
    bias2 = (rho**cfg.T) * (e0**2)
    noise_term = (cfg.lr**2) * cfg.sigma2_base * float(np.sum(alpha * phi))
    e2 = bias2 + noise_term

    # 強凸二次: f(w) - f* = (mu/2) e^2
    return 0.5 * cfg.mu * e2


def solve_optimal_bits_continuous(
    cfg: ToyQuadraticConfig,
    b_avg: float,
    d_eff: float | None = None,
) -> np.ndarray:
    """
    P(φ) の連続最適解を解析的に求める。

        minimize  sum_t alpha_t phi(b_t)
        s.t.      (1/T) sum_t b_t = b_avg

    phi(b) = 2^{-gamma b}, gamma = 2 / d_eff
    alpha_t = rho^{T-1-t}, rho = (1 - eta * mu)^2

    ここでは box 制約や単調制約は課さず、連続解をそのまま返す。
    """
    if d_eff is None:
        d_eff = cfg.d_eff

    T = cfg.T
    alpha = alpha_weights(cfg)

    gamma = 2.0 / float(d_eff)
    # 解析解:
    # b_t = (1/(gamma ln 2)) * ln(alpha_t * gamma ln 2) - const
    B = 1.0 / (gamma * np.log(2.0))  # = 1 / (gamma ln 2)
    C = B * np.log(alpha * gamma * np.log(2.0))  # shape (T,)

    # (1/T) sum_t b_t = b_avg になるように定数シフト
    # b_t = C_t - D とすると、
    #   (1/T) sum_t (C_t - D) = b_avg
    # => (1/T) sum_t C_t - D = b_avg
    # => D = (1/T) sum_t C_t - b_avg
    D = float(np.mean(C) - b_avg)

    b_opt = C - D  # shape (T,)
    return b_opt
