# schedule/schedules.py

from __future__ import annotations

import math
from typing import List


def make_fixed_schedule(T: int, b_fixed: int) -> List[int]:
    """
    最も単純な fixed-bit スケジュール。
    """
    return [int(b_fixed)] * T


def make_two_stage_schedule(T: int, b_low: int, b_high: int, b_avg: float) -> List[int]:
    """
    2-stage (low -> high) スケジュール。
    平均ビットが b_avg になるように T0 を決める。

        b_avg = (T0 * b_low + (T - T0) * b_high) / T
        => T0 = T * (b_high - b_avg) / (b_high - b_low)
    """
    if b_high == b_low:
        return [b_low] * T

    T0_float = T * (b_high - b_avg) / (b_high - b_low)
    T0 = int(round(T0_float))
    T0 = max(0, min(T0, T))
    return [b_low] * T0 + [b_high] * (T - T0)


def _build_log_schedule(T: int, b_min: int, b_max: int, gamma: float) -> List[int]:
    """
    内部用: 指定した gamma で log-increasing を生成して
    [b_min, b_max] にクリップ + 単調増加補正（DR1）する。
    """
    bs: List[int] = []
    for t in range(T):
        # t=0 で log(1) = 0 になる形
        b_cont = b_min + gamma * math.log(1.0 + t)
        b = int(round(b_cont))
        b = max(b_min, min(b, b_max))
        bs.append(b)

    # monotone correction (非減少にする) – DR1
    for t in range(1, T):
        if bs[t] < bs[t - 1]:
            bs[t] = bs[t - 1]

    return bs


def make_log_schedule(
    T: int,
    b_min: int,
    b_max: int,
    b_avg: float,
    max_gamma: float = 10.0,
    num_iters: int = 30,
) -> List[int]:
    """
    Route1 主役の log-increasing スケジュール。
    平均ビットが b_avg に近くなるよう gamma を二分探索で決める。

        b_t^cont = b_min + gamma * log(1 + t)

    その後 [b_min, b_max] にクリップし、単調増加補正を行う。
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if not (b_min <= b_avg <= b_max):
        # 若干ズレていてもなんとかすることはできるが、ここでは素直にエラー
        raise ValueError(f"b_avg must be in [b_min, b_max], got {b_avg}")

    lo, hi = 0.0, max_gamma

    def avg_bits(gamma: float) -> float:
        bs = _build_log_schedule(T, b_min, b_max, gamma)
        return sum(bs) / float(T)

    # 二分探索
    for _ in range(num_iters):
        mid = 0.5 * (lo + hi)
        avg = avg_bits(mid)
        if avg < b_avg:
            lo = mid
        else:
            hi = mid

    gamma_star = 0.5 * (lo + hi)
    return _build_log_schedule(T, b_min, b_max, gamma_star)


def make_zigzag_schedule(T: int, b_low: int, b_high: int) -> List[int]:
    """
    monotone をわざと破る悪い例 – zig-zag スケジュール。
    平均ビットは (b_low + b_high)/2。
    """
    return [b_low if (t % 2 == 0) else b_high for t in range(T)]


def make_decreasing_two_stage(
    T: int, b_low: int, b_high: int, b_avg: float
) -> List[int]:
    """
    two-stage の逆順 (high -> low)。DR1 を破るための対照実験用。
    """
    inc = make_two_stage_schedule(T, b_low, b_high, b_avg)
    return list(reversed(inc))


def make_opt_schedule_exponential_weights(
    T: int,
    b_min: int,
    b_max: int,
    b_avg: float,
    rho: float = 0.01,
    num_iters: int = 50,
) -> List[int]:
    """
    Route1 の Problem P(φ) を
      - α_k ≈ (1 - rho)^k （幾何減衰の重み）
      - φ(b) ≈ exp(-b)   （RD 型 φ の簡略版）
    でモデル化したときの「ほぼ最適」スケジュールを water-filling で求める。

    KKT 条件から、
        α_{T-1-t} * φ'(b_t) + λ = 0
      φ(b)=exp(-b) ⇒ φ'(b) = -exp(-b)
      ⇒ exp(-b_t) ∝ λ / α_{T-1-t}
      ⇒ b_t^cont = log(α_{T-1-t}) - log(λ)
    となるので、λ を二分探索で調整しつつ
    [b_min, b_max] でクリップしたものを用いる。

    戻り値は最近傍の整数 bit に丸めたスケジュール。
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if b_min > b_max:
        raise ValueError(f"b_min must be <= b_max, got {b_min}, {b_max}")

    # 目的関数の重み列（誤差 bound の α_{T-1-t} に対応）
    # t が大きいほど重みが大きくなる：α_{T-1-t} = (1-ρ)^{T-1-t}
    weights: List[float] = [(1.0 - rho) ** (T - 1 - t) for t in range(T)]
    B_target = b_avg * T

    def build_schedule_for_lambda(lmbda: float) -> List[float]:
        bs_cont: List[float] = []
        for w in weights:
            # log(α_{T-1-t}) - log(λ)
            b = math.log(max(w, 1e-12)) - math.log(lmbda)
            b = max(b_min, min(b_max, b))
            bs_cont.append(b)
        return bs_cont

    lo, hi = 1e-6, 1e6  # λ の探索レンジ（対数スケールなのでかなり広めでOK）
    for _ in range(num_iters):
        mid = 0.5 * (lo + hi)
        bs = build_schedule_for_lambda(mid)
        if sum(bs) > B_target:
            # bits を使いすぎ ⇒ λ を大きくして b_t を下げる
            lo = mid
        else:
            # bits が足りない ⇒ λ を小さくして b_t を上げる
            hi = mid

    bs_cont = build_schedule_for_lambda(0.5 * (lo + hi))
    # 最近傍整数に丸めたあと、最後に monotone 補正（DR1）をかける
    bs_int = [int(round(b)) for b in bs_cont]
    for t in range(1, T):
        if bs_int[t] < bs_int[t - 1]:
            bs_int[t] = bs_int[t - 1]

    # 最終的に平均ビットは b_avg に近いはずだが、多少ズレるのは許容。
    return bs_int
