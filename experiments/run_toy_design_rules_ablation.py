# experiments/run_toy_design_rules_ablation.py

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from core.toy_strong_convex import (
    ToyQuadraticConfig,
    compute_excess_risk,
    solve_optimal_bits_continuous,
)
from schedule.schedules import make_fixed_schedule, make_log_schedule


def _shift_to_target_mean(b: np.ndarray, b_avg: float) -> np.ndarray:
    """
    (1/T) sum b_t が b_avg になるように、一様シフトする。
    スケジュールの「形」は変えずに平均だけ合わせたいときに使う。
    """
    b = np.asarray(b, dtype=float)
    delta = b_avg - float(np.mean(b))
    return b + delta


def _random_permute(b: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    DR1 用: 同じ multiset の bit をランダム permute したスケジュールを作る。
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(b))
    return np.asarray(b, dtype=float)[perm]


def _linear_increasing_schedule(
    T: int, b_min: float, b_max: float, b_avg: float
) -> np.ndarray:
    """
    DR3 用: 単純な linearly increasing なスケジュールを作る。
        b_t^{lin} = b_min + (b_max - b_min) * t / (T - 1)
    としてから、平均が b_avg になるように一様シフトする。
    """
    if T == 1:
        base = np.array([b_avg], dtype=float)
    else:
        t = np.linspace(0.0, 1.0, T)
        base = b_min + (b_max - b_min) * t
    base = _shift_to_target_mean(base, b_avg=b_avg)
    return base


def _round_to_allowed(b: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """
    DR4 用: 連続的な bit を allowed の最近傍に丸める。
    """
    b = np.asarray(b, dtype=float)
    allowed = np.asarray(allowed, dtype=float)
    out = np.empty_like(b)
    for i, x in enumerate(b):
        idx = int(np.argmin((allowed - x) ** 2))
        out[i] = allowed[idx]
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Toy strongly-convex quadratic: Design Rules ablation "
            "(DR1/DR3/DR4) on bit schedules."
        )
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--b-avg", type=float, default=8.0)
    parser.add_argument("--b-min", type=float, default=2.0)
    parser.add_argument("--b-max", type=float, default=16.0)

    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--d-eff", type=float, default=1.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w-star", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0, help="random permute seed")
    args = parser.parse_args()

    T = args.rounds
    b_avg = args.b_avg

    cfg = ToyQuadraticConfig(
        T=T,
        mu=args.mu,
        lr=args.lr,
        sigma2_base=args.sigma2,
        d_eff=args.d_eff,
        w0=args.w0,
        w_star=args.w_star,
    )

    # =====================================================================
    # 1) 基準: P(φ) の連続最適解 (opt-cont) & 離散最適解 (opt-disc)
    # =====================================================================
    b_opt_cont = solve_optimal_bits_continuous(cfg, b_avg=b_avg)
    risk_opt_cont = compute_excess_risk(cfg, b_opt_cont)

    # DR4: 連続解を [b_min, b_max] にクリップして丸める
    b_opt_disc = np.clip(np.round(b_opt_cont), args.b_min, args.b_max)
    # 平均は若干ズレるので、再度 b_avg に合わせてシフト
    b_opt_disc = _shift_to_target_mean(b_opt_disc, b_avg=b_avg)
    risk_opt_disc = compute_excess_risk(cfg, b_opt_disc)

    # DR4 (実験で使う bit set {2,4,8,16,32} を想定した丸め)
    allowed_bits = np.array([2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)
    b_opt_disc_24816 = _round_to_allowed(b_opt_cont, allowed_bits)
    b_opt_disc_24816 = np.clip(b_opt_disc_24816, args.b_min, args.b_max)
    b_opt_disc_24816 = _shift_to_target_mean(b_opt_disc_24816, b_avg=b_avg)
    risk_opt_disc_24816 = compute_excess_risk(cfg, b_opt_disc_24816)

    # =====================================================================
    # 2) Route1 的な simple スケジュール: log / fixed
    # =====================================================================
    # log-increasing (DR3 の "good" 側)
    b_log = np.array(
        make_log_schedule(T, b_min=int(args.b_min), b_max=int(args.b_max), b_avg=b_avg),
        dtype=float,
    )

    # fixed (one of the most naive baselines)
    b_fixed = np.array(
        make_fixed_schedule(T, int(round(b_avg))),
        dtype=float,
    )

    # =====================================================================
    # 3) DR1: monotone vs random permute (multiset は同じ)
    # =====================================================================
    # log は monotone（非減少）なので、これをベーススケジュールとみなす
    b_log_perm = _random_permute(b_log, seed=args.seed)  # DR1: monotone を外した例

    # =====================================================================
    # 4) DR3: log vs linear-increasing
    # =====================================================================
    b_linear = _linear_increasing_schedule(
        T=T, b_min=args.b_min, b_max=args.b_max, b_avg=b_avg
    )

    # =====================================================================
    # 5) DR4: log を {2,4,8,16,32} に丸めた場合
    # =====================================================================
    b_log_disc_24816 = _round_to_allowed(b_log, allowed_bits)
    b_log_disc_24816 = np.clip(b_log_disc_24816, args.b_min, args.b_max)
    b_log_disc_24816 = _shift_to_target_mean(b_log_disc_24816, b_avg=b_avg)

    # =====================================================================
    # 6) すべてのスケジュールで excess risk を計算
    # =====================================================================
    schedules: Dict[str, np.ndarray] = {
        "opt-cont": b_opt_cont,
        "opt-disc": b_opt_disc,
        "opt-disc-24816": b_opt_disc_24816,
        "log": b_log,
        "log-permute": b_log_perm,
        "linear": b_linear,
        "fixed": b_fixed,
        "log-disc-24816": b_log_disc_24816,
    }

    risks: Dict[str, float] = {
        name: compute_excess_risk(cfg, b) for name, b in schedules.items()
    }

    print("=== Toy Design Rules Ablation (excess risk) ===")
    print(
        f"{'name':>15}  {'mean(b)':>8}  {'min(b)':>8}  {'max(b)':>8}  {'risk':>12}  {'ratio(opt)':>12}"
    )
    for name, b in schedules.items():
        r = risks[name]
        print(
            f"{name:>15}  "
            f"{float(np.mean(b)):8.3f}  "
            f"{float(np.min(b)):8.3f}  "
            f"{float(np.max(b)):8.3f}  "
            f"{r:12.6e}  "
            f"{r / risk_opt_cont:12.3f}"
        )

    # =====================================================================
    # 7) 可視化: bit スケジュールの形
    # =====================================================================
    t = np.arange(T)

    plt.figure(figsize=(8, 4))
    for name in ["opt-cont", "log", "linear", "log-permute"]:
        b = schedules[name]
        if name == "opt-cont":
            style = "-"
        elif name == "log":
            style = "--"
        elif name == "linear":
            style = ":"
        else:
            style = "-."
        plt.plot(t, b, linestyle=style, marker="o", label=name)

    plt.xlabel("round t")
    plt.ylabel("bits b_t")
    plt.title("Toy Design Rules Ablation: bit schedules (DR1/DR3)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_design_rules_bits.png")

    # =====================================================================
    # 8) 可視化: relative excess risk (opt-cont で正規化)
    # =====================================================================
    plt.figure(figsize=(8, 4))
    names = list(schedules.keys())
    ratios = [risks[n] / risk_opt_cont for n in names]
    x = np.arange(len(names))
    plt.bar(x, ratios)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("excess risk / optimal-cont")
    plt.title("Toy Design Rules Ablation: relative excess risk")
    plt.tight_layout()
    plt.savefig("toy_design_rules_risk.png")


if __name__ == "__main__":
    main()
