# experiments/run_toy_optimal_schedules.py

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
from schedule.schedules import (
    make_fixed_schedule,
    make_log_schedule,
    make_two_stage_schedule,
    make_zigzag_schedule,
)


def _shift_to_target_mean(b: np.ndarray, b_avg: float) -> np.ndarray:
    """
    (1/T) sum b_t が b_avg になるように、一様シフトするヘルパー。
    （スケジュールの「形」は変えずに平均だけ合わせたいとき用）
    """
    b = np.asarray(b, dtype=float)
    delta = b_avg - float(np.mean(b))
    return b + delta


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Toy strongly-convex quadratic: "
            "optimal bit schedule vs simple schedules (fixed/two-stage/log/zigzag)."
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

    # ---- 1) P(φ) の連続最適解 ----
    b_opt_cont = solve_optimal_bits_continuous(cfg, b_avg=b_avg)
    risk_opt_cont = compute_excess_risk(cfg, b_opt_cont)

    print("=== Continuous optimal schedule (P(phi) solution) ===")
    print(f"  mean(b) = {float(np.mean(b_opt_cont)):.4f}")
    print(
        f"  min(b), max(b) = {float(np.min(b_opt_cont)):.3f}, {float(np.max(b_opt_cont)):.3f}"
    )
    print(f"  excess risk = {risk_opt_cont:.6e}")

    # 離散版（丸め + 平均調整）
    b_opt_disc = np.clip(np.round(b_opt_cont), args.b_min, args.b_max)
    b_opt_disc = _shift_to_target_mean(b_opt_disc, b_avg)
    risk_opt_disc = compute_excess_risk(cfg, b_opt_disc)

    # ---- 2) 他のスケジュール ----
    schedules: Dict[str, np.ndarray] = {}

    # fixed
    schedules[f"fixed-{int(round(b_avg))}"] = np.array(
        make_fixed_schedule(T, int(round(b_avg))), dtype=float
    )

    # two-stage & log は b_low, b_high を指定して平均は b_avg 指定で作る
    b_low = int(max(args.b_min, 2))
    b_high = int(min(args.b_max, 16))

    schedules["two-stage"] = np.array(
        make_two_stage_schedule(T, b_low=b_low, b_high=b_high, b_avg=b_avg),
        dtype=float,
    )
    schedules["log"] = np.array(
        make_log_schedule(T, b_min=b_low, b_max=b_high, b_avg=b_avg),
        dtype=float,
    )

    # zigzag は {b_low, b_high} のみ指定。
    # 平均ビットは少しズレるので、シフトで b_avg に合わせる。
    schedules["zigzag"] = np.array(
        make_zigzag_schedule(T, b_low=b_low, b_high=b_high),
        dtype=float,
    )

    # 平均ビットを揃える（形はそのまま、全体をシフト）
    for name in list(schedules.keys()):
        schedules[name] = _shift_to_target_mean(schedules[name], b_avg=b_avg)

    # ---- 3) すべてのスケジュールで excess risk を計算 ----
    print("\n=== Schedules comparison (excess risk) ===")
    print(
        f"{'name':>15}  {'mean(b)':>8}  {'min(b)':>8}  {'max(b)':>8}  {'risk':>12}  {'ratio(opt)':>12}"
    )

    # 連続 optimal / 離散 optimal も含めて集計
    all_schedules = {
        "opt-cont": b_opt_cont,
        "opt-disc": b_opt_disc,
        **schedules,
    }

    risks = {}
    for name, b in all_schedules.items():
        r = compute_excess_risk(cfg, b)
        risks[name] = r
        print(
            f"{name:>15}  "
            f"{float(np.mean(b)):8.3f}  "
            f"{float(np.min(b)):8.3f}  "
            f"{float(np.max(b)):8.3f}  "
            f"{r:12.6e}  "
            f"{r / risk_opt_cont:12.3f}"
        )

    # ---- 4) スケジュール形状の可視化 ----
    t = np.arange(T)

    plt.figure(figsize=(8, 4))
    for name, b in all_schedules.items():
        if name == "opt-cont":
            style = "-"
        elif name == "opt-disc":
            style = "--"
        else:
            style = ":"
        plt.plot(t, b, linestyle=style, marker="o", label=name)

    plt.xlabel("round t")
    plt.ylabel("bits b_t")
    plt.title("Toy strongly-convex: bit schedules")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_optimal_schedules_bits.png")

    # excess risk のバー図もついでに（opt-cont で正規化）
    plt.figure(figsize=(6, 4))
    names = list(all_schedules.keys())
    ratios = [risks[n] / risk_opt_cont for n in names]
    x = np.arange(len(names))
    plt.bar(x, ratios)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("excess risk / optimal-cont")
    plt.title("Toy strongly-convex: relative excess risk")
    plt.tight_layout()
    plt.savefig("toy_optimal_schedules_risk.png")


if __name__ == "__main__":
    main()
