# experiments/run_toy_schedules.py

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt

from core.fl_toy import ToyFLConfig, run_toy_fl
from schedule.schedules import (
    make_decreasing_two_stage,
    make_fixed_schedule,
    make_log_schedule,
    make_two_stage_schedule,
    make_zigzag_schedule,
)


def main():
    parser = argparse.ArgumentParser(
        description="Toy FedAvg + QSGD: compare Route1-style bit schedules."
    )
    parser.add_argument("--rounds", type=int, default=30, help="Number of FL rounds.")
    parser.add_argument("--clients", type=int, default=20, help="Number of clients.")
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=5,
        help="Participating clients per round.",
    )
    parser.add_argument("--b-avg", type=float, default=8.0, help="Target average bits.")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    T = args.rounds
    b_avg = args.b_avg

    # 共通の FL 設定（Toy）
    config = ToyFLConfig(
        num_clients=args.clients,
        clients_per_round=args.clients_per_round,
        num_rounds=T,
        input_dim=20,
        hidden_dim=64,
        num_classes=2,
        local_epochs=1,
        batch_size=32,
        lr_local=0.1,
        bit=int(b_avg),  # fixed 用のデフォルト
        device=args.device,
        seed=args.seed,
    )

    # ---- bit スケジュールたち ----
    schedules: Dict[str, list[int]] = {}

    # 1) Fixed-bit baseline
    schedules[f"fixed-{int(b_avg)}"] = make_fixed_schedule(T, int(b_avg))

    # 2) Two-stage (low -> high) with the same average bits
    b_low = 4
    b_high = 16
    schedules["two-stage-4-16"] = make_two_stage_schedule(
        T, b_low=b_low, b_high=b_high, b_avg=b_avg
    )

    # 3) Log-increasing (Route1 主役)
    schedules["log-4-16"] = make_log_schedule(
        T,
        b_min=b_low,
        b_max=b_high,
        b_avg=b_avg,
    )

    # 4) Non-monotone な悪い例（平均ビットは必ずしも b_avg にはならない）
    schedules["zigzag-4-16"] = make_zigzag_schedule(T, b_low=4, b_high=16)
    schedules["decreasing-4-16"] = make_decreasing_two_stage(
        T, b_low=4, b_high=16, b_avg=b_avg
    )

    print("Bit schedules (first few entries):")
    for name, sched in schedules.items():
        avg = sum(sched) / len(sched)
        print(f"  {name}: avg={avg:.2f}, head={sched[:5]}")

    # ---- 実験実行 ----
    results = {}

    for name, bit_schedule in schedules.items():
        print(f"\nRunning schedule: {name}")
        logs = run_toy_fl(config, bit_schedule=bit_schedule)
        results[name] = logs
        print(
            f"  final best acc = {logs.best_test_acc[-1]:.4f}, "
            f"bits_norm_last = {logs.bits_cum_normalized[-1]:.3f}"
        )

    # ---- プロット: best-so-far accuracy vs normalized cumulative bits ----
    plt.figure()
    for name, logs in results.items():
        plt.plot(
            logs.bits_cum_normalized,
            logs.best_test_acc,
            marker="o",
            linestyle="-",
            label=name,
        )

    plt.xlabel("Cumulative uplink bits / B_FP")
    plt.ylabel("Best-so-far accuracy")
    plt.title("Toy FL: Route1-style bit schedules")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_route1_schedules.png")
    print("\nSaved figure to toy_route1_schedules.png")


if __name__ == "__main__":
    main()
