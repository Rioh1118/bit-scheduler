# experiments/run_cifar10_scaling.py

from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib.pyplot as plt

from core.fl_cifar10 import Cifar10FLConfig, run_cifar10_fl
from schedule.schedules import make_log_schedule


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 FL: Bits–Error scaling with Route1 log-increasing schedule."
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument(
        "--b-avgs",
        type=float,
        nargs="+",
        default=[4.0, 8.0, 16.0, 32.0],
        help="Average bits to sweep, e.g., 4 8 16 32",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iid", action="store_true", help="Use IID partition.")
    parser.add_argument("--non-iid", dest="iid", action="store_false")
    parser.set_defaults(iid=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha.")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    T = args.rounds

    # Route1-Log の Bits–Error scaling を見たいので、
    # ここでは log-increasing スケジュールだけを使う。
    b_min = 2
    b_max = 32

    results_B_over_FP: List[float] = []
    results_error: List[float] = []
    labels: List[str] = []

    for b_avg in args.b_avgs:
        print(f"\n=== Running log-schedule with b_avg={b_avg} ===")

        config = Cifar10FLConfig(
            num_clients=args.clients,
            clients_per_round=args.clients_per_round,
            num_rounds=T,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr_local=0.1,
            bit=int(b_avg),  # 使わないがログのために入れておく
            device=args.device,
            seed=args.seed,
            data_root="./data",
            iid=args.iid,
            dirichlet_alpha=args.alpha,
            num_workers=2,
        )

        # log-increasing スケジュールを生成
        bit_schedule = make_log_schedule(
            T=T,
            b_min=b_min,
            b_max=b_max,
            b_avg=b_avg,
        )
        avg_sched = sum(bit_schedule) / len(bit_schedule)
        print(f"  schedule avg bits ≈ {avg_sched:.3f}, head={bit_schedule[:5]}")

        logs = run_cifar10_fl(config, bit_schedule=bit_schedule)

        final_best_acc = logs.best_test_acc[-1]
        final_error = 1.0 - final_best_acc
        final_B_over_FP = logs.bits_cum_normalized[-1]

        print(
            f"  final best acc = {final_best_acc:.4f}, "
            f"error = {final_error:.4e}, "
            f"B/B_FP = {final_B_over_FP:.4f}"
        )

        results_B_over_FP.append(final_B_over_FP)
        results_error.append(final_error)
        labels.append(f"b_avg={b_avg}")

    # --- Bits–Error scaling plot (log–log) ---
    plt.figure()
    plt.loglog(
        results_B_over_FP,
        results_error,
        marker="o",
        linestyle="-",
    )

    # 各点に b_avg のラベルを少しずらして描く
    for x, y, lab in zip(results_B_over_FP, results_error, labels):
        plt.text(x * 1.05, y * 1.05, lab)

    plt.xlabel("Total uplink bits / B_FP (log scale)")
    plt.ylabel("Final best-so-far error  1 - acc (log scale)")
    plt.title("CIFAR-10 FL: Bits–Error scaling (Route1 log-increasing)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("cifar10_bits_error_scaling_loglog.png")
    print("\nSaved figure to cifar10_bits_error_scaling_loglog.png")


if __name__ == "__main__":
    main()
