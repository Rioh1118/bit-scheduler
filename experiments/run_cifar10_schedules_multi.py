# experiments/run_cifar10_schedules.py

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from core.fl_cifar10 import Cifar10FLConfig, run_cifar10_fl
from schedule.schedules import (
    make_decreasing_two_stage,
    make_fixed_schedule,
    make_log_schedule,
    make_opt_schedule_exponential_weights,
    make_two_stage_schedule,
    make_zigzag_schedule,
)


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 FL + QSGD: compare Route1-style bit schedules."
    )
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--b-avg", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to average over.",
    )
    parser.add_argument("--iid", action="store_true", help="Use IID partition.")
    parser.add_argument("--non-iid", dest="iid", action="store_false")
    parser.set_defaults(iid=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha.")
    parser.add_argument(
        "--rho-opt",
        type=float,
        default=0.01,
        help="Decay rate rho for theoretical weights in opt schedule.",
    )
    args = parser.parse_args()

    T = args.rounds
    b_avg = args.b_avg

    # ベースとなる FL 設定（seed だけ後で差し替える）
    base_config = Cifar10FLConfig(
        num_clients=args.clients,
        clients_per_round=args.clients_per_round,
        num_rounds=T,
        local_epochs=1,
        batch_size=64,
        lr_local=0.1,
        bit=int(b_avg),  # fixed-bit のベース値
        device=args.device,
        seed=args.seed,
        data_root="./data",
        iid=args.iid,
        dirichlet_alpha=args.alpha,
        num_workers=2,
    )

    # ---- bit schedules ----
    schedules: Dict[str, List[int]] = {}

    schedules[f"fixed-{int(b_avg)}"] = make_fixed_schedule(T, int(b_avg))

    b_low = 4
    b_high = 16
    schedules["two-stage-4-16"] = make_two_stage_schedule(
        T, b_low=b_low, b_high=b_high, b_avg=b_avg
    )
    schedules["log-4-16"] = make_log_schedule(
        T,
        b_min=b_low,
        b_max=b_high,
        b_avg=b_avg,
    )
    schedules["zigzag-4-16"] = make_zigzag_schedule(T, b_low=4, b_high=16)
    schedules["decreasing-4-16"] = make_decreasing_two_stage(
        T, b_low=4, b_high=16, b_avg=b_avg
    )
    schedules["opt-exp-4-16"] = make_opt_schedule_exponential_weights(
        T,
        b_min=b_low,
        b_max=b_high,
        b_avg=b_avg,
        rho=args.rho_opt,
    )

    print("Bit schedules (avg, head):")
    for name, sched in schedules.items():
        avg = sum(sched) / len(sched)
        print(f"  {name}: avg={avg:.2f}, head={sched[:5]}")

    # results[name] = List[FLLogs] （seed ごとのログを全部持つ）
    results: Dict[str, List] = {name: [] for name in schedules.keys()}

    # ---- multi-seed ループ ----
    for s in range(args.num_seeds):
        seed_val = args.seed + s
        cfg = replace(base_config, seed=seed_val)
        print(f"\n========== Seed {seed_val} ==========")

        for name, bit_schedule in schedules.items():
            print(f"\n=== Running schedule: {name} (seed={seed_val}) ===")
            logs = run_cifar10_fl(cfg, bit_schedule=bit_schedule)
            results[name].append(logs)
            print(
                f"  final best acc = {logs.best_test_acc[-1]:.4f}, "
                f"bits_norm_last = {logs.bits_cum_normalized[-1]:.3f}"
            )

    # 集約結果のサマリを出しておく（final best acc の mean/std）
    print("\n===== Aggregate over seeds =====")
    for name, logs_list in results.items():
        final_acc = np.array([logs.best_test_acc[-1] for logs in logs_list])
        bits_last = logs_list[0].bits_cum_normalized[-1]
        print(
            f"{name:16s}: "
            f"best_acc = {final_acc.mean():.4f} ± {final_acc.std():.4f} "
            f"(bits_norm_last = {bits_last:.3f}, n_seeds={len(logs_list)})"
        )

    # ========= Figure 1: best-so-far test acc vs normalized bits (mean ± std) =========
    plt.figure()
    for name, logs_list in results.items():
        # すべての seed で bits_cum_normalized は同じはずなので先頭を使う
        bits = np.array(logs_list[0].bits_cum_normalized)
        acc_mat = np.stack([np.array(l.best_test_acc) for l in logs_list], axis=0)
        acc_mean = acc_mat.mean(axis=0)
        acc_std = acc_mat.std(axis=0)

        plt.plot(bits, acc_mean, label=name)
        if args.num_seeds > 1:
            plt.fill_between(bits, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)

    plt.xlabel("Cumulative uplink bits / B_FP")
    plt.ylabel("Best-so-far test accuracy")
    plt.title("CIFAR-10 FL: Route1-style bit schedules (accuracy)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cifar10_route1_schedules.png")
    print("\nSaved figure to cifar10_route1_schedules.png")

    # ========= Figure 2: global training loss vs normalized bits (mean ± std) =========
    plt.figure()
    for name, logs_list in results.items():
        bits = np.array(logs_list[0].bits_cum_normalized)
        loss_mat = np.stack([np.array(l.train_loss) for l in logs_list], axis=0)
        loss_mean = loss_mat.mean(axis=0)
        loss_std = loss_mat.std(axis=0)

        plt.plot(bits, loss_mean, label=name)
        if args.num_seeds > 1:
            plt.fill_between(
                bits, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2
            )

    plt.xlabel("Cumulative uplink bits / B_FP")
    plt.ylabel("Global training loss (cross-entropy)")
    plt.title("CIFAR-10 FL: Route1-style bit schedules (global train loss)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cifar10_route1_schedules_train_loss.png")
    print("Saved figure to cifar10_route1_schedules_train_loss.png")

    # ========= Figure 3: bit schedule shapes b_t vs t =========
    plt.figure()
    for name, sched in schedules.items():
        plt.step(range(T), sched, where="post", label=name)

    plt.xlabel("Round t")
    plt.ylabel("Bit-width b_t")
    plt.title("Bit schedules (Route1-style)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cifar10_route1_bit_schedules.png")
    print("Saved figure to cifar10_route1_bit_schedules.png")


if __name__ == "__main__":
    main()
