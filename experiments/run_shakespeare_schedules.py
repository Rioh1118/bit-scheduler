# experiments/run_shakespeare_schedules.py

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt

from core.fl_shakespeare import ShakespeareFLConfig, run_shakespeare_fl
from schedule.schedules import (
    make_decreasing_two_stage,
    make_fixed_schedule,
    make_log_schedule,
    make_two_stage_schedule,
    make_zigzag_schedule,
)


def main():
    parser = argparse.ArgumentParser(
        description="Shakespeare FL + QSGD: compare Route1-style bit schedules."
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--b-avg", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=str, default="./data/shakespeare")
    parser.add_argument("--train-json", type=str, default="train.json")
    parser.add_argument("--test-json", type=str, default="test.json")
    args = parser.parse_args()

    T = args.rounds
    b_avg = args.b_avg

    config = ShakespeareFLConfig(
        num_clients=args.clients,
        clients_per_round=args.clients_per_round,
        num_rounds=T,
        local_epochs=1,
        batch_size=args.batch_size,
        lr_local=0.5,
        bit=int(b_avg),
        device=args.device,
        seed=args.seed,
        data_root=args.data_root,
        train_json=args.train_json,
        test_json=args.test_json,
        seq_len=args.seq_len,
        num_workers=2,
    )

    # ---- bit schedules ----
    schedules: Dict[str, list[int]] = {}

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

    print("Bit schedules (avg, head):")
    for name, sched in schedules.items():
        avg = sum(sched) / len(sched)
        print(f"  {name}: avg={avg:.2f}, head={sched[:5]}")

    results = {}

    for name, bit_schedule in schedules.items():
        print(f"\n=== Running schedule: {name} ===")
        logs = run_shakespeare_fl(config, bit_schedule=bit_schedule)
        results[name] = logs
        print(
            f"  final best acc = {logs.best_test_acc[-1]:.4f}, "
            f"bits_norm_last = {logs.bits_cum_normalized[-1]:.3f}"
        )

    # plot
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
    plt.ylabel("Best-so-far next-char accuracy")
    plt.title("Shakespeare FL: Route1-style bit schedules")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("shakespeare_route1_schedules.png")
    print("\nSaved figure to shakespeare_route1_schedules.png")


if __name__ == "__main__":
    main()
