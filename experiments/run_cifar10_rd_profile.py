# experiments/run_cifar10_rd_profile.py

from __future__ import annotations

import argparse
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from core.fl_cifar10 import Cifar10FLConfig, build_resnet18_cifar10, train_client_local
from data.data_cifar10_fl import Cifar10DataConfig, build_cifar10_federated_loaders
from quantization.rd_profile import estimate_phi_emp_from_states


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 FL: RD profile (phi_emp(b)) for QSGD updates."
    )
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iid", action="store_true", help="Use IID partition.")
    parser.add_argument("--non-iid", dest="iid", action="store_false")
    parser.set_defaults(iid=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha.")
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
        help="Bit-widths to evaluate RD profile for.",
    )
    args = parser.parse_args()

    # --- config & data ---
    config = Cifar10FLConfig(
        num_clients=args.clients,
        clients_per_round=args.clients_per_round,
        num_rounds=1,  # RD 用なのでラウンド数は使わない
        local_epochs=args.local_epochs,
        batch_size=64,
        lr_local=0.1,
        bit=8,  # ここでは使わない
        device=args.device,
        seed=args.seed,
        data_root="./data",
        iid=args.iid,
        dirichlet_alpha=args.alpha,
        num_workers=2,
    )

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device(config.device)

    data_config = Cifar10DataConfig(
        data_root=config.data_root,
        num_clients=config.num_clients,
        iid=config.iid,
        dirichlet_alpha=config.dirichlet_alpha,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    client_loaders, test_loader = build_cifar10_federated_loaders(data_config)

    # --- global model at "RD round" ---
    global_model = build_resnet18_cifar10(num_classes=10).to(device)

    # ここで何ラウンドか pre-train してから RD を測りたければ、
    # run_cifar10_fl と同様のループを先に回してもよい。
    # とりあえず v1 では "round 0 (初期モデル)" の Δ を見る。

    with torch.no_grad():
        global_state: Dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
            if torch.is_tensor(v) and v.is_floating_point()
        }

    # --- RD 用のクライアントをサンプル ---
    all_client_ids = list(range(config.num_clients))
    rng = random.Random(config.seed + 1)
    selected_ids: List[int] = rng.sample(all_client_ids, config.clients_per_round)

    print(f"Selected clients for RD profile: {selected_ids}")

    client_states: List[Dict[str, torch.Tensor]] = []

    for cid in selected_ids:
        client_loader = client_loaders[cid]

        # クライアントモデル作成 & 初期化
        client_model = build_resnet18_cifar10(num_classes=10).to(device)
        client_model.load_state_dict(global_model.state_dict())

        # ローカル学習
        train_client_local(client_model, client_loader, config, device)

        # RD 用に state_dict（FP tensor）を保存
        with torch.no_grad():
            client_state: Dict[str, torch.Tensor] = {
                k: v.detach().clone()
                for k, v in client_model.state_dict().items()
                if torch.is_tensor(v) and v.is_floating_point()
            }
        client_states.append(client_state)

    # --- φ_emp(b) を計算 ---
    bits_list = sorted(set(int(b) for b in args.bits))
    phi_emp = estimate_phi_emp_from_states(global_state, client_states, bits_list)

    print("\nEmpirical RD profile (phi_emp(b)):")
    for b in bits_list:
        print(f"  b={b:2d}: phi_emp={phi_emp[b]:.3e}")

    # --- 簡単なプロット: log(phi) vs bit ---
    xs = bits_list
    ys = [phi_emp[b] for b in bits_list]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.yscale("log")
    plt.xlabel("bit-width b")
    plt.ylabel("phi_emp(b)  (log scale)")
    plt.title("CIFAR-10 FL (Δ updates): empirical RD profile")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("cifar10_rd_profile.png")
    print("\nSaved figure to cifar10_rd_profile.png")


if __name__ == "__main__":
    main()
