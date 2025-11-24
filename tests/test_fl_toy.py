# tests/test_fl_toy.py

from core.fl_toy import ToyFLConfig, run_toy_fl
from schedule.schedules import make_log_schedule, make_two_stage_schedule


def test_run_toy_fl_runs_and_logs_shape():
    config = ToyFLConfig(
        num_clients=10,
        clients_per_round=3,
        num_rounds=5,
        input_dim=10,
        hidden_dim=16,
        num_classes=2,
        local_epochs=1,
        batch_size=16,
        lr_local=0.1,
        bit=8,
        device="cpu",
        seed=123,
    )

    logs = run_toy_fl(config)

    assert len(logs.test_acc) == config.num_rounds
    assert len(logs.best_test_acc) == config.num_rounds
    assert len(logs.bits_cum) == config.num_rounds
    assert len(logs.bits_cum_normalized) == config.num_rounds

    assert logs.d > 0
    assert logs.B_FP > 0.0

    for i in range(1, len(logs.bits_cum)):
        assert logs.bits_cum[i] >= logs.bits_cum[i - 1]


def test_run_toy_fl_respects_bit_schedule():
    """
    手で作った bit スケジュールを渡したとき、
    そのスケジュールに応じて 1 ラウンドあたりの bits が変わっていることを確認。
    """
    num_rounds = 6
    config = ToyFLConfig(
        num_clients=10,
        clients_per_round=3,
        num_rounds=num_rounds,
        input_dim=10,
        hidden_dim=16,
        num_classes=2,
        local_epochs=1,
        batch_size=16,
        lr_local=0.1,
        bit=8,  # ここは使われない（bit_schedule を渡すので）
        device="cpu",
        seed=42,
    )

    # 低ビット→高ビット→低ビットのカスタムスケジュール
    bit_schedule = [4, 4, 16, 16, 4, 4]

    logs = run_toy_fl(config, bit_schedule=bit_schedule)

    # ラウンドごとの bits (B_t) を計算
    per_round_bits = []
    for t, cum in enumerate(logs.bits_cum):
        if t == 0:
            per_round_bits.append(cum)
        else:
            per_round_bits.append(cum - logs.bits_cum[t - 1])

    assert len(per_round_bits) == num_rounds

    # インデックス 0,1 は b=4、2,3 は b=16、4,5 は b=4
    low_rounds = [0, 1, 4, 5]
    high_rounds = [2, 3]

    avg_low = sum(per_round_bits[i] for i in low_rounds) / len(low_rounds)
    avg_high = sum(per_round_bits[i] for i in high_rounds) / len(high_rounds)

    # 高ビットのラウンドの方が 1 ラウンドあたりのビット数が多いはず
    assert avg_high > avg_low, "High-bit rounds should consume more bits per round."
