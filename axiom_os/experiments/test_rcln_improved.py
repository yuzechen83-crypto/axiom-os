"""
改进 RCLN 测试脚本
1. RCLN 单元测试
2. RCLN 基准（forward 延迟、湍流训练）
3. RCLN vs PINN-LSTM 对比（湍流时空预测）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_unit_tests():
    """运行 RCLN 单元测试"""
    print("=" * 60)
    print("[1/4] RCLN 单元测试")
    print("=" * 60)
    from axiom_os.tests.test_rcln import (
        test_rcln_no_hard_core,
        test_rcln_with_hard_core,
        test_get_soft_activity,
        test_rcln_accepts_tensor,
        test_rcln_force_mlp_fallback,
        test_rcln_clifford_multivector_structure,
        test_rcln_spectral_soft_shell,
    )
    test_rcln_no_hard_core()
    test_rcln_with_hard_core()
    test_get_soft_activity()
    test_rcln_accepts_tensor()
    test_rcln_force_mlp_fallback()
    test_rcln_clifford_multivector_structure()
    test_rcln_spectral_soft_shell()
    print("  OK 所有 RCLN 单元测试通过\n")


def run_benchmark_rcln():
    """RCLN 基准：forward 延迟 + 湍流训练"""
    print("=" * 60)
    print("[2/4] RCLN 基准测试")
    print("=" * 60)
    import time
    import torch
    from axiom_os.layers.rcln import RCLNLayer

    # Forward 延迟
    rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, hard_core_func=None, lambda_res=1.0)
    x = torch.randn(1024, 4)
    for _ in range(10):
        _ = rcln(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = rcln(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / 100 * 1000
    throughput = 1024 * 100 / (time.perf_counter() - t0)
    print(f"  RCLN forward: {elapsed_ms:.4f} ms/batch, {throughput/1e6:.2f}M samples/s (batch=1024)")

    # 湍流训练（精简 50 epochs）
    try:
        from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
        from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
        from axiom_os.coach import coach_loss_torch

        coords, targets, _ = load_atmospheric_turbulence_3d(
            n_lat=3, n_lon=3, delta_deg=0.15, forecast_days=3, use_synthetic_if_fail=True
        )
        split = int(0.8 * len(coords))
        X_train = torch.from_numpy(coords[:split]).float()
        Y_train = torch.from_numpy(targets[:split]).float()
        u_mean = float(Y_train[:, 0].mean())
        v_mean = float(Y_train[:, 1].mean())
        hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=5.0, use_enhanced=False)
        rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, hard_core_func=hard_core, lambda_res=1.0)
        opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
        epochs = 50
        t0 = time.perf_counter()
        for ep in range(epochs):
            opt.zero_grad()
            pred = rcln(X_train)
            loss = torch.nn.functional.huber_loss(pred, Y_train, delta=1.0) + 0.15 * coach_loss_torch(pred, domain="fluids")
            loss.backward()
            opt.step()
        elapsed = time.perf_counter() - t0
        with torch.no_grad():
            pred_test = rcln(torch.from_numpy(coords[split:]).float()).numpy()
        mae = (abs(pred_test[:, 0] - targets[split:, 0]).mean() + abs(pred_test[:, 1] - targets[split:, 1]).mean()) / 2
        print(f"  湍流训练: {epochs} epochs, {elapsed:.2f}s, Test MAE={mae:.4f}")
    except Exception as e:
        print(f"  湍流训练: 跳过 ({e})")
    print()


def run_pinn_lstm_comparison():
    """RCLN vs PINN-LSTM 对比（湍流时空预测）"""
    print("=" * 60)
    print("[3/4] RCLN vs PINN-LSTM 湍流对比")
    print("=" * 60)
    from axiom_os.experiments.run_turbulence_pinn_lstm import main as pinn_main
    pinn_main()
    print()


def run_temporal_neuron_test():
    """TemporalPhysicsNeuron (PINN-LSTM 封装) 测试"""
    print("=" * 60)
    print("[4/4] TemporalPhysicsNeuron 测试")
    print("=" * 60)
    import torch
    from axiom_os.neurons.temporal import TemporalPhysicsNeuron

    def dummy_hard_core(x):
        v = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return torch.zeros(v.shape[0], 2, device=v.device)

    neuron = TemporalPhysicsNeuron(
        input_dim=4, hidden_dim=32, output_dim=2, seq_len=8,
        hard_core_func=dummy_hard_core, lambda_res=1.0,
    )
    x = torch.randn(16, 8, 4)  # (B, seq_len, 4)
    y = neuron(x)
    assert y.shape == (16, 2), f"Expected (16,2), got {y.shape}"
    activity = neuron.get_soft_activity()
    print(f"  TemporalPhysicsNeuron: input (16,8,4) -> output {y.shape}, activity={activity:.4f}")
    print("  OK TemporalPhysicsNeuron 测试通过\n")


def main():
    print("\n" + "=" * 60)
    print("改进 RCLN 测试套件")
    print("=" * 60 + "\n")

    run_unit_tests()
    run_benchmark_rcln()
    run_temporal_neuron_test()
    run_pinn_lstm_comparison()

    print("=" * 60)
    print("全部测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
