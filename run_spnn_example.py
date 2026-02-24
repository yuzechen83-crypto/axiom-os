"""
SPNN-Opt-Rev5 (Axiom-OS) 示例运行脚本
六阶段认知学习全链路 + 主脑调度演示
"""

import torch
import numpy as np
from spnn import SPNN
from spnn.training import SPNNTrainer
from spnn.training import SPNNLoss
from spnn.training.trainer import TrainingConfig


def main():
    print("=" * 60)
    print("SPNN-Opt-Rev5 (Axiom-OS) 示例")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 构造简单物理数据 (例如: 弹簧振动 x'' = -kx)
    n_samples = 500
    t = np.linspace(0, 10, n_samples).astype(np.float32)
    x = np.sin(t) * np.exp(-0.1 * t)  #  damped oscillation
    X = np.stack([t, x], axis=1)
    y = -x - 0.2 * np.cos(t) * np.exp(-0.1 * t)  # 近似加速度
    y = y.reshape(-1, 1).astype(np.float32)

    X_t = torch.as_tensor(X, device=device)
    y_t = torch.as_tensor(y, device=device)

    # 2. 创建 SPNN 模型
    model = SPNN(
        in_dim=2,
        hidden_dim=32,
        out_dim=1,
        num_rcln_layers=2,
        memory_capacity=1000,
        tau_active=0.1,
        device=device,
    )

    # 3. 训练
    config = TrainingConfig(
        phases=1,
        epochs_per_phase=3,
        batch_size=64,
    )
    trainer = SPNNTrainer(model, config=config)
    trainer.setup_optimizer()

    print("\n训练中...")
    for phase in range(config.phases):
        for ep in range(config.epochs_per_phase):
            loss_dict = trainer.train_epoch(X_t, y_t)
            loss = loss_dict.get("total", 0)
            print(f"  Phase {phase+1} Epoch {ep+1}: loss={loss:.6f}")

        if phase < config.phases - 1:
            trainer.advance_phase()

    # 4. 预测
    model.eval()
    with torch.no_grad():
        pred, _ = model(X_t[:10])
        print("\n预测结果 (前5个):")
        pred_np = pred.cpu().numpy()
        for i in range(5):
            print(f"  真实: {y[i,0]:.4f}  预测: {pred_np[i,0]:.4f}")

    # 5. 双路径验证
    from spnn.safety import DualPathValidator
    validator = DualPathValidator()
    final, info = validator.validate(pred[:5])
    print(f"\n双路径验证: route={info.get('route')}")

    print("\n" + "=" * 60)
    print("SPNN-Opt-Rev5 示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
