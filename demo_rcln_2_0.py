"""
RCLN 2.0 完整演示
展示四大进化方案：FNO / Clifford / KAN / Mamba
"""

import torch
import torch.nn as nn
from axiom_os.layers import (
    RCLNLayer,
    create_fno_rcln,
    create_clifford_rcln,
    create_kan_rcln,
    create_mamba_rcln,
    list_evolution_paths,
)


def demo_scheme_1_fno():
    """方案一：FNO-RCLN (算子化进化)"""
    print("\n" + "="*70)
    print("[方案一] FNO-RCLN: Operator Evolution")
    print("="*70)
    print("场景：流体动力学 - 学习涡度场演化")
    
    # 模拟 2D 流体场 (64x64)
    batch_size = 4
    H, W = 32, 32  # 简化的网格
    
    # FNO 需要 4D 输入 (B, C, H, W)
    vorticity_field = torch.randn(batch_size, 1, H, W)
    
    # 创建 FNO-RCLN (使用 spectral 作为简化演示)
    rcln = RCLNLayer(
        input_dim=H*W,
        hidden_dim=64,
        output_dim=H*W,
        net_type="spectral",  # 使用 spectral 作为 FNO 简化
    )
    
    # 展平处理 (简化示例)
    x = vorticity_field.view(batch_size, -1)
    y = rcln(x)
    
    print(f"输入: 涡度场 {vorticity_field.shape}")
    print(f"输出: 预测演化 {y.shape}")
    print(f"特点: 频率域操作，适合周期物理")
    print("✓ FNO-RCLN 演示完成")


def demo_scheme_2_clifford():
    """方案二：Clifford-RCLN (几何化进化)"""
    print("\n" + "="*70)
    print("[方案二] Clifford-RCLN: Geometric Evolution")
    print("="*70)
    print("场景：无人机控制 - 旋转等变动力学")
    
    # 输入：线速度 (3D) + 角速度 (3D) = 6D
    batch_size = 8
    velocity = torch.randn(batch_size, 6)
    
    # 创建 Clifford-RCLN
    rcln = create_clifford_rcln(
        input_dim=6,
        hidden_dim=16,
        output_dim=3,  # 输出：线加速度
        use_transformer=False,
    )
    
    acceleration = rcln(velocity)
    
    print(f"输入: 速度矢量 [vx, vy, vz, wx, wy, wz] {velocity.shape}")
    print(f"输出: 加速度 [ax, ay, az] {acceleration.shape}")
    print(f"特点: O(3) 等变，旋转不变")
    print("✓ Clifford-RCLN 演示完成")


def demo_scheme_3_kan():
    """方案三：KAN-RCLN (符号化进化)"""
    print("\n" + "="*70)
    print("[方案三] KAN-RCLN: Symbolic Evolution")
    print("="*70)
    print("场景：发现引擎 - 从数据提取物理公式")
    
    # 生成数据：F_drag = 0.47 * v^2 (二次阻力)
    v = torch.linspace(0, 10, 100).unsqueeze(1)
    f_drag_true = 0.47 * v ** 2
    
    # 创建 KAN-RCLN
    rcln = create_kan_rcln(
        input_dim=1,
        hidden_dim=4,
        output_dim=1,
        grid_size=6,
        spline_order=3,
    )
    
    # 训练
    optimizer = torch.optim.Adam(rcln.parameters(), lr=0.01)
    for epoch in range(300):
        optimizer.zero_grad()
        f_pred = rcln(v)
        loss = nn.functional.mse_loss(f_pred, f_drag_true)
        loss.backward()
        optimizer.step()
    
    print(f"训练完成，最终损失: {loss.item():.6f}")
    
    # 提取公式
    formula = rcln.extract_formula(var_names=["velocity"])
    if formula:
        print(f"\n🎯 发现公式:")
        print(f"   {formula['formula_str']}")
        print(f"   置信度: {formula['confidence']:.1%}")
        print(f"   组件: {len(formula['components'])} 项")
    
    # 验证
    v_test = torch.tensor([[5.0]])
    f_learned = rcln(v_test).item()
    f_true = 0.47 * 25
    print(f"\n验证 v=5:")
    print(f"   学习到的 F = {f_learned:.3f}")
    print(f"   真实的 F = {f_true:.3f}")
    print(f"   误差 = {abs(f_learned-f_true)/f_true*100:.1f}%")
    
    print(f"\n特点: 直接公式提取，白盒可解释")
    print("✓ KAN-RCLN 演示完成")


def demo_scheme_4_mamba():
    """方案四：Mamba-RCLN (序列化进化)"""
    print("\n" + "="*70)
    print("[方案四] Mamba-RCLN: Sequential Evolution")
    print("="*70)
    print("场景：长时程预测 - 连续时间动力学")
    
    # 生成序列数据：阻尼振荡
    seq_len = 100
    t = torch.linspace(0, 10, seq_len)
    x_seq = torch.sin(t) * torch.exp(-0.1*t)  # 衰减正弦
    
    # 创建 Mamba-RCLN
    rcln = create_mamba_rcln(
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        state_dim=16,
        n_layers=2,
    )
    
    # 单步预测
    x_input = x_seq[:50].unsqueeze(1)  # (50, 1)
    y_output = rcln.forward_sequence(x_input.unsqueeze(0))  # (1, 50, 1)
    
    print(f"输入序列: {x_input.shape[0]} 步")
    print(f"输出序列: {y_output.shape[1]} 步")
    print(f"特点: 线性复杂度 O(L)，支持 10k+ 步")
    print("✓ Mamba-RCLN 演示完成")


def demo_hard_core_integration():
    """演示硬核心 + 软壳的混合架构"""
    print("\n" + "="*70)
    print("[硬核心集成] Physics + Neural Hybrid")
    print("="*70)
    
    # 已知物理：简谐振子 (Hooke's law)
    def harmonic_oscillator(x):
        if hasattr(x, 'detach'):
            pos = x[:, 0:1]
            return -0.5 * pos  # F = -kx
        return -0.5 * x.values[:, 0:1]
    
    # 未知部分：非线性阻尼 (用 KAN 学习)
    rcln = RCLNLayer(
        input_dim=2,  # [position, velocity]
        hidden_dim=4,
        output_dim=1,  # acceleration
        net_type="kan",
        hard_core_func=harmonic_oscillator,
        lambda_res=0.3,  # Soft 部分权重
        kan_grid_size=5,
    )
    
    # 测试
    x = torch.tensor([[1.0, 0.5]])  # pos=1, vel=0.5
    y = rcln(x)
    
    print(f"输入: 位置={x[0,0].item():.2f}, 速度={x[0,1].item():.2f}")
    print(f"硬核心贡献: {-0.5 * x[0,0].item():.3f}")
    print(f"总输出 (加速度): {y[0,0].item():.3f}")
    print(f"公式: a = F_hard + 0.3 * F_soft")
    print("✓ 硬核心集成演示完成")


def demo_comparison():
    """对比四大方案"""
    print("\n" + "="*70)
    print("[方案对比] Four Evolution Paths")
    print("="*70)
    
    x = torch.randn(10, 4)
    
    schemes = [
        ("spectral", "Spectral", "频率域，周期物理"),
        ("clifford", "Clifford", "几何代数，旋转等变"),
        ("kan", "KAN", "符号提取，可解释"),
        ("mamba", "Mamba", "线性复杂度，序列"),
    ]
    
    print(f"\n输入: {x.shape}")
    print("-" * 70)
    
    for net_type, name, feature in schemes:
        try:
            if net_type == "kan":
                rcln = RCLNLayer(4, 8, 2, net_type=net_type, kan_grid_size=5)
            else:
                rcln = RCLNLayer(4, 8, 2, net_type=net_type)
            
            y = rcln(x)
            info = rcln.get_architecture_info()
            
            print(f"{name:12s} | 输出: {str(y.shape):15s} | {feature}")
        except Exception as e:
            print(f"{name:12s} | 错误: {e}")
    
    print("-" * 70)
    print("✓ 方案对比完成")


def main():
    """运行所有演示"""
    print("\n" + "="*70)
    print(" Axiom-OS RCLN 2.0 完整演示")
    print(" Four Evolution Paths: FNO / Clifford / KAN / Mamba")
    print("="*70)
    
    # 显示可用路径
    print("\n[可用进化路径]")
    list_evolution_paths()
    
    # 运行演示
    demo_scheme_1_fno()
    demo_scheme_2_clifford()
    demo_scheme_3_kan()
    demo_scheme_4_mamba()
    demo_hard_core_integration()
    demo_comparison()
    
    # 总结
    print("\n" + "="*70)
    print(" 演示完成!")
    print("="*70)
    print("""
总结:
┌─────────────────────────────────────────────────────────────────────┐
│  FNO      → 流体、气象、波传播 (频率域操作)                          │
│  Clifford → 机器人、电磁、分子 (几何等变)                           │
│  KAN      → 公式发现、控制、建模 (符号可解释)                        │
│  Mamba    → 长序列、实时控制、ODE (线性复杂度)                       │
└─────────────────────────────────────────────────────────────────────┘

所有方案保持统一接口:
    y = RCLNLayer(..., net_type="<path>")(x)
    
硬核心 + 软壳架构:
    y_total = y_hard + λ * y_soft
""")


if __name__ == "__main__":
    main()
