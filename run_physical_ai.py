"""
PhysicalAI 完整工作流测试
输入物理问题 → 输出物理解 + 置信度 + 可解释报告
"""

import numpy as np
from spnn import PhysicalAI

def main():
    print("=" * 60)
    print("PhysicalAI - 完整系统工作流")
    print("=" * 60)

    ai = PhysicalAI(in_dim=4, hidden_dim=32, out_dim=1, memory_capacity=1000)

    # 输入物理问题 (t, x, y, z) 或类似
    x = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    l_e = "harmonic_oscillator"  # 语义标签

    res = ai.solve(x, l_e=l_e)
    print("\n--- 输出物理解 ---")
    print(f"Result: {res.result}")
    print(f"Confidence: {res.confidence:.4f}")
    print("\n--- 可解释报告 ---")
    print(res.report)

    # 批量测试
    x_batch = np.random.randn(5, 4).astype(np.float32) * 0.5
    res2 = ai.solve(x_batch, l_e="batch_test")
    print("\n--- 批量结果 ---")
    print(f"Result shape: {res2.result.shape}")
    print(f"Confidence: {res2.confidence:.4f}")

    print("\n" + "=" * 60)
    print("PhysicalAI 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
