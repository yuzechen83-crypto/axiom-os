"""
Demo: 倒立双摆 - 爱因斯坦模块 + MPC 稳住
验证 MPC 能否将倒立摆稳定在垂直位置。
使用自定义双摆环境 (与 MPC 物理一致)。
可选: --fast 减少采样以加速
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.core.upi import UPIState
from axiom_os.orchestrator.mpc import (
    ImaginationMPC,
    double_pendulum_H,
    step_env,
    angle_normalize,
)

PI = np.pi


class AcrobotEnv:
    """
    自定义双摆环境，与 axiom_os MPC 物理模型一致。
    扭矩施加于底座 (第一关节)。
    目标：稳定在 q1=pi, q2=pi (垂直向上)
    """

    def __init__(
        self,
        H=None,
        dt=0.02,
        friction=0.1,
        noise_std=0.02,
        seed=None,
    ):
        self.H = H or double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
        self.dt = dt
        self.friction = friction
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros(2)
        self._p = np.zeros(2)

    def reset(self, q=None, p=None):
        """重置到初始状态。默认：接近直立 + 小扰动。返回 (obs, info) 兼容 Gymnasium"""
        if q is not None and p is not None:
            self._q = np.asarray(q, dtype=np.float64).ravel()[:2]
            self._p = np.asarray(p, dtype=np.float64).ravel()[:2]
        else:
            self._q = np.array([PI - 0.15 * self._rng.random(), PI - 0.15 * self._rng.random()])
            self._p = 0.1 * self._rng.standard_normal(2)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self._q, self._p]).astype(np.float32)

    def step(self, action):
        """执行一步。action: 扭矩 (标量)"""
        q_next, p_next = step_env(
            self._q, self._p, float(action), self.H,
            dt=self.dt, friction=self.friction, state_dim=2,
        )
        if self.noise_std > 0:
            q_next += self._rng.standard_normal(2) * self.noise_std
            p_next += self._rng.standard_normal(2) * self.noise_std

        self._q = q_next
        self._p = p_next
        obs = self._get_obs()

        # 简单奖励：偏离直立越远惩罚越大
        err = angle_normalize(self._q - PI)
        reward = -float(np.sum(err**2) + 0.1 * np.sum(self._p**2))
        done = False
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        """可选：返回状态用于外部绘图"""
        return self._get_obs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Fewer samples for quick demo")
    args = parser.parse_args()

    print("=" * 60)
    print("Demo: 倒立双摆 - Einstein + MPC")
    print("=" * 60)

    H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
    env = AcrobotEnv(H=H, dt=0.02, friction=0.1, noise_std=0.005, seed=42)

    n_samples = 100 if args.fast else 2000
    mpc = ImaginationMPC(
        H=H,
        horizon_steps=30 if args.fast else 80,
        n_samples=n_samples,
        dt=0.02,
        friction=0.1,
        action_std=2.0,
        target_state=np.array([PI, PI]),
        distance_threshold=0.5,
    )

    # 从接近直立开始 (PI - 0.02)
    obs, _ = env.reset(q=np.array([PI - 0.02, PI - 0.02]), p=np.array([0.0, 0.0]))
    q, p = obs[:2], obs[2:4]

    # 主循环
    t_max = 100 if args.fast else 200
    history = {"t": [], "q1": [], "q2": [], "action": [], "err": []}

    print("\n>>> Control loop: Observe -> Think (MPC) -> Act")
    for t in range(t_max):
        # 1. 观察
        state_upi = UPIState(
            values=torch.tensor(obs, dtype=torch.float64),
            units=[0, 0, 0, 0, 0],
            semantics="AcrobotState",
        )

        # 2. 思考 (MPC - 调用 Einstein 进行 N 次并行想象)
        action = mpc.plan(q, p)

        # 3. 行动
        obs, reward, done, truncated, info = env.step(action)
        q, p = obs[:2], obs[2:4]

        # 记录
        err = np.sqrt(np.sum(angle_normalize(q - PI) ** 2))
        history["t"].append(t * env.dt)
        history["q1"].append(q[0])
        history["q2"].append(q[1])
        history["action"].append(action)
        history["err"].append(err)

        if (t + 1) % 50 == 0:
            print(f"  t={t+1:4d}  q1={q[0]:.3f}  q2={q[1]:.3f}  err={err:.3f}  tau={action:.2f}")

    # 4. 可视化
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax = axes[0]
        ax.plot(history["t"], history["q1"], "b-", label="theta1")
        ax.axhline(PI, color="k", linestyle="--", alpha=0.5)
        ax.set_ylabel("theta1")
        ax.set_title("Acrobot: MPC Stabilization")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(history["t"], history["q2"], "g-", label="theta2")
        ax.axhline(PI, color="k", linestyle="--", alpha=0.5)
        ax.set_ylabel("theta2")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(history["t"], history["action"], "r-", label="torque")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(__file__).resolve().parent / "demo_acrobot.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved: {out_path}")
        plt.close()
    except ImportError:
        print("\n(matplotlib not available, skipping plot)")

    # 成功标准
    err_final = np.mean(history["err"][-50:])
    stabilized = err_final < 0.8
    print(f"\nFinal avg error (last 50 steps): {err_final:.3f}")
    print(f"Stabilized: {'YES' if stabilized else 'NO'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
