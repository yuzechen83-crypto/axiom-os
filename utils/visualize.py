"""
Visualization suite for Axiom-OS MuJoCo benchmarks.

- Robustness Heatmap: load sensitivity results JSON, plot 2D heatmap
  X = Friction, Y = Gravity, Color = Reward; save to docs/images/robustness_map.png.
- Replay Video: run best Humanoid model, capture frames, save as docs/images/humanoid_walk.gif.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_robustness_heatmap(
    results_path: str | Path,
    save_path: str | Path = "docs/images/robustness_map.png",
) -> None:
    """
    Load sensitivity results JSON and plot 2D heatmap: X=Friction, Y=Gravity, Color=Reward.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping robustness heatmap")
        return

    path = Path(results_path)
    if not path.exists():
        print(f"Results not found: {path}")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("walker2d_sensitivity", data.get("sensitivity", []))
    if not rows:
        print("No sensitivity rows in JSON")
        return

    g_vals = sorted(set(r["g"] for r in rows))
    f_vals = sorted(set(r["f"] for r in rows))
    if len(g_vals) * len(f_vals) != len(rows) and len(g_vals) > 1 and len(f_vals) > 1:
        # Scatter: not a full grid
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        g = [r["g"] for r in rows]
        f = [r["f"] for r in rows]
        rewards = [r["reward"] for r in rows]
        sc = ax.scatter(f, g, c=rewards, s=120, cmap="RdYlGn", edgecolors="k")
        ax.set_xlabel("Friction scale")
        ax.set_ylabel("Gravity scale")
        ax.set_title("Walker2d Robustness: Reward vs (Friction, Gravity)")
        plt.colorbar(sc, ax=ax, label="Total reward")
    else:
        # Grid heatmap
        Z = np.full((len(g_vals), len(f_vals)), np.nan)
        for r in rows:
            i = g_vals.index(r["g"])
            j = f_vals.index(r["f"])
            Z[i, j] = r["reward"]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(
            Z,
            extent=[min(f_vals), max(f_vals), min(g_vals), max(g_vals)],
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
        )
        ax.set_xlabel("Friction scale")
        ax.set_ylabel("Gravity scale")
        ax.set_title("Walker2d Robustness: Reward (Gravity × Friction)")
        plt.colorbar(im, ax=ax, label="Total reward")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Robustness heatmap saved to {out}")


def save_humanoid_replay_gif(
    save_path: str | Path = "docs/images/humanoid_walk.gif",
    steps: int = 400,
    seed: int = 42,
    policy=None,
    fps: int = 30,
) -> None:
    """
    Run Humanoid env, capture frames, save as GIF.
    If policy is None, use a random policy (for demo).
    """
    try:
        import gymnasium as gym
        from PIL import Image
    except ImportError as e:
        print(f"Need gymnasium and PIL for replay gif: {e}")
        return

    from axiom_os.envs import make_axiom_mujoco_env
    from axiom_os.envs.axiom_policy import PhysicsAwarePolicy, train_policy_reinforce

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if policy is None:
        env = make_axiom_mujoco_env("Humanoid-v5", seed=seed)
        obs, _ = env.reset(seed=seed)
        policy = PhysicsAwarePolicy(obs.shape[0], env.action_space.shape[0])
        train_policy_reinforce(policy, env, episodes=25, steps_per_episode=200, seed=seed)
        env.close()

    env = make_axiom_mujoco_env("Humanoid-v5", seed=seed, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(steps):
        action = policy.act(obs, deterministic=True, gravity_scale=1.0, friction_scale=1.0)
        obs, reward, term, trunc, info = env.step(action)
        frame = env.render()
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            frames.append(Image.fromarray(frame))
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    if not frames:
        print("No frames captured (render may return None). Skipping gif.")
        return
    # Limit size for gif
    frames = frames[:: max(1, len(frames) // 120)]
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Humanoid replay gif saved to {out}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Axiom-OS visualization")
    parser.add_argument("--heatmap", action="store_true", help="Plot robustness heatmap from sensitivity JSON")
    parser.add_argument("--replay", action="store_true", help="Record Humanoid walk gif")
    parser.add_argument("--results", default="sensitivity_results.json", help="Path to sensitivity JSON")
    parser.add_argument("--out-heatmap", default="docs/images/robustness_map.png")
    parser.add_argument("--out-gif", default="docs/images/humanoid_walk.gif")
    args = parser.parse_args()

    if args.heatmap:
        plot_robustness_heatmap(args.results, args.out_heatmap)
    if args.replay:
        save_humanoid_replay_gif(args.out_gif)
    if not args.heatmap and not args.replay:
        plot_robustness_heatmap(args.results, args.out_heatmap)
        print("Run with --replay to also generate humanoid_walk.gif (requires rendering).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
