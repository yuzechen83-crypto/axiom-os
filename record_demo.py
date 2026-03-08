"""
Axiom-OS Demo Video Recorder
30-second video: Left = pendulum animation, Right = torque (red) + angle (blue) curves.
Usage: python axiom_os/record_demo.py
Output: axiom_os/demo_video.mp4 (or .gif if ffmpeg not installed)
Requires: pip install matplotlib; ffmpeg for MP4 (optional)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from axiom_os.orchestrator.mpc import (
    ImaginationMPC,
    double_pendulum_H,
    step_env,
    angle_normalize,
)

PI = np.pi

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="5 sec, 100 samples (quick test)")
    args = parser.parse_args()

    print("=" * 60)
    print("Axiom-OS Demo Video Recorder")
    print("30 sec: Pendulum (left) | Torque + Angle (right)")
    print("=" * 60)

    H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
    n_samp = 100 if args.fast else 300
    mpc = ImaginationMPC(
        H=H,
        horizon_steps=50 if args.fast else 80,
        n_samples=n_samp,
        dt=0.02,
        friction=0.1,
        action_std=2.0,
        target_state=np.array([PI, PI]),
    )

    # Simulate: --fast = 5 sec, else 30 sec
    fps = 20
    duration_sec = 5 if args.fast else 30
    dt = 0.02
    n_steps = int(duration_sec / dt)
    frames_to_capture = fps * duration_sec
    step_per_frame = max(1, n_steps // frames_to_capture)

    q = np.array([PI - 0.02, PI - 0.02])
    p = np.array([0.0, 0.0])

    # Apply kick at t=2 sec
    kick_at = int(2.0 / dt)
    history = {"t": [], "q1": [], "q2": [], "tau": []}

    print("Running simulation...")
    for i in range(n_steps):
        if i == kick_at:
            p = p + np.random.uniform(-6, 6, 2)
        tau = mpc.plan(q, p)
        q, p = step_env(q, p, tau, H, dt=dt, friction=0.1, state_dim=2)
        t = (i + 1) * dt
        history["t"].append(t)
        history["q1"].append(q[0])
        history["q2"].append(q[1])
        history["tau"].append(tau)

    # Build frame indices
    frame_indices = list(range(0, n_steps, step_per_frame))[:frames_to_capture]

    # Setup figure: left pendulum, right curves
    fig = plt.figure(figsize=(12, 6))
    ax_pend = fig.add_subplot(121)
    ax_curves = fig.add_subplot(122)

    L1, L2 = 1.0, 1.0

    def draw_frame(idx):
        ax_pend.clear()
        ax_curves.clear()

        i = min(idx, len(history["t"]) - 1)
        q1, q2 = history["q1"][i], history["q2"][i]
        tau = history["tau"][i]
        t = history["t"][i]

        # Left: Pendulum
        x0, y0 = 0.0, 0.0
        x1 = L1 * np.sin(q1)
        y1 = -L1 * np.cos(q1)
        x2 = x1 + L2 * np.sin(q2)
        y2 = y1 - L2 * np.cos(q2)

        ax_pend.set_aspect("equal")
        ax_pend.plot([x0, x1], [y0, y1], "o-", color="blue", linewidth=4, markersize=10)
        ax_pend.plot([x1, x2], [y1, y2], "o-", color="green", linewidth=4, markersize=10)
        ax_pend.set_xlim(-2.5, 2.5)
        ax_pend.set_ylim(-2.5, 2.5)
        ax_pend.set_title(f"Acrobot  t={t:.1f}s")
        ax_pend.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax_pend.grid(True, alpha=0.3)

        # Right: Curves - dual y-axis (angle left, torque right)
        t_cur = np.array(history["t"][: i + 1])
        q1_cur = np.array(history["q1"][: i + 1])
        tau_cur = np.array(history["tau"][: i + 1])

        ax1 = ax_curves
        ax2 = ax_curves.twinx()
        ax1.plot(t_cur, q1_cur, "b-", linewidth=2, label="θ1")
        ax1.axhline(PI, color="b", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Angle θ1 (rad)", color="b")
        ax1.set_ylim(1.5, 4.8)
        ax1.tick_params(axis="y", labelcolor="b")
        ax2.plot(t_cur, tau_cur, "r-", linewidth=2, label="τ")
        ax2.set_ylabel("Torque τ", color="r")
        ax2.set_ylim(-25, 25)
        ax2.tick_params(axis="y", labelcolor="r")
        ax_curves.set_xlim(0, duration_sec)
        ax_curves.set_xlabel("Time (s)")
        ax_curves.set_title("Blue: Angle  |  Red: Torque")
        ax_curves.grid(True, alpha=0.3)

        plt.tight_layout()

    # Try FFMpeg first, else Pillow (gif)
    out_path = Path(__file__).resolve().parent / "demo_video.mp4"
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="Axiom-OS"))
        with writer.saving(fig, str(out_path), 150):
            for k, idx in enumerate(frame_indices):
                draw_frame(idx)
                writer.grab_frame()
                if (k + 1) % 60 == 0:
                    print(f"  Frame {k+1}/{len(frame_indices)}")
    except Exception as e:
        print(f"FFMpeg failed: {e}. Trying GIF...")
        out_path = Path(__file__).resolve().parent / "demo_video.gif"
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(out_path), 150):
            for k, idx in enumerate(frame_indices):
                draw_frame(idx)
                writer.grab_frame()
                if (k + 1) % 60 == 0:
                    print(f"  Frame {k+1}/{len(frame_indices)}")

    plt.close()
    print(f"\nSaved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
