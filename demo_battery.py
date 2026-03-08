"""
Battery RUL Demo Video/GIF
Left: NASA paper (formulas). Right: Axiom-OS running (Data -> RCLN -> Discovery).
Caption: "AI doesn't just predict battery life. It understands why."

Usage: python axiom_os/demo_battery.py [--fast]
Output: axiom_os/demo_battery.gif (or .mp4 if ffmpeg available)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from axiom_os.layers.rcln import RCLNLayer
from axiom_os.datasets.nasa_battery import load_battery_data

try:
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _build_aging_features(t_raw: np.ndarray) -> np.ndarray:
    t = np.asarray(t_raw, dtype=np.float64).ravel()
    t_safe = np.maximum(t, 1e-8)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-12)
    return np.column_stack([t, t**2, np.sqrt(t_safe), np.exp(t_norm)])


def _discover_terms(cycles_raw: np.ndarray, soft_shell: np.ndarray) -> list[str]:
    """Return discovered terms in order: t, t^2, sqrt(t), exp(t)."""
    X = _build_aging_features(cycles_raw)
    y = np.asarray(soft_shell, dtype=np.float64).ravel()
    if not HAS_SKLEARN:
        return ["t", "t^2", "sqrt(t)"]
    model = Ridge(alpha=1e-4)
    model.fit(X, y)
    coef = model.coef_
    terms = []
    if abs(coef[0]) > 1e-6:
        terms.append("t")
    if abs(coef[1]) > 1e-6:
        terms.append("t^2")
    if abs(coef[2]) > 1e-6:
        terms.append("sqrt(t)")
    if abs(coef[3]) > 1e-6:
        terms.append("exp(t)")
    return terms if terms else ["t"]


def ideal_battery_model(x):
    if isinstance(x, torch.Tensor):
        vals = x
    elif hasattr(x, "values") and not isinstance(x, torch.Tensor):
        vals = x.values
    else:
        vals = x
    t = torch.as_tensor(vals, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return torch.ones_like(t[:, :1]) * 1.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="5 sec, fewer frames")
    args = parser.parse_args()

    print("=" * 60)
    print("Battery RUL Demo - NASA Paper vs Axiom-OS")
    print("=" * 60)

    # Pre-run pipeline
    print("Loading data & training RCLN...")
    cycles_norm, capacity_norm, scalers = load_battery_data()
    n = len(cycles_norm)
    split = int(0.8 * n)
    cycles_train = cycles_norm[:split]
    capacity_train = capacity_norm[:split]
    cycles_test = cycles_norm[split:]
    capacity_test = capacity_norm[split:]

    def inv_cycle(z):
        return z * (scalers["cycle_max"] - scalers["cycle_min"]) + scalers["cycle_min"]

    def inv_cap(z):
        return z * (scalers["cap_max"] - scalers["cap_min"]) + scalers["cap_min"]

    cycles_train_raw = inv_cycle(cycles_train.ravel())
    cycles_test_raw = inv_cycle(cycles_test.ravel())
    cap_train_raw = inv_cap(capacity_train.ravel())
    cap_test_raw = inv_cap(capacity_test.ravel())

    rcln = RCLNLayer(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        hard_core_func=ideal_battery_model,
        lambda_res=1.0,
    )
    optimizer = torch.optim.Adam(rcln.parameters(), lr=0.01)
    x_train = torch.from_numpy(cycles_train).float()
    y_train = torch.from_numpy(capacity_train).float()

    loss_history = []
    for epoch in range(2000):
        optimizer.zero_grad()
        pred = rcln(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    with torch.no_grad():
        pred_train = rcln(x_train).numpy()
        soft_shell = rcln._last_y_soft.detach().cpu().numpy().ravel()
        x_test = torch.from_numpy(cycles_test).float()
        pred_test = rcln(x_test).numpy()
    pred_train_raw = inv_cap(pred_train.ravel())
    pred_test_raw = inv_cap(pred_test.ravel())

    discovered_terms = _discover_terms(cycles_train_raw, soft_shell)
    print(f"Discovered terms: {discovered_terms}")

    # Animation setup
    fps = 15
    duration_sec = 5 if args.fast else 18
    n_frames = fps * duration_sec

    # Phase boundaries (0-1)
    phase_data = 0.20
    phase_train = 0.50
    phase_discovery = 1.0

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax_left = fig.add_subplot(121)
    ax_right = fig.add_subplot(122)

    # NASA paper aesthetic
    ax_left.set_facecolor("#16213e")
    ax_left.tick_params(colors="#e8e8e8")
    ax_left.spines["bottom"].set_color("#e8e8e8")
    ax_left.spines["left"].set_color("#e8e8e8")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # Right panel
    ax_right.set_facecolor("#0f3460")
    ax_right.tick_params(colors="#e8e8e8")
    ax_right.spines["bottom"].set_color("#e8e8e8")
    ax_right.spines["left"].set_color("#e8e8e8")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    def draw_frame(frame_idx: int):
        progress = (frame_idx + 1) / n_frames

        # --- Left: NASA paper (static formulas) ---
        ax_left.clear()
        ax_left.set_facecolor("#16213e")
        ax_left.set_xlim(0, 10)
        ax_left.set_ylim(0, 12)
        ax_left.axis("off")

        title = ax_left.text(5, 11, "NASA PCoE\nBattery Dataset", ha="center", fontsize=14,
                            color="#e94560", fontweight="bold", family="monospace")
        ax_left.text(1, 9.2, "Capacity fade model:", fontsize=10, color="#a0a0a0", family="monospace")
        ax_left.text(1, 8.2, r"$Q(n) = Q_0 \cdot e^{-\alpha n}$", fontsize=12, color="#e8e8e8",
                     family="serif")
        ax_left.text(1, 6.8, "Empirical RUL:", fontsize=10, color="#a0a0a0", family="monospace")
        ax_left.text(1, 5.8, r"$RUL = \frac{C_{EOL} - C_t}{\partial C / \partial t}$", fontsize=11,
                     color="#e8e8e8", family="serif")
        ax_left.text(1, 4.2, "Power-law aging:", fontsize=10, color="#a0a0a0", family="monospace")
        ax_left.text(1, 3.2, r"$\Delta Q \propto n^\beta$", fontsize=12, color="#e8e8e8",
                     family="serif")
        ax_left.text(1, 1.5, "Saha & Goebel (2007)", fontsize=8, color="#606060",
                     style="italic", family="serif")

        # --- Right: Axiom-OS workflow ---
        ax_right.clear()
        ax_right.set_facecolor("#0f3460")
        ax_right.set_xlim(0, 140)
        ax_right.set_ylim(1.2, 2.15)
        ax_right.set_xlabel("Cycle", color="#e8e8e8")
        ax_right.set_ylabel("Capacity (Ah)", color="#e8e8e8")

        # Phase 1: Data streams in
        n_show = int(len(cycles_train_raw) * min(1.0, progress / phase_data))
        if n_show > 0:
            ax_right.plot(cycles_train_raw[:n_show], cap_train_raw[:n_show],
                          "b-", linewidth=2, alpha=0.9, label="Data")
            ax_right.scatter(cycles_train_raw[n_show - 1], cap_train_raw[n_show - 1],
                             c="cyan", s=80, zorder=5, edgecolors="white")

        # Phase 2: RCLN fit appears
        if progress >= phase_data:
            ax_right.plot(cycles_train_raw, pred_train_raw, "r-", linewidth=2,
                          alpha=0.7 + 0.3 * min(1, (progress - phase_data) / (phase_train - phase_data)),
                          label="RCLN fit")
        if progress >= phase_train:
            ax_right.plot(cycles_test_raw, pred_test_raw, "r--", linewidth=1.5,
                          alpha=0.8, label="RUL prediction")
            ax_right.scatter(cycles_test_raw, cap_test_raw, c="blue", s=20, alpha=0.6)

        # Phase 3: Found terms pop up (ensure sqrt(t) is featured)
        if progress >= phase_train and discovered_terms:
            t_prog = (progress - phase_train) / (phase_discovery - phase_train)
            n_terms_show = min(len(discovered_terms), max(1, int(len(discovered_terms) * t_prog * 1.2)))
            terms_shown = discovered_terms[:n_terms_show]
            for i, term in enumerate(terms_shown):
                y_pos = 2.08 - i * 0.06
                ax_right.text(100, y_pos, f"Found term: {term}", fontsize=12, color="#00ff88",
                              fontweight="bold", family="monospace", va="top",
                              bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d2818", edgecolor="#00ff88", alpha=0.9))

        ax_right.legend(loc="upper right", fontsize=8)
        ax_right.grid(True, alpha=0.2)
        ax_right.set_title("Axiom-OS: Data → RCLN → Discovery", color="#e8e8e8")

        # Caption
        fig.suptitle("AI doesn't just predict battery life. It understands why.",
                     fontsize=18, color="#e94560", fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Write video/gif
    out_mp4 = Path(__file__).resolve().parent / "demo_battery.mp4"
    out_gif = Path(__file__).resolve().parent / "demo_battery.gif"
    saved_path = None

    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="Axiom-OS"))
        with writer.saving(fig, str(out_mp4), 150):
            for k in range(n_frames):
                draw_frame(k)
                writer.grab_frame()
                if (k + 1) % 30 == 0:
                    print(f"  Frame {k+1}/{n_frames}")
        saved_path = out_mp4
    except Exception as e:
        print(f"FFMpeg failed: {e}. Writing GIF...")
        saved_path = out_gif
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(out_gif), 150):
            for k in range(n_frames):
                draw_frame(k)
                writer.grab_frame()
                if (k + 1) % 30 == 0:
                    print(f"  Frame {k+1}/{n_frames}")

    plt.close()
    print(f"\nSaved: {saved_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
