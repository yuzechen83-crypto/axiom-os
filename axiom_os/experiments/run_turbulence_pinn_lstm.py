"""
Turbulence PINN-LSTM Experiment
Physics-Informed LSTM for spatiotemporal wind prediction.
Compares: PINN-LSTM vs RCLN (MLP) vs MLP baseline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
from axiom_os.layers.pinn_lstm import PhysicsInformedLSTM, pde_residual_loss_temporal

# Coach: optional (proprietary)
try:
    from axiom_os.coach import coach_score, coach_loss_torch
    HAS_COACH = True
except ImportError:
    HAS_COACH = False


def build_sequences(
    coords: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    meta: dict,
) -> tuple:
    """
    Build (X_seq, Y) for LSTM from flat (t,x,y,z) -> (u,v) data.
    For synthetic: reshape from regular grid (meshgrid ij order: t,x,y,z, z fastest).
    For real: group by spatial point.
    # Try to detect grid structure from meta
    Returns: X_seq (N, seq_len, 4), Y (N, 2), valid mask
    """
    n = len(coords)
    if n < seq_len + 1:
        return None, None, None

    data_src = meta.get("data_source", "unknown")
    if data_src == "synthetic":
        # Synthetic: meshgrid(t,x,y,z, indexing="ij"), ravel C-order: z fastest
        # Default: n_t=24, n_x=n_y=n_z=4 -> 1536
        n_pts = n
        n_t_candidates = [24, 48, 72, 96, 12, 36]
        n_t, n_x, n_y, n_z = seq_len + 1, 2, 2, 2
        for nt in n_t_candidates:
            if n_pts % nt == 0:
                nxyz = n_pts // nt
                n_cube = int(round(nxyz ** (1/3)))
                if n_cube ** 3 == nxyz:
                    n_t, n_x, n_y, n_z = nt, n_cube, n_cube, n_cube
                    break
        if n_t < seq_len + 1:
            n_t = n_pts // 64 if n_pts >= 64 else seq_len + 1
            n_xyz = n_pts // n_t
            n_x = n_y = n_z = max(2, int(round(n_xyz ** (1/3))))

        n_xyz = n_t * n_x * n_y * n_z
        if n_xyz > n:
            n_xyz = n
        try:
            coords_4d = coords[:n_xyz].reshape(n_t, n_x, n_y, n_z, 4)
            targets_4d = targets[:n_xyz].reshape(n_t, n_x, n_y, n_z, 2)
        except ValueError:
            pass  # fall through to sliding window
        else:
            seqs, ys = [], []
            for ix in range(n_x):
                for iy in range(n_y):
                    for iz in range(n_z):
                        c_seq = coords_4d[:, ix, iy, iz, :]  # (n_t, 4)
                        t_seq = targets_4d[:, ix, iy, iz, :]  # (n_t, 2)
                        for i in range(n_t - seq_len):
                            seqs.append(c_seq[i : i + seq_len])
                            ys.append(t_seq[i + seq_len - 1])
            if seqs:
                X_seq = np.stack(seqs, axis=0).astype(np.float32)
                Y = np.stack(ys, axis=0).astype(np.float32)
                return X_seq, Y, np.ones(len(Y), dtype=bool)

    # Fallback: sliding window over flat data (works for any structure)
    seqs, ys = [], []
    for i in range(n - seq_len):
        seqs.append(coords[i : i + seq_len].astype(np.float32))
        ys.append(targets[i + seq_len - 1].astype(np.float32))
    X_seq = np.stack(seqs, axis=0)
    Y = np.stack(ys, axis=0)
    return X_seq, Y, np.ones(len(Y), dtype=bool)


def main():
    print("=" * 60)
    print("Turbulence PINN-LSTM Experiment")
    print("=" * 60)

    # Load data: use_synthetic=False for real API data (Open-Meteo)
    use_synthetic = False  # True = synthetic, False = real API
    if use_synthetic:
        from axiom_os.datasets.atmospheric_turbulence import _generate_synthetic_3d_turbulence
        coords, targets, meta = _generate_synthetic_3d_turbulence(
            n_t=48, n_x=4, n_y=4, n_z=4, seed=42,
        )
    else:
        coords, targets, meta = load_atmospheric_turbulence_3d(
            n_lat=3, n_lon=3, delta_deg=0.15, forecast_days=3,
            use_synthetic_if_fail=True,
        )
    n = len(coords)
    data_src = meta.get("data_source", "unknown")
    print(f"\nData: {n} points, source={data_src}")

    seq_len = 8
    result = build_sequences(coords, targets, seq_len, meta)
    if result[0] is None:
        print("ERROR: Could not build sequences (insufficient data)")
        return

    X_seq, Y, _ = result
    print(f"Sequences: {X_seq.shape[0]} samples, seq_len={seq_len}, input_dim={X_seq.shape[2]}")

    split = int(0.8 * len(X_seq))
    X_train_seq = torch.from_numpy(X_seq[:split]).float()
    Y_train = torch.from_numpy(Y[:split]).float()
    X_test_seq = torch.from_numpy(X_seq[split:]).float()
    Y_test = torch.from_numpy(Y[split:]).float()

    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    wind_mag = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
    thresh = max(float(np.percentile(wind_mag, 85)), 5.0)
    hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=thresh, use_enhanced=True)
    print(f"Hard Core: adaptive+enhanced, threshold={thresh:.1f}")

    # PINN-LSTM model
    model = PhysicsInformedLSTM(
        input_dim=4,
        hidden_dim=64,
        output_dim=2,
        seq_len=seq_len,
        hard_core_func=hard_core,
        lambda_res=1.0,
        num_layers=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=80)
    loss_fn = nn.HuberLoss(delta=1.0)
    w_u, w_v = 1.5, 1.0
    lambda_coach = 0.15 if HAS_COACH else 0.0
    lambda_pde = 0.02  # temporal continuity (reduced - can conflict with data fit)

    epochs = 800
    print(f"\nTraining PINN-LSTM: {epochs} epochs (λ_coach={lambda_coach}, λ_pde={lambda_pde})...")
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.set_lambda_decay(epoch, epochs, decay_min=0.6)
        opt.zero_grad()
        pred = model(X_train_seq)
        loss_u = loss_fn(pred[:, 0:1], Y_train[:, 0:1])
        loss_v = loss_fn(pred[:, 1:2], Y_train[:, 1:2])
        loss_data = w_u * loss_u + w_v * loss_v

        # PDE residual: temporal continuity (use previous step as u_prev)
        if lambda_pde > 0 and X_train_seq.shape[1] >= 2:
            # Approximate: u_prev from hard core at t-1
            x_prev = X_train_seq[:, -2, :]
            with torch.no_grad():
                u_prev = hard_core(x_prev)
                if not isinstance(u_prev, torch.Tensor):
                    u_prev = torch.as_tensor(u_prev, dtype=torch.float32, device=pred.device)
            dt = 1.0 / max(1, seq_len)
            l_pde = pde_residual_loss_temporal(pred, u_prev, dt=dt)
        else:
            l_pde = torch.tensor(0.0, device=pred.device)

        l_coach = torch.tensor(0.0, device=pred.device)
        if HAS_COACH and lambda_coach > 0:
            l_coach = coach_loss_torch(pred, domain="fluids")

        loss = loss_data + lambda_pde * l_pde + lambda_coach * l_coach
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step(loss.item())

        if (epoch + 1) % 200 == 0:
            with torch.no_grad():
                sc = coach_score(X_train_seq[:, -1].numpy(), pred.numpy(), domain="fluids") if HAS_COACH else 0.0
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f} data={loss_data.item():.6f} pde={l_pde.item():.6f} coach={l_coach.item():.4f} score={sc:.4f}")

    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        pred_test = model(X_test_seq).numpy()
    mae_u = np.mean(np.abs(pred_test[:, 0] - Y_test.numpy()[:, 0]))
    mae_v = np.mean(np.abs(pred_test[:, 1] - Y_test.numpy()[:, 1]))
    mae_pinn_lstm = (mae_u + mae_v) / 2
    coach_test = coach_score(X_test_seq[:, -1].numpy(), pred_test, domain="fluids") if HAS_COACH else 0.0

    print(f"\n>>> PINN-LSTM Test MAE: u={mae_u:.4f} v={mae_v:.4f} (avg={mae_pinn_lstm:.4f})")
    if HAS_COACH:
        print(f">>> Coach score (Test): {coach_test:.4f}")
    print(f">>> Training time: {elapsed:.2f}s")

    # Baselines: point-wise (use last step of sequence)
    X_train_pt = X_train_seq[:, -1, :]
    X_test_pt = X_test_seq[:, -1, :]

    # Baseline 1: RCLN (Hard Core + MLP) - point-wise, same as run_turbulence_full
    try:
        from axiom_os.layers.rcln import RCLNLayer
        rcln = RCLNLayer(
            input_dim=4, hidden_dim=64, output_dim=2,
            hard_core_func=hard_core, lambda_res=1.0, net_type="mlp",
        )
        opt_rcln = torch.optim.Adam(rcln.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt_rcln.zero_grad()
            p = rcln(X_train_pt)
            loss = nn.functional.huber_loss(p, Y_train, delta=1.0)
            loss.backward()
            opt_rcln.step()
        with torch.no_grad():
            pred_rcln = rcln(X_test_pt).numpy()
        mae_rcln = (np.mean(np.abs(pred_rcln[:, 0] - Y_test.numpy()[:, 0])) +
                    np.mean(np.abs(pred_rcln[:, 1] - Y_test.numpy()[:, 1]))) / 2
        print(f"\n>>> RCLN (Hard Core + MLP) MAE: {mae_rcln:.4f}")
    except Exception as e:
        mae_rcln = float("nan")
        print(f"\n>>> RCLN baseline: skip ({e})")

    # Baseline 2: MLP on last point only (no Hard Core)
    mlp = nn.Sequential(
        nn.Linear(4, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 2),
    )
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt_mlp.zero_grad()
        p = mlp(X_train_pt)
        loss = nn.functional.huber_loss(p, Y_train, delta=1.0)
        loss.backward()
        opt_mlp.step()
    with torch.no_grad():
        pred_mlp = mlp(X_test_pt).numpy()
    mae_mlp = (np.mean(np.abs(pred_mlp[:, 0] - Y_test.numpy()[:, 0])) +
               np.mean(np.abs(pred_mlp[:, 1] - Y_test.numpy()[:, 1]))) / 2
    print(f">>> MLP baseline (no Hard Core) MAE: {mae_mlp:.4f}")

    # Summary
    print("\n" + "-" * 40)
    print("Summary:")
    print(f"  PINN-LSTM:     {mae_pinn_lstm:.4f}")
    print(f"  RCLN (MLP):    {mae_rcln:.4f}" if not np.isnan(mae_rcln) else "  RCLN: skip")
    print(f"  MLP baseline:  {mae_mlp:.4f}")
    best = min(mae_pinn_lstm, mae_rcln if not np.isnan(mae_rcln) else mae_pinn_lstm, mae_mlp)
    print(f"  Best:          {best:.4f}")
    print("=" * 60)

    # 3D visualization: True vs PINN-LSTM Pred in (x, y, z) space
    # Coords for predictions: each sequence i predicts at coords[i+seq_len-1]
    n_seq = len(X_seq)
    coords_pred = coords[seq_len - 1 : seq_len - 1 + n_seq]  # (n_seq, 4)
    # Full model predictions (train + test)
    with torch.no_grad():
        pred_all = model(torch.from_numpy(X_seq).float()).numpy()
    Y_all = Y

    step = max(1, n_seq // 600)
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    sc1 = ax1.scatter(
        coords_pred[::step, 1], coords_pred[::step, 2], coords_pred[::step, 3],
        c=Y_all[::step, 0], cmap="RdBu_r", s=8, alpha=0.7,
    )
    ax1.set_xlabel("x (lat norm)")
    ax1.set_ylabel("y (lon norm)")
    ax1.set_zlabel("z (height norm)")
    ax1.set_title("True u wind (3D)")
    plt.colorbar(sc1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(132, projection="3d")
    sc2 = ax2.scatter(
        coords_pred[::step, 1], coords_pred[::step, 2], coords_pred[::step, 3],
        c=pred_all[::step, 0], cmap="RdBu_r", s=8, alpha=0.7,
    )
    ax2.set_xlabel("x (lat norm)")
    ax2.set_ylabel("y (lon norm)")
    ax2.set_zlabel("z (height norm)")
    ax2.set_title("PINN-LSTM Pred u (3D)")
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(133, projection="3d")
    sc3 = ax3.scatter(
        coords_pred[::step, 1], coords_pred[::step, 2], coords_pred[::step, 3],
        c=Y_all[::step, 1], cmap="viridis", s=8, alpha=0.7,
    )
    ax3.set_xlabel("x (lat norm)")
    ax3.set_ylabel("y (lon norm)")
    ax3.set_zlabel("z (height norm)")
    ax3.set_title("True v wind (3D)")
    plt.colorbar(sc3, ax=ax3, shrink=0.5)

    plt.suptitle(f"Turbulence 3D: {data_src} data | PINN-LSTM MAE u={mae_u:.3f} v={mae_v:.3f}")
    plt.tight_layout()
    out_3d = ROOT / "axiom_os" / "experiments" / "pinn_lstm_3d_plot.png"
    out_3d.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_3d, dpi=150)
    plt.close()
    print(f"\nSaved 3D plot: {out_3d}")

    # Second figure: Pred v (3D) + True vs Pred scatter
    fig2 = plt.figure(figsize=(14, 5))
    ax4 = fig2.add_subplot(121, projection="3d")
    sc4 = ax4.scatter(
        coords_pred[::step, 1], coords_pred[::step, 2], coords_pred[::step, 3],
        c=pred_all[::step, 1], cmap="viridis", s=8, alpha=0.7,
    )
    ax4.set_xlabel("x (lat norm)")
    ax4.set_ylabel("y (lon norm)")
    ax4.set_zlabel("z (height norm)")
    ax4.set_title("PINN-LSTM Pred v (3D)")
    plt.colorbar(sc4, ax=ax4, shrink=0.5)

    ax5 = fig2.add_subplot(122)
    ax5.scatter(Y_test.numpy()[:, 0], pred_test[:, 0], alpha=0.5, s=12, label="u", c="C0")
    ax5.scatter(Y_test.numpy()[:, 1], pred_test[:, 1], alpha=0.5, s=12, label="v", c="C1")
    ax5.plot([-10, 10], [-10, 10], "k--", alpha=0.5)
    ax5.set_xlabel("True (m/s)")
    ax5.set_ylabel("Pred (m/s)")
    ax5.set_title(f"True vs Pred (Test) | u MAE={mae_u:.3f} v MAE={mae_v:.3f}")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect("equal")
    plt.tight_layout()
    out_scatter = ROOT / "axiom_os" / "experiments" / "pinn_lstm_scatter.png"
    plt.savefig(out_scatter, dpi=150)
    plt.close()
    print(f"Saved scatter: {out_scatter}")


if __name__ == "__main__":
    main()
