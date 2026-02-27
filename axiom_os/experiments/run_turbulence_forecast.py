"""
Atmospheric Turbulence Forecasting - Strong Temporal Evolution
Real data, proper temporal sequences (group by spatial point), multi-step forecast.
Compares: PINN-LSTM vs RCLN vs Persistence vs Linear trend.
Visualization: line plots (time series), bar chart (MAE comparison) - no scatter.
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

from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d_temporal
from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
from axiom_os.layers.pinn_lstm import PhysicsInformedLSTM, pde_residual_loss_temporal

try:
    from axiom_os.coach import coach_score, coach_loss_torch
    HAS_COACH = True
except ImportError:
    HAS_COACH = False


def build_temporal_sequences(
    series_list: list,
    seq_len: int,
    forecast_horizon: int = 1,
    input_with_uv: bool = True,
) -> tuple:
    """
    Build sequences for temporal forecasting from per-spatial-point time series.
    Input: past seq_len steps of (t,x,y,z) or (t,x,y,z,u,v)
    Output: (u,v) at t + forecast_horizon
    Returns: X_seq (N, seq_len, D), Y (N, 2), coords_last (N, 4)
    """
    seqs, ys, coords_last = [], [], []
    for coords_ts, targets_ts in series_list:
        n_t = len(coords_ts)
        if n_t < seq_len + forecast_horizon:
            continue
        if input_with_uv:
            # Input: (t, x, y, z, u, v) = 6 dim
            full = np.hstack([coords_ts, targets_ts]).astype(np.float32)
            for i in range(n_t - seq_len - forecast_horizon + 1):
                seqs.append(full[i : i + seq_len])
                ys.append(targets_ts[i + seq_len + forecast_horizon - 1])
                coords_last.append(coords_ts[i + seq_len + forecast_horizon - 1])
        else:
            for i in range(n_t - seq_len - forecast_horizon + 1):
                seqs.append(coords_ts[i : i + seq_len].astype(np.float32))
                ys.append(targets_ts[i + seq_len + forecast_horizon - 1])
                coords_last.append(coords_ts[i + seq_len + forecast_horizon - 1])
    if not seqs:
        return None, None, None
    return np.stack(seqs), np.stack(ys), np.stack(coords_last)


def main():
    print("=" * 60)
    print("Atmospheric Turbulence Forecasting (Strong Temporal Evolution)")
    print("=" * 60)

    # Load real data - temporal structure (use_synthetic_if_fail=True if API unavailable)
    series_list, meta = load_atmospheric_turbulence_3d_temporal(
        n_lat=3, n_lon=3, delta_deg=0.12, forecast_days=5,
        use_synthetic_if_fail=True,  # Falls back to synthetic when API fails
    )
    n_series = len(series_list)
    n_t = meta.get("n_t", 0)
    print(f"\nData: {n_series} spatial points, ~{n_t} time steps each, source={meta.get('data_source','?')}")

    seq_len = 12  # longer context for temporal evolution
    forecast_horizon = 3  # predict 3 steps ahead
    input_with_uv = True  # include (u,v) history - strong temporal signal

    result = build_temporal_sequences(series_list, seq_len, forecast_horizon, input_with_uv)
    if result[0] is None:
        print("ERROR: Could not build sequences")
        return

    X_seq, Y, coords_last = result
    input_dim = X_seq.shape[2]  # 6 if input_with_uv else 4
    print(f"Sequences: {X_seq.shape[0]}, seq_len={seq_len}, input_dim={input_dim}, horizon={forecast_horizon}")

    split = int(0.75 * len(X_seq))
    X_train = torch.from_numpy(X_seq[:split]).float()
    Y_train = torch.from_numpy(Y[:split]).float()
    X_test = torch.from_numpy(X_seq[split:]).float()
    Y_test = torch.from_numpy(Y[split:]).float()

    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    wind_mag = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
    # Hard core expects (t,x,y,z) - 4 dim. For input_with_uv we have 6 dim, use last 4 from coords
    thresh = max(float(np.percentile(wind_mag, 85)), 5.0)
    hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=thresh, use_enhanced=True)

    coords_last_train = coords_last[:split]
    coords_last_test = coords_last[split:]

    # PINN-LSTM: input 6-dim (t,x,y,z,u,v). Use target coords for Hard Core (same info as RCLN)
    def _hard_core_wrapper(x):
        # x may be (B, 6) or (B, 4); hard_core needs (t,x,y,z)
        return hard_core(x[:, :4] if x.shape[-1] >= 4 else x)

    model = PhysicsInformedLSTM(
        input_dim=input_dim,
        hidden_dim=96,
        output_dim=2,
        seq_len=seq_len,
        hard_core_func=_hard_core_wrapper,
        lambda_res=1.0,
        num_layers=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=60)
    loss_fn = nn.HuberLoss(delta=1.0)
    lambda_coach = 0.1 if HAS_COACH else 0.0
    lambda_pde = 0.01

    epochs = 600
    print(f"\nTraining PINN-LSTM: {epochs} ep (horizon={forecast_horizon}, λ_pde={lambda_pde})...")
    t0 = time.perf_counter()
    coords_train_t = torch.from_numpy(coords_last_train).float()
    for epoch in range(epochs):
        model.set_lambda_decay(epoch, epochs, decay_min=0.6)
        opt.zero_grad()
        pred = model(X_train, target_coords=coords_train_t)
        loss_data = loss_fn(pred, Y_train)
        # PDE: temporal continuity with last observed (u,v)
        u_prev = X_train[:, -1, 4:6] if X_train.shape[2] >= 6 else hard_core(X_train[:, -1, :4])
        if not isinstance(u_prev, torch.Tensor):
            u_prev = torch.as_tensor(u_prev, dtype=torch.float32, device=pred.device)
        l_pde = pde_residual_loss_temporal(pred, u_prev, dt=float(forecast_horizon))
        l_coach = coach_loss_torch(pred, domain="fluids") if HAS_COACH else torch.tensor(0.0, device=pred.device)
        loss = loss_data + lambda_pde * l_pde + lambda_coach * l_coach
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step(loss.item())
        if (epoch + 1) % 150 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} data={loss_data.item():.4f}")

    elapsed = time.perf_counter() - t0
    coords_test_t = torch.from_numpy(coords_last_test).float()
    with torch.no_grad():
        pred_pinn = model(X_test, target_coords=coords_test_t).numpy()
    mae_pinn = (np.mean(np.abs(pred_pinn[:, 0] - Y_test.numpy()[:, 0])) +
                np.mean(np.abs(pred_pinn[:, 1] - Y_test.numpy()[:, 1]))) / 2
    print(f">>> PINN-LSTM MAE: {mae_pinn:.4f} ({elapsed:.1f}s)")

    # RCLN: point-wise at target time (t+horizon, x, y, z) - fair: same target as PINN-LSTM
    coords_test = torch.from_numpy(coords_last_test).float()
    try:
        from axiom_os.layers.rcln import RCLNLayer
        rcln = RCLNLayer(input_dim=4, hidden_dim=96, output_dim=2, hard_core_func=hard_core, lambda_res=1.0, net_type="mlp")
        opt_r = torch.optim.Adam(rcln.parameters(), lr=1e-3)
        coords_train = torch.from_numpy(coords_last_train).float()
        for _ in range(epochs):
            opt_r.zero_grad()
            p = rcln(coords_train)
            nn.functional.huber_loss(p, Y_train, delta=1.0).backward()
            opt_r.step()
        with torch.no_grad():
            pred_rcln = rcln(coords_test).numpy()
        mae_rcln = (np.mean(np.abs(pred_rcln[:, 0] - Y_test.numpy()[:, 0])) +
                    np.mean(np.abs(pred_rcln[:, 1] - Y_test.numpy()[:, 1]))) / 2
        print(f">>> RCLN (point-wise) MAE: {mae_rcln:.4f}")
    except Exception as e:
        mae_rcln = float("nan")
        print(f">>> RCLN: skip ({e})")

    # Persistence: predict = last observed (u,v)
    last_uv = X_test[:, -1, 4:6].numpy() if X_test.shape[2] >= 6 else np.tile([u_mean, v_mean], (len(Y_test), 1))
    mae_pers = (np.mean(np.abs(last_uv[:, 0] - Y_test.numpy()[:, 0])) +
                np.mean(np.abs(last_uv[:, 1] - Y_test.numpy()[:, 1]))) / 2
    print(f">>> Persistence (last uv) MAE: {mae_pers:.4f}")

    # Linear trend: fit last 3 points, extrapolate
    if X_test.shape[2] >= 6 and seq_len >= 3:
        u_last3 = X_test[:, -3:, 4].numpy()
        v_last3 = X_test[:, -3:, 5].numpy()
        t_idx = np.arange(3, dtype=np.float32)
        pred_lin = np.zeros_like(Y_test.numpy())
        for i in range(len(X_test)):
            pu = np.polyfit(t_idx, u_last3[i], 1)
            pv = np.polyfit(t_idx, v_last3[i], 1)
            pred_lin[i, 0] = np.polyval(pu, 3 + forecast_horizon - 1)
            pred_lin[i, 1] = np.polyval(pv, 3 + forecast_horizon - 1)
        mae_lin = (np.mean(np.abs(pred_lin[:, 0] - Y_test.numpy()[:, 0])) +
                   np.mean(np.abs(pred_lin[:, 1] - Y_test.numpy()[:, 1]))) / 2
        print(f">>> Linear trend MAE: {mae_lin:.4f}")
    else:
        mae_lin = float("nan")

    # Summary
    print("\n" + "-" * 50)
    results = {"PINN-LSTM": mae_pinn, "RCLN": mae_rcln, "Persistence": mae_pers, "Linear": mae_lin}
    for k, v in results.items():
        if not np.isnan(v):
            print(f"  {k}: {v:.4f}")
    best_name = min((k for k, v in results.items() if not np.isnan(v)), key=lambda k: results[k])
    print(f"  Best: {best_name} ({results[best_name]:.4f})")
    print("=" * 60)

    # Visualization: line plots (time series) + bar chart, no scatter
    out_dir = ROOT / "axiom_os" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart: MAE comparison
    fig1, ax = plt.subplots(figsize=(7, 4))
    names = [k for k, v in results.items() if not np.isnan(v)]
    vals = [results[k] for k in names]
    colors = ["#2ecc71" if k == "PINN-LSTM" else "#3498db" if k == "RCLN" else "#95a5a6" for k in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("MAE (m/s)")
    ax.set_title(f"Turbulence Forecast (horizon={forecast_horizon}h) | Real Data")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig1.savefig(out_dir / "turbulence_forecast_mae.png", dpi=150)
    plt.close()
    print(f"\nSaved: {out_dir / 'turbulence_forecast_mae.png'}")

    # 2. Line plot: sample time series True vs Pred (first 3 spatial points, last 48 steps)
    n_plot = min(3, n_series)
    fig2, axes = plt.subplots(n_plot, 2, figsize=(12, 3 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, -1)
    for spi in range(n_plot):
        coords_ts, targets_ts = series_list[spi]
        n_ts = len(targets_ts)
        if n_ts < seq_len + forecast_horizon + 20:
            continue
        # Get predictions for this series (find test indices that belong to this series)
        t_axis = np.arange(n_ts)
        axes[spi, 0].plot(t_axis, targets_ts[:, 0], "b-", label="True u", linewidth=1.5)
        axes[spi, 1].plot(t_axis, targets_ts[:, 1], "b-", label="True v", linewidth=1.5)
        # Overlay preds where we have them - need to match indices
        # Simplified: plot a segment where we have both true and pred
        seg_start = seq_len
        seg_end = min(n_ts - forecast_horizon, seg_start + 30)
        if seg_end <= seg_start:
            continue
        pred_seg_u, pred_seg_v = [], []
        for i in range(seg_start, seg_end):
            full = np.hstack([coords_ts, targets_ts])
            seq = full[i - seq_len : i]
            with torch.no_grad():
                inp = torch.from_numpy(seq).float().unsqueeze(0)
                tc = torch.from_numpy(coords_ts[i + seq_len + forecast_horizon - 1]).float().unsqueeze(0)
                out = model(inp, target_coords=tc).numpy()[0]
            pred_seg_u.append(out[0])
            pred_seg_v.append(out[1])
        t_seg = np.arange(seg_start + forecast_horizon - 1, seg_end + forecast_horizon - 1)
        if len(t_seg) != len(pred_seg_u):
            t_seg = np.arange(seg_start, seg_end)
        axes[spi, 0].plot(t_seg[: len(pred_seg_u)], pred_seg_u, "r--", label="PINN-LSTM u", linewidth=1)
        axes[spi, 1].plot(t_seg[: len(pred_seg_v)], pred_seg_v, "r--", label="PINN-LSTM v", linewidth=1)
        axes[spi, 0].set_ylabel("u (m/s)")
        axes[spi, 0].legend(loc="upper right", fontsize=8)
        axes[spi, 0].grid(True, alpha=0.3)
        axes[spi, 1].set_ylabel("v (m/s)")
        axes[spi, 1].legend(loc="upper right", fontsize=8)
        axes[spi, 1].grid(True, alpha=0.3)
    plt.suptitle("Time Series: True vs PINN-LSTM Forecast")
    plt.tight_layout()
    fig2.savefig(out_dir / "turbulence_forecast_series.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'turbulence_forecast_series.png'}")


if __name__ == "__main__":
    main()
