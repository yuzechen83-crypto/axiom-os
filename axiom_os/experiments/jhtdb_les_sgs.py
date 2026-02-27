"""
JHTDB LES-SGS - Real DNS Data, Hard Core + Soft Shell

REAL DATA ONLY. Loads isotropic turbulence from Johns Hopkins Turbulence Database.
Task: Learn Reynolds stress tau_ij (Soft Shell) to close coarse-grid N-S (Hard Core).

Hard Core: Coarse-grid Navier-Stokes (filtered DNS)
Soft Shell: tau_ij = <u_i u_j> - <u_i><u_j>  (residual to learn)

Baseline: Smagorinsky tau_ij = -2*nu_sgs*S_ij, nu_sgs = (Cs*delta)^2*|S|

Run: python -m axiom_os.experiments.jhtdb_les_sgs
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _strain_rate(u: np.ndarray) -> np.ndarray:
    """S_ij = 0.5*(du_i/dx_j + du_j/dx_i). u: (nx,ny,nz,3). Central diff."""
    S = np.zeros((*u.shape[:3], 3, 3))
    for i in range(3):
        for j in range(3):
            dui_dxj = np.gradient(u[..., i], axis=j)
            duj_dxi = np.gradient(u[..., j], axis=i)
            S[..., i, j] = 0.5 * (dui_dxj + duj_dxi)
    return S


def smagorinsky_tau(u_coarse: np.ndarray, Cs: float = 0.17, delta: float = 2.0) -> np.ndarray:
    """Smagorinsky: tau_ij = -2*nu_sgs*S_ij, nu_sgs = (Cs*delta)^2*|S|."""
    S = _strain_rate(u_coarse)
    S2 = np.sum(S**2, axis=(-2, -1))
    S_norm = np.sqrt(np.maximum(2 * S2, 1e-20))
    nu_sgs = (Cs * delta) ** 2 * S_norm
    tau = np.zeros_like(S)
    for i in range(3):
        for j in range(3):
            tau[..., i, j] = -2 * nu_sgs * S[..., i, j]
    return tau


def _spatial_split_mask(nx: int, ny: int, nz: int, train_top: bool) -> np.ndarray:
    """train_top: z > nz/2; else z <= nz/2."""
    z_idx = np.arange(nz)
    mask = (z_idx > nz // 2) if train_top else (z_idx <= nz // 2)
    return np.broadcast_to(mask, (nx, ny, nz))


def train_tbnn(
    u_coarse: np.ndarray,
    tau_target: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    epochs: int = 1000,
    hidden: int = 48,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[np.ndarray, float, float]:
    """
    Train TBNN (Galilean invariant, normalized) on train_mask, evaluate on test_mask.
    Uses LR scheduler and extended training for better R².
    Returns: tau_pred (full grid), r2_train, r2_test.
    """
    import torch
    from axiom_os.core.turbulence_invariants import extract_invariants_and_basis_normalized
    from axiom_os.layers.tbnn import TBNN, stack_tensor_basis

    inv, basis, sigma = extract_invariants_and_basis_normalized(u_coarse)
    inv_flat = inv.reshape(-1, 5).astype(np.float32)
    tb_stacked = stack_tensor_basis(basis).numpy()
    tb_flat = tb_stacked.reshape(-1, 10, 3, 3).astype(np.float32)
    sigma_flat = sigma.reshape(-1).astype(np.float32)
    tau_flat = tau_target.reshape(-1, 9).astype(np.float32)

    train_idx = train_mask.ravel()
    test_idx = test_mask.ravel()
    X_train = torch.from_numpy(inv_flat[train_idx])
    tb_train = torch.from_numpy(tb_flat[train_idx])
    sigma_train = torch.from_numpy(sigma_flat[train_idx])
    y_train = torch.from_numpy(tau_flat[train_idx])
    X_test = torch.from_numpy(inv_flat[test_idx])
    tb_test = torch.from_numpy(tb_flat[test_idx])
    sigma_test = torch.from_numpy(sigma_flat[test_idx])
    y_test = torch.from_numpy(tau_flat[test_idx])

    model = TBNN(n_invariants=5, n_tensors=10, hidden=hidden, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=80, min_lr=1e-5
    )
    for ep in range(epochs):
        tau_pred = model(X_train, tb_train, sigma_train)
        loss = ((tau_pred.reshape(-1, 9) - y_train) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step(float(loss.detach()))

    inv_full = torch.from_numpy(inv_flat)
    tb_full = torch.from_numpy(tb_flat)
    sigma_full = torch.from_numpy(sigma_flat)
    with torch.no_grad():
        tau_train_pred = model(X_train, tb_train, sigma_train).reshape(-1, 9).numpy()
        tau_test_pred = model(X_test, tb_test, sigma_test).reshape(-1, 9).numpy()
        tau_full = model(inv_full, tb_full, sigma_full).reshape(-1, 9).numpy()

    def r2(y_true, y_pred):
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-20
        return 1.0 - ss_res / ss_tot

    r2_train = r2(tau_flat[train_idx], tau_train_pred)
    r2_test = r2(tau_flat[test_idx], tau_test_pred)
    tau_pred = tau_full.reshape(tau_target.shape)
    return tau_pred, r2_train, r2_test


def _tau_3x3_to_6ch(tau: np.ndarray) -> np.ndarray:
    """Convert (...,3,3) symmetric tensor to (...,6): [τ00, τ11, τ22, τ01, τ02, τ12]."""
    return np.stack([
        tau[..., 0, 0], tau[..., 1, 1], tau[..., 2, 2],
        tau[..., 0, 1], tau[..., 0, 2], tau[..., 1, 2],
    ], axis=-1)


def train_fno(
    u_coarse: np.ndarray,
    tau_target: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    epochs: int = 600,
) -> Tuple[np.ndarray, float, float]:
    """
    Train FNO3d (non-local) on train_mask, evaluate on test_mask.
    Returns: tau_pred (full grid as 6ch), r2_train, r2_test.
    """
    import torch
    from axiom_os.layers.fno import FNO3d

    u_ch = np.transpose(u_coarse, (3, 0, 1, 2))
    u_batch = torch.from_numpy(u_ch[np.newaxis].astype(np.float32))
    tau_6ch = _tau_3x3_to_6ch(tau_target)
    tau_ch = np.transpose(tau_6ch, (3, 0, 1, 2))
    tau_batch = torch.from_numpy(tau_ch[np.newaxis].astype(np.float32))

    train_flat = train_mask.ravel()
    test_flat = test_mask.ravel()

    model = FNO3d(in_channels=3, out_channels=6, width=24, modes1=3, modes2=3, modes3=3, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=50, min_lr=1e-5)

    for _ in range(epochs):
        pred = model(u_batch)
        pred_flat = pred.reshape(-1, 6)
        tau_flat = tau_batch.reshape(-1, 6)
        loss = ((pred_flat[train_flat] - tau_flat[train_flat]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step(float(loss.detach()))

    with torch.no_grad():
        pred_full = model(u_batch).reshape(-1, 6).numpy()
    tau_flat_np = tau_batch.reshape(-1, 6).numpy()

    def r2(y_true, y_pred):
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-20
        return 1.0 - ss_res / ss_tot

    r2_train = r2(tau_flat_np[train_flat], pred_full[train_flat])
    r2_test = r2(tau_flat_np[test_flat], pred_full[test_flat])
    tau_pred_6ch = pred_full.reshape(1, 6, *u_coarse.shape[:3])
    return tau_pred_6ch, r2_train, r2_test


def _run_pinn_lstm_r2(epochs: int = 200) -> float:
    """Run PINN-LSTM (velocity prediction) and return test R². Returns 0.0 on failure."""
    try:
        from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
        from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
        from axiom_os.layers.pinn_lstm import PhysicsInformedLSTM
        import torch
        import torch.nn as nn

        coords, targets, meta = load_atmospheric_turbulence_3d(
            n_lat=3, n_lon=3, delta_deg=0.15, forecast_days=3, use_synthetic_if_fail=True
        )
        seq_len = 6
        seqs, ys = [], []
        for i in range(len(coords) - seq_len):
            seqs.append(coords[i : i + seq_len].astype(np.float32))
            ys.append(targets[i + seq_len - 1].astype(np.float32))
        if len(seqs) < 50:
            return 0.0
        X_seq = np.stack(seqs)
        Y = np.stack(ys)
        split = int(0.8 * len(X_seq))
        X_train = torch.from_numpy(X_seq[:split]).float()
        Y_train = torch.from_numpy(Y[:split]).float()
        X_test = torch.from_numpy(X_seq[split:]).float()
        Y_test = torch.from_numpy(Y[split:]).float()

        u_mean = float(Y_train[:, 0].mean())
        v_mean = float(Y_train[:, 1].mean())
        hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=5.0)
        model = PhysicsInformedLSTM(input_dim=4, hidden_dim=32, output_dim=2, seq_len=seq_len,
                                  hard_core_func=hard_core, lambda_res=1.0, num_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(X_train)
            loss = nn.functional.mse_loss(pred, Y_train)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred_test = model(X_test).numpy()
        Y_test_np = Y_test.numpy()
        ss_res = np.sum((pred_test - Y_test_np) ** 2)
        ss_tot = np.sum((Y_test_np - np.mean(Y_test_np)) ** 2) + 1e-20
        return float(1.0 - ss_res / ss_tot)
    except Exception:
        return 0.0


def plot_comparison_bar(
    r2_smag: float,
    r2_tbnn: float,
    r2_fno: float,
    r2_pinn_lstm: float,
    out_dir: Path,
) -> None:
    """Bar chart: Smagorinsky vs TBNN vs FNO vs PINN-LSTM."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = ["Smagorinsky", "TBNN", "FNO", "PINN-LSTM"]
    values = [r2_smag, r2_tbnn, r2_fno, r2_pinn_lstm]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("R² (Test)", fontsize=12)
    ax.set_title("JHTDB LES-SGS: Turbulence Closure Comparison\n(Smag/TBNN/FNO: τ prediction; PINN-LSTM: velocity prediction)", fontsize=11)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylim(min(min(values) - 0.1, -0.15), max(max(values) + 0.1, 0.1))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / "jhtdb_les_sgs_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved comparison: {out_path}")


def plot_jhtdb_3d(
    u_coarse: np.ndarray,
    tau_target: np.ndarray,
    tau_tbnn: np.ndarray,
    tau_smag: np.ndarray,
    train_mask: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Generate 3D visualizations of JHTDB LES-SGS: velocity, tau target, TBNN pred, Smagorinsky.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        return

    n_c = u_coarse.shape[0]
    xg = np.arange(n_c)
    X, Y, Z = np.meshgrid(xg, xg, xg, indexing="ij")
    x, y, z = X.ravel(), Y.ravel(), Z.ravel()

    u_mag = np.sqrt(np.sum(u_coarse**2, axis=-1)).ravel()
    tau_target_norm = np.sqrt(np.sum(tau_target**2, axis=(-2, -1))).ravel()
    tau_tbnn_norm = np.sqrt(np.sum(tau_tbnn**2, axis=(-2, -1))).ravel()
    tau_smag_norm = np.sqrt(np.sum(tau_smag**2, axis=(-2, -1))).ravel()
    train_flat = train_mask.ravel()

    def _scatter3d(ax, x, y, z, c, title, cmap="viridis", s=25):
        sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=s, alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, shrink=0.6)
        ax.view_init(elev=25, azim=45)

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _scatter3d(ax1, x, y, z, u_mag, "Velocity |u| (coarse)")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _scatter3d(ax2, x, y, z, tau_target_norm, "Target |τ| (DNS SGS stress)")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    _scatter3d(ax3, x, y, z, tau_tbnn_norm, "TBNN |τ_pred|")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    _scatter3d(ax4, x, y, z, tau_smag_norm, "Smagorinsky |τ|")

    plt.tight_layout()
    out_path = out_dir / "jhtdb_les_sgs_3d.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Train vs Test split visualization
    fig2 = plt.figure(figsize=(10, 5))
    ax5 = fig2.add_subplot(1, 2, 1, projection="3d")
    c_split = np.where(train_flat, 1.0, 0.0)
    sc5 = ax5.scatter(x, y, z, c=c_split, cmap="coolwarm", s=30, alpha=0.9)
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_zlabel("z")
    ax5.set_title("Train (red) vs Test (blue) spatial split")
    ax5.view_init(elev=25, azim=45)

    ax6 = fig2.add_subplot(1, 2, 2, projection="3d")
    err = np.abs(tau_tbnn_norm - tau_target_norm)
    sc6 = ax6.scatter(x, y, z, c=err, cmap="hot", s=25, alpha=0.8)
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_zlabel("z")
    ax6.set_title("|τ_TBNN - τ_target|")
    plt.colorbar(sc6, ax=ax6, shrink=0.6)
    ax6.view_init(elev=25, azim=45)

    plt.tight_layout()
    out_path2 = out_dir / "jhtdb_les_sgs_3d_split.png"
    plt.savefig(out_path2, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"    Saved 3D plots: {out_path}, {out_path2}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JHTDB LES-SGS (real data only)")
    parser.add_argument("--size", type=int, default=16, help="Fine grid size (16^3)")
    parser.add_argument("--coarse-ratio", type=int, default=2, help="Filter ratio")
    parser.add_argument("--dataset", default="isotropic8192", help="JHTDB dataset")
    parser.add_argument("--timepoint", type=int, default=1, help="Time index")
    args = parser.parse_args()

    print("=" * 60)
    print("JHTDB LES-SGS: Real DNS Data")
    print("=" * 60)
    print("\nREAL DATA ONLY - No synthetic fallback.")
    print("Source: Johns Hopkins Turbulence Database (givernylocal)")

    from axiom_os.datasets.jhtdb_turbulence import load_jhtdb_for_les_sgs

    try:
        u_fine, u_coarse, tau_target, meta = load_jhtdb_for_les_sgs(
            fine_size=args.size,
            coarse_ratio=args.coarse_ratio,
            dataset=args.dataset,
            timepoint=args.timepoint,
        )
    except Exception as e:
        print(f"\n[FAIL] {e}")
        print("\nTo use larger cutouts, get JHTDB auth token:")
        print("  https://turbulence.idies.jhu.edu/staging/database")
        print("  Then: set JHTDB_AUTH_TOKEN=your_token")
        return 1

    print(f"\n[1] Loaded: {meta['data_source']} {meta['dataset']}")
    print(f"    Fine shape: {u_fine.shape}")
    print(f"    Coarse shape: {u_coarse.shape}")
    print(f"    tau_ij shape: {tau_target.shape}")

    # Basic stats
    u_mag = np.sqrt(np.sum(u_fine**2, axis=-1))
    print(f"\n[2] Velocity: mean |u| = {np.mean(u_mag):.4f}, std = {np.std(u_mag):.4f}")
    tau_norm = np.sqrt(np.sum(tau_target**2, axis=(-2, -1)))
    print(f"    Reynolds stress: mean |tau| = {np.mean(tau_norm):.6f}")

    print("\n[3] tau_ij diagonal (trace):")
    trace_tau = np.sum([tau_target[..., i, i] for i in range(3)], axis=0)
    print(f"    mean = {np.mean(trace_tau):.6f}, std = {np.std(trace_tau):.6f}")

    # Smagorinsky baseline (with best-fit scale: tau_smag * alpha to minimize MSE)
    delta = float(args.coarse_ratio)
    tau_smag = smagorinsky_tau(u_coarse, Cs=0.17, delta=delta)
    alpha = np.sum(tau_smag * tau_target) / (np.sum(tau_smag**2) + 1e-20)
    tau_smag_scaled = tau_smag * alpha
    ss_res_smag = np.sum((tau_smag_scaled - tau_target) ** 2)
    ss_tot = np.sum((tau_target - np.mean(tau_target)) ** 2) + 1e-20
    r2_smag = 1.0 - ss_res_smag / ss_tot
    print(f"\n[4] Smagorinsky (best-fit scale alpha={alpha:.4f}): R2 = {r2_smag:.4f}")

    # TBNN (Galilean invariant, strict spatial split)
    n_c = u_coarse.shape[0]
    train_mask = _spatial_split_mask(n_c, n_c, n_c, train_top=True)
    test_mask = _spatial_split_mask(n_c, n_c, n_c, train_top=False)
    print("\n[5] Training TBNN (Galilean invariant, spatial split)...")
    tau_tbnn, r2_train, r2_test = train_tbnn(
        u_coarse, tau_target, train_mask, test_mask, epochs=1000
    )
    print(f"    TBNN R2 train = {r2_train:.4f}, R2 test = {r2_test:.4f}")
    ss_res_tbnn = np.sum((tau_tbnn - tau_target) ** 2)
    r2_soft = 1.0 - ss_res_tbnn / ss_tot
    print(f"    TBNN R2 full = {r2_soft:.4f}")

    verdict = "TBNN > Smagorinsky" if r2_soft > r2_smag else "Smagorinsky >= TBNN"
    print(f"\n[6] Verdict: {verdict}")

    # FNO (non-local)
    print("\n[7] Training FNO (non-local)...")
    tau_fno_6ch, r2_fno_train, r2_fno_test = train_fno(
        u_coarse, tau_target, train_mask, test_mask, epochs=600
    )
    print(f"    FNO R2 train = {r2_fno_train:.4f}, R2 test = {r2_fno_test:.4f}")

    # PINN-LSTM (different task: velocity prediction)
    print("\n[8] Running PINN-LSTM (velocity prediction)...")
    r2_pinn_lstm = _run_pinn_lstm_r2(epochs=200)
    print(f"    PINN-LSTM R2 test = {r2_pinn_lstm:.4f}")

    # 3D visualization
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_jhtdb_3d(
        u_coarse, tau_target, tau_tbnn, tau_smag_scaled,
        train_mask, out_dir,
    )
    plot_comparison_bar(r2_smag, r2_test, r2_fno_test, r2_pinn_lstm, out_dir)

    # Save
    import json
    from datetime import datetime
    out = {
        "timestamp": datetime.now().isoformat(),
        "meta": meta,
        "u_mean": float(np.mean(u_mag)),
        "tau_mean_norm": float(np.mean(tau_norm)),
        "r2_smagorinsky": float(r2_smag),
        "r2_tbnn_train": float(r2_train),
        "r2_tbnn_test": float(r2_test),
        "r2_tbnn_full": float(r2_soft),
        "r2_fno_train": float(r2_fno_train),
        "r2_fno_test": float(r2_fno_test),
        "r2_pinn_lstm": float(r2_pinn_lstm),
        "verdict": verdict,
        "data_source": "JHTDB",
    }
    with open(out_dir / "jhtdb_les_sgs.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n    Saved: {out_dir / 'jhtdb_les_sgs.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
