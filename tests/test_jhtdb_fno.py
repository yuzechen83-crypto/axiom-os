"""
JHTDB LES-SGS FNO Test - Non-local Turbulence Modeling

FNO maps full velocity field u(·) -> stress field τ(·).
Input: (B, 3, H, W, D) velocity
Target: (B, 6, H, W, D) symmetric stress [τ11, τ22, τ33, τ12, τ13, τ23]

Spatial split: train z>nz/2, test z<=nz/2.

Run: python -m pytest tests/test_jhtdb_fno.py -v -s
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tau_3x3_to_6ch(tau: np.ndarray) -> np.ndarray:
    """Convert (...,3,3) symmetric tensor to (...,6): [τ00, τ11, τ22, τ01, τ02, τ12]."""
    return np.stack([
        tau[..., 0, 0], tau[..., 1, 1], tau[..., 2, 2],
        tau[..., 0, 1], tau[..., 0, 2], tau[..., 1, 2],
    ], axis=-1)


def _spatial_split_mask(nx: int, ny: int, nz: int, train_top: bool) -> np.ndarray:
    """train_top: z > nz/2; else z <= nz/2."""
    z_idx = np.arange(nz)
    mask = (z_idx > nz // 2) if train_top else (z_idx <= nz // 2)
    return np.broadcast_to(mask, (nx, ny, nz))


def test_fno3d_forward():
    """Test FNO3d forward pass."""
    import torch
    from axiom_os.layers.fno import FNO3d

    torch.manual_seed(42)
    model = FNO3d(in_channels=3, out_channels=6, width=16, modes1=2, modes2=2, modes3=2, n_layers=2)
    x = torch.randn(2, 3, 8, 8, 8) * 0.1
    out = model(x)
    assert out.shape == (2, 6, 8, 8, 8)
    assert torch.isfinite(out).all()


def test_jhtdb_fno_spatial_split():
    """
    Train FNO on JHTDB LES-SGS with spatial split.
    Expectation: FNO R²_test should surpass TBNN (potentially > 0.6) due to non-local context.
    """
    from axiom_os.datasets.jhtdb_turbulence import load_jhtdb_for_les_sgs
    from axiom_os.layers.fno import FNO3d
    import torch

    u_fine, u_coarse, tau_target, meta = load_jhtdb_for_les_sgs(
        fine_size=16, coarse_ratio=2, dataset="isotropic8192", timepoint=1
    )
    n_c = u_coarse.shape[0]
    train_mask = _spatial_split_mask(n_c, n_c, n_c, train_top=True)
    test_mask = _spatial_split_mask(n_c, n_c, n_c, train_top=False)

    # Input: (1, 3, H, W, D), Target: (1, 6, H, W, D)
    u_ch = np.transpose(u_coarse, (3, 0, 1, 2))  # (3,8,8,8)
    u_batch = torch.from_numpy(u_ch[np.newaxis].astype(np.float32))
    tau_6ch = _tau_3x3_to_6ch(tau_target)
    tau_ch = np.transpose(tau_6ch, (3, 0, 1, 2))  # (6,8,8,8)
    tau_batch = torch.from_numpy(tau_ch[np.newaxis].astype(np.float32))

    train_flat = train_mask.ravel()
    test_flat = test_mask.ravel()

    model = FNO3d(in_channels=3, out_channels=6, width=24, modes1=3, modes2=3, modes3=3, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=50, min_lr=1e-5)

    epochs = 600
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
    pred_test = pred_full[test_flat]
    tau_test = tau_flat_np[test_flat]

    ss_res = np.sum((pred_test - tau_test) ** 2)
    ss_tot = np.sum((tau_test - np.mean(tau_test)) ** 2) + 1e-20
    r2_test = 1.0 - ss_res / ss_tot

    n_train = int(np.sum(train_flat))
    n_test = int(np.sum(test_flat))
    assert n_train >= 100
    assert n_test >= 100
    print(f"\nFNO strict eval: train n={n_train}, test n={n_test}, R2_test={r2_test:.4f}")
    assert r2_test > -1.0, "FNO should not be catastrophically worse than mean"
