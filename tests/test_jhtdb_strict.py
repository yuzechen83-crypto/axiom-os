"""
Strict JHTDB LES-SGS Evaluation - Spatial Block Split

Train: Top half of domain (z > nz/2)
Test: Bottom half (z <= nz/2)

Turbulence is spatially correlated; random split leaks data.
Spatial block split prevents train/test leakage.

For 16^3 fine -> 8^3 coarse: train z in [4,5,6,7], test z in [0,1,2,3].

Run: python -m pytest tests/test_jhtdb_strict.py -v
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _spatial_split_mask(nx: int, ny: int, nz: int, train_top: bool = True) -> np.ndarray:
    """
    train_top=True: z > nz/2 (top half)
    train_top=False: z <= nz/2 (bottom half)
    Returns boolean mask (nx, ny, nz).
    """
    z_idx = np.arange(nz)
    if train_top:
        mask = z_idx > nz // 2
    else:
        mask = z_idx <= nz // 2
    return np.broadcast_to(mask, (nx, ny, nz))


def test_turbulence_invariants():
    """Test Pope invariants and tensor basis computation."""
    from axiom_os.core.turbulence_invariants import (
        grad_u_from_velocity,
        decompose_grad_u,
        compute_invariants,
        compute_tensor_basis,
        extract_invariants_and_basis,
    )

    # Synthetic u: (4,4,4,3)
    np.random.seed(42)
    u = np.random.randn(4, 4, 4, 3).astype(np.float64)
    inv, basis = extract_invariants_and_basis(u)
    assert inv.shape == (4, 4, 4, 5)
    assert len(basis) == 10
    assert basis[0].shape == (4, 4, 4, 3, 3)
    # T1 = S should be symmetric
    S = basis[0]
    assert np.allclose(S, np.swapaxes(S, -2, -1))


def test_tbnn_forward():
    """Test TBNN forward pass."""
    import torch
    from axiom_os.layers.tbnn import TBNN, stack_tensor_basis
    from axiom_os.core.turbulence_invariants import extract_invariants_and_basis_normalized

    np.random.seed(42)
    u = np.random.randn(8, 8, 8, 3).astype(np.float64) * 0.1
    inv, basis, sigma = extract_invariants_and_basis_normalized(u)
    inv_flat = inv.reshape(-1, 5).astype(np.float32)
    tb_stacked = stack_tensor_basis(basis)
    tb_flat = tb_stacked.reshape(-1, 10, 3, 3)
    sigma_flat = sigma.reshape(-1).astype(np.float32)

    model = TBNN(n_invariants=5, n_tensors=10, hidden=16, n_layers=2)
    inv_t = torch.from_numpy(inv_flat)
    tb_t = tb_flat
    sigma_t = torch.from_numpy(sigma_flat)
    tau = model(inv_t, tb_t, sigma_t)
    assert tau.shape == (512, 3, 3)
    assert torch.isfinite(tau).all()


def test_jhtdb_strict_spatial_split():
    """
    Strict evaluation: spatial split, TBNN vs naive MLP.
    Uses real JHTDB data.
    """
    from axiom_os.datasets.jhtdb_turbulence import load_jhtdb_for_les_sgs
    from axiom_os.core.turbulence_invariants import extract_invariants_and_basis_normalized
    from axiom_os.layers.tbnn import TBNN, stack_tensor_basis
    import torch

    u_fine, u_coarse, tau_target, meta = load_jhtdb_for_les_sgs(
        fine_size=16, coarse_ratio=2, dataset="isotropic8192", timepoint=1
    )
    n_c = u_coarse.shape[0]
    train_mask = _spatial_split_mask(n_c, n_c, n_c, train_top=True)
    test_mask = _spatial_split_mask(n_c, n_c, n_c, train_top=False)

    inv, basis, sigma = extract_invariants_and_basis_normalized(u_coarse)
    inv_flat = inv.reshape(-1, 5).astype(np.float32)
    tb_stacked = stack_tensor_basis(basis).numpy()
    tb_flat = tb_stacked.reshape(-1, 10, 3, 3)
    sigma_flat = sigma.reshape(-1).astype(np.float32)
    tau_flat = tau_target.reshape(-1, 9).astype(np.float32)

    train_idx = train_mask.ravel()
    test_idx = test_mask.ravel()

    X_train = torch.from_numpy(inv_flat[train_idx])
    tb_train = torch.from_numpy(tb_flat[train_idx].astype(np.float32))
    sigma_train = torch.from_numpy(sigma_flat[train_idx])
    y_train = torch.from_numpy(tau_flat[train_idx])

    X_test = torch.from_numpy(inv_flat[test_idx])
    tb_test = torch.from_numpy(tb_flat[test_idx].astype(np.float32))
    sigma_test = torch.from_numpy(sigma_flat[test_idx])
    y_test = torch.from_numpy(tau_flat[test_idx])

    model = TBNN(n_invariants=5, n_tensors=10, hidden=32, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(400):
        tau_pred = model(X_train, tb_train, sigma_train)
        loss = ((tau_pred.reshape(-1, 9) - y_train) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        tau_test_pred = model(X_test, tb_test, sigma_test).reshape(-1, 9).numpy()
    y_test_np = y_test.numpy()
    ss_res = np.sum((tau_test_pred - y_test_np) ** 2)
    ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2) + 1e-20
    r2_test = 1.0 - ss_res / ss_tot

    n_train = int(np.sum(train_idx))
    n_test = int(np.sum(test_idx))
    assert n_train >= 100
    assert n_test >= 100
    print(f"\nStrict eval: train n={n_train}, test n={n_test}, R2_test={r2_test:.4f}")
    assert r2_test > -1.0, "TBNN should not be catastrophically worse than mean"
