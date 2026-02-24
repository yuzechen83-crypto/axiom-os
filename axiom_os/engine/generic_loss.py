"""
GENERIC Loss - Physics-Consistent Training
Enforces degeneracy conditions: L·∇S = 0, M·∇E = 0
"""

import torch


def generic_loss(
    pred_dz: torch.Tensor,
    true_dz: torch.Tensor,
    L: torch.Tensor,
    M: torch.Tensor,
    grad_E: torch.Tensor,
    grad_S: torch.Tensor,
    lambda_phys: float = 1.0,
) -> torch.Tensor:
    """
    GENERIC-consistent loss: data fitting + degeneracy constraints.

    Args:
        pred_dz: Predicted dz/dt from model
        true_dz: Ground truth dz/dt
        L: Poisson matrix (anti-symmetric)
        M: Friction matrix (PSD)
        grad_E: ∇E
        grad_S: ∇S
        lambda_phys: Weight for physics constraints

    Returns:
        loss = MSE(pred, true) + lambda_phys * (||L·∇S||² + ||M·∇E||²)
    """
    # 1. Data fitting
    mse = torch.mean((pred_dz - true_dz) ** 2)

    # 2. Degeneracy: L·∇S = 0
    # L @ grad_S: (dim, dim) @ (B, dim)^T -> (B, dim)
    L_grad_S = torch.einsum("ij,bj->bi", L, grad_S)
    deg_1 = torch.mean(L_grad_S ** 2)

    # 3. Degeneracy: M·∇E = 0
    M_grad_E = torch.einsum("ij,bj->bi", M, grad_E)
    deg_2 = torch.mean(M_grad_E ** 2)

    return mse + lambda_phys * (deg_1 + deg_2)


def generic_loss_from_aux(
    pred_dz: torch.Tensor,
    true_dz: torch.Tensor,
    aux: dict,
    lambda_phys: float = 1.0,
) -> torch.Tensor:
    """
    Convenience: compute generic_loss from aux dict returned by GENERICSystem.dynamics(return_aux=True).
    """
    return generic_loss(
        pred_dz=pred_dz,
        true_dz=true_dz,
        L=aux["L"],
        M=aux["M"],
        grad_E=aux["grad_E"],
        grad_S=aux["grad_S"],
        lambda_phys=lambda_phys,
    )
