"""
SPNN-Opt-Rev5 数学体系补充与校准
- 量纲重正化算子 R(d)
- 安全运算 ⊕_ε
- 高阶可微门控 (erf / GELU-like)
- 震荡检测 (自相关 + 符号切换)
- 物理感知梯度裁剪
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum

# Θ_safe = 浮点数最大值的 1%
THETA_SAFE = float(np.finfo(np.float64).max) * 0.01
EPS = 1e-8


# ============ 一、量纲重正化算子 R(d) ============

def renormalization_scale(
    d: Union[torch.Tensor, np.ndarray],
    s: Union[torch.Tensor, np.ndarray],
    sigma: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """
    R(d) = ∏_i (s_i^d_i · exp(-d_i²/(2σ_i²)))
    d: 量纲幂次向量 [d1,d2,d3,d4,d5]
    s: 特征尺度 [s1,...,s5]
    sigma: 标准差 [σ1,...,σ5]
    """
    d = torch.as_tensor(d, dtype=torch.float64)
    s = torch.as_tensor(s, dtype=torch.float64)
    sigma = torch.as_tensor(sigma, dtype=torch.float64)
    if d.dim() == 1:
        d = d.unsqueeze(0)
    term = (s ** d) * torch.exp(-(d ** 2) / (2 * sigma ** 2 + EPS))
    return term.prod(dim=-1)


def normalize_to_cognitive_manifold(
    v: torch.Tensor,
    d: torch.Tensor,
    s: torch.Tensor,
    sigma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    v_norm = v / R(d),  d_norm = 0
    """
    R = renormalization_scale(d, s, sigma)
    R = R.view(-1, *([1] * (v.dim() - 1)))
    return v / (R + EPS), torch.zeros_like(d)


# ============ 二、安全运算 ============

def safe_add(x: torch.Tensor, y: torch.Tensor, theta_safe: float = THETA_SAFE) -> torch.Tensor:
    """
    x ⊕_ε y = x+y if |x|,|y| < Θ_safe else logsumexp(x,y)
    """
    mask = (x.abs() < theta_safe) & (y.abs() < theta_safe)
    normal = x + y
    large = torch.logsumexp(torch.stack([x, y], dim=-1), dim=-1)
    if large.dim() < normal.dim():
        large = large.expand_as(normal)
    return torch.where(mask, normal, large)


def safe_subtract(x: torch.Tensor, y: torch.Tensor, theta_safe: float = THETA_SAFE) -> torch.Tensor:
    return safe_add(x, -y, theta_safe)


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    n = torch.norm(x, dim=dim, keepdim=keepdim)
    return torch.clamp(n, min=EPS)


# ============ 三、高阶可微门控 ============

def erf_gate(rho_w: torch.Tensor, kappa: float = 10.0, rho_thresh: float = 0.5) -> torch.Tensor:
    """
    g(ρ_w) = 1/2 + 1/2·erf(κ·(ρ_w - ρ_thresh)/√2)
    g ∈ C^∞, lim g=0 (ρ→-∞), lim g=1 (ρ→∞)
    """
    return 0.5 + 0.5 * torch.erf(kappa * (rho_w - rho_thresh) / (2 ** 0.5))


def gelu_like_gate(
    rho_w: torch.Tensor,
    kappa: float = 10.0,
    rho_thresh: float = 0.5,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    g = σ(κ·(ρ-ρ_thresh)) + α·ρ·σ'(κ·(ρ-ρ_thresh))
    """
    u = kappa * (rho_w - rho_thresh)
    sig = torch.sigmoid(u)
    sig_prime = sig * (1 - sig)
    return sig + alpha * rho_w * sig_prime


# ============ 四、DAS (Differentiable Axiom Shield) ============

class DifferentiableAxiomShield(torch.autograd.Function):
    """
    可微公理势阱
    shielded = (1-g)*soft + g*hard
    potential = 0.5 * λ * ρ_w² * g.detach()
    """

    @staticmethod
    def forward(ctx, soft_out, hard_out, rho_w, gate, lambda_coeff):
        ctx.save_for_backward(soft_out, hard_out, gate, rho_w)
        ctx.lambda_coeff = lambda_coeff
        shielded = (1 - gate) * soft_out + gate * hard_out
        rho_sq = (rho_w ** 2).flatten(1).sum(dim=1, keepdim=True)
        pot = 0.5 * lambda_coeff * rho_sq * gate.flatten(1).mean(dim=1, keepdim=True).detach()
        potential = pot.expand_as(soft_out)
        return shielded, potential

    @staticmethod
    def backward(ctx, grad_shielded, grad_potential):
        soft_out, hard_out, gate, rho_w = ctx.saved_tensors
        lambda_coeff = ctx.lambda_coeff
        gs = gate
        dg = grad_shielded
        grad_soft = dg * (1 - gs)
        grad_hard = dg * gs
        if grad_potential is not None and grad_potential.numel() > 0:
            gp = grad_potential * lambda_coeff * rho_w * gs
            grad_soft = grad_soft + gp
            grad_hard = grad_hard - gp
        return grad_soft, grad_hard, None, None, None


def apply_das(
    soft_out: torch.Tensor,
    hard_out: torch.Tensor,
    kappa: float = 10.0,
    rho_thresh: float = 0.5,
    lambda_coeff: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply DAS with erf gate"""
    residual = safe_subtract(soft_out, hard_out)
    rho_w = safe_norm(residual, keepdim=True)
    if rho_w.dim() < soft_out.dim():
        rho_w = rho_w.expand_as(soft_out)
    gate = erf_gate(rho_w, kappa, rho_thresh)
    if gate.dim() < soft_out.dim():
        gate = gate.expand_as(soft_out)
    return DifferentiableAxiomShield.apply(soft_out, hard_out, rho_w, gate, lambda_coeff)


# ============ 五、震荡检测 ============

class OscillationPattern(Enum):
    EXPLORATORY = 1  # 探索性 (有益)
    CONVERGING = 2   # 收敛性 (正常)
    DIVERGING = 3    # 发散性 (危险)


@dataclass
class OscillationResult:
    is_oscillation: bool
    pattern: Optional[OscillationPattern]
    chi: float
    f_sign: float
    ratio: float


def autocorrelation(loss_history: List[float], tau: int) -> float:
    """χ(ℓ,τ) = (1/(τ-1)) Σ Corr(ℓ(t-k), ℓ(t-k-1))"""
    if len(loss_history) < tau or tau < 2:
        return 0.0
    recent = np.array(loss_history[-tau:])
    corrs = []
    for k in range(1, tau - 1):
        a, b = recent[-(k+1):-1], recent[-(k+2):-2]
        if len(a) > 0 and len(b) > 0 and a.std() > 1e-12 and b.std() > 1e-12:
            corrs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(corrs)) if corrs else 0.0


def sign_switch_frequency(loss_history: List[float], tau: int) -> float:
    """f_sign = (1/(τ-1)) Σ I(sign(Δℓ(t-k)) ≠ sign(Δℓ(t-k-1)))"""
    if len(loss_history) < tau or tau < 2:
        return 0.0
    recent = np.array(loss_history[-tau:])
    deltas = np.diff(recent)
    switches = np.sum(np.diff(np.sign(deltas)) != 0)
    return switches / max(1, len(deltas) - 1)


def detect_oscillation(
    loss_history: List[float],
    tau: int = 10,
    theta_osc: float = 10.0,
    chi_thresh: float = 0.0,
) -> OscillationResult:
    """
    TrueOscillation ⟺ (ratio > Θ_osc) ∧ (χ < χ_thresh) ∧ (f_sign > 0.3)
    Pattern: 1=探索性, 2=收敛性, 3=发散性
    """
    if len(loss_history) < tau:
        return OscillationResult(False, None, 0.0, 0.0, 1.0)
    recent = loss_history[-tau:]
    ratio = max(recent) / (min(recent) + EPS)
    chi = autocorrelation(loss_history, tau)
    f_sign = sign_switch_frequency(loss_history, tau)

    is_osc = ratio > theta_osc and chi < chi_thresh and f_sign > 0.3

    pattern = None
    if is_osc:
        trough = min(recent)
        peak = max(recent)
        if -0.5 < chi < 0 and 0.3 < f_sign < 0.6 and trough < peak * 0.8:
            pattern = OscillationPattern.EXPLORATORY
        elif chi > 0 and f_sign < 0.2 and trough >= peak * 0.95:
            pattern = OscillationPattern.CONVERGING
        elif chi < -0.7 and f_sign > 0.8 and trough > peak * 1.1:
            pattern = OscillationPattern.DIVERGING

    return OscillationResult(is_osc, pattern, chi, f_sign, ratio)


# ============ 六、物理感知梯度裁剪 ============

def physical_significance(g: torch.Tensor, units: Optional[torch.Tensor] = None) -> torch.Tensor:
    """PhysicalSignificance(g): 基于量纲的梯度重要性"""
    return torch.ones_like(g)
    # 可扩展: 根据 units 加权


def physical_aware_clip_grad(
    grad: torch.Tensor,
    G_max: float,
    physical_factor: float = 1.0,
) -> torch.Tensor:
    """
    g_clipped = g · min(1, G_max/||g|| · PhysicalSignificance(g) / PhysicalSignificance_max)
    """
    norm = grad.norm() + EPS
    sig = physical_significance(grad)
    sig_max = sig.max() + EPS
    scale = min(1.0, G_max / norm * (sig / sig_max).mean() * physical_factor)
    return grad * scale
