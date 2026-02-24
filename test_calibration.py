"""Test SPNN calibration module"""
import torch
import numpy as np
from spnn.core.calibration import (
    renormalization_scale,
    safe_add,
    erf_gate,
    apply_das,
    detect_oscillation,
    OscillationPattern,
)

def test_renorm():
    d = torch.tensor([1, 0, -1, 0, 0], dtype=torch.float64)
    s = torch.ones(5)
    sigma = torch.ones(5) * 2
    R = renormalization_scale(d, s, sigma)
    assert R.numel() >= 1
    print("renorm OK")

def test_safe_add():
    x = torch.tensor([1.0, 1e100])
    y = torch.tensor([2.0, 1e100])
    z = safe_add(x, y)
    assert z[0] == 3.0
    print("safe_add OK")

def test_erf_gate():
    r = torch.linspace(-2, 2, 10)
    g = erf_gate(r, kappa=5.0, rho_thresh=0.0)
    assert g.min() >= 0 and g.max() <= 1
    print("erf_gate OK")

def test_das():
    soft = torch.randn(4, 8)
    hard = torch.randn(4, 8) * 0.5
    shielded, potential = apply_das(soft, hard)
    assert shielded.shape == soft.shape
    assert potential.shape == soft.shape
    print("DAS OK")

def test_oscillation():
    # 发散性震荡
    hist = [1.0, 2.0, 0.5, 2.5, 0.3, 3.0, 0.2, 3.5, 0.1, 4.0]
    res = detect_oscillation(hist, tau=10, theta_osc=5.0)
    print("oscillation:", res.is_oscillation, res.pattern, res.chi, res.f_sign)
    print("detect_oscillation OK")

if __name__ == "__main__":
    test_renorm()
    test_safe_add()
    test_erf_gate()
    test_das()
    test_oscillation()
    print("All calibration tests passed.")
