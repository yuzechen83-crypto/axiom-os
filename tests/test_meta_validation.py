"""
Unit tests for Meta-Validation: SPARC, MetaProjectionLayer, verify_mashd.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_sparc_load_galaxy():
    """Load mock SPARC galaxy."""
    from axiom_os.datasets.sparc import load_sparc_galaxy

    d = load_sparc_galaxy("NGC_6503", use_mock_if_fail=True)
    assert "R" in d and "V_obs" in d and "V_gas" in d and "V_disk" in d and "V_bulge" in d
    assert len(d["R"]) > 5
    assert d["R"].min() >= 0


def test_sparc_V_def_sq():
    """V_def^2 = V_obs^2 - (V_gas^2 + V_disk^2 + V_bulge^2)."""
    from axiom_os.datasets.sparc import load_sparc_galaxy

    d = load_sparc_galaxy("NGC_3198")
    V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
    V_obs_sq = d["V_obs"]**2
    V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0)
    assert np.all(V_def_sq >= 0)


def test_meta_projection_layer():
    """MetaProjectionLayer forward."""
    from axiom_os.layers.meta_kernel import MetaProjectionLayer

    layer = MetaProjectionLayer(n_z_points=21, L_init=10.2, k0_init=1.0)
    rho = torch.rand(10) * 100
    out = layer(rho)
    assert out.shape == (10,)
    assert torch.all(out >= 0)


def test_meta_projection_model():
    """MetaProjectionModel forward."""
    from axiom_os.layers.meta_kernel import MetaProjectionModel

    model = MetaProjectionModel(L_init=10.2)
    r = torch.rand(20) * 20 + 0.5
    V_bary_sq = torch.rand(20) * 5000 + 100
    out = model(r, V_bary_sq)
    assert out.shape == (20,)


def test_discovery_check():
    """discovery_check_kernel_shape returns dict."""
    from axiom_os.engine.discovery_check import discovery_check_kernel_shape
    from axiom_os.datasets.sparc import load_sparc_galaxy

    d = load_sparc_galaxy("NGC_6503")
    R = d["R"]
    V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
    V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1)
    V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0)
    result = discovery_check_kernel_shape(R, V_def_sq, V_bary_sq)
    assert "formula" in result and "r2" in result and "match_score" in result
