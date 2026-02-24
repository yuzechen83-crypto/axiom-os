"""
Axiom-OS v4.0: The Holographic Generator - Unit Tests
Flow Matching, Tensor Network (MERA), Topological Early Warning.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_flow_matching_loss():
    """MetaFlowMatching: CFM loss and sampling."""
    from axiom_os.engine.flow_matching import MetaFlowMatching, conditional_flow_matching_loss

    model = MetaFlowMatching(state_dim=4, hidden_dim=32, n_layers=3)
    x_simple = torch.randn(16, 4) * 0.3
    x_complex = torch.randn(16, 4)
    loss = model.loss(x_simple, x_complex)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_flow_matching_sample():
    """MetaFlowMatching: ODE sampling from z=1 to z=0."""
    from axiom_os.engine.flow_matching import MetaFlowMatching

    model = MetaFlowMatching(state_dim=4, hidden_dim=32)
    samples = model.sample(batch_size=8, n_steps=5)
    assert samples.shape == (8, 4)


def test_tensor_net_forward():
    """HolographicTensorNet: forward pass."""
    from axiom_os.layers.tensor_net import HolographicTensorNet

    tn = HolographicTensorNet(input_dim=4, output_dim=2, chi=4, n_layers=2)
    x = torch.randn(10, 4)
    y = tn(x)
    assert y.shape == (10, 2)


def test_mera_soft_shell():
    """MERASoftShell: RCLN-compatible soft shell."""
    from axiom_os.layers.tensor_net import MERASoftShell

    mera = MERASoftShell(input_dim=4, hidden_dim=16, output_dim=2)
    x = torch.randn(10, 4)
    y = mera(x)
    assert y.shape == (10, 2)


def test_rcln_mera():
    """RCLN with net_type='mera'."""
    from axiom_os.layers.rcln import RCLNLayer

    rcln = RCLNLayer(input_dim=4, hidden_dim=32, output_dim=2, net_type="mera")
    x = torch.randn(10, 4)
    y = rcln(x)
    assert y.shape == (10, 2)


def test_topology_compute_persistence():
    """compute_persistence: returns dict with betti0, betti1."""
    from axiom_os.engine.topology import compute_persistence

    # 2D activation map
    act = np.random.randn(8, 8).astype(np.float64)
    pers = compute_persistence(act)
    assert "betti0" in pers
    assert "betti1" in pers
    assert "n_components" in pers
    assert "n_loops" in pers


def test_topological_early_warning():
    """TopologicalEarlyWarning: update and check_alert."""
    from axiom_os.engine.topology import TopologicalEarlyWarning

    tew = TopologicalEarlyWarning(threshold_derivative=5.0, window_size=3)
    for z in [1.0, 0.8, 0.6, 0.4, 0.2]:
        act = np.random.randn(6, 6).astype(np.float64)
        tew.update(z, act)
    alert, msg = tew.check_alert()
    assert isinstance(alert, bool)
    assert isinstance(msg, str)


def test_flow_matching_vector_field():
    """VectorFieldNet: v_z(x) output shape."""
    from axiom_os.engine.flow_matching import VectorFieldNet

    vnet = VectorFieldNet(state_dim=6, hidden_dim=32)
    x = torch.randn(5, 6)
    z = torch.rand(5)
    v = vnet(x, z)
    assert v.shape == (5, 6)
