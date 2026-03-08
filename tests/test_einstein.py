"""
Unit tests for Einstein Core (Symplectic & Metriplectic Reasoner).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.core.einstein import (
    EinsteinCore,
    SymplecticIntegrator,
    MetriplecticStructure,
    metriplectic_from_H,
)


def harmonic_H(qp):
    """Hamiltonian for 1D harmonic oscillator: H = 0.5*p^2 + 0.5*k*q^2"""
    qp = np.asarray(qp).ravel()
    k = 1.0
    q, p = qp[0], qp[1]
    return 0.5 * p * p + 0.5 * k * q * q


def test_step_leapfrog():
    """Test leapfrog integration preserves structure."""
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    dt = 0.01
    n_steps = 1000

    q, p = q0.copy(), p0.copy()
    for _ in range(n_steps):
        q, p = EinsteinCore.step_leapfrog(harmonic_H, q, p, dt, state_dim=1)

    H_final = harmonic_H(np.concatenate([q, p]))
    H_initial = harmonic_H(np.concatenate([q0, p0]))
    # Energy should be conserved (finite-diff grad adds small drift)
    assert abs(H_final - H_initial) < 1e-4


def test_step_leapfrog_multi_step():
    """Test multi-step leapfrog produces periodic motion."""
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    dt = 0.02
    # One period for k=1 is T = 2*pi
    n_steps = int(2 * np.pi / dt)

    q, p = q0.copy(), p0.copy()
    for _ in range(n_steps):
        q, p = EinsteinCore.step_leapfrog(harmonic_H, q, p, dt, state_dim=1)

    # After one period, should return close to start
    assert abs(q[0] - q0[0]) < 0.1
    assert abs(p[0] - p0[0]) < 0.1


def test_decompose_dynamics_placeholder():
    """Test decompose_dynamics returns placeholder."""
    einstein = EinsteinCore(state_dim=2)

    def dummy_model(x):
        return np.array([x[1], -x[0]])  # Simple oscillator

    H_func, diss_func = einstein.decompose_dynamics(dummy_model)
    assert H_func is None
    assert diss_func is None


def test_symplectic_integrator_compat():
    """Test SymplecticIntegrator still works (backward compat)."""
    def grad_H(qp):
        qp = np.asarray(qp).ravel()
        q, p = qp[0], qp[1]
        dH_dq = q  # dH/dq = k*q
        dH_dp = p  # dH/dp = p
        return np.array([dH_dq]), np.array([dH_dp])

    integrator = SymplecticIntegrator(dt=0.01)
    q, p = np.array([1.0]), np.array([0.0])
    q_new, p_new = integrator.step(q, p, harmonic_H, grad_H)
    assert q_new.shape == (1,)
    assert p_new.shape == (1,)


def test_metriplectic_from_H():
    """Test metriplectic_from_H builds structure from Hamiltonian."""
    struct = metriplectic_from_H(harmonic_H, state_dim=1)
    x = np.array([1.0, 0.0])
    dx = struct.dynamics(x)
    assert dx.shape == (2,)
    # Hamiltonian case: L=J, M=0, so dx = J @ grad_E = (dE/dp, -dE/dq)
    E_val = struct.E_func(x)
    assert abs(E_val - 0.5) < 1e-5


def test_metriplectic_degeneracy():
    """Test degeneracy L∇S=0 and M∇E=0."""
    def E(x):
        return 0.5 * (x[0]**2 + x[1]**2)
    def S(x):
        return 0.0  # constant

    def L_raw(x):
        return np.array([[0, 1], [-1, 0]])  # symplectic J

    def M_raw(x):
        return np.array([[0.1, 0], [0, 0.1]])  # small dissipation

    struct = MetriplecticStructure(E, S, L_raw, M_raw, dim=2)
    x = np.array([1.0, 0.5])
    LgS, MgE = struct.degeneracy_violation(x)
    # S=const => grad_S=0 => L∇S=0
    assert LgS < 1e-10
    # M projects orthogonally to ∇E => M∇E=0
    assert MgE < 1e-10


def test_step_metriplectic():
    """Test step_metriplectic integration."""
    struct = metriplectic_from_H(harmonic_H, state_dim=1)
    x = np.array([1.0, 0.0])
    x_new = EinsteinCore.step_metriplectic(struct, x, dt=0.01)
    assert x_new.shape == (2,)
    # Energy should be approximately conserved (Euler step has drift)
    E_old = struct.E_func(x)
    E_new = struct.E_func(x_new)
    assert abs(E_new - E_old) < 0.1


if __name__ == "__main__":
    test_step_leapfrog()
    test_step_leapfrog_multi_step()
    test_decompose_dynamics_placeholder()
    test_symplectic_integrator_compat()
    test_metriplectic_from_H()
    test_metriplectic_degeneracy()
    test_step_metriplectic()
    print("All Einstein tests passed.")
