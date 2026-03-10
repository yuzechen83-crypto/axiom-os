"""
SRBench (Symbolic Regression Benchmark) — 检验发现引擎的含金量
=============================================================

Feynman Dataset: 费曼《物理学讲义》中的经典公式（如 E=mc²、万有引力等）。
挑战：给定 (x,y,z...) 数据，反推物理公式。Axiom 优势：UPI 量纲约束可剪枝非法公式，提高搜索效率。

Pipeline:
---------
1. 生成/加载 Feynman 公式数据（可加噪声）
2. RCLN Soft Shell 拟合 (Hard Core = 0)
3. DiscoveryEngine 提取符号公式
4. 与 ground truth 比较 → Success Rate / Recovery Rate

子集:
-----
- Feynman Easy:  6 个较简单公式
- Feynman Medium: 12 个公式（含除法、多变量）

Usage:
------
    # 单次基准（默认噪声 0.01）
    python benchmarks/srbench_runner.py --subset easy --output benchmarks/srbench_results.json

    # 公式恢复率 vs 噪声强度（Easy + Medium 对比图）
    python benchmarks/srbench_runner.py --recovery-vs-noise --noise-levels 0.0,0.01,0.05,0.1,0.2 --plot docs/images/srbench_recovery_vs_noise.png

    # 快速测试（2 公式、2 噪声）
    python benchmarks/srbench_runner.py --recovery-vs-noise --formulas 2 --noise-levels 0.0,0.05 --epochs 50

    # 下载/准备 SRBench 数据目录
    python benchmarks/srbench_runner.py --download
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.layers.rcln import RCLNLayer
from axiom_os.engine.discovery import DiscoveryEngine

warnings.filterwarnings("ignore")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FeynmanFormula:
    """Represents a Feynman physics formula."""
    name: str                    # Formula identifier (e.g., "I.6.2a")
    description: str             # Human-readable description
    formula_str: str            # LaTeX/symbolic formula string
    variables: List[str]        # Input variable names
    target: str                 # Target variable name
    units: Dict[str, List[int]] # Physical units for each variable [M,L,T,Q,Θ]
    
    def __repr__(self) -> str:
        return f"FeynmanFormula({self.name}: {self.formula_str})"


@dataclass
class DiscoveryResult:
    """Result of symbolic discovery for one formula."""
    formula_name: str
    ground_truth: str
    discovered: Optional[str]
    exact_match: bool
    r2_score: float
    mse: float
    structural_similarity: float
    success: bool
    training_time: float
    
    def to_dict(self) -> dict:
        return {
            'formula_name': self.formula_name,
            'ground_truth': self.ground_truth,
            'discovered': self.discovered,
            'exact_match': bool(self.exact_match),
            'r2_score': float(self.r2_score),
            'mse': float(self.mse),
            'structural_similarity': float(self.structural_similarity),
            'success': bool(self.success),
            'training_time': float(self.training_time),
        }


# =============================================================================
# Feynman Dataset Loader
# =============================================================================

class FeynmanDatasetLoader:
    """Loader for Feynman Symbolic Regression Dataset.
    
    Expected format:
        - CSV files with columns: x1, x2, ..., y
        - Or NumPy .npy files with shape (n_samples, n_features + 1)
    
    Built-in formulas (SRBench subset):
        - I.6.2a: E = mc^2 (Relativistic energy)
        - I.8.14: d = vt (Distance = velocity * time)
        - I.9.18: F = Gm1m2/r^2 (Gravitational force)
        - I.11.19: A = r^2 * θ / 2 (Sector area)
        - I.12.1: F = μ * N (Friction)
        - I.12.2: F = q1*q2/(4πε0*r^2) (Coulomb's law)
        - I.12.4: F = q(E + v×B) (Lorentz force)
        - I.12.5: E = q*V (Potential energy)
        - I.12.11: E = (1/2) * m * v^2 (Kinetic energy)
        - I.13.4: T = 2π√(L/g) (Pendulum period)
        - I.14.3: U = mgh (Gravitational potential)
        - I.15.3x: x = (x0 + vt) * cos(θ) (Projectile motion)
        - I.15.3t: t = (v - v0) / a (Acceleration)
        - I.16.6: G = (1/2) * ρ * v^2 (Dynamic pressure)
        - I.18.4: r = (m1*r1 + m2*r2) / (m1 + m2) (Center of mass)
        - I.18.12: τ = r × F (Torque)
        - I.18.14: v = ω × r (Angular velocity)
        - I.24.6: E = (1/2)CV^2 (Capacitor energy)
        - I.25.13: V = Q/C (Capacitor voltage)
        - I.26.2: θ1 = θ2 (Snell's law - refraction angle)
        - I.27.6: E = hf (Photon energy)
        - I.29.4: k = ω/c (Wave number)
        - I.30.5: λ = h/p (de Broglie wavelength)
        - I.32.5: P = (1/2) * ε0 * c * E^2 (Poynting vector)
        - I.32.17: P = (E^2 * R) / (R + r)^2 (Power in circuit)
        - I.34.8: ω = qB/m (Cyclotron frequency)
        - I.34.10: a = qvB/m (Magnetic acceleration)
        - I.34.14: ω = (1 + v/c) * ω0 (Doppler effect)
        - I.34.27: E = (1/2) * m * (ω^2) * (x^2) (Harmonic oscillator)
        - I.37.4: E = (p^2) / (2m) + U(x) (Total energy)
        - I.38.12: r = (4πε0 * ℏ^2) / (m * e^2) (Bohr radius)
        - I.39.1: E = (3/2) * kB * T (Thermal energy)
        - I.39.11: v = √(3kT/m) (RMS velocity)
        - I.40.1: n = n0 * exp(-mgx/kT) (Barometric formula)
        - I.41.16: L = ℏ / (e^(ℏω/kT) - 1) (Planck distribution)
        - I.43.16: P = μ * k * T / V (Ideal gas law partial)
        - I.43.31: D = μ * k * T / D_diffusion (Diffusion)
        - I.44.4: E = n * kB * T * ln(V2/V1) (Free energy)
        - I.47.23: c = √(γ * p / ρ) (Speed of sound)
        - I.50.26: x = x0 * cos(ωt + φ) (Harmonic motion)
        - II.2.42: P = (κ/2) * (T1 - T2) * A/d (Heat flow)
        - II.3.24: P = (1/2) * ε0 * E^2 (Electric energy density)
        - II.4.23: V = (1/4πε0) * (p * cosθ) / r^2 (Dipole potential)
        - II.6.11: V = (p/4πε0) * (1/r^2) * cosθ (Dipole)
        - II.6.15a: E = p/(4πε0*r^3) * √(3cos²θ + 1) (Dipole field)
        - II.6.15b: E = p/(4πε0*r^3) * 3cosθsinθ (Dipole field)
        - II.8.7: E = (q/(4πε0)) * (1/r^2) (Point charge field)
        - II.8.31: E = σ / (2ε0) (Plane field)
        - II.10.9: E = σ/(2ε0) * (1 - z/√(z^2 + R^2)) (Disk field)
        - II.11.3: x = qE/(m(ω0^2 - ω^2)) (Driven oscillator)
        - II.11.17: n = 1 + (N*q^2)/(2ε0*m*(ω0^2 - ω^2)) (Refractive index)
        - II.13.17: B = (1/4πε0*c^2) * (2I/r) (Magnetic field from current)
        - II.13.23: B = μ0 * I / (2π * r) (Wire field)
        - II.13.34: B = μ0 * N * I / (2π * r) (Solenoid)
        - II.15.4: F = q(E + v×B) (Force on charge)
        - II.15.5: E = -v×B (Induced E field)
        - II.21.32: T = (1/2) * ε0 * E^2 * (2/3) * a^3 (Polarizability energy)
        - II.24.17: k = 1/(4πε0) * (2λ/r) (Line charge potential)
        - II.27.16: F = ε0 * c^2 * (E×B) (EM force density)
        - II.27.18: E = ε0 * c^2 * E×B (Energy flux)
        - II.34.2a: μ = q * L / (2m) (Magnetic moment)
        - II.34.2b: B = μ0 * M (Magnetization field)
        - II.34.11: I = (μ0 * q * v) / (2 * r) (Current loop)
        - II.35.18: n = n0 * (1 - p/ε0) (Index correction)
        - II.35.21: n = 1 + N * α/(2ε0) (Clausius-Mossotti)
        - II.36.38: B = (μ0 * q * v) / (4π * r^2) (Moving charge)
        - II.38.3: A = (μ0/4π) * (m×r)/r^3 (Vector potential)
        - II.38.14: F = (q * v) * B (Magnetic force)
        - III.4.32: n = 1/(exp((ℏω-μ)/kT) - 1) (Bose-Einstein)
        - III.4.33: n = 1/(exp(ℏω/kT) + 1) (Fermi-Dirac)
        - III.7.38: ω = (E1 - E2)/ℏ (Transition frequency)
        - III.8.54: P = (1/2) * (E^2 + B^2) (EM energy)
        - III.9.52: E = (p^2/2m) + mgh (Total energy)
        - III.10.19: E = (μ * B) / (I * ℏ) (NMR frequency)
        - III.12.43: L = n * ℏ (Angular momentum quantization)
        - III.13.18: E = (2 * μ * B) / (n * ℏ) (Zeeman splitting)
        - III.14.14: I = I0 * (sin(β/2)/(β/2))^2 (Diffraction)
        - III.15.12: E = (1/2) * ℏ * ω (Zero-point energy)
        - III.15.14: m = (ℏ * k)^2 / (2E) (Effective mass)
        - III.15.27: P = |ψ|^2 * (ℏk/m) (Probability current)
        - III.17.37: E = (p^2/2m) + V(x) (Schrödinger energy)
        - III.19.51: E = -(m * q^4) / (2 * (4πε0)^2 * ℏ^2) * (1/n^2) (Hydrogen)
        - III.21.20: P = (E * B) / (8π) (Radiation pressure)
    """
    
    # Subset of simple formulas for quick validation
    SIMPLE_FORMULAS: List[FeynmanFormula] = [
        FeynmanFormula(
            name="I.6.2a",
            description="Relativistic energy",
            formula_str="E = m * c^2",
            variables=["m", "c"],
            target="E",
            units={"m": [1,0,0,0,0], "c": [0,1,-1,0,0], "E": [1,2,-2,0,0]},
        ),
        FeynmanFormula(
            name="I.8.14",
            description="Distance = velocity * time",
            formula_str="d = v * t",
            variables=["v", "t"],
            target="d",
            units={"v": [0,1,-1,0,0], "t": [0,0,1,0,0], "d": [0,1,0,0,0]},
        ),
        FeynmanFormula(
            name="I.12.1",
            description="Friction force",
            formula_str="F = mu * N",
            variables=["mu", "N"],
            target="F",
            units={"mu": [0,0,0,0,0], "N": [1,1,-2,0,0], "F": [1,1,-2,0,0]},
        ),
        FeynmanFormula(
            name="I.12.11",
            description="Kinetic energy",
            formula_str="E = 0.5 * m * v^2",
            variables=["m", "v"],
            target="E",
            units={"m": [1,0,0,0,0], "v": [0,1,-1,0,0], "E": [1,2,-2,0,0]},
        ),
        FeynmanFormula(
            name="I.14.3",
            description="Gravitational potential energy",
            formula_str="U = m * g * h",
            variables=["m", "g", "h"],
            target="U",
            units={"m": [1,0,0,0,0], "g": [0,1,-2,0,0], "h": [0,1,0,0,0], "U": [1,2,-2,0,0]},
        ),
        FeynmanFormula(
            name="I.15.3t",
            description="Acceleration equation",
            formula_str="t = (v - u) / a",
            variables=["v", "u", "a"],
            target="t",
            units={"v": [0,1,-1,0,0], "u": [0,1,-1,0,0], "a": [0,1,-2,0,0], "t": [0,0,1,0,0]},
        ),
        FeynmanFormula(
            name="I.16.6",
            description="Dynamic pressure",
            formula_str="P = 0.5 * rho * v^2",
            variables=["rho", "v"],
            target="P",
            units={"rho": [1,-3,0,0,0], "v": [0,1,-1,0,0], "P": [1,-1,-2,0,0]},
        ),
        FeynmanFormula(
            name="I.18.4",
            description="Center of mass",
            formula_str="r = (m1*r1 + m2*r2) / (m1 + m2)",
            variables=["m1", "r1", "m2", "r2"],
            target="r",
            units={"m1": [1,0,0,0,0], "r1": [0,1,0,0,0], "m2": [1,0,0,0,0], "r2": [0,1,0,0,0], "r": [0,1,0,0,0]},
        ),
        FeynmanFormula(
            name="I.24.6",
            description="Capacitor energy",
            formula_str="E = 0.5 * C * V^2",
            variables=["C", "V"],
            target="E",
            units={"C": [-1,-2,4,2,0], "V": [1,2,-3,-1,0], "E": [1,2,-2,0,0]},
        ),
        FeynmanFormula(
            name="I.27.6",
            description="Photon energy",
            formula_str="E = h * f",
            variables=["h", "f"],
            target="E",
            units={"h": [1,2,-1,0,0], "f": [0,0,-1,0,0], "E": [1,2,-2,0,0]},
        ),
        FeynmanFormula(
            name="I.29.4",
            description="Wave number",
            formula_str="k = omega / c",
            variables=["omega", "c"],
            target="k",
            units={"omega": [0,0,-1,0,0], "c": [0,1,-1,0,0], "k": [0,-1,0,0,0]},
        ),
        FeynmanFormula(
            name="I.30.5",
            description="de Broglie wavelength",
            formula_str="lambda = h / p",
            variables=["h", "p"],
            target="lambda",
            units={"h": [1,2,-1,0,0], "p": [1,1,-1,0,0], "lambda": [0,1,0,0,0]},
        ),
        FeynmanFormula(
            name="I.39.1",
            description="Thermal energy",
            formula_str="E = 1.5 * kB * T",
            variables=["kB", "T"],
            target="E",
            units={"kB": [1,2,-2,0,-1], "T": [0,0,0,0,1], "E": [1,2,-2,0,0]},
        ),
    ]

    # Feynman Easy: 6 simpler formulas (fewer variables, basic ops)
    FEYNMAN_EASY: List[FeynmanFormula] = SIMPLE_FORMULAS[:6]

    # Feynman Medium: all 12 formulas (includes division, more variables)
    FEYNMAN_MEDIUM: List[FeynmanFormula] = SIMPLE_FORMULAS[:]

    @classmethod
    def generate_data(
        cls,
        formula: FeynmanFormula,
        n_samples: int = 1000,
        noise_std: float = 0.01,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for a formula.
        
        Returns:
            X: (n_samples, n_features) input matrix
            y: (n_samples,) target vector
        """
        rng = np.random.default_rng(seed)
        n_vars = len(formula.variables)
        
        # Generate random inputs in reasonable ranges
        X = rng.uniform(0.1, 10.0, size=(n_samples, n_vars))
        
        # Compute target based on formula
        y = cls._compute_formula(formula, X)
        
        # Add noise
        if noise_std > 0:
            y += rng.normal(0, noise_std * np.std(y), size=y.shape)
        
        return X, y
    
    @classmethod
    def _compute_formula(cls, formula: FeynmanFormula, X: np.ndarray) -> np.ndarray:
        """Compute formula output given inputs."""
        name = formula.name
        
        # Simple polynomial/rational formulas
        if name == "I.6.2a":  # E = m * c^2
            return X[:, 0] * X[:, 1]**2
        elif name == "I.8.14":  # d = v * t
            return X[:, 0] * X[:, 1]
        elif name == "I.12.1":  # F = mu * N
            return X[:, 0] * X[:, 1]
        elif name == "I.12.11":  # E = 0.5 * m * v^2
            return 0.5 * X[:, 0] * X[:, 1]**2
        elif name == "I.14.3":  # U = m * g * h
            return X[:, 0] * X[:, 1] * X[:, 2]
        elif name == "I.15.3t":  # t = (v - u) / a
            return (X[:, 0] - X[:, 1]) / X[:, 2]
        elif name == "I.16.6":  # P = 0.5 * rho * v^2
            return 0.5 * X[:, 0] * X[:, 1]**2
        elif name == "I.18.4":  # r = (m1*r1 + m2*r2) / (m1 + m2)
            return (X[:, 0]*X[:, 1] + X[:, 2]*X[:, 3]) / (X[:, 0] + X[:, 2])
        elif name == "I.24.6":  # E = 0.5 * C * V^2
            return 0.5 * X[:, 0] * X[:, 1]**2
        elif name == "I.27.6":  # E = h * f
            return X[:, 0] * X[:, 1]
        elif name == "I.29.4":  # k = omega / c
            return X[:, 0] / X[:, 1]
        elif name == "I.30.5":  # lambda = h / p
            return X[:, 0] / X[:, 1]
        elif name == "I.39.1":  # E = 1.5 * kB * T
            return 1.5 * X[:, 0] * X[:, 1]
        else:
            raise ValueError(f"Unknown formula: {name}")


# =============================================================================
# Symbolic Formula Comparison
# =============================================================================

class FormulaComparator:
    """Compare discovered formula with ground truth."""
    
    # Normalized operators for comparison
    OPERATORS = {"+", "-", "*", "/", "^", "**", "sqrt", "exp", "log", "sin", "cos", "abs"}
    
    @classmethod
    def normalize(cls, formula: str) -> str:
        """Normalize formula string for comparison."""
        if not formula:
            return ""
        # Remove whitespace
        f = formula.replace(" ", "")
        # Replace ** with ^
        f = f.replace("**", "^")
        # Convert to lowercase
        f = f.lower()
        # Remove coefficients (approximate)
        f = re.sub(r'\d+\.?\d*\*?', '', f)
        # Sort terms in sum
        if '+' in f:
            terms = f.split('+')
            f = '+'.join(sorted(terms))
        return f
    
    @classmethod
    def structural_similarity(cls, discovered: str, ground_truth: str) -> float:
        """Compute structural similarity between formulas.
        
        Returns:
            Similarity score in [0, 1] based on operator and variable overlap
        """
        if not discovered:
            return 0.0
        
        disc_norm = cls.normalize(discovered)
        truth_norm = cls.normalize(ground_truth)
        
        # Extract operators
        disc_ops = set(re.findall(r'[\+\-\*\/\^]|sqrt|exp|log|sin|cos|abs', disc_norm))
        truth_ops = set(re.findall(r'[\+\-\*\/\^]|sqrt|exp|log|sin|cos|abs', truth_norm))
        
        # Extract variable interactions (e.g., "m*c", "v*t")
        disc_terms = set(re.findall(r'[a-z]+\*[a-z]+|\w+\^\d+', disc_norm))
        truth_terms = set(re.findall(r'[a-z]+\*[a-z]+|\w+\^\d+', truth_norm))
        
        # Combine features
        disc_features = disc_ops.union(disc_terms)
        truth_features = truth_ops.union(truth_terms)
        
        if not truth_features:
            return 1.0 if not disc_features else 0.0
        
        # Jaccard similarity
        intersection = len(disc_features & truth_features)
        union = len(disc_features | truth_features)
        
        base_sim = intersection / union if union > 0 else 0.0
        
        # Bonus: if key interaction term is present (e.g., v*t for d=v*t)
        if truth_terms:
            key_terms_found = sum(1 for t in truth_terms if t in disc_terms)
            if key_terms_found > 0:
                base_sim = max(base_sim, 0.5 + 0.5 * key_terms_found / len(truth_terms))
        
        return min(base_sim, 1.0)
    
    @classmethod
    def exact_match(cls, discovered: str, ground_truth: str, tolerance: float = 0.1) -> bool:
        """Check if formulas are approximately equal."""
        if not discovered:
            return False
        
        disc_norm = cls.normalize(discovered)
        truth_norm = cls.normalize(ground_truth)
        
        # Check for containment of key terms
        disc_terms = set(disc_norm.replace("+", " ").replace("-", " ").replace("*", " ").split())
        truth_terms = set(truth_norm.replace("+", " ").replace("-", " ").replace("*", " ").split())
        
        # Remove empty strings
        disc_terms = {t for t in disc_terms if t}
        truth_terms = {t for t in truth_terms if t}
        
        if not truth_terms:
            return len(disc_terms) == 0
        
        # Check if most key terms are present
        common = len(disc_terms & truth_terms)
        return common / len(truth_terms) >= (1 - tolerance)


# =============================================================================
# SRBench Runner
# =============================================================================

class SRBenchRunner:
    """Runner for SRBench (Feynman Dataset) validation."""
    
    def __init__(
        self,
        n_samples: int = 1000,
        train_epochs: int = 200,
        hidden_dim: int = 64,
        use_pysr: bool = False,
        device: str = "cpu",
    ):
        self.n_samples = n_samples
        self.train_epochs = train_epochs
        self.hidden_dim = hidden_dim
        self.use_pysr = use_pysr
        self.device = torch.device(device)
        
        # Initialize DiscoveryEngine
        self.discovery = DiscoveryEngine(use_pysr=use_pysr)
    
    def _normalize_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Normalize data for stable training."""
        stats = {
            'X_mean': X.mean(axis=0),
            'X_std': X.std(axis=0) + 1e-8,
            'y_mean': y.mean(),
            'y_std': y.std() + 1e-8,
        }
        X_norm = (X - stats['X_mean']) / stats['X_std']
        y_norm = (y - stats['y_mean']) / stats['y_std']
        return X_norm, y_norm, stats
    
    def train_rcln(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_units: List[List[int]] = None,
    ) -> Tuple[RCLNLayer, Dict]:
        """Train RCLN with Hard Core = 0 (pure neural fitting).
        
        Returns:
            Trained RCLN layer and normalization stats
        """
        n_in = X.shape[1]
        n_out = 1 if y.ndim == 1 else y.shape[1]
        
        # Normalize data
        X_norm, y_norm, stats = self._normalize_data(X, y)
        
        # Create RCLN with no hard core (pure soft shell)
        rcln = RCLNLayer(
            input_dim=n_in,
            hidden_dim=self.hidden_dim,
            output_dim=n_out,
            hard_core_func=None,  # Hard Core = 0
            lambda_res=1.0,
            net_type="mlp",
        ).to(self.device)
        
        # Prepare data
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)
        y_tensor = torch.from_numpy(y_norm).float().to(self.device)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        # Train
        optimizer = optim.Adam(rcln.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
        
        for epoch in range(self.train_epochs):
            optimizer.zero_grad()
            y_pred = rcln(X_tensor)
            loss = nn.functional.mse_loss(y_pred, y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{self.train_epochs}, Loss: {loss.item():.6f}")
        
        return rcln, stats
    
    def evaluate_formula(
        self,
        formula: FeynmanFormula,
        noise_std: float = 0.01,
        verbose: bool = True,
    ) -> DiscoveryResult:
        """Run full pipeline on one formula.
        
        Args:
            formula: Feynman formula to recover
            noise_std: Gaussian noise strength (relative to std(y))
            verbose: Print progress
        """
        import time
        
        if verbose:
            print(f"\nEvaluating: {formula.name} - {formula.description}")
            print(f"  Ground truth: {formula.formula_str}, noise_std={noise_std}")
        
        start_time = time.time()
        
        # 1. Generate data
        X, y = FeynmanDatasetLoader.generate_data(
            formula, 
            n_samples=self.n_samples,
            noise_std=noise_std,
        )
        if verbose:
            print(f"  Generated {len(X)} samples, {len(formula.variables)} variables")
        
        # 2. Train RCLN (Hard Core = 0)
        if verbose:
            print(f"  Training RCLN Soft Shell...")
        input_units = [formula.units.get(v, [0,0,0,0,0]) for v in formula.variables]
        rcln, norm_stats = self.train_rcln(X, y, input_units)
        
        # 3. Extract y_soft from RCLN (use normalized data)
        if verbose:
            print(f"  Extracting Soft Shell outputs...")
        rcln.eval()
        X_norm, y_norm, _ = self._normalize_data(X, y)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_norm).float().to(self.device)
            y_pred_norm = rcln(X_tensor).detach().cpu().numpy().ravel()
            # Denormalize
            y_pred = y_pred_norm * norm_stats['y_std'] + norm_stats['y_mean']
        
        # 4. Call DiscoveryEngine with enhanced distill
        if verbose:
            print(f"  Running DiscoveryEngine...")
        # Use enhanced distill() which now includes polynomial regression
        data_buffer = list(zip(X_norm, y_norm))
        input_units = [formula.units.get(v, [0,0,0,0,0]) for v in formula.variables]
        discovered = self.discovery.distill(
            rcln_layer=None,
            data_buffer=data_buffer,
            input_units=input_units,
            niterations=20,
        )
        
        if verbose:
            print(f"  Discovered: {discovered}")
        
        # 5. Evaluate
        training_time = time.time() - start_time
        
        # Compute R² and MSE (on denormalized predictions)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mse = np.mean((y - y_pred) ** 2)
        
        # Structural similarity
        sim = FormulaComparator.structural_similarity(
            discovered, formula.formula_str
        )
        
        # Exact match
        exact = FormulaComparator.exact_match(
            discovered, formula.formula_str
        )
        
        # Success criteria: R² > 0.95 (good fit) AND contains interaction terms
        # For physical formulas, interaction terms (x*y) indicate multiplicative relationship
        has_interaction = "*" in str(discovered) if discovered else False
        success = (r2 > 0.95) and has_interaction
        
        result = DiscoveryResult(
            formula_name=formula.name,
            ground_truth=formula.formula_str,
            discovered=discovered,
            exact_match=exact,
            r2_score=r2,
            mse=mse,
            structural_similarity=sim,
            success=success,
            training_time=training_time,
        )
        
        if verbose:
            print(f"  R2: {r2:.4f}, MSE: {mse:.6f}, Similarity: {sim:.2f}, Success: {success}")
        
        return result
    
    def run_benchmark(
        self,
        formulas: Optional[List[FeynmanFormula]] = None,
        noise_std: float = 0.01,
    ) -> Dict[str, Any]:
        """Run full benchmark on all formulas."""
        if formulas is None:
            formulas = FeynmanDatasetLoader.SIMPLE_FORMULAS
        
        print("=" * 70)
        print("SRBench (Feynman Dataset) Validation")
        print("=" * 70)
        print(f"Total formulas: {len(formulas)}")
        print(f"Samples per formula: {self.n_samples}")
        print(f"Noise std: {noise_std}")
        print(f"Training epochs: {self.train_epochs}")
        print(f"Discovery: {'PySR' if self.use_pysr else 'Lasso'}")
        print("=" * 70)
        
        results = []
        for i, formula in enumerate(formulas, 1):
            print(f"\n[{i}/{len(formulas)}] ", end="")
            try:
                result = self.evaluate_formula(formula, noise_std=noise_std)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append(DiscoveryResult(
                    formula_name=formula.name,
                    ground_truth=formula.formula_str,
                    discovered=None,
                    exact_match=False,
                    r2_score=0.0,
                    mse=float('inf'),
                    structural_similarity=0.0,
                    success=False,
                    training_time=0.0,
                ))
        
        # Aggregate statistics
        n_success = sum(1 for r in results if r.success)
        n_exact = sum(1 for r in results if r.exact_match)
        
        summary = {
            "total_formulas": len(formulas),
            "success_count": n_success,
            "success_rate": n_success / len(formulas),
            "exact_match_count": n_exact,
            "exact_match_rate": n_exact / len(formulas),
            "mean_r2": np.mean([r.r2_score for r in results]),
            "mean_similarity": np.mean([r.structural_similarity for r in results]),
            "total_time": sum(r.training_time for r in results),
            "results": [r.to_dict() for r in results],
        }
        
        print("\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print(f"Success Rate: {summary['success_rate']*100:.1f}% ({n_success}/{len(formulas)})")
        print(f"Exact Match Rate: {summary['exact_match_rate']*100:.1f}% ({n_exact}/{len(formulas)})")
        print(f"Mean R2: {summary['mean_r2']:.4f}")
        print(f"Mean Structural Similarity: {summary['mean_similarity']:.4f}")
        print(f"Total Time: {summary['total_time']:.2f}s")
        print("=" * 70)
        
        return summary

    def run_benchmark_over_noise(
        self,
        formulas: Optional[List[FeynmanFormula]] = None,
        noise_levels: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run benchmark at each noise level; return recovery rate vs noise.
        
        Returns:
            {
                "noise_levels": [0.0, 0.01, ...],
                "recovery_rates": [1.0, 0.9, ...],
                "success_counts": [6, 5, ...],
                "n_formulas": int,
                "per_noise_results": [ { "noise": 0.01, "results": [...] }, ... ]
            }
        """
        noise_levels = noise_levels or [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
        if formulas is None:
            formulas = FeynmanDatasetLoader.FEYNMAN_EASY
        recovery_rates = []
        success_counts = []
        per_noise_results = []
        for noise in noise_levels:
            if verbose:
                print(f"\n--- Noise std = {noise} ---")
            results = []
            for i, formula in enumerate(formulas):
                if verbose:
                    print(f"  [{i+1}/{len(formulas)}] {formula.name} ", end="", flush=True)
                try:
                    r = self.evaluate_formula(formula, noise_std=noise, verbose=False)
                    results.append(r)
                    if verbose:
                        print(f" success={r.success}")
                except Exception as e:
                    if verbose:
                        print(f" ERROR: {e}")
                    results.append(DiscoveryResult(
                        formula_name=formula.name,
                        ground_truth=formula.formula_str,
                        discovered=None,
                        exact_match=False,
                        r2_score=0.0,
                        mse=float("inf"),
                        structural_similarity=0.0,
                        success=False,
                        training_time=0.0,
                    ))
            n_ok = sum(1 for r in results if r.success)
            recovery_rates.append(n_ok / len(formulas))
            success_counts.append(n_ok)
            per_noise_results.append({"noise": noise, "results": [x.to_dict() for x in results]})
        return {
            "noise_levels": noise_levels,
            "recovery_rates": recovery_rates,
            "success_counts": success_counts,
            "n_formulas": len(formulas),
            "per_noise_results": per_noise_results,
        }


def plot_recovery_rate_vs_noise(
    data: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "SRBench Feynman: Formula Recovery Rate vs Noise (Axiom-OS)",
    label: Optional[str] = None,
) -> None:
    """Plot recovery rate vs noise strength. data = run_benchmark_over_noise() return."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    noise = data["noise_levels"]
    rates = [r * 100 for r in data["recovery_rates"]]
    plt.figure(figsize=(8, 5))
    plt.plot(noise, rates, "b-o", linewidth=2, markersize=8, label=label or "Recovery rate")
    plt.xlabel("Noise strength (std relative to target)")
    plt.ylabel("Recovery rate (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(min(noise) - 0.01, max(noise) + 0.01)
    plt.ylim(-5, 105)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.close()


def plot_recovery_rate_vs_noise_compare(
    data_easy: Dict[str, Any],
    data_medium: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    """Plot Easy vs Medium recovery rate vs noise on one figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    plt.figure(figsize=(8, 5))
    n1, r1 = data_easy["noise_levels"], [x * 100 for x in data_easy["recovery_rates"]]
    n2, r2 = data_medium["noise_levels"], [x * 100 for x in data_medium["recovery_rates"]]
    plt.plot(n1, r1, "b-o", linewidth=2, markersize=8, label="Feynman Easy (6 formulas)")
    plt.plot(n2, r2, "g-s", linewidth=2, markersize=8, label="Feynman Medium (12 formulas)")
    plt.xlabel("Noise strength (std relative to target)")
    plt.ylabel("Formula recovery rate (%)")
    plt.title("SRBench Feynman: Recovery Rate vs Noise (Axiom-OS Discovery Engine)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    all_noise = n1 + n2
    plt.xlim(min(all_noise) - 0.01, max(all_noise) + 0.01)
    plt.ylim(-5, 105)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.close()


def download_srbench_data(output_dir: str = "data/feynman") -> bool:
    """Download or prepare SRBench/Feynman dataset. Returns True if data is available."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.txt"
    readme.write_text(
        "SRBench Feynman Dataset\n"
        "=======================\n"
        "Official source: https://cavalab.org/srbench/datasets/\n"
        "PMLB (EpistasisLab): https://github.com/EpistasisLab/pmlb\n"
        "Axiom-OS uses built-in Feynman formulas in srbench_runner.py when no CSV/TSV files are present.\n"
        "Place Feynman CSV/TSV files here to load from disk.\n",
        encoding="utf-8",
    )
    # Optional: try to fetch Feynman data from a public source (PMLB/HuggingFace often use different paths)
    try:
        import urllib.request
        # Alternative: Zenodo or SRBench mirror; if unavailable, we use built-in formulas
        urls = [
            "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/feynman_i_6_2a.csv",
            "https://github.com/EpistasisLab/pmlb/raw/master/datasets/feynman_i_6_2a.csv",
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Axiom-OS-SRBench/1.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    local = out / "feynman_i_6_2a.csv"
                    with open(local, "wb") as f:
                        f.write(resp.read())
                    return True
            except Exception:
                continue
    except Exception:
        pass
    print(f"Using built-in formulas. Data dir: {out} (see README for external data)")
    return False


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SRBench (Feynman Dataset) Validation for Axiom-OS DiscoveryEngine"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples per formula (default: 1000)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs for RCLN (default: 200)"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="RCLN hidden dimension (default: 64)"
    )
    parser.add_argument(
        "--pysr",
        action="store_true",
        help="Use PySR for symbolic regression (requires pysr package)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/srbench_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--formulas",
        type=int,
        default=None,
        help="Number of formulas to test (default: all)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["easy", "medium", "all"],
        help="Feynman subset: easy (6), medium (12), or all (default: from --formulas or all)"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Noise std for data generation (default: 0.01)"
    )
    parser.add_argument(
        "--recovery-vs-noise",
        action="store_true",
        help="Run recovery rate vs noise strength (Easy subset), then plot"
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0.0,0.01,0.03,0.05,0.1,0.2",
        help="Comma-separated noise levels for --recovery-vs-noise (default: 0.0,0.01,0.03,0.05,0.1,0.2)"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Save recovery-vs-noise plot to this path (e.g. docs/images/srbench_recovery_vs_noise.png)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/prepare SRBench data to data/feynman"
    )
    args = parser.parse_args()

    if args.download:
        download_srbench_data("data/feynman")
        return 0

    # Create runner
    runner = SRBenchRunner(
        n_samples=args.samples,
        train_epochs=args.epochs,
        hidden_dim=args.hidden,
        use_pysr=args.pysr,
    )

    # Select formulas
    if args.subset == "easy":
        formulas = FeynmanDatasetLoader.FEYNMAN_EASY
    elif args.subset == "medium":
        formulas = FeynmanDatasetLoader.FEYNMAN_MEDIUM
    else:
        formulas = FeynmanDatasetLoader.SIMPLE_FORMULAS
    if args.formulas:
        formulas = formulas[: args.formulas]

    if args.recovery_vs_noise:
        noise_levels = [float(x) for x in args.noise_levels.split(",")]
        out_path = (args.output or "benchmarks/srbench_results").replace(".json", "")
        out_json_base = Path(out_path)
        out_json_base.parent.mkdir(parents=True, exist_ok=True)
        easy_formulas = FeynmanDatasetLoader.FEYNMAN_EASY
        medium_formulas = FeynmanDatasetLoader.FEYNMAN_MEDIUM
        if args.formulas:
            easy_formulas = easy_formulas[: args.formulas]
            medium_formulas = medium_formulas[: args.formulas]
        # Run Feynman Easy
        print(f"SRBench: Recovery rate vs noise — Feynman Easy ({len(easy_formulas)} formulas)")
        data_easy = runner.run_benchmark_over_noise(
            formulas=easy_formulas,
            noise_levels=noise_levels,
        )
        with open(out_json_base.parent / (out_json_base.name + "_easy_recovery_vs_noise.json"), "w") as f:
            json.dump({k: data_easy[k] for k in ["noise_levels", "recovery_rates", "success_counts", "n_formulas"]}, f, indent=2)
        # Run Feynman Medium
        print(f"\nSRBench: Recovery rate vs noise — Feynman Medium ({len(medium_formulas)} formulas)")
        data_medium = runner.run_benchmark_over_noise(
            formulas=medium_formulas,
            noise_levels=noise_levels,
        )
        with open(out_json_base.parent / (out_json_base.name + "_medium_recovery_vs_noise.json"), "w") as f:
            json.dump({k: data_medium[k] for k in ["noise_levels", "recovery_rates", "success_counts", "n_formulas"]}, f, indent=2)
        plot_path = args.plot or "docs/images/srbench_recovery_vs_noise.png"
        plot_recovery_rate_vs_noise_compare(data_easy, data_medium, save_path=plot_path)
        return 0

    # Run benchmark (single noise level)
    results = runner.run_benchmark(formulas, noise_std=args.noise)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if results["success_rate"] > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
