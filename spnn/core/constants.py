"""
Physical Constants - Universal Scale Anchoring (B_universal)
普朗克尺度归一化基准
"""

import numpy as np
from typing import Dict

# SI units: m, kg, s, K, A
# B_universal = (c, G, ℏ, k_B, ε_0)

# Speed of light [m/s]
c = 299792458.0

# Gravitational constant [m³/(kg·s²)]
G = 6.67430e-11

# Reduced Planck constant [J·s]
hbar = 1.054571817e-34

# Boltzmann constant [J/K]
k_B = 1.380649e-23

# Vacuum permittivity [F/m]
epsilon_0 = 8.8541878128e-12

# Derived Planck units
ℓ_P = np.sqrt(hbar * G / c**3)      # Planck length ~ 1.616e-35 m
t_P = np.sqrt(hbar * G / c**5)      # Planck time ~ 5.391e-44 s
m_P = np.sqrt(hbar * c / G)         # Planck mass ~ 2.176e-8 kg

# Machine epsilon for numerical stability
EPSILON = 1e-8
EPSILON_MACHINE = np.finfo(np.float64).eps

# Scale tolerance for dimensional consistency
TAU_SCALE = 1e-6

# Default dimension exponents: [L, M, T, Θ, I] (length, mass, time, temp, current)
DIMENSION_BASIS = ["L", "M", "T", "Θ", "I"]

UNIVERSAL_CONSTANTS: Dict[str, float] = {
    "c": c,
    "G": G,
    "hbar": hbar,
    "k_B": k_B,
    "epsilon_0": epsilon_0,
    "ℓ_P": ℓ_P,
    "t_P": t_P,
    "m_P": m_P,
}
