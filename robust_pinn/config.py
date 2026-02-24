"""
SPNN-Opt-Rev5 Robust PINN - Global Configuration
All physics computations use torch.float64 (Double Precision).
Physical Constants from Document Section 4.1.
"""

import torch

# --- ENFORCE DOUBLE PRECISION ---
DTYPE = torch.float64

# --- Physical Constants (SI units) ---
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
ell_P = (hbar * G / c**3) ** 0.5      # Planck length ~ 1.616e-35 m
t_P = (hbar * G / c**5) ** 0.5       # Planck time ~ 5.391e-44 s
m_P = (hbar * c / G) ** 0.5          # Planck mass ~ 2.176e-8 kg

# Machine epsilon for numerical stability
EPSILON = 1e-8

# --- Hyperparameters ---
HIDDEN_DIM = 256
NUM_LAYERS = 6
ACTIVATION = "silu"

# Soft-Melt (Document Section 2.2)
SOFT_MELT_KAPPA = 10.0   # stiffness of g(D) = sigmoid(kappa * (D - epsilon))
SOFT_MELT_EPSILON = 0.1  # threshold distance

# Training
MAX_GRAD_NORM = 1.0      # clip_grad_norm_
WARMUP_EPOCHS = 100      # Soft-Start: w_phys = 0 initially
W_DATA = 1.0
W_PHYS = 0.1            # ramp up over warmup
W_BC = 0.1

# Loss balancing (GradNorm)
GRADNORM_ALPHA = 1.5    # target ratio exponent

# Boundary
GHOST_MARGIN = 0.05     # points slightly outside domain
BC_NEUMANN_WEIGHT = 1.0
