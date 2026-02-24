"""Quick SPNN structure test"""
import sys
print("Testing SPNN imports...")
from spnn import SPNN, PhysicalScaleSystem, RCLN, Hippocampus, AxiomOS
print("Imports OK")

import torch
model = SPNN(in_dim=4, hidden_dim=16, out_dim=1, num_rcln_layers=1, memory_capacity=100)
x = torch.randn(8, 4)
y, aux = model(x)
print(f"Forward OK: input {x.shape} -> output {y.shape}")
print("SPNN structure test passed.")
