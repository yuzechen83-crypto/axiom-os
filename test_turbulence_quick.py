"""Quick turbulence training test - 5 epochs, small data"""
import torch
from spnn import SPNN
from spnn.training.turbulence import run_turbulence_training, TurbulenceConfig

model = SPNN(in_dim=2, hidden_dim=32, out_dim=1, num_rcln_layers=1, memory_capacity=500)
config = TurbulenceConfig(n_t=15, n_x=20, n_modes=4)
run_turbulence_training(model, config=config, epochs=5, batch_size=64, lambda_phys=0.0)
print("OK")
