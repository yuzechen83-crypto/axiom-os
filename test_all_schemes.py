"""
RCLN 2.0 - Test All Four Evolution Schemes
"""

import torch
from axiom_os.layers import RCLNLayer

print("="*70)
print("RCLN 2.0 - Four Evolution Schemes Test")
print("="*70)

x = torch.randn(5, 4)

# Test 1: Spectral
print("\n[1] Spectral-RCLN")
rcln = RCLNLayer(4, 8, 2, net_type='spectral')
y = rcln(x)
print(f"    {x.shape} -> {y.shape} : OK")

# Test 2: Clifford
print("\n[2] Clifford-RCLN")
rcln = RCLNLayer(4, 8, 2, net_type='clifford')
y = rcln(x)
print(f"    {x.shape} -> {y.shape} : OK")

# Test 3: KAN
print("\n[3] KAN-RCLN")
rcln = RCLNLayer(4, 8, 2, net_type='kan', kan_grid_size=5)
y = rcln(x)
print(f"    {x.shape} -> {y.shape} : OK")
formula = rcln.extract_formula(['a', 'b', 'c', 'd'])
if formula:
    print(f"    Formula extraction: OK")

# Test 4: Mamba
print("\n[4] Mamba-RCLN")
rcln = RCLNLayer(4, 8, 2, net_type='mamba')
y = rcln(x)
print(f"    {x.shape} -> {y.shape} : OK")

# Test with Hard Core
print("\n[5] With Physics Hard Core")
def hard_core(x):
    return -0.5 * x if hasattr(x, 'detach') else -0.5 * x.values

rcln = RCLNLayer(4, 8, 4, net_type='kan', hard_core_func=hard_core, lambda_res=0.5)
y = rcln(x)
print(f"    {x.shape} -> {y.shape} : OK")
print(f"    y = F_hard + 0.5 * F_soft")

print("\n" + "="*70)
print("All Four Evolution Schemes Verified!")
print("="*70)
