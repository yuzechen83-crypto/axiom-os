#!/usr/bin/env python3
"""Simple test of Isaac Sim environment without torch."""

import sys
import os

# Add project path
sys.path.insert(0, r'C:\Users\ASUS\PycharmProjects\PythonProject1')

print("=" * 60)
print("Isaac Sim Environment Test")
print("=" * 60)
print()

# Test imports
print("[1] Testing imports...")
try:
    import numpy as np
    print(f"    [OK] numpy {np.__version__}")
except ImportError as e:
    print(f"    [FAIL] numpy: {e}")

try:
    # Try new API first
    from isaacsim.core.api import World
    print("    [OK] isaacsim.core.api.World")
    API_VERSION = "4.0+"
except ImportError as e:
    print(f"    [FAIL] isaacsim.core.api.World: {e}")
    try:
        from omni.isaac.core import World
        print("    [OK] omni.isaac.core.World (legacy)")
        API_VERSION = "legacy"
    except ImportError as e2:
        print(f"    [FAIL] omni.isaac.core.World: {e2}")
        API_VERSION = "none"

print(f"\n[2] Isaac Sim API Version: {API_VERSION}")

if API_VERSION == "none":
    print("\n[ERROR] Isaac Sim not available. Running in MOCK mode.")
    print("To use Isaac Sim, please run from Isaac Sim environment.")
    sys.exit(0)

# Test basic functionality
print("\n[3] Testing World creation...")
try:
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0,
    )
    print("    [OK] World created")
    
    # Add ground plane
    world.scene.add_default_ground_plane(z_position=0.0)
    print("    [OK] Ground plane added")
    
    # Reset world
    world.reset()
    print("    [OK] World reset")
    
    # Step once
    world.step(render=False)
    print("    [OK] World step executed")
    
    print("\n[4] Isaac Sim environment is working!")
    
except Exception as e:
    print(f"    [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Test complete")
print("=" * 60)
