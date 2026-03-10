#!/usr/bin/env python3
"""
Isaac Sim Standalone Example - Minimal working demo
This script can be run directly from Isaac Sim's script editor or via kit
"""

import sys
print("=" * 60)
print("Isaac Sim Standalone Example")
print("=" * 60)
print()

# Check if running in Isaac Sim
if "omni" not in sys.modules and "isaacsim" not in sys.modules:
    print("[ERROR] This script must be run from within Isaac Sim!")
    print()
    print("Usage methods:")
    print("1. Isaac Sim Script Editor: Window -> Script Editor -> Open -> Run")
    print("2. Kit command line: kit.exe app.kit --exec python this_script.py")
    print()
    input("Press Enter to exit...")
    sys.exit(1)

# Imports
try:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.nucleus import get_assets_root_path
    print("[OK] Isaac Sim 4.0+ API loaded")
except ImportError:
    try:
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        print("[OK] Legacy Isaac Sim API loaded")
    except ImportError as e:
        print(f"[ERROR] Cannot load Isaac Sim API: {e}")
        sys.exit(1)

import numpy as np

print("\n[1] Creating World...")
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0/60.0,
    rendering_dt=1.0/60.0,
)
print("    World created")

print("\n[2] Adding ground plane...")
world.scene.add_default_ground_plane(z_position=0.0)
print("    Ground plane added")

print("\n[3] Loading Go1 robot...")
try:
    assets_root = get_assets_root_path()
    if assets_root:
        go1_path = assets_root + "/Isaac/Robots/Unitree/Go1/go1.usd"
        print(f"    Asset path: {go1_path}")
        
        add_reference_to_stage(usd_path=go1_path, prim_path="/World/Go1")
        print("    Robot reference added to stage")
    else:
        print("    [WARNING] Could not get assets root path")
except Exception as e:
    print(f"    [ERROR] Failed to load robot: {e}")

print("\n[4] Resetting world...")
world.reset()
print("    World reset complete")

print("\n[5] Running simulation steps...")
for i in range(100):
    world.step(render=True)
    if i % 20 == 0:
        print(f"    Step {i}/100")

print("\n" + "=" * 60)
print("Demo complete! You should see:")
print("- A gray ground plane")
print("- A Unitree Go1 robot standing on the ground")
print("=" * 60)

# Keep window open
print("\nPress Ctrl+C in console or close window to exit")
while True:
    world.step(render=True)
