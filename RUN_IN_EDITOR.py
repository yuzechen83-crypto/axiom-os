#!/usr/bin/env python3
"""
INSTRUCTIONS: How to run in Isaac Sim Editor
============================================

Method 1: Using Isaac Sim Script Editor (Easiest)
-------------------------------------------------
1. Open Isaac Sim (isaac-sim.bat)
2. Wait for it to fully load (you see the empty stage)
3. Go to Menu: Window -> Script Editor
4. Click "Open" button
5. Select this file: isaac_standalone_example.py
6. Click "Run" button
7. You should see the robot appear in the viewport

Method 2: Copy-Paste
--------------------
1. Open Isaac Sim
2. Open Script Editor (Window -> Script Editor)
3. Copy the code from isaac_standalone_example.py
4. Paste into Script Editor
5. Click "Run"

Troubleshooting:
---------------

Q: Window opens but nothing appears
A: Wait 1-2 minutes for extensions to load, then try Method 1

Q: "No module named isaacsim"
A: You must run from within Isaac Sim, not standalone Python

Q: Robot doesn't appear
A: Check that Go1 USD asset exists in Nucleus
   Alternative: Use a simple cube first to test

Q: View is black/empty
A: Try resetting camera:
   - Press F to focus on selected object
   - Or use Viewport controls to zoom out

Demo Code:
----------
"""

# Copy the code below into Isaac Sim Script Editor:

demo_code = '''
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path

print("Creating World...")
world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

print("Adding ground...")
world.scene.add_default_ground_plane(z_position=0.0)

print("Loading robot...")
try:
    assets_root = get_assets_root_path()
    go1_path = assets_root + "/Isaac/Robots/Unitree/Go1/go1.usd"
    add_reference_to_stage(usd_path=go1_path, prim_path="/World/Go1")
    print(f"Loaded from: {go1_path}")
except Exception as e:
    print(f"Error: {e}")

print("Resetting...")
world.reset()

print("Running 100 steps...")
for i in range(100):
    world.step(render=True)

print("Done!")
'''

print(__doc__)
print("\n" + "="*60)
print("SAMPLE CODE TO COPY INTO ISAAC SIM:")
print("="*60)
print(demo_code)
print("="*60)
