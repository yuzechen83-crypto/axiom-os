#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axiom-OS x Isaac Sim - Working Version
Compatible with Isaac Sim 4.0+
"""

# Import SimulationApp (correct path for Isaac Sim 4.0+)
try:
    from isaacsim.simulation_app import SimulationApp
    print("[OK] Imported SimulationApp from isaacsim.simulation_app")
except ImportError as e1:
    print("[ERROR] Cannot import SimulationApp:", str(e1))
    print("[ERROR] Please run this script with Isaac Sim's python.bat")
    import sys
    sys.exit(1)

# Start simulation app
print("Starting SimulationApp...")
simulation_app = SimulationApp({"headless": False})
print("[OK] SimulationApp started")

# Import other modules
import numpy as np

try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    API = "isaacsim"
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    API = "omni.isaac"

print("[OK] API:", API)

# Create world
print("Creating world...")
world = World(stage_units_in_meters=1.0)
print("[OK] World created")

# Add ground
world.scene.add_default_ground_plane()
print("[OK] Ground added")

# Add robot
print("Loading robot...")
assets_root = get_assets_root_path()
if assets_root:
    usd_path = assets_root + "/Isaac/Robots/Unitree/Go1/go1.usd"
    add_reference_to_stage(usd_path=usd_path, prim_path="/World/Go1")
    print("[OK] Robot loaded from:", usd_path)
else:
    print("[ERROR] Cannot get assets root path")
    simulation_app.close()
    exit(1)

# Create robot handle
try:
    go1 = Articulation(prim_paths_expr="/World/Go1", name="Go1")
    go1.set_world_poses(positions=np.array([[0.0, 0.0, 0.4]]))
    print("[OK] Robot handle created")
except Exception as e:
    print("[WARN] Articulation failed, using basic prim:", str(e))
    go1 = None

# Reset
world.reset()
print("[OK] World reset")

# Simulation loop
print("\n" + "=" * 50)
print("Starting simulation loop")
print("=" * 50)

step_count = 0
for i in range(500):
    world.step(render=True)
    step_count = i
    
    # Change pose at specific steps
    if go1 and i == 100:
        try:
            pose = np.array([[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, -0.8, 1.5, -0.1, -0.8, 1.5]])
            go1.set_joint_positions(pose)
            print("[Step 100] Standing pose")
        except Exception as e:
            print("[WARN] Cannot set pose:", str(e))
    
    if go1 and i == 300:
        try:
            pose = np.array([[0.0, 1.0, -2.0, 0.0, 1.0, -2.0, 0.0, -1.0, 2.0, 0.0, -1.0, 2.0]])
            go1.set_joint_positions(pose)
            print("[Step 300] Crouch pose")
        except Exception as e:
            print("[WARN] Cannot set pose:", str(e))
    
    if i % 100 == 0:
        print("Step", i, "/500")

print("\n" + "=" * 50)
print("Simulation complete!")
print("Total steps:", step_count)
print("Close window or press Ctrl+C to exit")
print("=" * 50)

# Keep running
while simulation_app.is_running():
    world.step(render=True)
    simulation_app.update()

simulation_app.close()
