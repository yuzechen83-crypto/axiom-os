#!/usr/bin/env python3
"""Test Isaac Sim modules availability."""

print("Testing Isaac Sim 4.0+ modules (isaacsim.*)...")

try:
    import isaacsim
    print('[OK] isaacsim available')
except ImportError as e:
    print(f'[WARNING] isaacsim: {e}')

try:
    import omni
    print('[OK] omni available')
except ImportError as e:
    print(f'[WARNING] omni: {e}')

# Try to initialize kit
print("\nTrying to initialize Kit...")
try:
    from omni.isaac.kit import SimulationApp
    print('[OK] SimulationApp available')
    
    # Initialize with minimal config
    config = {"headless": True}
    simulation_app = SimulationApp(config)
    print('[OK] SimulationApp initialized')
    
    # Now try importing isaac modules
    try:
        from omni.isaac.core import World
        print('[OK] omni.isaac.core.World available after init')
    except ImportError as e:
        print(f'[WARNING] omni.isaac.core.World: {e}')
    
    simulation_app.close()
    print('[OK] SimulationApp closed')
    
except ImportError as e:
    print(f'[WARNING] SimulationApp: {e}')

print("\nCheck complete")
