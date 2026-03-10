# Axiom-OS x NVIDIA Isaac Sim - Digital Twin

Photorealistic, high-fidelity Sim-to-Real verification demo for Axiom-OS controlling Unitree Go1 quadruped robot.

## Overview

This module connects Axiom-OS (Brain) to NVIDIA Isaac Sim (Body) to create a digital twin for:
- **Real-time control** of Unitree Go1 quadruped
- **Domain randomization** (friction, latency, external forces)
- **Discovery hooks** for adaptive learning
- **Ray-traced video recording**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Axiom-OS Digital Twin                   │
├─────────────────────────────────────────────────────────────┤
│  Axiom-OS Brain                                             │
│  ├─ Soft Shell (RCLN Neural Network)                        │
│  ├─ Hard Core (Physics Constraints)                         │
│  ├─ MPC Controller (Fallback)                               │
│  └─ Discovery Logger (Adaptive Learning)                    │
│                          │                                  │
│                          ▼                                  │
│  UPI Interface (Unified Physical Interface)                 │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              NVIDIA Isaac Sim (Omniverse)                   │
│  ├─ IsaacGo1Env (Environment Wrapper)                       │
│  ├─ Unitree Go1 USD Asset                                   │
│  ├─ Physics Scene (Ground, Gravity)                         │
│  ├─ Chaos Injector (Domain Randomization)                   │
│  │   ├─ Friction Shift (Ice ↔ Concrete)                     │
│  │   ├─ Action Latency (20ms delay)                         │
│  │   └─ External Forces (Kicks)                             │
│  └─ Ray-traced Camera                                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **NVIDIA Isaac Sim 5.1.0** (or 4.x) installed  
   - Launcher: `%LOCALAPPDATA%\ov\pkg\isaac-sim-5.1.0`  
   - Workstation default: `C:\isaac-sim` (Windows)
2. **Axiom-OS** repository cloned
3. Python environment with PyTorch

### Running the Demo

```bash
# Navigate to Isaac Sim directory
cd /path/to/isaac-sim

# Run with Isaac Sim's Python (Windows)
python.bat run_isaac_demo.py

# Run with Isaac Sim's Python (Linux)
./python.sh run_isaac_demo.py

# Full path example (Windows)
C:\Users\ASUS\isaac-sim\python.bat C:\Users\ASUS\PycharmProjects\PythonProject1\run_isaac_demo.py
```

### Command Line Options

```bash
# Quick test (100 steps)
python.bat run_isaac_demo.py --test

# Longer simulation
python.bat run_isaac_demo.py --steps 5000

# With pre-trained model
python.bat run_isaac_demo.py --model path/to/axiom_model.pt

# Headless mode (no rendering)
python.bat run_isaac_demo.py --headless

# Disable video recording
python.bat run_isaac_demo.py --no-video

# Adjust discovery threshold
python.bat run_isaac_demo.py --threshold 0.3
```

## Files

### `simulation/isaac_env.py`

Environment wrapper providing Gym-like interface:

```python
from simulation.isaac_env import IsaacGo1Env, Go1Config

# Create environment
config = Go1Config(
    action_delay_frames=2,  # 33ms latency
    friction_range=(0.2, 1.0),  # Ice to Concrete
)

env = IsaacGo1Env(
    headless=False,
    render_resolution=(1920, 1080),
    enable_raytracing=True,
    config=config,
)

# Use environment
obs = env.reset()
for _ in range(1000):
    action = policy.predict(obs)  # 12D joint torques
    obs, reward, done, info = env.step(action)
```

### `run_isaac_demo.py`

Main execution script connecting Axiom-OS to Isaac Sim.

## Key Features

### 1. Chaos Injector (Domain Randomization)

The `ChaosInjector` simulates real-world messiness:

```python
# Friction randomization every 200 steps
friction = np.random.uniform(0.2, 1.0)  # Ice to Concrete

# Action latency simulation
delayed_action = chaos_injector.delay_action(action)  # 2 frames delay

# External perturbations
kick_force = np.random.uniform(-20, 20, size=3)  # Random kicks
```

### 2. Discovery Hook

When prediction error exceeds threshold (e.g. when walking on ice), the demo logs:

```
Axiom-OS Discovery: Friction coefficient changed. Adapting...
   [DISCOVERY #1] Friction coefficient changed to 0.25 (Ice) (error: 0.823)
```

### 3. UPI State Interface

Observations are wrapped in UPIState with physical units:

```python
obs = UPIState(
    values=torch.tensor([...]),  # 43 dimensions
    units=[0, 0, 0, 0, 0],       # Unitless (normalized)
    semantics="go1_observation:pos,ori,vel,joint_states,imu"
)
```

Observation space (43D):
- Base position: 3D
- Base orientation (quaternion): 4D
- Linear velocity: 3D
- Angular velocity: 3D
- Joint positions: 12D
- Joint velocities: 12D
- IMU acceleration: 3D
- IMU gyroscope: 3D

### 4. Ray Traced Rendering

Automatic ray tracing setup for photorealistic output:

```python
settings.set_bool("/rtx/pathtracing/enabled", True)
settings.set_int("/rtx/pathtracing/maxBounces", 4)
```

## Output Files

After running the demo:

```
axiom_go1_demo.mp4              # Demo video (if recording enabled)
outputs/
  discovery/
    discovery_YYYYMMDD_HHMMSS.json  # Discovery log
  demo_summary.json               # Run statistics
```

## Troubleshooting

### "Isaac Sim imports not available"

Ensure you're running with Isaac Sim's Python:
```bash
# Wrong
python run_isaac_demo.py

# Correct
./python.sh run_isaac_demo.py  # Linux
python.bat run_isaac_demo.py    # Windows
```

### "Cannot find Unitree Go1 asset"

The asset path is set to:
```python
go1_usd_path = "/Isaac/Robots/Unitree/Go1/go1.usd"
```

If using a custom Nucleus server:
```python
config = Go1Config(
    nucleus_server="omniverse://your-server/NVIDIA/Assets/Isaac/2023.1.1"
)
```

### Memory Issues

Reduce resolution or disable ray tracing:
```python
env = IsaacGo1Env(
    render_resolution=(1280, 720),
    enable_raytracing=False,
)
```

## Development

### Testing without Isaac Sim

The code supports mock mode for development:

```python
# Will run with mock observations if Isaac Sim not available
from simulation.isaac_env import IsaacGo1Env
env = IsaacGo1Env(headless=True)  # Mock mode
```

### Adding New Chaos Types

Extend `ChaosInjector.inject_chaos()`:

```python
def inject_chaos(self, world, robot, step_idx):
    # Add new chaos type
    if step_idx % 300 == 0:
        self._apply_mass_variation(robot)
    
    # ... existing chaos
```

## References

- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Unitree Go1 Robot](https://www.unitree.com/products/go1)
- [Axiom-OS Core Documentation](../README.md)
