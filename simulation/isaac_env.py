"""
IsaacGo1Env - NVIDIA Isaac Sim Environment Wrapper for Axiom-OS

This module wraps the complex Omniverse API into a simple interface
compatible with Axiom-OS for controlling Unitree Go1 quadruped robot.

Compatible with Isaac Sim 4.0+ / 5.1.0 (isaacsim.* namespace; 5.1.0 prefers Core Experimental API).

Usage (inside Isaac Sim Python environment):
    ./python.sh simulation/isaac_env.py

Author: Axiom-OS Simulation Team
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque
import time

# Handle Isaac Sim imports gracefully - Try new API first, then legacy
try:
    # Isaac Sim 4.0+ API
    from isaacsim.core.api import World
    from isaacsim.core.prims import XFormPrim
    try:
        from isaacsim.core.prims import Articulation as IsaacArticulation
    except ImportError:
        IsaacArticulation = None
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.sensors.physics import IMUSensor
    
    # Try to import robot-specific modules (optional)
    try:
        from isaacsim.robots.articulations import UnitreeGo1
        HAS_UNITREE_GO1 = True
    except ImportError:
        HAS_UNITREE_GO1 = False
    
    IS_ISAAC_AVAILABLE = True
    ISAAC_API_VERSION = "5.1"  # 4.0+ / 5.1.0 isaacsim.* namespace
    
except ImportError as e1:
    # Legacy API (Isaac Sim 2022.2/2023.1)
    try:
        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.sensor import IMUSensor
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.viewports import set_camera_view
        try:
            from omni.isaac.core.prims import Articulation as IsaacArticulation
        except ImportError:
            from omni.isaac.core.robots import Robot as IsaacArticulation
        try:
            from omni.isaac.quadruped.robots import UnitreeGo1
            HAS_UNITREE_GO1 = True
        except ImportError:
            HAS_UNITREE_GO1 = False
        
        IS_ISAAC_AVAILABLE = True
        ISAAC_API_VERSION = "legacy"
        
    except ImportError as e2:
        print(f"[WARNING] Isaac Sim imports not available: {e2}")
        print("[WARNING] Running in mock mode for development/testing")
        IS_ISAAC_AVAILABLE = False
        ISAAC_API_VERSION = "none"
        HAS_UNITREE_GO1 = False
        IsaacArticulation = None

# Axiom-OS imports with graceful fallback
try:
    import torch
    from axiom_os.core.upi import UPIState, Units
    from axiom_os.neurons.base import BaseNeuron
    AXIOM_AVAILABLE = True
except ImportError:
    print("[WARNING] Axiom-OS not available, using mock UPI")
    AXIOM_AVAILABLE = False


@dataclass
class Go1Config:
    """Configuration for Unitree Go1 robot."""
    # Robot parameters
    num_joints: int = 12
    default_position: Tuple[float, float, float] = (0.0, 0.0, 0.4)
    default_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    
    # Physics parameters
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float = 1.0 / 60.0  # 60 Hz simulation
    
    # Sensor noise (Gaussian std dev)
    position_noise: float = 0.005  # meters
    velocity_noise: float = 0.02   # m/s
    orientation_noise: float = 0.01  # rad
    imu_accel_noise: float = 0.1   # m/s^2
    imu_gyro_noise: float = 0.05   # rad/s
    joint_pos_noise: float = 0.01  # rad
    joint_vel_noise: float = 0.05  # rad/s
    
    # Action delay (frames)
    action_delay_frames: int = 2  # ~33ms at 60Hz
    
    # Domain randomization ranges
    friction_range: Tuple[float, float] = (0.2, 1.0)  # Ice to Concrete
    mass_range: Tuple[float, float] = (0.9, 1.1)  # +/- 10%
    
    # Asset paths - try multiple possible locations
    go1_usd_paths: List[str] = None
    nucleus_servers: List[str] = None
    
    def __post_init__(self):
        if self.go1_usd_paths is None:
            self.go1_usd_paths = [
                "/Isaac/Robots/Unitree/Go1/go1.usd",
                "/Isaac/Robots/Unitree/go1.usd",
                "/Assets/Robots/Unitree/Go1/go1.usd",
            ]
        if self.nucleus_servers is None:
            self.nucleus_servers = [
                "omniverse://localhost/NVIDIA/Assets/Isaac/4.0",
                "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1",
                "omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.1",
            ]


class ChaosInjector:
    """
    Domain Randomization and Chaos Injection for Sim-to-Real verification.
    
    Implements:
    - Friction randomization (Ice to Concrete)
    - Action latency simulation
    - External perturbation forces
    """
    
    def __init__(self, config: Go1Config):
        self.config = config
        self.step_idx = 0
        self.current_friction = 0.8  # Default concrete
        
        # Action delay buffer
        self.action_buffer: deque = deque(maxlen=config.action_delay_frames + 1)
        for _ in range(config.action_delay_frames):
            self.action_buffer.append(np.zeros(config.num_joints))
    
    def inject_chaos(self, world, robot, step_idx: int) -> Dict[str, any]:
        """
        Inject chaos into the simulation for robustness testing.
        
        Args:
            world: Isaac Sim World object
            robot: Robot object
            step_idx: Current simulation step
            
        Returns:
            Dict containing chaos state information
        """
        self.step_idx = step_idx
        chaos_state = {
            'friction_changed': False,
            'friction_value': self.current_friction,
            'kick_applied': False,
            'kick_force': np.zeros(3),
        }
        
        # 1. Friction Shift every 200 steps
        if step_idx % 200 == 0 and step_idx > 0:
            self.current_friction = np.random.uniform(
                *self.config.friction_range
            )
            self._update_ground_friction(world, self.current_friction)
            chaos_state['friction_changed'] = True
            chaos_state['friction_value'] = self.current_friction
            surface = 'Ice' if self.current_friction < 0.4 else 'Rough' if self.current_friction > 0.8 else 'Concrete'
            print(f"[ChaosInjector] Friction changed to {self.current_friction:.2f} ({surface})")
        
        # 2. External Force "Kick" every 500 steps
        if step_idx % 500 == 100:  # Offset from friction change
            kick_force = np.random.uniform(-20, 20, size=3)
            kick_force[2] = abs(kick_force[2]) + 5  # Mostly upward
            self._apply_kick(robot, kick_force)
            chaos_state['kick_applied'] = True
            chaos_state['kick_force'] = kick_force
            print(f"[ChaosInjector] Kick applied: {kick_force}")
        
        return chaos_state
    
    def _update_ground_friction(self, world, friction: float):
        """Update ground plane friction coefficient."""
        if not IS_ISAAC_AVAILABLE:
            return
        try:
            # Get ground plane and update friction
            ground = world.scene.get_object("ground_plane")
            if ground and hasattr(ground, 'get_applied_physics_material'):
                material = ground.get_applied_physics_material()
                if material:
                    material.set_friction_combine_mode("average")
                    material.set_dynamic_friction(friction)
                    material.set_static_friction(friction)
        except Exception as e:
            print(f"[ChaosInjector] Failed to update friction: {e}")
    
    def _apply_kick(self, robot, force: np.ndarray):
        """Apply external force to robot trunk."""
        if not IS_ISAAC_AVAILABLE:
            return
        try:
            # Apply force to trunk/base link
            if hasattr(robot, 'base_link') and robot.base_link is not None:
                robot.base_link.apply_force(force)
            elif hasattr(robot, '_base_link') and robot._base_link is not None:
                robot._base_link.apply_force(force)
        except Exception as e:
            print(f"[ChaosInjector] Failed to apply kick: {e}")
    
    def delay_action(self, action: np.ndarray) -> np.ndarray:
        """
        Implement action delay to simulate communication lag.
        
        Args:
            action: Current action (joint torques/positions)
            
        Returns:
            Delayed action from buffer
        """
        self.action_buffer.append(action.copy())
        return self.action_buffer.popleft()


class IsaacGo1Env:
    """
    Isaac Sim Environment for Unitree Go1 Quadruped.
    
    Provides a Gym-like interface compatible with Axiom-OS:
    - reset(): Reset robot to initial pose
    - step(action): Execute one simulation step
    - get_obs(): Return UPIState with sensor readings
    """
    
    def __init__(
        self,
        headless: bool = False,
        render_resolution: Tuple[int, int] = (1920, 1080),
        enable_raytracing: bool = True,
        config: Optional[Go1Config] = None,
    ):
        self.config = config or Go1Config()
        self.headless = headless
        self.render_resolution = render_resolution
        self.enable_raytracing = enable_raytracing
        
        self.world: Optional[World] = None
        self.robot = None
        self.imu: Optional[IMUSensor] = None
        self.camera = None
        
        self.chaos_injector = ChaosInjector(self.config)
        self.step_count = 0
        
        # Observation history for discovery
        self.obs_history: deque = deque(maxlen=100)
        self.prediction_errors: List[float] = []
        
        # Initialize if Isaac is available
        if IS_ISAAC_AVAILABLE:
            self._initialize_isaac()
        else:
            print("[IsaacGo1Env] Running in MOCK mode")
    
    def _initialize_isaac(self):
        """Initialize Isaac Sim world and robot."""
        print(f"[IsaacGo1Env] Initializing with Isaac Sim API {ISAAC_API_VERSION}")
        
        # Enable required extensions
        try:
            enable_extension("isaacsim.core.api")
            enable_extension("isaacsim.sensors.physics")
        except:
            try:
                enable_extension("omni.isaac.core")
                enable_extension("omni.isaac.sensor")
            except:
                pass  # Extensions might already be enabled
        
        # Create world
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.dt,
            rendering_dt=self.config.dt,
        )
        
        # Setup physics scene
        self.world.scene.add_default_ground_plane(
            z_position=0.0,
            name="ground_plane",
            static_friction=0.8,
            dynamic_friction=0.8,
        )
        
        # Load Go1 robot
        self._load_robot()
        
        # Setup camera
        self._setup_camera()
        
        # Setup rendering
        if self.enable_raytracing:
            self._setup_raytracing()
        
        print("[IsaacGo1Env] Isaac Sim initialized successfully")
    
    def _load_robot(self):
        """Load Unitree Go1 robot from USD asset."""
        try:
            # Try to find the asset
            assets_root = None
            try:
                assets_root = get_assets_root_path()
            except:
                pass
            
            go1_path = None
            if assets_root:
                for path in self.config.go1_usd_paths:
                    test_path = assets_root + path
                    # We can't actually check if path exists in Omniverse
                    # So just use the first one
                    go1_path = test_path
                    break
            
            if not go1_path:
                # Fallback to direct Nucleus path
                for server in self.config.nucleus_servers:
                    for path in self.config.go1_usd_paths:
                        test_path = server + path
                        go1_path = test_path
                        break
                    if go1_path:
                        break
            
            print(f"[IsaacGo1Env] Loading Go1 from: {go1_path}")
            
            # Add robot to stage (built-in Nucleus asset path)
            add_reference_to_stage(usd_path=go1_path, prim_path="/World/Go1")
            
            # Create robot instance: prefer Articulation (isaacsim/omni API), then UnitreeGo1, then generic
            if IsaacArticulation is not None:
                try:
                    self.robot = IsaacArticulation(
                        prim_paths_expr="/World/Go1",
                        name="go1",
                    )
                    self.world.scene.add(self.robot)
                except Exception as e:
                    print(f"[IsaacGo1Env] Articulation failed: {e}")
                    self._create_generic_robot()
            elif HAS_UNITREE_GO1:
                try:
                    self.robot = UnitreeGo1(
                        prim_path="/World/Go1",
                        name="go1",
                        position=self.config.default_position,
                        orientation=self.config.default_orientation,
                    )
                    self.world.scene.add(self.robot)
                except Exception as e:
                    print(f"[IsaacGo1Env] UnitreeGo1 failed: {e}")
                    self._create_generic_robot()
            else:
                self._create_generic_robot()
            
            # Setup IMU - try different sensor APIs
            try:
                self.imu = IMUSensor(
                    prim_path="/World/Go1/trunk",
                    name="go1_imu",
                    frequency=100,
                )
            except Exception as e:
                print(f"[IsaacGo1Env] IMU setup failed: {e}")
                self.imu = None
            
            print("[IsaacGo1Env] Robot loaded successfully")
            
        except Exception as e:
            print(f"[IsaacGo1Env] Failed to load robot: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_generic_robot(self):
        """Create a generic robot wrapper if Articulation/UnitreeGo1 unavailable."""
        if IsaacArticulation is not None:
            try:
                self.robot = IsaacArticulation(
                    prim_paths_expr="/World/Go1",
                    name="go1",
                )
                self.world.scene.add(self.robot)
                return
            except Exception:
                pass
        try:
            from omni.isaac.core.robots import Robot
            self.robot = Robot(
                prim_path="/World/Go1",
                name="go1",
                position=self.config.default_position,
                orientation=self.config.default_orientation,
            )
        except Exception:
            self.robot = XFormPrim(
                prim_path="/World/Go1",
                name="go1",
                position=self.config.default_position,
            )
        self.world.scene.add(self.robot)
    
    def _setup_camera(self):
        """Setup tracking camera for robot (smooth follow for demo video)."""
        self._camera_offset_eye = np.array([2.5, 2.5, 1.8])   # Camera position offset from robot
        self._camera_offset_target = np.array([0.0, 0.0, 0.2])  # Look-at offset (above base)
        try:
            if ISAAC_API_VERSION == "legacy":
                from omni.isaac.core.utils.viewports import set_camera_view
                set_camera_view(
                    eye=self._camera_offset_eye.copy(),
                    target=np.array([0.0, 0.0, 0.4]) + self._camera_offset_target,
                    camera_prim_path="/OmniverseKit_Persp",
                )
            print("[IsaacGo1Env] Camera setup complete (tracking enabled)")
        except Exception as e:
            print(f"[IsaacGo1Env] Camera setup failed: {e}")
    
    def _update_camera_track(self):
        """Update camera to smoothly track the robot (call each step for demo video)."""
        if not IS_ISAAC_AVAILABLE or self.robot is None:
            return
        try:
            if hasattr(self.robot, "get_world_pose"):
                pos, _ = self.robot.get_world_pose()
                # Handle both (3,) and (1,3) return shapes
                if pos.ndim == 2:
                    pos = pos[0]
                target = pos + self._camera_offset_target
                eye = pos + self._camera_offset_eye
                if ISAAC_API_VERSION == "legacy":
                    from omni.isaac.core.utils.viewports import set_camera_view
                    set_camera_view(
                        eye=eye,
                        target=target,
                        camera_prim_path="/OmniverseKit_Persp",
                    )
        except Exception:
            pass
    
    def _setup_raytracing(self):
        """Enable ray tracing for photorealistic rendering."""
        try:
            import carb
            
            # Get render settings
            settings = carb.settings.get_settings()
            
            # Enable ray tracing
            settings.set_bool("/rtx/pathtracing/enabled", True)
            settings.set_int("/rtx/pathtracing/maxSamplesPerFrame", 4)
            settings.set_int("/rtx/pathtracing/maxBounces", 4)
            settings.set_int("/rtx/pathtracing/aaSamples", 4)
            
            # Enable shadows and reflections
            settings.set_bool("/rtx/shadows/enabled", True)
            settings.set_bool("/rtx/reflections/enabled", True)
            
            # Set lighting
            settings.set_float("/rtx/scene/sky_intensity", 1.0)
            settings.set_float("/rtx/scene/sun_intensity", 2.0)
            
            print("[IsaacGo1Env] Ray tracing enabled")
            
        except Exception as e:
            print(f"[IsaacGo1Env] Ray tracing setup failed: {e}")
    
    def reset(self) -> "UPIState":
        """
        Reset environment to initial state.
        
        Returns:
            UPIState: Initial observation
        """
        self.step_count = 0
        self.obs_history.clear()
        self.prediction_errors.clear()
        
        if not IS_ISAAC_AVAILABLE:
            # Mock mode: return dummy observation
            return self._create_mock_obs()
        
        # Reset world
        self.world.reset()
        
        # Reset robot pose to (0, 0, 0.4) as per Master Prompt
        pos_arr = np.array(self.config.default_position, dtype=np.float64)
        ori_arr = np.array(self.config.default_orientation, dtype=np.float64)
        if hasattr(self.robot, 'set_world_poses'):
            try:
                self.robot.set_world_poses(positions=pos_arr.reshape(1, 3), orientations=ori_arr.reshape(1, 4))
            except TypeError:
                self.robot.set_world_poses(positions=pos_arr.reshape(1, 3))
        elif hasattr(self.robot, 'set_world_pose'):
            self.robot.set_world_pose(position=pos_arr, orientation=ori_arr)
        
        if hasattr(self.robot, 'set_linear_velocity'):
            self.robot.set_linear_velocity(np.zeros(3))
        if hasattr(self.robot, 'set_angular_velocity'):
            self.robot.set_angular_velocity(np.zeros(3))
        
        # Reset joint states
        if hasattr(self.robot, 'set_joint_positions'):
            try:
                self.robot.set_joint_positions(np.zeros(self.config.num_joints))
                self.robot.set_joint_velocities(np.zeros(self.config.num_joints))
            except:
                pass
        
        # Reset chaos injector
        self.chaos_injector = ChaosInjector(self.config)
        
        # Step once to stabilize
        self.world.step(render=not self.headless)
        
        return self.get_obs()
    
    def get_obs(self) -> "UPIState":
        """
        Get current observation with sensor noise.
        
        Returns:
            UPIState containing:
            - Base position (3D)
            - Base orientation (quaternion, 4D)
            - Base linear velocity (3D)
            - Base angular velocity (3D)
            - Joint positions (12D)
            - Joint velocities (12D)
            - IMU acceleration (3D)
            - IMU gyroscope (3D)
        """
        if not IS_ISAAC_AVAILABLE:
            return self._create_mock_obs()
        
        try:
            # Get robot state
            pos = np.zeros(3)
            ori = np.array([1.0, 0.0, 0.0, 0.0])
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)
            
            if hasattr(self.robot, 'get_world_pose'):
                try:
                    pos, ori = self.robot.get_world_pose()
                    if pos.ndim == 2:
                        pos, ori = pos[0], ori[0]
                except Exception:
                    pass
            
            if hasattr(self.robot, 'get_linear_velocity'):
                try:
                    lin_vel = self.robot.get_linear_velocity()
                except:
                    pass
            
            if hasattr(self.robot, 'get_angular_velocity'):
                try:
                    ang_vel = self.robot.get_angular_velocity()
                except:
                    pass
            
            # Get joint states
            joint_pos = np.zeros(self.config.num_joints)
            joint_vel = np.zeros(self.config.num_joints)
            
            if hasattr(self.robot, 'get_joint_positions'):
                try:
                    jp = self.robot.get_joint_positions()
                    joint_pos = np.array(jp).flatten()[: self.config.num_joints]
                except Exception:
                    pass
            
            if hasattr(self.robot, 'get_joint_velocities'):
                try:
                    jv = self.robot.get_joint_velocities()
                    joint_vel = np.array(jv).flatten()[: self.config.num_joints]
                except Exception:
                    pass
            
            # Get IMU data
            imu_accel = lin_vel.copy()  # Fallback
            imu_gyro = ang_vel.copy()
            
            if self.imu:
                try:
                    imu_data = self.imu.get_current_frame()
                    if imu_data:
                        if hasattr(imu_data, 'accel'):
                            imu_accel = np.array(imu_data.accel)
                        if hasattr(imu_data, 'gyro'):
                            imu_gyro = np.array(imu_data.gyro)
                except:
                    pass
            
            # Add Gaussian noise (Sim-to-Real)
            pos += np.random.normal(0, self.config.position_noise, 3)
            ori += np.random.normal(0, self.config.orientation_noise, 4)
            ori = ori / (np.linalg.norm(ori) + 1e-8)  # Renormalize quaternion
            lin_vel += np.random.normal(0, self.config.velocity_noise, 3)
            ang_vel += np.random.normal(0, self.config.velocity_noise, 3)
            joint_pos += np.random.normal(0, self.config.joint_pos_noise, self.config.num_joints)
            joint_vel += np.random.normal(0, self.config.joint_vel_noise, self.config.num_joints)
            imu_accel += np.random.normal(0, self.config.imu_accel_noise, 3)
            imu_gyro += np.random.normal(0, self.config.imu_gyro_noise, 3)
            
            # Concatenate all observations
            obs_values = np.concatenate([
                pos,           # 3
                ori,           # 4
                lin_vel,       # 3
                ang_vel,       # 3
                joint_pos,     # 12
                joint_vel,     # 12
                imu_accel,     # 3
                imu_gyro,      # 3
            ])  # Total: 43 dimensions
            
        except Exception as e:
            print(f"[IsaacGo1Env] Error getting observation: {e}")
            obs_values = np.zeros(43)
        
        # Create UPIState
        if AXIOM_AVAILABLE:
            # All values are dimensionless in normalized observation space
            obs = UPIState(
                values=torch.tensor(obs_values, dtype=torch.float32),
                units=[0, 0, 0, 0, 0],  # Unitless for neural network input
                semantics="go1_observation:pos,ori,vel,joint_states,imu"
            )
        else:
            obs = MockUPIState(obs_values)
        
        # Store in history
        self.obs_history.append(obs_values)
        
        return obs
    
    def _create_mock_obs(self) -> "UPIState":
        """Create mock observation for testing without Isaac."""
        obs_values = np.random.randn(43).astype(np.float32)
        if AXIOM_AVAILABLE:
            return UPIState(
                values=torch.tensor(obs_values),
                units=[0, 0, 0, 0, 0],
                semantics="mock_observation"
            )
        return MockUPIState(obs_values)
    
    def step(self, action: np.ndarray) -> Tuple["UPIState", float, bool, Dict]:
        """
        Execute one simulation step.
        
        Args:
            action: Joint torques or positions (12D array)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1
        
        if not IS_ISAAC_AVAILABLE:
            # Mock mode
            obs = self._create_mock_obs()
            reward = 0.0
            done = False
            info = {'mock': True}
            return obs, reward, done, info
        
        # Apply chaos injection
        chaos_state = self.chaos_injector.inject_chaos(
            self.world, self.robot, self.step_count
        )
        
        # Apply action delay
        delayed_action = self.chaos_injector.delay_action(action)
        
        # Apply action to robot
        try:
            if hasattr(self.robot, 'set_joint_efforts'):
                self.robot.set_joint_efforts(delayed_action)
            elif hasattr(self.robot, 'apply_action'):
                self.robot.apply_action(delayed_action)
        except Exception as e:
            print(f"[IsaacGo1Env] Failed to apply action: {e}")
        
        # Step physics
        self.world.step(render=not self.headless)
        
        # Smooth camera tracking for demo video
        self._update_camera_track()
        
        # Get observation
        obs = self.get_obs()
        
        # Compute reward (simple height-based)
        try:
            if hasattr(self.robot, 'get_world_pose'):
                pos, _ = self.robot.get_world_pose()
                height_reward = pos[2]  # Reward for staying upright
            else:
                height_reward = 0.4
        except:
            height_reward = 0.0
        
        # Check termination
        done = height_reward < 0.2  # Fall detection
        
        # Info dict
        info = {
            'step': self.step_count,
            'chaos': chaos_state,
            'action_delayed': not np.array_equal(action, delayed_action),
        }
        
        return obs, height_reward, done, info
    
    def render(self, mode: str = "rgb") -> Optional[np.ndarray]:
        """
        Render current frame.
        
        Args:
            mode: "rgb" or "depth"
            
        Returns:
            Rendered frame as numpy array
        """
        if not IS_ISAAC_AVAILABLE or self.headless:
            return None
        
        # Rendering is handled by Isaac Sim viewport
        return None
    
    def close(self):
        """Clean up resources."""
        if IS_ISAAC_AVAILABLE and self.world is not None:
            try:
                self.world.stop()
            except:
                pass
            print("[IsaacGo1Env] Environment closed")
    
    def get_prediction_error(self, expected_obs: np.ndarray) -> float:
        """
        Calculate prediction error for discovery.
        
        Args:
            expected_obs: Expected observation from Axiom model
            
        Returns:
            Mean squared error
        """
        if len(self.obs_history) < 2:
            return 0.0
        
        actual_obs = self.obs_history[-1]
        error = np.mean((actual_obs - expected_obs) ** 2)
        self.prediction_errors.append(error)
        
        return error


class MockUPIState:
    """Mock UPIState for testing without Axiom-OS."""
    def __init__(self, values: np.ndarray):
        self.values = values
        self.units = [0, 0, 0, 0, 0]
        self.semantics = "mock"


# Check torch availability
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass


def test_env():
    """Test environment wrapper."""
    print("=" * 60)
    print("Testing IsaacGo1Env")
    print("=" * 60)
    
    env = IsaacGo1Env(headless=True)
    
    # Test reset
    print("\n[TEST] Reset environment...")
    obs = env.reset()
    print(f"  Observation shape: {obs.values.shape if hasattr(obs, 'values') else len(obs)}")
    
    # Test step
    print("\n[TEST] Step simulation...")
    action = np.random.randn(12) * 10  # Random torques
    obs, reward, done, info = env.step(action)
    print(f"  Reward: {reward:.4f}, Done: {done}")
    print(f"  Chaos state: {info.get('chaos', {})}")
    
    # Test multiple steps with chaos
    print("\n[TEST] Running 250 steps (to trigger chaos)...")
    for i in range(250):
        action = np.random.randn(12) * 5
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"  Episode terminated at step {i}")
            break
    
    print("\n[TEST] Complete!")
    env.close()


if __name__ == "__main__":
    test_env()
