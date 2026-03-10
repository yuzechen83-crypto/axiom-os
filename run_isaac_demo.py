"""
Axiom-OS x NVIDIA Isaac Sim - Digital Twin Demo
Main Execution Loop for Unitree Go1 Control

This script connects Axiom-OS (Brain) to Isaac Sim (Body) to create
a photorealistic Sim-to-Real verification demo.

Usage (inside Isaac Sim Python environment):
    Windows: python.bat run_isaac_demo.py
    Linux:   ./python.sh run_isaac_demo.py

Features:
- Real-time control of Unitree Go1 quadruped
- Domain randomization (friction, latency, external forces)
- Discovery hook for adaptive learning
- Ray-traced video recording (axiom_go1_demo.mp4)

Author: Axiom-OS Simulation Team
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from collections import deque

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set EXP_PATH before any isaac import (Isaac Sim 5.1.0 Script Editor uses os.environ["EXP_PATH"] at import time)
# Isaac Sim 5.1.0: Launcher install = %LOCALAPPDATA%\ov\pkg\isaac-sim-5.1.0, workstation = C:\isaac-sim
def _default_isaac_root():
    env = os.environ.get("ISAAC_PATH") or os.environ.get("CARB_APP_PATH")
    if env:
        return env
    local = os.environ.get("LOCALAPPDATA", "")
    pkg_51 = Path(local) / "ov" / "pkg" / "isaac-sim-5.1.0" if local else None
    if pkg_51 and pkg_51.exists():
        return str(pkg_51.resolve())
    if Path("C:/isaac-sim").exists():
        return "C:/isaac-sim"
    return r"C:\Users\ASUS\isaacsim\_build\windows-x86_64\release"

_isaac_root = _default_isaac_root()
if "EXP_PATH" not in os.environ:
    os.environ["EXP_PATH"] = _isaac_root
if "ISAAC_PATH" not in os.environ:
    os.environ["ISAAC_PATH"] = _isaac_root
if "CARB_APP_PATH" not in os.environ:
    os.environ["CARB_APP_PATH"] = _isaac_root

# Bootstrap Isaac Sim when run as main (required before any omni/isaac imports)
_SIMULATION_APP = None
if __name__ == "__main__":
    try:
        from isaacsim.simulation_app import SimulationApp
        _exp_path = os.environ["EXP_PATH"]
        _SIMULATION_APP = SimulationApp({
            "headless": False,
            "width": 1280,
            "height": 720,
            "renderer": "RayTracedLighting",
            "EXP_PATH": _exp_path,
        })
        print("[AXIOM-OS] Isaac Sim SimulationApp started")
    except ImportError:
        print("[AXIOM-OS] Not in Isaac Sim environment; run with: ./python.sh run_isaac_demo.py")

# Simulation imports (after SimulationApp when running in Isaac)
from simulation.isaac_env import IsaacGo1Env, Go1Config, IS_ISAAC_AVAILABLE

# Axiom-OS imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    from axiom_os.core.upi import UPIState, Units
    from axiom_os.neurons.base import BaseNeuron
    AXIOM_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Axiom-OS not fully available: {e}")
    print("[WARNING] Running with mock controller")
    AXIOM_AVAILABLE = False


class AxiomPolicy:
    """
    Axiom-OS Policy Wrapper.
    
    Loads pre-trained Soft Shell model or uses MPC fallback.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_mpc: bool = True,
        device: str = "cuda" if AXIOM_AVAILABLE and torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.use_mpc = use_mpc
        self.model_path = model_path
        
        # Load model or create MPC controller
        self.model = self._load_model(model_path)
        
        # State for MPC
        self.prev_action = np.zeros(12)
        self.target_height = 0.4
        
        print(f"[AxiomPolicy] Initialized (device: {device})")
        if self.model:
            print("[AxiomPolicy] Using Neural Network controller")
        elif use_mpc:
            print("[AxiomPolicy] Using MPC controller")
        else:
            print("[AxiomPolicy] Using Random controller")
    
    def _load_model(self, model_path: Optional[str]) -> Optional[nn.Module]:
        """Load pre-trained Axiom model."""
        if not AXIOM_AVAILABLE or model_path is None:
            return None
        
        try:
            if os.path.exists(model_path):
                model = torch.load(model_path, map_location=self.device)
                model.eval()
                print(f"[AxiomPolicy] Loaded model from {model_path}")
                return model
        except Exception as e:
            print(f"[AxiomPolicy] Failed to load model: {e}")
        
        return None
    
    def predict(self, obs: UPIState) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            obs: UPIState containing robot observations
            
        Returns:
            Action array (12 joint torques/positions)
        """
        # Extract values from UPIState
        if hasattr(obs, 'values'):
            obs_tensor = obs.values
            if isinstance(obs_tensor, torch.Tensor):
                obs_tensor = obs_tensor.cpu().numpy()
        else:
            obs_tensor = np.array(obs)
        
        if self.model is not None:
            # Neural network inference
            with torch.no_grad():
                obs_torch = torch.tensor(obs_tensor, dtype=torch.float32).to(self.device)
                if obs_torch.dim() == 1:
                    obs_torch = obs_torch.unsqueeze(0)
                action = self.model(obs_torch).cpu().numpy().flatten()
        
        elif self.use_mpc:
            # Simple MPC: maintain height and balance
            action = self._mpc_controller(obs_tensor)
        
        else:
            # Random action for testing
            action = np.random.randn(12) * 5
        
        # Clip action to safe range
        action = np.clip(action, -50, 50)
        
        self.prev_action = action
        return action
    
    def _mpc_controller(self, obs: np.ndarray) -> np.ndarray:
        """
        Simple Model Predictive Control for standing balance.
        
        Args:
            obs: Observation array [pos(3), ori(4), lin_vel(3), ang_vel(3), 
                                     joint_pos(12), joint_vel(12), imu_accel(3), imu_gyro(3)]
        
        Returns:
            Joint torques
        """
        # Extract relevant states
        height = obs[2]  # z-position
        orientation = obs[3:7]  # quaternion
        ang_vel = obs[10:13]  # angular velocity
        joint_pos = obs[13:25]  # joint positions
        joint_vel = obs[25:37]  # joint velocities
        
        # PD control for balance
        # Target: height = 0.4m, upright orientation
        height_error = self.target_height - height
        
        # Simple balance: use joint positions to correct orientation
        # This is a simplified version - real MPC would use trajectory optimization
        kp = 20.0  # Position gain
        kd = 2.0   # Velocity gain
        
        # Generate target joint positions based on orientation error
        # Extract roll and pitch from quaternion (simplified)
        w, x, y, z = orientation
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        # Clamp to valid range for arcsin
        pitch_val = np.clip(2*(w*y - z*x), -1.0, 1.0)
        pitch = np.arcsin(pitch_val)
        
        # Correction torques based on orientation
        target_joints = np.zeros(12)
        
        # Front legs
        target_joints[0:3] = np.array([roll * 0.5, pitch * 0.3, -0.5])  # FL
        target_joints[3:6] = np.array([roll * 0.5, pitch * 0.3, -0.5])  # FR
        # Back legs
        target_joints[6:9] = np.array([roll * 0.5, -pitch * 0.3, -0.5])  # RL
        target_joints[9:12] = np.array([roll * 0.5, -pitch * 0.3, -0.5])  # RR
        
        # PD control
        position_error = target_joints - joint_pos
        torque = kp * position_error - kd * joint_vel
        
        # Add height correction (simplified as standing force)
        height_correction = np.ones(12) * height_error * 10
        
        return torque + height_correction


class DiscoveryLogger:
    """
    Discovery Hook for Axiom-OS.
    
    Monitors prediction errors and logs discoveries when
    the model encounters unexpected conditions.
    """
    
    def __init__(
        self,
        error_threshold: float = 0.5,
        window_size: int = 50,
        log_dir: str = "outputs/discovery"
    ):
        self.error_threshold = error_threshold
        self.window_size = window_size
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.error_history: deque = deque(maxlen=window_size)
        self.discovery_count = 0
        self.discoveries: List[Dict] = []
        
        # Discovery log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"discovery_{timestamp}.json"
        
        print(f"[DiscoveryLogger] Initialized (threshold: {error_threshold})")
    
    def log_step(
        self,
        step: int,
        obs: np.ndarray,
        predicted_obs: np.ndarray,
        chaos_state: Dict
    ) -> Optional[str]:
        """
        Log a step and check for discoveries.
        
        Returns:
            Discovery message if threshold exceeded, None otherwise
        """
        # Calculate prediction error
        error = np.mean((obs - predicted_obs) ** 2)
        self.error_history.append(error)
        
        # Check for discovery
        if error > self.error_threshold:
            self.discovery_count += 1
            
            # Determine cause
            cause = self._infer_cause(chaos_state, error)
            
            discovery = {
                'timestamp': time.time(),
                'step': step,
                'error': float(error),
                'cause': cause,
                'chaos_state': {
                    k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                    for k, v in chaos_state.items()
                }
            }
            self.discoveries.append(discovery)
            
            # Save immediately
            self._save_discoveries()
            
            message = f"[DISCOVERY #{self.discovery_count}] {cause} (error: {error:.3f})"
            return message
        
        return None
    
    def _infer_cause(self, chaos_state: Dict, error: float) -> str:
        """Infer the cause of high prediction error."""
        causes = []
        
        if chaos_state.get('friction_changed', False):
            friction = chaos_state.get('friction_value', 0.8)
            surface = 'Ice' if friction < 0.4 else 'Concrete'
            causes.append(f"Friction coefficient changed to {friction:.2f} ({surface})")
        
        if chaos_state.get('kick_applied', False):
            force = chaos_state.get('kick_force', [0, 0, 0])
            magnitude = np.linalg.norm(force)
            causes.append(f"External force applied (magnitude: {magnitude:.1f}N)")
        
        if chaos_state.get('action_delayed', False):
            causes.append("Action latency detected")
        
        if not causes:
            if error > self.error_threshold * 2:
                causes.append("Unknown high-error condition")
            else:
                causes.append("Model uncertainty")
        
        return "; ".join(causes)
    
    def _save_discoveries(self):
        """Save discoveries to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump({
                'metadata': {
                    'threshold': self.error_threshold,
                    'total_discoveries': self.discovery_count,
                    'window_size': self.window_size,
                },
                'discoveries': self.discoveries
            }, f, indent=2)
    
    def get_stats(self) -> Dict:
        """Get discovery statistics."""
        if not self.error_history:
            return {'mean_error': 0, 'max_error': 0, 'discoveries': 0}
        
        return {
            'mean_error': float(np.mean(self.error_history)),
            'max_error': float(np.max(self.error_history)),
            'discoveries': self.discovery_count,
        }


class VideoRecorder:
    """
    Video recorder for Isaac Sim with ray tracing.
    Uses viewport capture when running inside Isaac Sim, else collects env.render() frames.
    """
    
    def __init__(
        self,
        output_path: str = "axiom_go1_demo.mp4",
        fps: int = 30,
        resolution: tuple = (1920, 1080)
    ):
        self.output_path = str(Path(output_path).resolve())
        self.fps = fps
        self.resolution = resolution
        self.frames: List[np.ndarray] = []
        self._temp_dir: Optional[Path] = None
        self._frame_idx = 0
        self._use_viewport_capture = False
        if IS_ISAAC_AVAILABLE:
            try:
                from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                self._get_active_viewport = get_active_viewport
                self._capture_viewport_to_file = capture_viewport_to_file
                self._use_viewport_capture = True
            except Exception:
                pass
        print(f"[VideoRecorder] Output: {self.output_path} @ {fps}fps (viewport_capture={self._use_viewport_capture})")
    
    def record_frame(self, env: IsaacGo1Env):
        """Capture a frame from the environment (viewport or env.render)."""
        if self._use_viewport_capture and hasattr(self, "_capture_viewport_to_file"):
            try:
                vp = self._get_active_viewport()
                if vp is None:
                    return
                self._temp_dir = self._temp_dir or Path(self.output_path).parent / "viewport_frames"
                self._temp_dir.mkdir(parents=True, exist_ok=True)
                path = str(self._temp_dir / f"frame_{self._frame_idx:05d}.png")
                self._capture_viewport_to_file(vp, path)
                self._frame_idx += 1
                try:
                    from PIL import Image
                    img = np.array(Image.open(path))
                    self.frames.append(img)
                except Exception:
                    pass
            except Exception:
                pass
            return
        frame = env.render(mode="rgb") if hasattr(env, "render") else None
        if frame is not None:
            self.frames.append(frame)
    
    def save(self):
        """Save recorded frames to video file (axiom_go1_demo.mp4)."""
        if not self.frames:
            print("[VideoRecorder] No frames to save")
            return
        try:
            import cv2
            h, w = self.resolution[1], self.resolution[0]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            for frame in self.frames:
                if not isinstance(frame, np.ndarray):
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if (frame.shape[1], frame.shape[0]) != (w, h):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
            writer.release()
            print(f"[VideoRecorder] Saved {len(self.frames)} frames to {self.output_path}")
        except ImportError:
            self._save_frames_as_images()
        except Exception as e:
            print(f"[VideoRecorder] Failed to save video: {e}")
    
    def _save_frames_as_images(self):
        """Fallback: save frames as individual images."""
        try:
            from PIL import Image
        except ImportError:
            print("[VideoRecorder] No PIL/OpenCV; skipping save")
            return
        frames_dir = Path(self.output_path).stem + "_frames"
        Path(frames_dir).mkdir(exist_ok=True)
        for i, frame in enumerate(self.frames):
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
                img.save(f"{frames_dir}/frame_{i:05d}.png")
        print(f"[VideoRecorder] Saved {len(self.frames)} frames to {frames_dir}/")


def setup_raytracing_env():
    """Setup environment for ray-traced rendering."""
    if not IS_ISAAC_AVAILABLE:
        return
    
    try:
        import carb
        settings = carb.settings.get_settings()
        
        # Ray tracing settings
        settings.set_bool("/rtx/pathtracing/enabled", True)
        settings.set_int("/rtx/pathtracing/maxSamplesPerFrame", 4)
        settings.set_int("/rtx/pathtracing/maxBounces", 4)
        settings.set_int("/rtx/pathtracing/aaSamples", 4)
        
        # Lighting
        settings.set_float("/rtx/scene/sky_intensity", 1.0)
        settings.set_float("/rtx/scene/sun_intensity", 2.0)
        
        # Shadows
        settings.set_bool("/rtx/shadows/enabled", True)
        
        print("[Setup] Ray tracing enabled")
        
    except Exception as e:
        print(f"[Setup] Failed to enable ray tracing: {e}")


def run_demo(
    num_steps: int = 2000,
    model_path: Optional[str] = None,
    use_mpc: bool = True,
    record_video: bool = True,
    headless: bool = False,
    discovery_threshold: float = 0.5,
):
    """
    Main demo execution loop.
    
    Args:
        num_steps: Number of simulation steps
        model_path: Path to pre-trained Axiom model
        use_mpc: Use MPC if no model provided
        record_video: Enable video recording
        headless: Run without rendering
        discovery_threshold: Error threshold for discovery
    """
    print("=" * 70)
    print("Axiom-OS x NVIDIA Isaac Sim - Digital Twin Demo")
    print("Unitree Go1 Quadruped Control")
    print("=" * 70)
    
    if not IS_ISAAC_AVAILABLE:
        print("\n[WARNING] Isaac Sim not available - running in MOCK mode")
        print("[WARNING] To run with Isaac Sim, use: ./python.sh run_isaac_demo.py\n")
    
    # Setup ray tracing
    setup_raytracing_env()
    
    # Initialize environment
    print("\n[1/5] Initializing IsaacGo1Env...")
    config = Go1Config(
        action_delay_frames=2,  # ~33ms latency
        friction_range=(0.2, 1.0),  # Ice to Concrete
    )
    env = IsaacGo1Env(
        headless=headless,
        render_resolution=(1920, 1080),
        enable_raytracing=True,
        config=config,
    )
    
    # Initialize Axiom policy
    print("[2/5] Loading Axiom-OS policy...")
    policy = AxiomPolicy(
        model_path=model_path,
        use_mpc=use_mpc,
    )
    
    # Initialize discovery logger
    print("[3/5] Initializing Discovery logger...")
    discovery = DiscoveryLogger(
        error_threshold=discovery_threshold,
        log_dir="outputs/discovery"
    )
    
    # Initialize video recorder
    recorder = None
    if record_video and not headless:
        print("[4/5] Initializing Video recorder...")
        recorder = VideoRecorder(
            output_path="axiom_go1_demo.mp4",
            fps=30,
            resolution=(1920, 1080)
        )
    else:
        print("[4/5] Video recording disabled")
    
    # Main loop
    print(f"[5/5] Running simulation for {num_steps} steps...")
    print("-" * 70)
    
    obs = env.reset()
    episode_reward = 0
    steps_completed = 0
    
    try:
        for step in range(num_steps):
            # Axiom inference
            action = policy.predict(obs)
            
            # Store observation for prediction error calculation
            obs_before = obs.values.cpu().numpy() if hasattr(obs, 'values') else np.array(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps_completed += 1
            
            # Get observation after step
            obs_after = obs.values.cpu().numpy() if hasattr(obs, 'values') else np.array(obs)
            
            # Discovery: check prediction error
            # Simple prediction: assume next obs = current obs + small change
            predicted_obs = obs_before  # Simplified prediction
            discovery_msg = discovery.log_step(
                step=step,
                obs=obs_after,
                predicted_obs=predicted_obs,
                chaos_state=info.get('chaos', {})
            )
            
            if discovery_msg:
                # Master Prompt: exact log when error > threshold (e.g. walking on ice)
                if info.get("chaos", {}).get("friction_changed"):
                    print("\nAxiom-OS Discovery: Friction coefficient changed. Adapting...")
                print(f"   [DISCOVERY] {discovery_msg}\n")
            
            # Record frame
            if recorder and step % 2 == 0:  # Record at 30fps (assuming 60Hz sim)
                recorder.record_frame(env)
            
            # Progress logging
            if step % 100 == 0:
                stats = discovery.get_stats()
                print(f"Step {step:5d} | Reward: {episode_reward:8.2f} | "
                      f"Discoveries: {stats['discoveries']} | "
                      f"Avg Error: {stats['mean_error']:.4f}")
            
            # Check termination
            if done:
                print(f"\n[Episode terminated at step {step}]")
                print(f"  Reason: Robot fell or unstable")
                break
            
            # Small delay for real-time factor (optional)
            # time.sleep(1/60)
    
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("Simulation Complete")
        print("=" * 70)
        print(f"Steps completed: {steps_completed}")
        print(f"Total reward: {episode_reward:.2f}")
        
        stats = discovery.get_stats()
        print(f"Discoveries triggered: {stats['discoveries']}")
        print(f"Average prediction error: {stats['mean_error']:.4f}")
        print(f"Max prediction error: {stats['max_error']:.4f}")
        
        # Save video
        if recorder:
            print("\nSaving video...")
            recorder.save()
        
        # Save final stats
        env.close()
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'steps_completed': steps_completed,
            'total_reward': float(episode_reward),
            'discoveries': stats['discoveries'],
            'mean_error': stats['mean_error'],
            'max_error': stats['max_error'],
        }
        
        summary_path = Path("outputs/demo_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        print(f"Discovery log: {discovery.log_file}")
        
        return summary


def main():
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(
        description="Axiom-OS x NVIDIA Isaac Sim - Digital Twin Demo"
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=2000,
        help='Number of simulation steps (default: 2000)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to pre-trained Axiom model'
    )
    
    parser.add_argument(
        '--mpc', '-mpc',
        action='store_true',
        default=True,
        help='Use MPC controller if no model provided'
    )
    
    parser.add_argument(
        '--no-mpc',
        action='store_true',
        help='Disable MPC (use random controller)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without rendering'
    )
    
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Disable video recording'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Discovery error threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--test', '-test',
        action='store_true',
        help='Run quick test (100 steps)'
    )
    
    # Ignore Kit/Omniverse args (e.g. --/app/...) so they don't cause parse errors
    _argv = [a for a in sys.argv[1:] if not (a.startswith("--/") and "/" in a[3:])]
    args = parser.parse_args(_argv)
    
    # Quick test mode
    if args.test:
        args.steps = 100
        args.headless = False
        args.no_video = True
        print("[TEST MODE] Running quick test (100 steps)")
    
    # Run demo
    run_demo(
        num_steps=args.steps,
        model_path=args.model,
        use_mpc=not args.no_mpc,
        record_video=not args.no_video,
        headless=args.headless,
        discovery_threshold=args.threshold,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if _SIMULATION_APP is not None:
            _SIMULATION_APP.close()
            print("[AXIOM-OS] Isaac Sim closed.")
