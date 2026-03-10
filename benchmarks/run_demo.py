"""Quick demo of Axiom-OS Gym Adapter"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.gym_adapter import AxiomAgent, run_episode, plot_episode
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make('Acrobot-v1')

# Create agent with better parameters for swing-up
agent = AxiomAgent(
    env_name='Acrobot-v1',
    horizon_steps=50,
    n_samples=500,
    use_adaptation=True,
)

print('=' * 60)
print('Axiom-OS Acrobot Demo (300 steps)')
print('=' * 60)

traj = run_episode(env, agent, max_steps=300, render=False, verbose=True)

print('\n' + '=' * 60)
print('Results')
print('=' * 60)
print(f'Episode Length: {traj["length"]}')
print(f'Total Reward: {traj["total_reward"]:.2f}')
print(f'Success: {traj["success"]}')

# Check final state
final_q = traj['states_q'][-1]
final_err = np.sqrt(np.sum((np.array(final_q) - np.pi) ** 2))
print(f'Final angles: q1={final_q[0]:.3f}, q2={final_q[1]:.3f}')
print(f'Target: pi={np.pi:.3f}')
print(f'Final error: {final_err:.3f} rad')

# Plot
plot_episode(traj, save_path='benchmarks/demo_result.png')
print('\nPlot saved to: benchmarks/demo_result.png')

env.close()
