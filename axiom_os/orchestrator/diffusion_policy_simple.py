# -*- coding: utf-8 -*-
"""
Diffusion Policy Controller for Axiom-OS v4.0 - Simplified Version
SOTA: Generative Control via Denoising Diffusion
"""

from typing import Optional, Tuple, Dict, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class DiffusionPolicyConfig:
    """Configuration for Diffusion Policy"""
    obs_dim: int = 4
    obs_horizon: int = 2
    action_dim: int = 1
    action_horizon: int = 16
    num_diffusion_steps: int = 20
    hidden_dim: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SimpleConditionalNet(nn.Module):
    """
    Simple conditional MLP for action denoising.
    Input: [noisy_actions, timestep_embedding, observation]
    Output: predicted noise
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        action_horizon: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        
        input_dim = action_horizon * action_dim + hidden_dim + obs_dim
        output_dim = action_horizon * action_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        noisy_actions: torch.Tensor,  # [B, T, action_dim]
        timesteps: torch.Tensor,      # [B]
        obs_cond: torch.Tensor,       # [B, obs_dim]
    ) -> torch.Tensor:
        B = noisy_actions.shape[0]
        
        # Flatten actions
        actions_flat = noisy_actions.view(B, -1)  # [B, T * action_dim]
        
        # Time embedding
        t_emb = self.time_mlp(timesteps)  # [B, hidden_dim]
        
        # Concatenate all inputs
        x = torch.cat([actions_flat, t_emb, obs_cond], dim=-1)  # [B, input_dim]
        
        # Forward
        output = self.net(x)  # [B, T * action_dim]
        
        # Reshape back
        noise_pred = output.view(B, self.action_horizon, self.action_dim)
        
        return noise_pred


class DiffusionPolicy(nn.Module):
    """Diffusion Policy for Control"""
    
    def __init__(self, config: Optional[DiffusionPolicyConfig] = None):
        super().__init__()
        self.config = config or DiffusionPolicyConfig()
        cfg = self.config
        
        # Noise prediction network
        self.noise_pred_net = SimpleConditionalNet(
            action_dim=cfg.action_dim,
            obs_dim=cfg.obs_dim,
            action_horizon=cfg.action_horizon,
            hidden_dim=cfg.hidden_dim,
        )
        
        # Diffusion schedule
        self.num_train_steps = cfg.num_diffusion_steps
        self._setup_noise_schedule()
        
        self.to(cfg.device)
    
    def _setup_noise_schedule(self):
        """Setup beta and alpha schedules for diffusion"""
        steps = torch.arange(self.num_train_steps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / self.num_train_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    
    def forward_diffusion(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        x_noisy = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return x_noisy, noise
    
    def training_step(self, obs, actions):
        """Single training step"""
        B = obs.shape[0]
        device = obs.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_train_steps, (B,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Forward diffusion
        x_noisy, _ = self.forward_diffusion(actions, t, noise)
        
        # Predict noise
        noise_pred = self.noise_pred_net(x_noisy, t, obs)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {'loss': loss}
    
    @torch.no_grad()
    def sample(self, obs, num_samples=1):
        """Sample action sequence"""
        self.eval()
        device = obs.device
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        
        if num_samples > 1:
            obs = obs.repeat_interleave(num_samples, dim=0)
            B = B * num_samples
        
        T = self.config.action_horizon
        action_dim = self.config.action_dim
        
        # Start from random noise
        x = torch.randn(B, T, action_dim, device=device)
        
        # Reverse diffusion
        for i in reversed(range(self.num_train_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.noise_pred_net(x, t, obs)
            
            # Compute x_{t-1}
            alpha_t = self.alphas[t][:, None, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
            
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -10, 10)
            
            if i > 0:
                noise = torch.randn_like(x)
                variance = torch.sqrt(self.posterior_variance[t])[:, None, None] * noise
            else:
                variance = 0
            
            x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(1 - alpha_t) * noise_pred + variance
        
        if num_samples > 1:
            x = x.reshape(-1, num_samples, T, action_dim)
        
        return x
    
    def get_action(self, obs, num_samples=100, **kwargs):
        """Get action for control"""
        obs_t = torch.from_numpy(obs).float().to(self.config.device)
        action_seq = self.sample(obs_t, num_samples=num_samples)
        return action_seq[0, 0].cpu().numpy() if action_seq.dim() == 3 else action_seq[0, 0, 0].cpu().numpy()


class DiffusionPolicyController:
    """High-level controller interface"""
    
    def __init__(self, config=None):
        self.config = config or DiffusionPolicyConfig()
        self.policy = DiffusionPolicy(self.config)
        self.optimizer = None
        self.global_step = 0
    
    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-4, weight_decay=1e-6)
    
    def train_step(self, batch):
        if self.optimizer is None:
            self.setup_optimizer()
        
        self.policy.train()
        self.optimizer.zero_grad()
        
        loss_dict = self.policy.training_step(batch['obs'], batch['actions'])
        loss = loss_dict['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        self.global_step += 1
        return loss.item()
    
    def act(self, obs, **kwargs):
        return self.policy.get_action(obs, **kwargs)


def test_diffusion_policy():
    """Test Diffusion Policy"""
    print("=" * 70)
    print("Diffusion Policy Test - Axiom-OS v4.0 (Simplified)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    config = DiffusionPolicyConfig(
        obs_dim=4,
        action_dim=1,
        action_horizon=16,
        num_diffusion_steps=20,
        device=device,
    )
    
    print("\n[1] Creating Diffusion Policy...")
    policy = DiffusionPolicy(config)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    print("\n[2] Testing training step...")
    B = 32
    obs = torch.randn(B, config.obs_dim, device=device)
    actions = torch.randn(B, config.action_horizon, config.action_dim, device=device)
    
    loss_dict = policy.training_step(obs, actions)
    print(f"  Training loss: {loss_dict['loss'].item():.6f}")
    
    print("\n[3] Testing sampling...")
    obs_single = torch.randn(1, config.obs_dim, device=device)
    sampled_actions = policy.sample(obs_single)
    print(f"  Single sample shape: {sampled_actions.shape}")
    
    print("\n[4] Testing batch sampling...")
    obs_batch = torch.randn(8, config.obs_dim, device=device)
    batch_actions = policy.sample(obs_batch)
    print(f"  Batch sample shape: {batch_actions.shape}")
    print(f"  Mean action: {batch_actions.mean().item():.3f}")
    print(f"  Std action: {batch_actions.std().item():.3f}")
    
    print("\n[5] Testing multi-modal sampling...")
    multi_actions = policy.sample(obs_single, num_samples=10)
    print(f"  Multi-sample shape: {multi_actions.shape}")
    print(f"  Mean: {multi_actions.mean().item():.3f}, Std: {multi_actions.std().item():.3f}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Diffusion Policy ready!")
    print("=" * 70)


if __name__ == "__main__":
    test_diffusion_policy()
