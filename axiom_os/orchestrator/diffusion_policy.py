# -*- coding: utf-8 -*-
"""
Diffusion Policy Controller for Axiom-OS v4.0
SOTA: Generative Control via Denoising Diffusion

Replaces MPPI sampling with conditional diffusion for:
- Multi-modal action distributions
- Smooth trajectory generation
- Contact-rich tasks

Reference: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (Chi et al., 2023)
"""

from typing import Optional, Tuple, Dict, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from einops import rearrange


@dataclass
class DiffusionPolicyConfig:
    """Configuration for Diffusion Policy"""
    # Observation
    obs_dim: int = 4
    obs_horizon: int = 2  # Number of past observations to condition on
    
    # Action
    action_dim: int = 1
    action_horizon: int = 16  # Length of action sequence to predict
    pred_horizon: int = 16
    
    # Diffusion
    num_diffusion_steps: int = 20
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Network
    noise_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 4
    
    # Training
    num_train_steps: int = 100000
    batch_size: int = 64
    learning_rate: float = 1e-4
    
    # Inference
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


class ConditionalUNet1D(nn.Module):
    """
    1D Conditional UNet for action denoising.
    
    Architecture:
    - Input: Noisy action sequence [B, T, action_dim]
    - Conditioning: Observation encoding [B, obs_dim]
    - Output: Predicted noise [B, T, action_dim]
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        noise_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(noise_dim),
            nn.Linear(noise_dim, noise_dim * 4),
            nn.Mish(),
            nn.Linear(noise_dim * 4, noise_dim),
        )
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_layers):
            self.down_blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.Mish(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.Mish(),
            ))
        
        # Middle block
        self.middle_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )
        
        # Time and observation conditioning injection
        self.time_cond_proj = nn.Linear(noise_dim, hidden_dim)
        self.obs_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_layers):
            self.up_blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.Mish(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.Mish(),
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: [B, T, action_dim] - noisy action sequence
            timesteps: [B] - diffusion timesteps
            obs_cond: [B, obs_dim] - observation conditioning
        
        Returns:
            noise_pred: [B, T, action_dim] - predicted noise
        """
        B, T, action_dim = noisy_actions.shape
        
        # DEBUG
        print(f"DEBUG forward: noisy_actions.shape={noisy_actions.shape}, timesteps.shape={timesteps.shape}, obs_cond.shape={obs_cond.shape}")
        
        # Ensure obs_cond is batched correctly
        if obs_cond.dim() == 1:
            obs_cond = obs_cond.unsqueeze(0)
        if obs_cond.shape[0] != B:
            # Repeat obs_cond to match batch size
            obs_cond = obs_cond.repeat(B // obs_cond.shape[0], 1)
        
        # Encode time and observation
        t_emb = self.time_mlp(timesteps)  # [B, noise_dim]
        obs_emb = self.obs_encoder(obs_cond)  # [B, hidden_dim]
        
        # Project input to hidden dimension
        print(f"DEBUG: About to call input_proj with shape {noisy_actions.shape}, action_dim={action_dim}, self.action_dim={self.action_dim}")
        x = self.input_proj(noisy_actions)  # [B, T, hidden_dim]
        x = rearrange(x, 'b t c -> b c t')  # [B, hidden_dim, T] for Conv1d
        
        # Add conditioning
        t_cond = self.time_cond_proj(t_emb)[:, :, None]  # [B, hidden_dim, 1]
        obs_cond_proj = self.obs_cond_proj(obs_emb)[:, :, None]  # [B, hidden_dim, 1]
        x = x + t_cond + obs_cond_proj
        
        # Downsampling with skip connections
        skips = []
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        
        # Middle
        x = self.middle_block(x)
        
        # Upsampling with skip connections
        for block in self.up_blocks:
            x = torch.cat([x, skips.pop()], dim=1)  # Concatenate skip connection
            x = block(x)
        
        # Output
        x = rearrange(x, 'b c t -> b t c')  # [B, T, hidden_dim]
        noise_pred = self.output_proj(x)  # [B, T, action_dim]
        
        return noise_pred


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy for Control
    
    Generates action sequences by iteratively denoising random noise,
    conditioned on current observation.
    """
    
    def __init__(self, config: Optional[DiffusionPolicyConfig] = None):
        super().__init__()
        self.config = config or DiffusionPolicyConfig()
        cfg = self.config
        
        # Noise prediction network
        self.noise_pred_net = ConditionalUNet1D(
            action_dim=cfg.action_dim,
            obs_dim=cfg.obs_dim,
            noise_dim=cfg.noise_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
        )
        
        # Diffusion schedule
        self.num_train_steps = cfg.num_diffusion_steps
        self._setup_noise_schedule()
        
        self.to(cfg.device)
    
    def _setup_noise_schedule(self):
        """Setup beta and alpha schedules for diffusion"""
        cfg = self.config
        
        if cfg.beta_schedule == "squaredcos_cap_v2":
            # Improved schedule from "Improved Denoising Diffusion Probabilistic Models"
            steps = torch.arange(cfg.num_diffusion_steps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / cfg.num_diffusion_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_diffusion_steps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # For sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def forward_diffusion(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: [B, T, action_dim] - clean action sequence
            t: [B] - timestep
            noise: optional noise
        
        Returns:
            x_noisy, noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        x_noisy = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return x_noisy, noise
    
    def training_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step
        
        Args:
            obs: [B, obs_dim] - observations
            actions: [B, T, action_dim] - action sequences
        
        Returns:
            Dict with loss
        """
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
    def sample(
        self,
        obs: torch.Tensor,
        num_samples: int = 1,
        return_chain: bool = False,
    ) -> torch.Tensor:
        """
        Sample action sequence using DDPM sampling
        
        Args:
            obs: [obs_dim] or [B, obs_dim] - observation
            num_samples: number of samples per observation
            return_chain: whether to return full diffusion chain
        
        Returns:
            actions: [B, T, action_dim] or [num_samples, T, action_dim] if B==1
        """
        self.eval()
        device = obs.device
        
        # Handle input shape
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        
        # Expand for multiple samples
        if num_samples > 1:
            obs = obs.repeat_interleave(num_samples, dim=0)  # [B*num_samples, obs_dim]
            total_batch = B * num_samples
        else:
            total_batch = B
        
        T = self.config.action_horizon
        action_dim = self.config.action_dim
        
        # Start from random noise
        x = torch.randn(total_batch, T, action_dim, device=device)
        
        # DEBUG
        print(f"DEBUG sample: x.shape={x.shape}, obs.shape={obs.shape}, B={B}, total_batch={total_batch}")
        
        chain = [x] if return_chain else None
        
        # Reverse diffusion
        for i in reversed(range(self.num_train_steps)):
            t = torch.full((total_batch,), i, device=device, dtype=torch.long)
            
            # Predict noise
            # DEBUG
            # print(f"DEBUG iteration {i}: x.shape={x.shape}, t.shape={t.shape}")
            noise_pred = self.noise_pred_net(x, t, obs)
            
            # Compute x_{t-1}
            alpha_t = self.alphas[t][:, None, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
            beta_t = self.betas[t][:, None, None]
            
            # Mean of p(x_{t-1} | x_t)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -10, 10)
            
            if i > 0:
                noise = torch.randn_like(x)
                variance = torch.sqrt(self.posterior_variance[t]) * noise
            else:
                variance = 0
            
            x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(1 - alpha_t) * noise_pred + variance
            
            if return_chain:
                chain.append(x)
        
        # Reshape if num_samples > 1 and B == 1
        if num_samples > 1 and B == 1:
            x = x.reshape(num_samples, T, action_dim)
        elif num_samples > 1:
            x = x.reshape(B, num_samples, T, action_dim)
        
        if return_chain:
            return x, chain
        return x
    
    @torch.no_grad()
    def get_action(
        self,
        obs: torch.Tensor,
        num_samples: int = 100,
        best_k: Optional[int] = None,
        cost_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Get action for control, with optional cost-based selection
        
        Args:
            obs: observation
            num_samples: number of diffusion samples
            best_k: select top-k actions (for MPPI-like behavior)
            cost_fn: cost function for action selection
        
        Returns:
            action: [action_dim] - first action of best sequence
        """
        # Sample multiple action sequences
        action_seqs = self.sample(obs, num_samples=num_samples)  # [num_samples, T, action_dim]
        
        if cost_fn is not None and best_k is not None:
            # Evaluate costs (MPPI-style selection)
            costs = torch.tensor([cost_fn(a) for a in action_seqs])
            _, top_k_idx = torch.topk(costs, best_k, largest=False)
            best_actions = action_seqs[top_k_idx]
            action = best_actions.mean(dim=0)[0]  # Average best, take first
        else:
            # Just take the first sample's first action
            action = action_seqs[0, 0]
        
        return action


class DiffusionPolicyController:
    """
    High-level controller interface for Axiom-OS
    Replaces MPC with Diffusion Policy
    """
    
    def __init__(self, config: Optional[DiffusionPolicyConfig] = None):
        self.config = config or DiffusionPolicyConfig()
        self.policy = DiffusionPolicy(self.config)
        self.optimizer = None
        self.global_step = 0
    
    def setup_optimizer(self):
        """Setup optimizer for training"""
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-6,
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
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
    
    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get action for control
        
        Args:
            obs: observation array
            **kwargs: additional arguments for get_action
        
        Returns:
            action: numpy array
        """
        obs_t = torch.from_numpy(obs).float().to(self.config.device)
        action = self.policy.get_action(obs_t, **kwargs)
        return action.cpu().numpy()
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'global_step': self.global_step,
            'config': self.config,
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.policy.load_state_dict(checkpoint['policy'])
        if checkpoint['optimizer'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']


def test_diffusion_policy():
    """Test Diffusion Policy"""
    print("=" * 70)
    print("Diffusion Policy Test - Axiom-OS v4.0")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Config
    config = DiffusionPolicyConfig(
        obs_dim=4,
        action_dim=1,
        action_horizon=16,
        num_diffusion_steps=20,
        device=device,
    )
    
    # Create policy
    print("\n[1] Creating Diffusion Policy...")
    policy = DiffusionPolicy(config)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test training step
    print("\n[2] Testing training step...")
    B = 32
    obs = torch.randn(B, config.obs_dim, device=device)
    actions = torch.randn(B, config.action_horizon, config.action_dim, device=device)
    
    loss_dict = policy.training_step(obs, actions)
    print(f"  Training loss: {loss_dict['loss'].item():.6f}")
    
    # Test sampling
    print("\n[3] Testing sampling...")
    obs_single = torch.randn(config.obs_dim, device=device)
    sampled_actions = policy.sample(obs_single, num_samples=1)
    print(f"  Sampled action sequence shape: {sampled_actions.shape}")
    print(f"  Action range: [{sampled_actions.min():.3f}, {sampled_actions.max():.3f}]")
    
    # Test batch sampling
    print("\n[4] Testing batch sampling (batch_size=8)...")
    obs_batch = torch.randn(8, config.obs_dim, device=device)
    batch_actions = policy.sample(obs_batch)  # [8, 16, 1]
    print(f"  Batch sample shape: {batch_actions.shape}")
    print(f"  Mean action: {batch_actions.mean().item():.3f}")
    print(f"  Std action: {batch_actions.std().item():.3f}")
    
    # Test controller interface
    print("\n[5] Testing controller interface...")
    controller = DiffusionPolicyController(config)
    obs_np = np.random.randn(config.obs_dim)
    action_np = controller.act(obs_np)
    print(f"  Controller output shape: {action_np.shape}")
    print(f"  Action value: {action_np[0]:.3f}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Diffusion Policy ready for Axiom-OS v4.0!")
    print("=" * 70)
    print("\nKey Features:")
    print("  - Conditional UNet for action denoising")
    print("  - Multi-modal action distribution")
    print("  - Smooth trajectory generation")
    print("  - Drop-in replacement for MPC")


if __name__ == "__main__":
    test_diffusion_policy()
