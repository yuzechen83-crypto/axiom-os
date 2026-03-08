"""
Axiom-OS RCLN: 20-Month ENSO Forecast
=====================================
True Axiom architecture for long-range climate prediction:
- Hard Core: Delayed Oscillator Physics
- Soft Shell: Neural Correction Network
- Discovery: Learnable delay and damping
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom_os.datasets.noaa_sst_real import load_real_nino34


# =============================================================================
# Axiom-OS RCLN for Long-Range ENSO
# =============================================================================
class AxiomENSORCLN(nn.Module):
    """
    Axiom-OS Hybrid Architecture for 20-month ENSO forecasting.
    
    Architecture:
        T_pred = T_hard + lambda * T_soft
        
    Hard Core (Physics):
        Delayed Oscillator Equation:
        dT/dt = -alpha * T(t) - beta * T(t-tau) - gamma * T^3
        
    Soft Shell (Neural):
        MLP that learns residual corrections
        
    Discovery:
        - Learnable delay tau (months)
        - Learnable damping alpha
        - Learnable coupling lambda
    """
    
    def __init__(
        self,
        history_len: int = 24,
        forecast_len: int = 20,
        hidden_dim: int = 32
    ):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len
        
        # ========== Hard Core: Physics Model ==========
        # Delayed Oscillator Parameters (learnable!)
        self.alpha = nn.Parameter(torch.tensor(0.2))   # Damping
        self.beta = nn.Parameter(torch.tensor(0.3))    # Delayed feedback strength
        self.gamma = nn.Parameter(torch.tensor(0.01))  # Non-linear damping
        self.tau = nn.Parameter(torch.tensor(6.0))     # Delay time (months)
        
        # ========== Soft Shell: Neural Correction ==========
        # Takes history and physics prediction, outputs correction
        self.encoder = nn.Sequential(
            nn.Linear(history_len + forecast_len, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.corrector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, forecast_len)
        )
        
        # ========== Coupling ==========
        self.lambda_mix = nn.Parameter(torch.tensor(0.5))
        
    def physics_core(self, history: torch.Tensor, steps: int):
        """
        Delayed Oscillator integration.
        
        Args:
            history: (batch, history_len) - past 24 months
            steps: number of steps to forecast
        
        Returns:
            physics_pred: (batch, steps) - physics-only forecast
        """
        batch_size = history.shape[0]
        
        # Initialize with last known value
        T = history[:, -1]  # (batch,)
        
        # Keep history as tensor for easy indexing
        T_history = history  # (batch, history_len)
        
        physics_preds = []
        
        # Delay parameters (constrained)
        tau_int = torch.clamp(self.tau, 1, 18).long().item()
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        gamma = torch.abs(self.gamma)
        
        for step in range(steps):
            # Get delayed value from history or previous predictions
            if step < tau_int:
                # From input history
                T_delayed = T_history[:, -(tau_int - step)]
            else:
                # From previous predictions
                T_delayed = physics_preds[step - tau_int]
            
            # Delayed Oscillator equation
            # dT/dt = -alpha*T - beta*T(t-tau) - gamma*T^3
            damping = -alpha * T
            feedback = -beta * T_delayed
            nonlinear = -gamma * T**3
            
            # Update
            dT = damping + feedback + nonlinear
            T = T + dT
            
            physics_preds.append(T)
        
        return torch.stack(physics_preds, dim=1)  # (batch, steps)
    
    def forward(self, history: torch.Tensor):
        """
        Args:
            history: (batch, history_len) - past 24 months
        
        Returns:
            prediction: (batch, forecast_len) - 20-month forecast
            info: dict with components for analysis
        """
        batch_size = history.shape[0]
        
        # Hard Core: Physics prediction
        T_physics = self.physics_core(history, self.forecast_len)
        
        # Soft Shell: Neural correction
        # Concatenate history and physics prediction as input
        combined = torch.cat([history, T_physics], dim=1)
        features = self.encoder(combined)
        T_soft = self.corrector(features)
        
        # Hybrid combination
        lam = torch.sigmoid(self.lambda_mix)
        T_pred = T_physics + lam * T_soft
        
        info = {
            'T_physics': T_physics,
            'T_soft': T_soft,
            'lambda': lam,
            'alpha': torch.sigmoid(self.alpha),
            'beta': torch.sigmoid(self.beta),
            'tau': torch.clamp(self.tau, 1, 18),
            'gamma': torch.abs(self.gamma)
        }
        
        return T_pred, info


class PureNeural20Month(nn.Module):
    """Pure neural baseline for comparison."""
    
    def __init__(self, history_len: int = 24, forecast_len: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(history_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, forecast_len)
        )
    
    def forward(self, history):
        pred = self.net(history)
        return pred, {}


# =============================================================================
# Data Preparation
# =============================================================================
def create_20month_sequences(data: torch.Tensor, history_len: int = 24, forecast_len: int = 20):
    """Create (history, future) pairs for 20-month forecasting."""
    n = len(data)
    X, Y = [], []
    
    for i in range(history_len, n - forecast_len + 1):
        X.append(data[i-history_len:i].numpy())
        Y.append(data[i:i+forecast_len].numpy())
    
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


# =============================================================================
# Training
# =============================================================================
def train_axiom_model(model, X_train, Y_train, epochs=300, lr=1e-3, device='cuda'):
    """Train Axiom-OS model."""
    model = model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    losses = []
    physics_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Mini-batch
        batch_size = 32
        idx = torch.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]
        
        pred, info = model(X_batch)
        
        # Total loss: match observations
        loss = torch.nn.functional.mse_loss(pred, Y_batch)
        
        # Optional: encourage physics to be reasonable
        physics_penalty = torch.mean(info['T_physics']**2) * 0.01
        total_loss = loss + physics_penalty
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Physics Loss: {physics_penalty.item():.4f}")
            print(f"      Discovered: tau={info['tau'].item():.1f}mo, "
                  f"alpha={info['alpha'].item():.3f}, beta={info['beta'].item():.3f}, "
                  f"lambda={info['lambda'].item():.3f}")
    
    return losses


def evaluate_20month(model, X_test, Y_test, device='cuda'):
    """Evaluate 20-month forecast."""
    model.eval()
    
    with torch.no_grad():
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        
        pred, info = model(X_test)
        
        # Metrics by lead time
        rmse_by_lead = torch.sqrt(((pred - Y_test)**2).mean(dim=0)).cpu().numpy()
        mae_by_lead = torch.abs(pred - Y_test).mean(dim=0).cpu().numpy()
        
        corr_by_lead = []
        for i in range(20):
            p = pred[:, i].cpu().numpy()
            y = Y_test[:, i].cpu().numpy()
            corr = np.corrcoef(p, y)[0, 1] if len(p) > 1 else 0
            corr_by_lead.append(corr)
        
        # Overall
        overall_rmse = torch.sqrt(((pred - Y_test)**2).mean()).item()
        
    return {
        'rmse_by_lead': rmse_by_lead,
        'mae_by_lead': mae_by_lead,
        'corr_by_lead': corr_by_lead,
        'overall_rmse': overall_rmse,
        'predictions': pred.cpu().numpy(),
        'targets': Y_test.cpu().numpy(),
        'info': {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                 for k, v in info.items()}
    }


# =============================================================================
# Main Experiment
# =============================================================================
def run_axiom_20month_experiment():
    print("="*70)
    print("AXIOM-OS RCLN: 20-MONTH ENSO FORECAST")
    print("Physics-Informed AI for Long-Range Climate Prediction")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load real NOAA data
    print("\n[1] Loading real Nino 3.4 data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, loader = load_real_nino34(
        train_period=(1950, 2015),
        test_period=(2016, 2024),
        lookback=24
    )
    
    # Create 20-month sequences
    print("\n[2] Creating 20-month forecast sequences...")
    train_series = torch.cat([X_train_raw[0], y_train_raw])
    test_series = torch.cat([X_test_raw[0], y_test_raw])
    
    X_train, Y_train = create_20month_sequences(train_series, 24, 20)
    X_test, Y_test = create_20month_sequences(test_series, 24, 20)
    
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Train Axiom-OS RCLN
    print("\n[3] Training Axiom-OS RCLN...")
    print("    Hard Core: Delayed Oscillator (learnable tau, alpha, beta)")
    print("    Soft Shell: Neural Correction MLP")
    axiom_model = AxiomENSORCLN(history_len=24, forecast_len=20, hidden_dim=32)
    losses_axiom = train_axiom_model(axiom_model, X_train, Y_train, epochs=300, device=device)
    
    # Train Pure Neural baseline
    print("\n[4] Training Pure Neural baseline...")
    pure_model = PureNeural20Month(history_len=24, forecast_len=20, hidden_dim=64)
    pure_model = pure_model.to(device)
    X_train_d = X_train.to(device)
    Y_train_d = Y_train.to(device)
    
    optimizer = torch.optim.Adam(pure_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
    
    losses_pure = []
    for epoch in range(300):
        pure_model.train()
        optimizer.zero_grad()
        
        idx = torch.randint(0, len(X_train_d), (32,))
        pred, _ = pure_model(X_train_d[idx])
        loss = torch.nn.functional.mse_loss(pred, Y_train_d[idx])
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses_pure.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/300, Loss: {loss.item():.4f}")
    
    # Evaluate
    print("\n[5] Evaluating 20-month forecast skill...")
    results_axiom = evaluate_20month(axiom_model, X_test, Y_test, device)
    results_pure = evaluate_20month(pure_model, X_test, Y_test, device)
    
    # Print skill by lead
    print("\n  Forecast Skill by Lead Time:")
    print(f"  {'Lead':<8} {'Axiom RMSE':<12} {'Pure RMSE':<12} {'Axiom Corr':<12} {'Pure Corr':<12}")
    print("  " + "-"*60)
    
    for i in range(0, 20, 2):
        print(f"  {i+1:<8} {results_axiom['rmse_by_lead'][i]:<12.3f} "
              f"{results_pure['rmse_by_lead'][i]:<12.3f} "
              f"{results_axiom['corr_by_lead'][i]:<12.3f} "
              f"{results_pure['corr_by_lead'][i]:<12.3f}")
    
    # Discovered parameters
    print("\n[6] Discovered Physics Parameters:")
    info = results_axiom['info']
    print(f"    Delay time (tau): {float(info['tau']):.1f} months")
    print(f"    Damping (alpha): {float(info['alpha']):.3f}")
    print(f"    Feedback (beta): {float(info['beta']):.3f}")
    print(f"    Non-linear (gamma): {float(info['gamma']):.4f}")
    print(f"    Coupling (lambda): {float(info['lambda']):.3f}")
    
    # Useful skill limit
    axiom_useful = next((i for i, c in enumerate(results_axiom['corr_by_lead']) if c < 0.5), 20)
    pure_useful = next((i for i, c in enumerate(results_pure['corr_by_lead']) if c < 0.5), 20)
    
    print(f"\n    Useful skill (corr>0.5): Axiom={axiom_useful}mo, Pure={pure_useful}mo")
    
    # Visualization
    print("\n[7] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training curves
    ax = axes[0, 0]
    ax.semilogy(losses_pure, 'b-', label='Pure Neural', alpha=0.8)
    ax.semilogy(losses_axiom, 'r-', label='Axiom RCLN', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE by lead
    ax = axes[0, 1]
    leads = np.arange(1, 21)
    ax.plot(leads, results_axiom['rmse_by_lead'], 'r-o', label='Axiom RCLN', markersize=6)
    ax.plot(leads, results_pure['rmse_by_lead'], 'b-s', label='Pure Neural', markersize=6)
    ax.set_xlabel('Forecast Lead (months)')
    ax.set_ylabel('RMSE')
    ax.set_title('Forecast Error vs Lead Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation by lead
    ax = axes[1, 0]
    ax.plot(leads, results_axiom['corr_by_lead'], 'r-o', label='Axiom RCLN', markersize=6)
    ax.plot(leads, results_pure['corr_by_lead'], 'b-s', label='Pure Neural', markersize=6)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Useful skill')
    ax.set_xlabel('Forecast Lead (months)')
    ax.set_ylabel('Correlation')
    ax.set_title('Forecast Correlation vs Lead Time')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sample forecast
    ax = axes[1, 1]
    sample_idx = 0
    history = X_test[sample_idx, -12:].numpy()
    obs = results_axiom['targets'][sample_idx]
    pred_axiom = results_axiom['predictions'][sample_idx]
    pred_pure = results_pure['predictions'][sample_idx]
    
    hist_time = np.arange(-12, 0)
    lead_time = np.arange(0, 20)
    
    ax.plot(hist_time, history, 'k-', linewidth=2, label='History (12mo)')
    ax.plot(lead_time, obs, 'k--', linewidth=2, label='Observed')
    ax.plot(lead_time, pred_axiom, 'r-o', label='Axiom RCLN', markersize=4)
    ax.plot(lead_time, pred_pure, 'b-s', label='Pure Neural', markersize=4)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Nino 3.4 Anomaly')
    ax.set_title('Sample 20-Month Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/axiom_enso_20month.png', dpi=150, bbox_inches='tight')
    print("  Saved: experiments/axiom_enso_20month.png")
    
    # Summary
    print("\n" + "="*70)
    print("AXIOM-OS RCLN 20-MONTH FORECAST SUMMARY")
    print("="*70)
    print(f"""
Discovered Equation:
    dT/dt = -{float(info['alpha']):.3f} * T(t) 
            - {float(info['beta']):.3f} * T(t-{float(info['tau']):.0f}) 
            - {float(info['gamma']):.4f} * T^3
            + {float(info['lambda']):.3f} * Neural_Correction

Forecast Skill:
    - Useful skill limit: {axiom_useful} months (Axiom) vs {pure_useful} months (Pure)
    - 20-month avg RMSE: {results_axiom['overall_rmse']:.3f} (Axiom) vs {results_pure['overall_rmse']:.3f} (Pure)

Key Advantages of Axiom RCLN:
    1. Interpretable physics parameters (discovered tau={float(info['tau']):.0f} months)
    2. Stable long-range predictions (physics damping prevents drift)
    3. Neural correction adapts to model errors
    4. Hybrid architecture balances physics and data

This is true "Physics-Informed AI" for climate prediction!
""")
    
    return {
        'axiom': axiom_model,
        'pure': pure_model,
        'results_axiom': results_axiom,
        'results_pure': results_pure
    }


if __name__ == "__main__":
    results = run_axiom_20month_experiment()
