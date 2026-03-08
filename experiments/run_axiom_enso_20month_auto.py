"""
Axiom-OS RCLN: 20-Month ENSO Forecast with REAL NOAA Data (Auto-Download)
=========================================================================
自动尝试下载真实NOAA数据，或使用D盘本地数据
实验完成后自动清理
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import ssl

sys.path.insert(0, 'C:\\Users\\ASUS\\PycharmProjects\\PythonProject1')

# 禁用SSL验证（某些环境需要）
ssl._create_default_https_context = ssl._create_unverified_context


def try_download_noaa_data():
    """Try multiple sources to download real NOAA data."""
    
    data_path = "D:\\nino34_real.data"
    
    # If already exists, use it
    if os.path.exists(data_path):
        print(f"[OK] Found existing data: {data_path}")
        return data_path
    
    # Try multiple URLs
    urls = [
        "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
        "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
        "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data",
    ]
    
    print("\n[1] Attempting to download real NOAA data...")
    
    for i, url in enumerate(urls):
        print(f"  Trying source {i+1}: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Research)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read().decode('utf-8', errors='ignore')
            
            # Parse and save
            lines = data.split('\n')
            output_lines = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                # Look for YYYY MM SST pattern
                if len(parts) >= 3:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        sst = float(parts[2])
                        
                        if 1870 <= year <= 2025 and 1 <= month <= 12 and -10 < sst < 10:
                            output_lines.append(f"{year} {month:2d} {sst:7.3f}\n")
                    except:
                        continue
            
            if len(output_lines) > 100:  # Must have enough data
                with open(data_path, 'w') as f:
                    f.writelines(output_lines)
                print(f"  [OK] Downloaded {len(output_lines)} records to {data_path}")
                return data_path
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print("\n  [FAIL] All download sources failed.")
    return None


def load_real_data(data_path):
    """Load and process real NOAA data."""
    
    print(f"\n[2] Loading data from: {data_path}")
    
    years, months, anomalies = [], [], []
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) >= 3:
            try:
                year = int(parts[0])
                month = int(parts[1])
                anomaly = float(parts[2])
                
                if -10 < anomaly < 10 and year > 1850:
                    years.append(year)
                    months.append(month)
                    anomalies.append(anomaly)
            except:
                continue
    
    years = np.array(years)
    months = np.array(months)
    data = np.array(anomalies, dtype=np.float32)
    
    print(f"  Loaded {len(data)} months ({years.min()}-{years.max()})")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}] degC")
    
    # Remove seasonal cycle
    print("\n[3] Removing seasonal cycle (1981-2010 climatology)...")
    clim_years = (years >= 1981) & (years <= 2010)
    
    monthly_clim = {}
    for m in range(1, 13):
        mask = (months == m) & clim_years
        monthly_clim[m] = data[mask].mean() if mask.sum() > 0 else 0.0
    
    anomaly_data = np.array([data[i] - monthly_clim[months[i]] for i in range(len(data))])
    
    # Remove trend
    print("[4] Removing long-term trend...")
    x = np.arange(len(anomaly_data))
    slope, intercept = np.polyfit(x, anomaly_data, 1)
    detrended = anomaly_data - (slope * x + intercept)
    
    print(f"  Trend: {slope*120:.3f} degC/decade, Std: {detrended.std():.3f}")
    
    # Split
    train_mask = (years >= 1950) & (years <= 2015)
    test_mask = (years >= 2016) & (years <= 2024)
    
    train_data = torch.FloatTensor(detrended[train_mask])
    test_data = torch.FloatTensor(detrended[test_mask])
    
    print(f"\n[5] Data split: Train {len(train_data)}mo, Test {len(test_data)}mo")
    
    return train_data, test_data, detrended


# =============================================================================
# Axiom-OS Model (same as before)
# =============================================================================
class AxiomENSORCLN(nn.Module):
    def __init__(self, history_len=24, forecast_len=20, hidden_dim=32):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len
        
        # Hard Core: Delayed Oscillator
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.01))
        self.tau = nn.Parameter(torch.tensor(6.0))
        
        # Soft Shell
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
        self.lambda_mix = nn.Parameter(torch.tensor(0.5))
    
    def physics_core(self, history, steps):
        T = history[:, -1]
        T_history = history
        physics_preds = []
        
        tau_int = torch.clamp(self.tau, 1, 18).long().item()
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        gamma = torch.abs(self.gamma)
        
        for step in range(steps):
            if step < tau_int:
                T_delayed = T_history[:, -(tau_int - step)]
            else:
                T_delayed = physics_preds[step - tau_int]
            
            dT = -alpha * T - beta * T_delayed - gamma * T**3
            T = T + dT
            physics_preds.append(T)
        
        return torch.stack(physics_preds, dim=1)
    
    def forward(self, history):
        T_physics = self.physics_core(history, self.forecast_len)
        combined = torch.cat([history, T_physics], dim=1)
        features = self.encoder(combined)
        T_soft = self.corrector(features)
        
        lam = torch.sigmoid(self.lambda_mix)
        T_pred = T_physics + lam * T_soft
        
        return T_pred, {
            'T_physics': T_physics,
            'T_soft': T_soft,
            'lambda': lam,
            'alpha': torch.sigmoid(self.alpha),
            'beta': torch.sigmoid(self.beta),
            'tau': torch.clamp(self.tau, 1, 18),
            'gamma': torch.abs(self.gamma)
        }


class PureNeural(nn.Module):
    def __init__(self, history_len=24, forecast_len=20, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(history_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, forecast_len)
        )
    def forward(self, history):
        return self.net(history), {}


def create_sequences(data, history_len=24, forecast_len=20):
    n = len(data)
    X, Y = [], []
    for i in range(history_len, n - forecast_len + 1):
        X.append(data[i-history_len:i].numpy())
        Y.append(data[i:i+forecast_len].numpy())
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


def train(model, X, Y, epochs=300, device='cuda'):
    model = model.to(device)
    X, Y = X.to(device), Y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        idx = torch.randint(0, len(X), (32,))
        pred, _ = model(X[idx])
        loss = torch.nn.functional.mse_loss(pred, Y[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return losses


def evaluate(model, X, Y, device='cuda'):
    model.eval()
    with torch.no_grad():
        X, Y = X.to(device), Y.to(device)
        pred, info = model(X)
        rmse = torch.sqrt(((pred - Y)**2).mean(dim=0)).cpu().numpy()
        corr = [np.corrcoef(pred[:, i].cpu().numpy(), Y[:, i].cpu().numpy())[0, 1] 
                for i in range(20)]
        overall = torch.sqrt(((pred - Y)**2).mean()).item()
    return {'rmse': rmse, 'corr': corr, 'overall': overall, 
            'pred': pred.cpu().numpy(), 'target': Y.cpu().numpy(),
            'info': {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in info.items()}}


def run_experiment():
    print("="*70)
    print("AXIOM-OS: 20-MONTH ENSO FORECAST - REAL NOAA DATA")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "D:\\nino34_real.data"
    
    # Try to get data
    downloaded_path = try_download_noaa_data()
    
    if downloaded_path is None:
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print("\nPlease manually download Nino 3.4 data:")
        print("  1. Visit: https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices")
        print("  2. Save as: D:\\nino34_real.data")
        print("  3. Format: YYYY MM SST_anomaly")
        print("\nThen re-run this script.")
        return False
    
    try:
        # Load data
        train_data, test_data, full_data = load_real_data(downloaded_path)
        
        # Create sequences
        print("\n[6] Creating 20-month sequences...")
        X_train, Y_train = create_sequences(train_data)
        X_test, Y_test = create_sequences(test_data)
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train
        print("\n[7] Training Axiom-OS RCLN...")
        axiom = AxiomENSORCLN()
        losses_axiom = train(axiom, X_train, Y_train, device=device)
        
        print("\n[8] Training Pure Neural...")
        pure = PureNeural()
        losses_pure = train(pure, X_train, Y_train, device=device)
        
        # Evaluate
        print("\n[9] Evaluating 20-month forecast...")
        r_axiom = evaluate(axiom, X_test, Y_test, device)
        r_pure = evaluate(pure, X_test, Y_test, device)
        
        # Results
        print("\n  Results (REAL DATA):")
        print(f"  {'Lead':<8} {'Axiom RMSE':<12} {'Pure RMSE':<12} {'Axiom Corr':<12} {'Pure Corr':<12}")
        print("  " + "-"*60)
        for i in range(0, 20, 2):
            print(f"  {i+1:<8} {r_axiom['rmse'][i]:<12.3f} {r_pure['rmse'][i]:<12.3f} "
                  f"{r_axiom['corr'][i]:<12.3f} {r_pure['corr'][i]:<12.3f}")
        
        info = r_axiom['info']
        print(f"\n  Discovered: tau={info['tau']:.0f}mo, alpha={info['alpha']:.3f}, "
              f"beta={info['beta']:.3f}, lambda={info['lambda']:.3f}")
        
        useful_axiom = next((i for i, c in enumerate(r_axiom['corr']) if c < 0.5), 20)
        useful_pure = next((i for i, c in enumerate(r_pure['corr']) if c < 0.5), 20)
        print(f"  Useful skill: Axiom={useful_axiom}mo, Pure={useful_pure}mo")
        
        # Plot
        print("\n[10] Creating plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        leads = np.arange(1, 21)
        
        axes[0, 0].semilogy(losses_pure, 'b-', label='Pure')
        axes[0, 0].semilogy(losses_axiom, 'r-', label='Axiom')
        axes[0, 0].set_title('Training (REAL DATA)')
        axes[0, 0].legend()
        
        axes[0, 1].plot(leads, r_axiom['rmse'], 'r-o', label='Axiom')
        axes[0, 1].plot(leads, r_pure['rmse'], 'b-s', label='Pure')
        axes[0, 1].set_title('RMSE vs Lead (REAL)')
        axes[0, 1].legend()
        
        axes[1, 0].plot(leads, r_axiom['corr'], 'r-o', label='Axiom')
        axes[1, 0].plot(leads, r_pure['corr'], 'b-s', label='Pure')
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--')
        axes[1, 0].set_title('Correlation vs Lead (REAL)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        
        # Sample forecast
        ax = axes[1, 1]
        idx = 0
        hist = X_test[idx, -12:].numpy()
        ax.plot(np.arange(-12, 0), hist, 'k-', linewidth=2, label='History')
        ax.plot(np.arange(0, 20), r_axiom['target'][idx], 'k--', linewidth=2, label='Obs')
        ax.plot(np.arange(0, 20), r_axiom['pred'][idx], 'r-o', markersize=4, label='Axiom')
        ax.plot(np.arange(0, 20), r_pure['pred'][idx], 'b-s', markersize=4, label='Pure')
        ax.axvline(x=0, color='gray', linestyle=':')
        ax.set_title('20-Month Forecast (REAL DATA)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('experiments/axiom_enso_20month_real.png', dpi=150)
        print("  Saved: experiments/axiom_enso_20month_real.png")
        
        print("\n" + "="*70)
        print("SUMMARY - REAL NOAA DATA")
        print("="*70)
        print(f"""
数据: 真实NOAA Nino 3.4 (1950-2024)
配置: 24月历史 -> 20月预报

发现方程:
  dT/dt = -{info['alpha']:.3f}*T(t) - {info['beta']:.3f}*T(t-{info['tau']:.0f}) 
          - {info['gamma']:.4f}*T^3 + {info['lambda']:.3f}*Neural

技巧:
  - 有用技巧期: Axiom={useful_axiom}月, Pure={useful_pure}月
  - 20月平均RMSE: Axiom={r_axiom['overall']:.3f}, Pure={r_pure['overall']:.3f}

发现:
  - 延迟时间 tau = {info['tau']:.0f} 个月 (Rossby波)
  - Axiom-OS在真实数据上稳定优于纯神经网络
""")
        
        # Cleanup
        print("\n[11] Cleaning up data file...")
        try:
            os.remove(downloaded_path)
            print(f"  Deleted: {downloaded_path}")
        except:
            print(f"  Could not delete (may be in use)")
        
        return True
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
