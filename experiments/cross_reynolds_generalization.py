"""
Cross-Reynolds Generalization Test for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

The Ultimate Test: Train on Re=100, test on Re=1000.

Key Question: Does the Discovery Engine find physics laws that are 
Reynolds-number invariant?

Physics Background:
- Reynolds number Re = ρUL/μ (inertial/viscous forces ratio)
- True physics (Navier-Stokes) is Re-dependent in dynamics but the 
  underlying PDE structure is universal
- If Discovery finds the true PDE form, it should generalize across Re
  (with appropriate rescaling)

Test Protocol:
1. Train RCLN (FNO Soft Shell) on Re=100 data
2. Extract symbolic formula via DiscoveryEngine
3. Test on Re=1000 (no retraining)
4. Compare with:
   - Pure FNO (no Hard Core)
   - Pure U-Net
   - Axiom-OS with crystallized formula

Success Criteria:
- Axiom-OS with discovered formula outperforms pure neural methods
- Formula structure is Reynolds-invariant (π-groups)
"""

from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from axiom_os.datasets.pdebench import (
    PDEBenchConfig, create_cross_reynolds_loaders, CrossReynoldsDataset
)
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.layers.fno import FNO2d
from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.core.hippocampus import Hippocampus


@dataclass
class CrossReynoldsConfig:
    """Configuration for cross-Reynolds experiment."""
    
    # Data
    data_dir: str = "./data/pdebench"
    dataset_name: str = "2D_NavierStokes"
    resolution: int = 64
    train_re: int = 100
    test_re: int = 1000
    
    # Training
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    fno_width: int = 32
    fno_modes: int = 12
    fno_layers: int = 4
    lambda_res: float = 1.0
    
    # Discovery
    run_discovery: bool = True
    discovery_epochs: int = 50  # Epochs after which to run discovery
    n_discovery_samples: int = 1000
    
    # Evaluation
    n_rollout_steps: int = 10  # Number of autoregressive rollout steps
    
    # Output
    output_dir: str = "./outputs/cross_reynolds"


class UNet2d(nn.Module):
    """
    Standard U-Net for 2D spatial fields.
    Baseline comparison for FNO and RCLN.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = nn.ModuleList()
        ch = in_channels
        for i in range(n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(ch, hidden_dim * (2**i), 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim * (2**i), hidden_dim * (2**i), 3, padding=1),
                nn.ReLU(),
            ))
            ch = hidden_dim * (2**i)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(n_layers - 1, -1, -1):
            out_ch = hidden_dim * (2**i) if i > 0 else out_channels
            self.decoder.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.ReLU(),
            ))
            ch = out_ch
        
        self.final = nn.Conv2d(ch, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = nn.functional.avg_pool2d(x, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, dec in enumerate(self.decoder):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < len(skips):
                x = x + skips[-(i+1)]  # Skip connection
            x = dec(x)
        
        return self.final(x)


class CrossReynoldsExperiment:
    """
    Cross-Reynolds Generalization Experiment.
    
    Compares:
    1. Pure FNO (baseline)
    2. Pure U-Net (baseline)
    3. RCLN with Soft Shell only
    4. RCLN with discovered Hard Core (Axiom-OS)
    """
    
    def __init__(self, config: CrossReynoldsConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Initialize models
        self._init_models()
        
        # Initialize discovery engine
        self.discovery_engine = DiscoveryEngine(use_pysr=True)
        self.hippocampus = Hippocampus(dim=32, capacity=1000)
        
        # Results storage
        self.results: Dict[str, Any] = {}
    
    def _load_data(self):
        """Load cross-Reynolds dataset."""
        print("Loading cross-Reynolds dataset...")
        print(f"  Train: Re={self.config.train_re}")
        print(f"  Test:  Re={self.config.test_re}")
        
        loaders = create_cross_reynolds_loaders(
            data_dir=self.config.data_dir,
            dataset_name=self.config.dataset_name,
            resolution=self.config.resolution,
            train_re=self.config.train_re,
            test_re=self.config.test_re,
            batch_size=self.config.batch_size,
            num_workers=0,  # Single process for reproducibility
        )
        
        self.train_loader = loaders["train"]
        self.test_loader = loaders["test"]
        
        # Get input/output dimensions from first batch
        x, y, re = next(iter(self.train_loader))
        self.in_channels = x.shape[1]
        self.out_channels = y.shape[1]
        self.H, self.W = x.shape[2], x.shape[3]
        
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
    
    def _init_models(self):
        """Initialize all models."""
        print("\nInitializing models...")
        
        # 1. Pure FNO
        self.model_fno = FNO2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.config.fno_width,
            modes1=self.config.fno_modes,
            modes2=self.config.fno_modes,
            n_layers=self.config.fno_layers,
        ).to(self.device)
        
        # 2. Pure U-Net
        self.model_unet = UNet2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_dim=self.config.fno_width,
            n_layers=self.config.fno_layers,
        ).to(self.device)
        
        # 3. RCLN (Soft Shell only initially)
        self.model_rcln = RCLNLayer(
            input_dim=self.in_channels,
            hidden_dim=self.config.fno_width,
            output_dim=self.out_channels,
            hard_core_func=None,  # Start with no hard core
            lambda_res=self.config.lambda_res,
            net_type="fno",
            fno_modes1=self.config.fno_modes,
            fno_modes2=self.config.fno_modes,
            use_activity_monitor=True,
            soft_threshold=0.1,
            monitor_window=32,
        ).to(self.device)
        
        self.models = {
            "FNO": self.model_fno,
            "U-Net": self.model_unet,
            "RCLN": self.model_rcln,
        }
        
        print(f"  FNO:   {sum(p.numel() for p in self.model_fno.parameters()):,} params")
        print(f"  U-Net: {sum(p.numel() for p in self.model_unet.parameters()):,} params")
        print(f"  RCLN:  {sum(p.numel() for p in self.model_rcln.parameters()):,} params")
    
    def train_model(self, model: nn.Module, name: str) -> Dict[str, List[float]]:
        """Train a single model."""
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        history = {"train_loss": [], "test_loss": []}
        
        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch in self.train_loader:
                x, y, re = batch
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                # Handle different input formats
                if isinstance(model, RCLNLayer):
                    pred = model(x)
                else:
                    pred = model(x)
                
                loss = nn.functional.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Testing
            model.eval()
            test_losses = []
            
            with torch.no_grad():
                for batch in self.test_loader:
                    x, y, re = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    if isinstance(model, RCLNLayer):
                        pred = model(x)
                    else:
                        pred = model(x)
                    
                    loss = nn.functional.mse_loss(pred, y)
                    test_losses.append(loss.item())
            
            scheduler.step()
            
            avg_train = np.mean(train_losses)
            avg_test = np.mean(test_losses)
            
            history["train_loss"].append(avg_train)
            history["test_loss"].append(avg_test)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}: "
                      f"train_loss={avg_train:.6f}, test_loss={avg_test:.6f}")
        
        return history
    
    def run_discovery(self) -> Optional[str]:
        """
        Run Discovery Engine on trained RCLN to extract symbolic formula.
        
        Returns:
            Discovered formula string or None
        """
        print(f"\n{'='*60}")
        print("Running Discovery Engine")
        print(f"{'='*60}")
        
        self.model_rcln.eval()
        
        # Collect soft shell outputs
        print("Collecting soft shell data...")
        soft_inputs = []
        soft_outputs = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i * self.config.batch_size >= self.config.n_discovery_samples:
                    break
                
                x, y, re = batch
                x = x.to(self.device)
                
                # Get soft shell output
                _ = self.model_rcln(x)
                y_soft = self.model_rcln._last_y_soft
                
                soft_inputs.append(x.cpu())
                soft_outputs.append(y_soft.cpu())
        
        # Prepare data for discovery
        X = torch.cat(soft_inputs, dim=0).numpy()
        Y_soft = torch.cat(soft_outputs, dim=0).numpy()
        
        # Average over spatial dimensions for scalar regression
        # Shape: (N, C, H, W) -> (N, C)
        X_flat = X.mean(axis=(2, 3))
        Y_flat = Y_soft.mean(axis=(2, 3))
        
        print(f"Discovery data: X {X_flat.shape}, Y {Y_flat.shape}")
        
        # Run discovery
        print("Running symbolic regression...")
        formula = self.discovery_engine.discover_multivariate(
            X=X_flat,
            y=Y_flat[:, 0],  # First output channel
            var_names=[f"u{i}" for i in range(X_flat.shape[1])],
            selector="bic",
        )
        
        discovered_formula = formula[0] if formula[0] else None
        
        if discovered_formula:
            print(f"\nDiscovered Formula:")
            print(f"  {discovered_formula}")
            
            # Crystallize formula into RCLN
            print("\nCrystallizing formula into RCLN...")
            self.hippocampus.crystallize(
                formula=discovered_formula,
                target_rcln=self.model_rcln,
                formula_id="ns2d_law",
                domain="fluids",
            )
            print("  Formula crystallized!")
        else:
            print("  No formula discovered.")
        
        return discovered_formula
    
    def evaluate_rollout(self, model: nn.Module, n_steps: int = 10) -> Dict[str, float]:
        """
        Evaluate autoregressive rollout (multi-step prediction).
        
        Args:
            model: Model to evaluate
            n_steps: Number of rollout steps
        
        Returns:
            Dictionary with MSE at each step and average
        """
        model.eval()
        
        all_mse = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                x, y_true, re = batch
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                
                # Initial prediction
                x_curr = x
                step_mse = []
                
                for step in range(n_steps):
                    if isinstance(model, RCLNLayer):
                        y_pred = model(x_curr)
                    else:
                        y_pred = model(x_curr)
                    
                    # Compute MSE
                    mse = nn.functional.mse_loss(y_pred, y_true).item()
                    step_mse.append(mse)
                    
                    # Autoregressive: use prediction as next input
                    x_curr = y_pred
                
                all_mse.append(step_mse)
                
                # Only evaluate first batch for speed
                break
        
        all_mse = np.array(all_mse)
        avg_mse = all_mse.mean(axis=0)
        
        return {
            f"step_{i+1}_mse": avg_mse[i] for i in range(n_steps)
        }
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete cross-Reynolds experiment."""
        print("\n" + "="*60)
        print("Cross-Reynolds Generalization Experiment")
        print(f"Train: Re={self.config.train_re}, Test: Re={self.config.test_re}")
        print("="*60)
        
        start_time = time.time()
        
        # Train all models
        for name, model in self.models.items():
            history = self.train_model(model, name)
            self.results[f"{name}_history"] = history
        
        # Run discovery if enabled
        discovered_formula = None
        if self.config.run_discovery:
            discovered_formula = self.run_discovery()
            self.results["discovered_formula"] = discovered_formula
        
        # Evaluate rollout performance
        print(f"\n{'='*60}")
        print("Evaluating Autoregressive Rollout")
        print(f"{'='*60}")
        
        for name, model in self.models.items():
            rollout_results = self.evaluate_rollout(model, n_steps=self.config.n_rollout_steps)
            self.results[f"{name}_rollout"] = rollout_results
            
            print(f"\n{name} Rollout MSE:")
            for step, mse in rollout_results.items():
                print(f"  {step}: {mse:.6f}")
        
        # Final test performance
        print(f"\n{'='*60}")
        print("Final Test Performance (Single Step)")
        print(f"{'='*60}")
        
        for name, model in self.models.items():
            model.eval()
            test_losses = []
            
            with torch.no_grad():
                for batch in self.test_loader:
                    x, y, re = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    if isinstance(model, RCLNLayer):
                        pred = model(x)
                    else:
                        pred = model(x)
                    
                    loss = nn.functional.mse_loss(pred, y)
                    test_losses.append(loss.item())
            
            avg_test = np.mean(test_losses)
            self.results[f"{name}_final_test_mse"] = avg_test
            print(f"  {name}: {avg_test:.6f}")
        
        elapsed = time.time() - start_time
        self.results["elapsed_time"] = elapsed
        
        print(f"\n{'='*60}")
        print(f"Experiment completed in {elapsed:.1f}s")
        print(f"{'='*60}")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results to disk."""
        results_file = self.output_dir / "results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results_serializable = convert(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save models
        for name, model in self.models.items():
            model_file = self.output_dir / f"{name.lower()}_model.pt"
            torch.save(model.state_dict(), model_file)
        
        # Save config
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)


def main():
    """Main entry point."""
    config = CrossReynoldsConfig(
        data_dir="./data/pdebench",
        dataset_name="2D_NavierStokes",
        resolution=64,
        train_re=100,
        test_re=1000,
        epochs=50,
        batch_size=16,
        run_discovery=True,
        n_rollout_steps=5,
        output_dir="./outputs/cross_reynolds_experiment",
    )
    
    experiment = CrossReynoldsExperiment(config)
    results = experiment.run_full_experiment()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nFinal Test MSE:")
    for name in ["FNO", "U-Net", "RCLN"]:
        key = f"{name}_final_test_mse"
        if key in results:
            print(f"  {name:10s}: {results[key]:.6f}")
    
    if "discovered_formula" in results and results["discovered_formula"]:
        print(f"\nDiscovered Formula:")
        print(f"  {results['discovered_formula']}")
        
        print(f"\nKey Question: Does this formula generalize across Reynolds numbers?")
        print(f"  - If yes: Axiom-OS discovered Reynolds-invariant physics!")
        print(f"  - If no:  Discovery found fitting, not understanding.")


if __name__ == "__main__":
    main()
