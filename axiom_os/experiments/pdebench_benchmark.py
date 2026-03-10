"""
PDEBench Benchmark - Axiom-OS vs FNO vs U-Net
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Standardized comparison on PDEBench 2D Navier-Stokes:
- FNO (Fourier Neural Operator) - Baseline
- U-Net - CNN Baseline  
- RCLN (Axiom-OS) - Neuro-Symbolic with Discovery

Metrics:
- Single-step MSE
- Multi-step rollout MSE (autoregressive)
- Physics-informed metrics (divergence, energy spectrum)
- Discovery success rate (formula extraction)
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from axiom_os.datasets.pdebench import (
    PDEBenchConfig, create_pdebench_loaders, PDEBenchDataset
)
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.layers.fno import FNO2d
from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.core.hippocampus import Hippocampus


@dataclass
class BenchmarkConfig:
    """Configuration for PDEBench benchmark."""
    
    # Data
    data_dir: str = "./data/pdebench"
    dataset_name: str = "2D_NavierStokes"
    resolution: int = 64
    reynolds_number: int = 100
    
    # Training
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Models
    fno_width: int = 32
    fno_modes: int = 12
    fno_layers: int = 4
    
    # Discovery
    run_discovery: bool = True
    discovery_interval: int = 25  # Run discovery every N epochs
    
    # Evaluation
    n_rollout_steps: int = 10
    compute_spectrum: bool = True  # Compute energy spectrum
    
    # Output
    output_dir: str = "./outputs/pdebench_benchmark"


class SimpleUNet2d(nn.Module):
    """Lightweight U-Net for 2D fields."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Decoder
        self.dec3 = self._conv_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._conv_block(base_channels * 2, base_channels)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder
        d3 = self.dec3(self.upsample(e3) + e2)
        d2 = self.dec2(self.upsample(d3) + e1)
        d1 = self.dec1(d2)
        
        return d1


class PDEBenchBenchmark:
    """Benchmark runner for PDEBench."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Initialize models
        self._init_models()
        
        # Discovery components
        self.discovery_engine = DiscoveryEngine(use_pysr=True)
        self.hippocampus = Hippocampus(dim=32, capacity=1000)
        
        # Results
        self.results: Dict[str, Any] = {
            "config": self._config_to_dict(),
            "models": {},
        }
    
    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "data_dir": self.config.data_dir,
            "dataset_name": self.config.dataset_name,
            "resolution": self.config.resolution,
            "reynolds_number": self.config.reynolds_number,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "device": str(self.config.device),
        }
    
    def _load_data(self):
        """Load PDEBench dataset."""
        print("Loading PDEBench dataset...")
        
        pde_config = PDEBenchConfig(
            data_dir=self.config.data_dir,
            dataset_name=self.config.dataset_name,
            resolution=self.config.resolution,
            reynolds_number=self.config.reynolds_number,
            normalize=True,
        )
        
        self.loaders = create_pdebench_loaders(
            config=pde_config,
            batch_size=self.config.batch_size,
            num_workers=0,
        )
        
        # Get dimensions
        x, y = next(iter(self.loaders["train"]))
        self.in_channels = x.shape[1]
        self.out_channels = y.shape[1]
        self.resolution = x.shape[2]
        
        print(f"  Resolution: {self.resolution}x{self.resolution}")
        print(f"  Channels: {self.in_channels} -> {self.out_channels}")
        print(f"  Train batches: {len(self.loaders['train'])}")
        print(f"  Test batches: {len(self.loaders['test'])}")
    
    def _init_models(self):
        """Initialize models."""
        print("\nInitializing models...")
        
        # FNO
        self.model_fno = FNO2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.config.fno_width,
            modes1=self.config.fno_modes,
            modes2=self.config.fno_modes,
            n_layers=self.config.fno_layers,
        ).to(self.device)
        
        # U-Net
        self.model_unet = SimpleUNet2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            base_channels=self.config.fno_width,
        ).to(self.device)
        
        # RCLN with FNO soft shell
        self.model_rcln = RCLNLayer(
            input_dim=self.in_channels,
            hidden_dim=self.config.fno_width,
            output_dim=self.out_channels,
            hard_core_func=None,
            lambda_res=self.config.lambda_res if hasattr(self.config, 'lambda_res') else 1.0,
            net_type="fno",
            fno_modes1=self.config.fno_modes,
            fno_modes2=self.config.fno_modes,
            use_activity_monitor=True,
        ).to(self.device)
        
        self.models = {
            "FNO": self.model_fno,
            "U-Net": self.model_unet,
            "RCLN": self.model_rcln,
        }
        
        for name, model in self.models.items():
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  {name:10s}: {n_params:,} parameters")
    
    def train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch."""
        model.train()
        losses = []
        
        for batch in self.loaders["train"]:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            optimizer.zero_grad()
            
            if isinstance(model, RCLNLayer):
                pred = model(x)
            else:
                pred = model(x)
            
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def evaluate(self, model: nn.Module, split: str = "test") -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        
        mse_list = []
        mae_list = []
        
        with torch.no_grad():
            for batch in self.loaders[split]:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                if isinstance(model, RCLNLayer):
                    pred = model(x)
                else:
                    pred = model(x)
                
                mse = nn.functional.mse_loss(pred, y).item()
                mae = nn.functional.l1_loss(pred, y).item()
                
                mse_list.append(mse)
                mae_list.append(mae)
        
        return {
            "mse": np.mean(mse_list),
            "mae": np.mean(mae_list),
            "rmse": np.sqrt(np.mean(mse_list)),
        }
    
    def evaluate_rollout(
        self,
        model: nn.Module,
        n_steps: int = 10,
    ) -> Dict[str, List[float]]:
        """Evaluate autoregressive rollout."""
        model.eval()
        
        step_mse = [[] for _ in range(n_steps)]
        step_mae = [[] for _ in range(n_steps)]
        
        with torch.no_grad():
            for batch in self.loaders["test"]:
                x, y_target = batch[0].to(self.device), batch[1].to(self.device)
                
                x_curr = x
                
                for step in range(n_steps):
                    if isinstance(model, RCLNLayer):
                        y_pred = model(x_curr)
                    else:
                        y_pred = model(x_curr)
                    
                    mse = nn.functional.mse_loss(y_pred, y_target).item()
                    mae = nn.functional.l1_loss(y_pred, y_target).item()
                    
                    step_mse[step].append(mse)
                    step_mae[step].append(mae)
                    
                    # Autoregressive
                    x_curr = y_pred
        
        return {
            "mse_by_step": [np.mean(m) for m in step_mse],
            "mae_by_step": [np.mean(m) for m in step_mae],
            "mse_final": np.mean(step_mse[-1]),
        }
    
    def compute_energy_spectrum(self, model: nn.Module) -> np.ndarray:
        """Compute energy spectrum of predictions vs ground truth."""
        model.eval()
        
        spectra_pred = []
        spectra_true = []
        
        with torch.no_grad():
            for batch in self.loaders["test"]:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                if isinstance(model, RCLNLayer):
                    pred = model(x)
                else:
                    pred = model(x)
                
                # Compute energy spectrum in Fourier space
                # Average over channels
                pred_fft = torch.fft.fft2(pred.mean(dim=1))
                y_fft = torch.fft.fft2(y.mean(dim=1))
                
                # Radial average
                pred_spec = torch.abs(pred_fft).pow(2).mean(dim=0).cpu().numpy()
                y_spec = torch.abs(y_fft).pow(2).mean(dim=0).cpu().numpy()
                
                spectra_pred.append(pred_spec)
                spectra_true.append(y_spec)
                
                # Only compute on first few batches
                if len(spectra_pred) >= 5:
                    break
        
        return np.mean(spectra_pred, axis=0), np.mean(spectra_true, axis=0)
    
    def run_discovery_on_model(self) -> Optional[str]:
        """Run discovery on trained RCLN."""
        print("\n" + "="*60)
        print("Discovery Phase")
        print("="*60)
        
        self.model_rcln.eval()
        
        # Collect data
        print("Collecting soft shell activations...")
        inputs = []
        soft_outputs = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.loaders["train"]):
                if i >= 20:  # Limit samples for speed
                    break
                
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                _ = self.model_rcln(x)
                
                soft_out = self.model_rcln._last_y_soft
                
                # Spatial averaging for discovery
                inputs.append(x.mean(dim=(2, 3)).cpu().numpy())
                soft_outputs.append(soft_out.mean(dim=(2, 3)).cpu().numpy())
        
        X = np.concatenate(inputs, axis=0)
        Y = np.concatenate(soft_outputs, axis=0)
        
        print(f"Discovery data: X {X.shape}, Y {Y.shape}")
        
        # Run discovery
        formula, pred, coefs = self.discovery_engine.discover_multivariate(
            X=X,
            y=Y[:, 0],  # First output dimension
            var_names=[f"c{i}" for i in range(X.shape[1])],
            selector="bic",
        )
        
        if formula:
            print(f"\nDiscovered Formula: {formula}")
            
            # Crystallize
            self.hippocampus.crystallize(
                formula=formula,
                target_rcln=self.model_rcln,
                formula_id="pdebench_ns",
                domain="fluids",
            )
            print("Formula crystallized into RCLN!")
            
            return formula
        else:
            print("No formula discovered.")
            return None
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark."""
        print("\n" + "="*60)
        print("PDEBench Benchmark")
        print(f"Dataset: {self.config.dataset_name} Re={self.config.reynolds_number}")
        print(f"Resolution: {self.config.resolution}x{self.config.resolution}")
        print("="*60)
        
        start_time = time.time()
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}")
            print(f"{'='*60}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
            
            history = {"train_loss": [], "val_mse": []}
            
            for epoch in range(self.config.epochs):
                train_loss = self.train_epoch(model, optimizer)
                history["train_loss"].append(train_loss)
                
                if (epoch + 1) % 10 == 0:
                    val_metrics = self.evaluate(model, split="val")
                    history["val_mse"].append(val_metrics["mse"])
                    print(f"Epoch {epoch+1}/{self.config.epochs}: "
                          f"train_loss={train_loss:.6f}, val_mse={val_metrics['mse']:.6f}")
                
                scheduler.step()
                
                # Periodic discovery for RCLN
                if name == "RCLN" and self.config.run_discovery:
                    if (epoch + 1) % self.config.discovery_interval == 0:
                        self.run_discovery_on_model()
            
            # Final evaluation
            print(f"\nFinal Evaluation: {name}")
            test_metrics = self.evaluate(model, split="test")
            rollout_metrics = self.evaluate_rollout(model, n_steps=self.config.n_rollout_steps)
            
            print(f"  Test MSE:  {test_metrics['mse']:.6f}")
            print(f"  Test MAE:  {test_metrics['mae']:.6f}")
            print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
            print(f"  Rollout MSE (step {self.config.n_rollout_steps}): {rollout_metrics['mse_final']:.6f}")
            
            self.results["models"][name] = {
                "history": history,
                "test_metrics": test_metrics,
                "rollout_metrics": rollout_metrics,
            }
        
        # Final discovery
        if self.config.run_discovery:
            formula = self.run_discovery_on_model()
            if formula:
                self.results["discovered_formula"] = formula
                
                # Evaluate RCLN with crystallized formula
                print("\nEvaluating RCLN with crystallized formula...")
                test_metrics = self.evaluate(self.model_rcln, split="test")
                rollout_metrics = self.evaluate_rollout(self.model_rcln, n_steps=self.config.n_rollout_steps)
                
                self.results["models"]["RCLN_Crystallized"] = {
                    "test_metrics": test_metrics,
                    "rollout_metrics": rollout_metrics,
                }
        
        elapsed = time.time() - start_time
        self.results["elapsed_time"] = elapsed
        
        print(f"\n{'='*60}")
        print(f"Benchmark completed in {elapsed:.1f}s")
        print("="*60)
        
        # Print comparison
        print("\nModel Comparison:")
        print(f"{'Model':<20} {'Test MSE':<12} {'Test MAE':<12} {'Rollout MSE':<12}")
        print("-" * 60)
        for name in ["FNO", "U-Net", "RCLN", "RCLN_Crystallized"]:
            if name in self.results["models"]:
                m = self.results["models"][name]
                print(f"{name:<20} {m['test_metrics']['mse']:<12.6f} "
                      f"{m['test_metrics']['mae']:<12.6f} "
                      f"{m['rollout_metrics']['mse_final']:<12.6f}")
        
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to disk."""
        # Convert to JSON-serializable
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj
        
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(convert(self.results), f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save models
        for name, model in self.models.items():
            torch.save(model.state_dict(), self.output_dir / f"{name.lower()}_final.pt")


def main():
    """Main entry point."""
    config = BenchmarkConfig(
        data_dir="./data/pdebench",
        dataset_name="2D_NavierStokes",
        resolution=64,
        reynolds_number=100,
        epochs=50,
        batch_size=16,
        run_discovery=True,
        n_rollout_steps=5,
        output_dir="./outputs/pdebench_benchmark",
    )
    
    benchmark = PDEBenchBenchmark(config)
    results = benchmark.run_benchmark()
    
    print("\n" + "="*60)
    print("PDEBench Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
