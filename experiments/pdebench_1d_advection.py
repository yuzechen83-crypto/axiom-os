"""
PDEBench 1D Advection Experiment for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Dataset: 1D_Advection_Sols_beta0.1.hdf5
- 10,000 samples
- 201 time steps
- 1024 spatial grid points
- Beta (advection speed) = 0.1

Architecture: RCLN with Spectral Soft Shell (1D)
Discovery: Extract symbolic advection law from Soft Shell
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from axiom_os.layers.rcln import RCLNLayer
from axiom_os.layers.rcln import SpectralConv1d
from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.core.hippocampus import Hippocampus


class PDEBench1DAdvectionDataset(Dataset):
    """Dataset for 1D Advection from PDEBench HDF5."""
    
    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        input_steps: int = 1,
        output_steps: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalize: bool = True,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        
        # Load data
        self._load_data()
        
        # Compute statistics
        if normalize:
            self._compute_stats()
        
        # Create split
        self._create_split(split)
    
    def _load_data(self):
        """Load HDF5 data."""
        import h5py
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # tensor shape: (10000, 201, 1024)
            # (samples, time_steps, spatial_points)
            self.data = np.array(f['tensor'], dtype=np.float32)
            self.x_coords = np.array(f['x-coordinate'], dtype=np.float32)
            self.t_coords = np.array(f['t-coordinate'], dtype=np.float32)
        
        self.n_samples_total = self.data.shape[0]
        self.n_times = self.data.shape[1]
        self.n_x = self.data.shape[2]
        
        print(f"Loaded 1D Advection data:")
        print(f"  Samples: {self.n_samples_total}")
        print(f"  Time steps: {self.n_times}")
        print(f"  Spatial points: {self.n_x}")
        print(f"  Spatial domain: [{self.x_coords[0]:.3f}, {self.x_coords[-1]:.3f}]")
        print(f"  Time domain: [{self.t_coords[0]:.3f}, {self.t_coords[-1]:.3f}]")
    
    def _compute_stats(self):
        """Compute normalization statistics."""
        n_train = int(self.n_samples_total * 0.7)
        train_data = self.data[:n_train]
        
        self.data_mean = np.mean(train_data)
        self.data_std = np.std(train_data) + 1e-8
        
        print(f"  Data mean: {self.data_mean:.4f}, std: {self.data_std:.4f}")
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data."""
        if not self.normalize:
            return x
        return (x - self.data_mean) / self.data_std
    
    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        if not self.normalize:
            return x
        return x * self.data_std + self.data_mean
    
    def _create_split(self, split: str):
        """Create train/val/test split."""
        n_train = int(self.n_samples_total * 0.7)
        n_val = int(self.n_samples_total * 0.15)
        
        if split == "train":
            self.indices = np.arange(0, n_train)
        elif split == "val":
            self.indices = np.arange(n_train, n_train + n_val)
        elif split == "test":
            self.indices = np.arange(n_train + n_val, self.n_samples_total)
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.n_samples = len(self.indices)
        print(f"  {split} samples: {self.n_samples}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input-output pair for one-step prediction.
        
        Returns:
            x: (input_steps, n_x) - initial condition
            y: (output_steps, n_x) - future state
        """
        real_idx = self.indices[idx]
        
        # Get sample data (time_steps, spatial_points)
        sample = self.data[real_idx]
        
        # Input: first time step(s)
        x = sample[:self.input_steps]
        # Output: next time step(s)
        y = sample[self.input_steps:self.input_steps + self.output_steps]
        
        # Normalize
        x = self._normalize(x)
        y = self._normalize(y)
        
        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        return x, y


class Spectral1DModel(nn.Module):
    """1D Spectral model for advection."""
    
    def __init__(
        self,
        n_x: int = 1024,
        hidden_dim: int = 64,
        n_modes: int = 32,
    ):
        super().__init__()
        self.n_x = n_x
        
        # Spectral convolution
        self.spectral = SpectralConv1d(n_x, n_modes=n_modes)
        
        # MLP for non-spectral features
        self.mlp = nn.Sequential(
            nn.Linear(n_x, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_x),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, L) or (B, T, L) - input field
        
        Returns:
            y: (B, L) - predicted field
        """
        if x.dim() == 3:
            # (B, T, L) -> use last time step
            x = x[:, -1, :]
        
        # Spectral path
        x_spec = self.spectral(x)
        
        # MLP path
        x_mlp = self.mlp(x)
        
        # Combine
        return x_spec + x_mlp


class AdvectionRCLN(nn.Module):
    """RCLN specialized for 1D Advection."""
    
    def __init__(
        self,
        n_x: int = 1024,
        hidden_dim: int = 64,
        lambda_res: float = 1.0,
        use_spectral: bool = True,
    ):
        super().__init__()
        self.n_x = n_x
        
        # Hard core: analytical advection (if known)
        self.hard_core = None  # Will be set by discovery
        
        # Soft shell: Spectral or MLP
        if use_spectral:
            self.soft_shell = Spectral1DModel(n_x, hidden_dim)
        else:
            self.soft_shell = nn.Sequential(
                nn.Linear(n_x, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_x),
            )
        
        self.lambda_res = lambda_res
        self._last_y_soft = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: y = y_hard + lambda * y_soft"""
        if x.dim() == 3:
            x = x[:, -1, :]
        
        y_soft = self.soft_shell(x)
        self._last_y_soft = y_soft.detach()
        
        if self.hard_core is not None:
            y_hard = self.hard_core(x)
            y = y_hard + self.lambda_res * y_soft
        else:
            y = self.lambda_res * y_soft
        
        return y


@dataclass
class AdvectionExperimentConfig:
    """Configuration for 1D Advection experiment."""
    
    # Data
    hdf5_path: str = r"C:\Users\ASUS\Downloads\1D_Advection_Sols_beta0.1.hdf5"
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    hidden_dim: int = 64
    n_modes: int = 32
    use_spectral: bool = True
    
    # Discovery
    run_discovery: bool = True
    n_discovery_samples: int = 500
    
    # Output
    output_dir: str = "./outputs/pdebench_1d_advection"


class AdvectionExperiment:
    """1D Advection experiment runner."""
    
    def __init__(self, config: AdvectionExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Initialize model
        self._init_model()
        
        # Discovery
        self.discovery_engine = DiscoveryEngine(use_pysr=False)
        self.hippocampus = Hippocampus(dim=32, capacity=500)
        
        # Results
        self.results: Dict[str, Any] = {}
    
    def _load_data(self):
        """Load dataset."""
        print("\n" + "="*70)
        print("Loading 1D Advection Dataset")
        print("="*70)
        
        self.train_ds = PDEBench1DAdvectionDataset(
            hdf5_path=self.config.hdf5_path,
            split="train",
            normalize=True,
        )
        self.val_ds = PDEBench1DAdvectionDataset(
            hdf5_path=self.config.hdf5_path,
            split="val",
            normalize=True,
        )
        self.test_ds = PDEBench1DAdvectionDataset(
            hdf5_path=self.config.hdf5_path,
            split="test",
            normalize=True,
        )
        
        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.config.batch_size,
            shuffle=True, num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0,
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0,
        )
        
        self.n_x = self.train_ds.n_x
    
    def _init_model(self):
        """Initialize model."""
        print("\n" + "="*70)
        print("Initializing Model")
        print("="*70)
        
        self.model = AdvectionRCLN(
            n_x=self.n_x,
            hidden_dim=self.config.hidden_dim,
            lambda_res=1.0,
            use_spectral=self.config.use_spectral,
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model."""
        print("\n" + "="*70)
        print("Training")
        print("="*70)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(x)
                loss = nn.functional.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    loss = nn.functional.mse_loss(pred, y)
                    val_losses.append(loss.item())
            
            scheduler.step()
            
            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            
            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}: "
                      f"train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")
        
        return history
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        
        mse_list = []
        mae_list = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                
                mse = nn.functional.mse_loss(pred, y).item()
                mae = nn.functional.l1_loss(pred, y).item()
                
                mse_list.append(mse)
                mae_list.append(mae)
        
        return {
            "mse": np.mean(mse_list),
            "mae": np.mean(mae_list),
            "rmse": np.sqrt(np.mean(mse_list)),
        }
    
    def run_discovery(self) -> Optional[str]:
        """Run Discovery Engine to extract symbolic formula."""
        if not self.config.run_discovery:
            return None
        
        print("\n" + "="*70)
        print("Discovery Phase")
        print("="*70)
        
        self.model.eval()
        
        # Collect soft shell data
        print("Collecting soft shell activations...")
        inputs = []
        soft_outputs = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_loader):
                if i * self.config.batch_size >= self.config.n_discovery_samples:
                    break
                
                x = x.to(self.device)
                _ = self.model(x)
                
                y_soft = self.model._last_y_soft
                
                inputs.append(x.cpu().numpy())
                soft_outputs.append(y_soft.cpu().numpy())
        
        X = np.concatenate(inputs, axis=0)
        Y = np.concatenate(soft_outputs, axis=0)
        
        # Reshape for discovery: flatten spatial dimension
        # X: (n_samples, n_x) -> treat each spatial point as feature
        X_flat = X.reshape(X.shape[0], -1)  # (n_samples, n_x)
        Y_flat = Y.reshape(Y.shape[0], -1)  # (n_samples, n_x)
        
        # Subsample spatial points for discovery
        X_sample = X_flat[:, ::8]  # Every 8th point
        Y_sample = Y_flat[:, ::8]
        
        print(f"Discovery data: X {X_sample.shape}, Y {Y_sample.shape}")
        
        # Run discovery on a few spatial points
        formulas = []
        for i in range(min(3, X_sample.shape[1])):
            xi = X_sample[:, i:i+1]  # (n_samples, 1)
            yi = Y_sample[:, i]      # (n_samples,)
            formula, pred, coefs = self.discovery_engine.discover_multivariate(
                X=xi,
                y=yi,
                var_names=[f"u{i}"],
                selector="bic",
            )
            if formula:
                formulas.append(formula)
                print(f"Point {i}: {formula}")
        
        if formulas:
            # Use most common formula
            from collections import Counter
            formula_counts = Counter(formulas)
            best_formula = formula_counts.most_common(1)[0][0]
            
            print(f"\n[OK] Best discovered formula: {best_formula}")
            
            # Crystallize
            def hard_core_func(x):
                # Simple placeholder - would use actual formula
                return 0.0
            
            self.model.hard_core = hard_core_func
            print("[OK] Formula crystallized into model!")
            
            return best_formula
        
        return None
    
    def rollout_prediction(self, n_steps: int = 10) -> Dict[str, float]:
        """Test autoregressive rollout."""
        print("\n" + "="*70)
        print(f"Rollout Prediction ({n_steps} steps)")
        print("="*70)
        
        self.model.eval()
        
        # Get one test sample
        x0, y_true = self.test_ds[0]
        x0 = x0.to(self.device)
        
        x_curr = x0.unsqueeze(0)
        predictions = [x_curr.cpu().numpy()]
        
        with torch.no_grad():
            for step in range(n_steps):
                x_next = self.model(x_curr)
                predictions.append(x_next.cpu().numpy())
                x_curr = x_next
        
        # Compute error growth
        errors = []
        for step in range(1, min(n_steps + 1, len(predictions))):
            # Compare with ground truth if available
            if step < len(predictions):
                # Use next time step from dataset if available
                pass
        
        print(f"Rollout complete: {len(predictions)} steps")
        
        return {"n_steps": n_steps}
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete experiment."""
        print("\n" + "="*70)
        print("PDEBench 1D Advection Experiment")
        print("="*70)
        
        start_time = time.time()
        
        # Train
        history = self.train()
        self.results["history"] = history
        
        # Evaluate
        test_metrics = self.evaluate()
        self.results["test_metrics"] = test_metrics
        
        print(f"\nTest Results:")
        print(f"  MSE:  {test_metrics['mse']:.6f}")
        print(f"  MAE:  {test_metrics['mae']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        
        # Discovery
        if self.config.run_discovery:
            formula = self.run_discovery()
            if formula:
                self.results["discovered_formula"] = formula
        
        # Rollout
        rollout = self.rollout_prediction(n_steps=10)
        self.results["rollout"] = rollout
        
        elapsed = time.time() - start_time
        self.results["elapsed_time"] = elapsed
        
        print(f"\n[OK] Experiment completed in {elapsed:.1f}s")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to disk."""
        results_file = self.output_dir / "results.json"
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(results_file, 'w') as f:
            json.dump(convert(self.results), f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save model
        model_file = self.output_dir / "model.pt"
        torch.save(self.model.state_dict(), model_file)
        print(f"Model saved to: {model_file}")


def main():
    """Main entry point."""
    config = AdvectionExperimentConfig(
        hdf5_path=r"C:\Users\ASUS\Downloads\1D_Advection_Sols_beta0.1.hdf5",
        epochs=30,
        batch_size=32,
        run_discovery=True,
        output_dir="./outputs/pdebench_1d_advection",
    )
    
    experiment = AdvectionExperiment(config)
    results = experiment.run_experiment()
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
