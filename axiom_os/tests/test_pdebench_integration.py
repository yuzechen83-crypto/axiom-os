"""
Test PDEBench Integration for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Tests:
1. Data loading
2. Cross-Reynolds dataset
3. Model training loop
4. Discovery on RCLN
"""

import sys
from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch
import pytest

# Add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.pdebench import (
    PDEBenchConfig, PDEBenchDataset, CrossReynoldsDataset,
    create_pdebench_loaders, create_cross_reynolds_loaders
)
from axiom_os.experiments.cross_reynolds_generalization import (
    CrossReynoldsConfig, CrossReynoldsExperiment, UNet2d
)
from axiom_os.layers.fno import FNO2d
from axiom_os.layers.rcln import RCLNLayer


class MockPDEBenchData:
    """Create mock PDEBench data for testing without downloading."""
    
    @staticmethod
    def create_hdf5(filepath: Path, n_samples: int = 100, resolution: int = 32, re: int = 100):
        """Create mock HDF5 file in PDEBench format."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Create synthetic Navier-Stokes-like data
            # Input: velocity field at t=0
            inputs = np.random.randn(n_samples, resolution, resolution, 2).astype(np.float32)
            # Output: velocity field at t=dt
            outputs = inputs + 0.1 * np.random.randn(n_samples, resolution, resolution, 2).astype(np.float32)
            
            f.create_dataset('input', data=inputs)
            f.create_dataset('output', data=outputs)
            
            # Attributes
            f.attrs['Re'] = float(re)
            f.attrs['viscosity'] = 1.0 / re
        
        return filepath


class TestPDEBenchDataLoading:
    """Test PDEBench data loading functionality."""
    
    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory with mock data."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock data files
        MockPDEBenchData.create_hdf5(
            temp_dir / "2D_NavierStokes_Re100_32x32.h5",
            n_samples=100, resolution=32, re=100
        )
        MockPDEBenchData.create_hdf5(
            temp_dir / "2D_NavierStokes_Re1000_32x32.h5",
            n_samples=100, resolution=32, re=1000
        )
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_pdebench_config(self):
        """Test PDEBenchConfig creation."""
        config = PDEBenchConfig(
            data_dir="./data/pdebench",
            resolution=64,
            reynolds_number=100,
        )
        
        assert config.resolution == 64
        assert config.reynolds_number == 100
        assert config.normalize == True
    
    def test_dataset_loading(self, temp_data_dir):
        """Test PDEBenchDataset loading."""
        config = PDEBenchConfig(
            data_dir=temp_data_dir,
            dataset_name="2D_NavierStokes",
            resolution=32,
            reynolds_number=100,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        
        # Test train split
        train_ds = PDEBenchDataset(config, split="train")
        assert len(train_ds) == 70  # 70% of 100
        
        # Test getting item
        x, y = train_ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (2, 32, 32)  # (C, H, W)
        assert y.shape == (2, 32, 32)
    
    def test_cross_reynolds_dataset(self, temp_data_dir):
        """Test CrossReynoldsDataset."""
        ds = CrossReynoldsDataset(
            data_dir=temp_data_dir,
            dataset_name="2D_NavierStokes",
            resolution=32,
            train_re=100,
            test_re=1000,
            split="train",
        )
        
        assert len(ds) == 100
        
        x, y, re = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert re == 100  # Training data is Re=100
    
    def test_dataloader_creation(self, temp_data_dir):
        """Test DataLoader creation."""
        config = PDEBenchConfig(
            data_dir=temp_data_dir,
            resolution=32,
            reynolds_number=100,
        )
        
        loaders = create_pdebench_loaders(config, batch_size=8, num_workers=0)
        
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders
        
        # Test batch
        x, y = next(iter(loaders["train"]))
        assert x.shape[0] <= 8  # Batch size
        assert x.shape[1:] == (2, 32, 32)  # (C, H, W)


class TestModels:
    """Test models on PDEBench-like data."""
    
    def test_fno_forward(self):
        """Test FNO2d forward pass."""
        model = FNO2d(
            in_channels=2,
            out_channels=2,
            width=16,
            modes1=4,
            modes2=4,
            n_layers=2,
        )
        
        # Test input
        x = torch.randn(4, 2, 32, 32)  # (B, C, H, W)
        y = model(x)
        
        assert y.shape == (4, 2, 32, 32)
    
    def test_unet_forward(self):
        """Test UNet2d forward pass."""
        model = UNet2d(
            in_channels=2,
            out_channels=2,
            hidden_dim=16,
            n_layers=2,
        )
        
        x = torch.randn(4, 2, 32, 32)
        y = model(x)
        
        assert y.shape == (4, 2, 32, 32)
    
    def test_rcln_forward(self):
        """Test RCLNLayer with FNO soft shell."""
        model = RCLNLayer(
            input_dim=2,
            hidden_dim=16,
            output_dim=2,
            net_type="fno",
            fno_modes1=4,
            fno_modes2=4,
        )
        
        x = torch.randn(4, 2, 32, 32)
        y = model(x)
        
        assert y.shape == (4, 2, 32, 32)


class TestTraining:
    """Test training functionality."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for training test."""
        # Synthetic data: (B, C, H, W)
        X = torch.randn(20, 2, 32, 32)
        Y = X + 0.1 * torch.randn(20, 2, 32, 32)  # Slight perturbation
        return X, Y
    
    def test_fno_training_step(self, mock_data):
        """Test single training step for FNO."""
        X, Y = mock_data
        
        model = FNO2d(in_channels=2, out_channels=2, width=16, modes1=4, modes2=4, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_rcln_training_step(self, mock_data):
        """Test single training step for RCLN."""
        X, Y = mock_data
        
        model = RCLNLayer(
            input_dim=2, hidden_dim=16, output_dim=2,
            net_type="fno", fno_modes1=4, fno_modes2=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestDiscovery:
    """Test Discovery Engine integration."""
    
    def test_discovery_engine_init(self):
        """Test DiscoveryEngine initialization."""
        from axiom_os.engine.discovery import DiscoveryEngine
        
        engine = DiscoveryEngine(use_pysr=False)  # Skip PySR dependency
        assert engine is not None
    
    def test_discovery_on_synthetic_data(self):
        """Test discovery on simple synthetic data."""
        from axiom_os.engine.discovery import DiscoveryEngine
        
        engine = DiscoveryEngine(use_pysr=False)
        
        # Simple linear relationship: y = 2*x0 + 3*x1
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * np.random.randn(100)
        
        formula, pred, coefs = engine.discover_multivariate(
            X=X, y=y, var_names=["x0", "x1"], selector="bic"
        )
        
        assert formula is not None
        print(f"Discovered formula: {formula}")


def run_tests():
    """Run all tests."""
    print("="*60)
    print("PDEBench Integration Tests")
    print("="*60)
    
    # Check if h5py is available
    try:
        import h5py
    except ImportError:
        print("WARNING: h5py not installed. Some tests will be skipped.")
        print("Install with: pip install h5py")
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
