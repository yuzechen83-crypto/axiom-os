"""
PDEBench Data Loader - SciML Standard Benchmark
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

PDEBench is the "ImageNet" for AI4Science, providing standardized PDE datasets:
- 2D Navier-Stokes (incompressible, periodic BC)
- 1D/2D/3D Burgers
- Shallow Water Equations
- etc.

Key Feature: Cross-Reynolds generalization test (Re=100 train -> Re=1000 test).
This tests whether the Discovery Engine found physics laws that are 
Reynolds-number invariant (truth should be).

Data format: HDF5 with structure:
    - input: (n_samples, H, W, T_in, C) - initial condition
    - output: (n_samples, H, W, T_out, C) - future state
    - viscosity, Reynolds number metadata

Reference: https://github.com/pdebench/PDEBench
"""

from typing import Tuple, Optional, Dict, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class PDEBenchConfig:
    """Configuration for PDEBench dataset loading."""
    
    # Data paths
    data_dir: Union[str, Path] = "./data/pdebench"
    dataset_name: str = "2D_NavierStokes"  # or "1D_Burgers", "3D_NavierStokes"
    
    # Grid parameters
    resolution: int = 64  # 64 or 128 for 2D NS
    n_input_steps: int = 1  # Number of input time steps
    n_output_steps: int = 1  # Number of output time steps (prediction horizon)
    
    # Physical parameters
    reynolds_number: Optional[int] = None  # None = load all available Re
    viscosity: Optional[float] = None  # Alternative to Re
    
    # Data preprocessing
    normalize: bool = True
    normalize_method: str = "standard"  # "standard", "minmax", "none"
    
    # Train/test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)


class PDEBenchDataset(Dataset):
    """
    PDEBench Dataset for 2D Navier-Stokes and other PDEs.
    
    Loads HDF5 data with lazy loading support for large datasets.
    """
    
    def __init__(
        self,
        config: PDEBenchConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_reynolds: bool = False,
    ):
        self.config = config
        self.split = split
        self.transform = transform
        self.return_reynolds = return_reynolds
        
        # Load data
        self._load_data()
        
        # Compute statistics for normalization
        if config.normalize:
            self._compute_stats()
        
        # Create train/val/test split
        self._create_split()
    
    def _load_data(self):
        """Load HDF5 data from PDEBench format."""
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "PDEBench requires h5py. Install: pip install h5py"
            )
        
        # Construct file path
        # PDEBench naming: 2D_NavierStokes_cond_Re100_64x64.h5
        res_str = f"{self.config.resolution}x{self.config.resolution}"
        
        if self.config.reynolds_number is not None:
            re_str = f"Re{self.config.reynolds_number}"
            filename = f"{self.config.dataset_name}_{re_str}_{res_str}.h5"
        else:
            # Load all available Reynolds numbers
            filename = f"{self.config.dataset_name}_*.h5"
        
        filepath = self.config.data_dir / filename
        
        # Check if file exists, if not try to find any matching file
        if not filepath.exists():
            if filepath.parent.exists():
                # Try to find any matching file
                matching_files = list(filepath.parent.glob(filename.replace("*.h5", "*.h5")))
                if matching_files:
                    filepath = matching_files[0]
                    warnings.warn(f"Using found file: {filepath}")
                else:
                    raise FileNotFoundError(
                        f"PDEBench data not found at {filepath}. "
                        f"Please download from https://github.com/pdebench/PDEBench"
                    )
            else:
                raise FileNotFoundError(
                    f"PDEBench data directory not found: {self.config.data_dir}. "
                    f"Please download from https://github.com/pdebench/PDEBench"
                )
        
        # Load HDF5
        with h5py.File(filepath, 'r') as f:
            # PDEBench format: 'input' and 'output' datasets
            if 'input' in f:
                self.inputs = np.array(f['input'], dtype=np.float32)
                self.outputs = np.array(f['output'], dtype=np.float32)
            elif 'data' in f:
                # Alternative format: single data array
                data = np.array(f['data'], dtype=np.float32)
                # Assume last dim is time steps, split into input/output
                n_time = data.shape[-1]
                self.inputs = data[..., :self.config.n_input_steps]
                self.outputs = data[..., self.config.n_input_steps:self.config.n_input_steps + self.config.n_output_steps]
            else:
                raise ValueError(f"Unknown PDEBench format. Keys: {list(f.keys())}")
            
            # Load metadata if available
            self.metadata = {}
            if 'viscosity' in f.attrs:
                self.metadata['viscosity'] = float(f.attrs['viscosity'])
            if 'Re' in f.attrs or 'reynolds' in f.attrs:
                self.metadata['Re'] = float(f.attrs.get('Re', f.attrs.get('reynolds', 0)))
            
            # Try to infer Reynolds number from filename
            if 'Re' not in self.metadata:
                import re
                re_match = re.search(r'Re(\d+)', str(filepath))
                if re_match:
                    self.metadata['Re'] = int(re_match.group(1))
        
        # Ensure correct shape: (n_samples, H, W, C) or (n_samples, H, W, T, C)
        if self.inputs.ndim == 3:
            # Add channel dimension: (n_samples, H, W) -> (n_samples, H, W, 1)
            self.inputs = self.inputs[..., np.newaxis]
            self.outputs = self.outputs[..., np.newaxis]
        
        self.n_samples_total = self.inputs.shape[0]
        
        print(f"Loaded PDEBench data: {filepath}")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Output shape: {self.outputs.shape}")
        print(f"  Metadata: {self.metadata}")
    
    def _compute_stats(self):
        """Compute normalization statistics."""
        # Compute on training portion to avoid data leakage
        n_train = int(self.n_samples_total * self.config.train_ratio)
        
        train_inputs = self.inputs[:n_train]
        train_outputs = self.outputs[:n_train]
        
        if self.config.normalize_method == "standard":
            self.input_mean = np.mean(train_inputs, axis=(0, 1, 2), keepdims=True)
            self.input_std = np.std(train_inputs, axis=(0, 1, 2), keepdims=True) + 1e-8
            self.output_mean = np.mean(train_outputs, axis=(0, 1, 2), keepdims=True)
            self.output_std = np.std(train_outputs, axis=(0, 1, 2), keepdims=True) + 1e-8
        elif self.config.normalize_method == "minmax":
            self.input_min = np.min(train_inputs, axis=(0, 1, 2), keepdims=True)
            self.input_max = np.max(train_inputs, axis=(0, 1, 2), keepdims=True)
            self.output_min = np.min(train_outputs, axis=(0, 1, 2), keepdims=True)
            self.output_max = np.max(train_outputs, axis=(0, 1, 2), keepdims=True)
    
    def _normalize(self, x: np.ndarray, is_input: bool = True) -> np.ndarray:
        """Normalize data."""
        if not self.config.normalize:
            return x
        
        if self.config.normalize_method == "standard":
            if is_input:
                return (x - self.input_mean) / self.input_std
            else:
                return (x - self.output_mean) / self.output_std
        elif self.config.normalize_method == "minmax":
            if is_input:
                return (x - self.input_min) / (self.input_max - self.input_min + 1e-8)
            else:
                return (x - self.output_min) / (self.output_max - self.output_min + 1e-8)
        return x
    
    def _denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Denormalize output predictions."""
        if not self.config.normalize:
            return x
        
        if self.config.normalize_method == "standard":
            return x * self.output_std + self.output_mean
        elif self.config.normalize_method == "minmax":
            return x * (self.output_max - self.output_min + 1e-8) + self.output_min
        return x
    
    def _create_split(self):
        """Create train/val/test indices."""
        n_train = int(self.n_samples_total * self.config.train_ratio)
        n_val = int(self.n_samples_total * self.config.val_ratio)
        
        if self.split == "train":
            self.indices = np.arange(0, n_train)
        elif self.split == "val":
            self.indices = np.arange(n_train, n_train + n_val)
        elif self.split == "test":
            self.indices = np.arange(n_train + n_val, self.n_samples_total)
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.n_samples = len(self.indices)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                              Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Get a sample.
        
        Returns:
            (input, output) or (input, output, Reynolds_number)
        """
        real_idx = self.indices[idx]
        
        x = self.inputs[real_idx]
        y = self.outputs[real_idx]
        
        # Normalize
        x = self._normalize(x, is_input=True)
        y = self._normalize(y, is_input=False)
        
        # Convert to torch tensors and permute to (C, H, W)
        # Input shape: (H, W, C) or (H, W, T, C) -> we take last time step if T > 1
        if x.ndim == 4:  # (H, W, T, C)
            x = x[:, :, -1, :]  # Take last time step
        if y.ndim == 4:
            y = y[:, :, -1, :]
        
        # Permute: (H, W, C) -> (C, H, W)
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.from_numpy(y).permute(2, 0, 1).float()
        
        if self.transform:
            x = self.transform(x)
        
        if self.return_reynolds:
            re = self.metadata.get('Re', 0.0)
            return x, y, float(re)
        return x, y


class CrossReynoldsDataset(Dataset):
    """
    Cross-Reynolds dataset for testing generalization.
    
    Train on one Reynolds number, test on another.
    This tests if the model learned physics laws that are Reynolds-invariant.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_name: str = "2D_NavierStokes",
        resolution: int = 64,
        train_re: int = 100,
        test_re: int = 1000,
        split: str = "train",
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.resolution = resolution
        self.train_re = train_re
        self.test_re = test_re
        self.split = split
        
        # Load both datasets
        self._load_both_reynolds()
        
        # Normalize using training data statistics
        if normalize:
            self._compute_stats()
    
    def _load_both_reynolds(self):
        """Load both train and test Reynolds number data."""
        import h5py
        
        res_str = f"{self.resolution}x{self.resolution}"
        
        # Load train (low Re)
        train_file = self.data_dir / f"{self.dataset_name}_Re{self.train_re}_{res_str}.h5"
        if not train_file.exists():
            raise FileNotFoundError(f"Train data not found: {train_file}")
        
        with h5py.File(train_file, 'r') as f:
            self.train_inputs = np.array(f['input'], dtype=np.float32)
            self.train_outputs = np.array(f['output'], dtype=np.float32)
        
        # Load test (high Re)
        test_file = self.data_dir / f"{self.dataset_name}_Re{self.test_re}_{res_str}.h5"
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        with h5py.File(test_file, 'r') as f:
            self.test_inputs = np.array(f['input'], dtype=np.float32)
            self.test_outputs = np.array(f['output'], dtype=np.float32)
        
        print(f"Cross-Reynolds dataset loaded:")
        print(f"  Train Re={self.train_re}: {self.train_inputs.shape}")
        print(f"  Test Re={self.test_re}: {self.test_inputs.shape}")
    
    def _compute_stats(self):
        """Compute normalization stats from training data only."""
        self.input_mean = np.mean(self.train_inputs, axis=(0, 1, 2), keepdims=True)
        self.input_std = np.std(self.train_inputs, axis=(0, 1, 2), keepdims=True) + 1e-8
        self.output_mean = np.mean(self.train_outputs, axis=(0, 1, 2), keepdims=True)
        self.output_std = np.std(self.train_outputs, axis=(0, 1, 2), keepdims=True) + 1e-8
    
    def _normalize(self, x: np.ndarray, is_input: bool = True) -> np.ndarray:
        """Normalize using train stats."""
        if is_input:
            return (x - self.input_mean) / self.input_std
        else:
            return (x - self.output_mean) / self.output_std
    
    def _denormalize(self, x: np.ndarray, is_output: bool = True) -> np.ndarray:
        """Denormalize."""
        if is_output:
            return x * self.output_std + self.output_mean
        else:
            return x * self.input_std + self.input_mean
    
    def __len__(self) -> int:
        if self.split == "train":
            return len(self.train_inputs)
        else:
            return len(self.test_inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get sample with Reynolds number label."""
        if self.split == "train":
            x = self.train_inputs[idx]
            y = self.train_outputs[idx]
            re = self.train_re
        else:
            x = self.test_inputs[idx]
            y = self.test_outputs[idx]
            re = self.test_re
        
        # Normalize
        x = self._normalize(x, is_input=True)
        y = self._normalize(y, is_input=False)
        
        # Handle shape
        if x.ndim == 4:
            x = x[:, :, -1, :]
        if y.ndim == 4:
            y = y[:, :, -1, :]
        
        # Permute to (C, H, W)
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.from_numpy(y).permute(2, 0, 1).float()
        
        return x, y, re


def create_pdebench_loaders(
    config: PDEBenchConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    return_reynolds: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for PDEBench.
    
    Args:
        config: PDEBench configuration
        batch_size: Batch size
        num_workers: Number of data loading workers
        return_reynolds: Whether to return Reynolds number in batches
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    train_ds = PDEBenchDataset(config, split="train", return_reynolds=return_reynolds)
    val_ds = PDEBenchDataset(config, split="val", return_reynolds=return_reynolds)
    test_ds = PDEBenchDataset(config, split="test", return_reynolds=return_reynolds)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def create_cross_reynolds_loaders(
    data_dir: Union[str, Path],
    dataset_name: str = "2D_NavierStokes",
    resolution: int = 64,
    train_re: int = 100,
    test_re: int = 1000,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for cross-Reynolds generalization test.
    
    Returns:
        Dictionary with 'train' (Re=100) and 'test' (Re=1000) DataLoaders
    """
    train_ds = CrossReynoldsDataset(
        data_dir=data_dir, dataset_name=dataset_name, resolution=resolution,
        train_re=train_re, test_re=test_re, split="train",
    )
    test_ds = CrossReynoldsDataset(
        data_dir=data_dir, dataset_name=dataset_name, resolution=resolution,
        train_re=train_re, test_re=test_re, split="test",
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    return {
        "train": train_loader,
        "test": test_loader,
    }
