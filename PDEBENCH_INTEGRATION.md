# PDEBench Integration for Axiom-OS

## Overview

PDEBench is the "ImageNet" for AI4Science - a standardized benchmark for Physics-Informed Machine Learning. This integration enables Axiom-OS to:

1. **Load standardized PDE datasets** (2D Navier-Stokes, Burgers, etc.)
2. **Run cross-Reynolds generalization tests** - the ultimate test for physics discovery
3. **Compare against baselines** (FNO, U-Net) on equal footing
4. **Validate Discovery Engine** - does it find Reynolds-invariant laws?

## Installation

```bash
# Install PDEBench dependencies
pip install h5py

# Optional: Install PDEBench official package for data download
pip install git+https://github.com/pdebench/PDEBench.git

# Optional: PySR for symbolic regression (discovery)
pip install pysr
```

## Data Download

Download PDEBench datasets from [official repository](https://github.com/pdebench/PDEBench):

```bash
# Create data directory
mkdir -p data/pdebench

# Download 2D Navier-Stokes (example)
# Files follow naming convention: 2D_NavierStokes_Re{Re}_{resolution}x{resolution}.h5
wget https://darus.uni-stuttgart.de/file.xhtml?... -O data/pdebench/2D_NavierStokes_Re100_64x64.h5
wget https://darus.uni-stuttgart.de/file.xhtml?... -O data/pdebench/2D_NavierStokes_Re1000_64x64.h5
```

## Quick Start

### 1. Basic Data Loading

```python
from axiom_os.datasets.pdebench import PDEBenchConfig, create_pdebench_loaders

# Configure dataset
config = PDEBenchConfig(
    data_dir="./data/pdebench",
    dataset_name="2D_NavierStokes",
    resolution=64,
    reynolds_number=100,
    normalize=True,
)

# Create data loaders
loaders = create_pdebench_loaders(
    config=config,
    batch_size=16,
    num_workers=4,
)

# Use in training
for x, y in loaders["train"]:
    # x: (B, C, H, W) - input velocity field
    # y: (B, C, H, W) - target velocity field
    pass
```

### 2. Cross-Reynolds Generalization Test

The **key test**: Train on Re=100, test on Re=1000.

```python
from axiom_os.experiments.cross_reynolds_generalization import (
    CrossReynoldsConfig, CrossReynoldsExperiment
)

# Configure experiment
config = CrossReynoldsConfig(
    data_dir="./data/pdebench",
    train_re=100,      # Low Reynolds (laminar-ish)
    test_re=1000,      # High Reynolds (turbulent)
    epochs=100,
    run_discovery=True,  # Enable symbolic discovery
)

# Run experiment
experiment = CrossReynoldsExperiment(config)
results = experiment.run_full_experiment()

# Results include:
# - FNO baseline performance
# - U-Net baseline performance  
# - RCLN (Axiom-OS) performance
# - Discovered symbolic formula
# - Rollout errors (autoregressive prediction)
```

### 3. Benchmark Against FNO/U-Net

```python
from axiom_os.experiments.pdebench_benchmark import (
    BenchmarkConfig, PDEBenchBenchmark
)

config = BenchmarkConfig(
    data_dir="./data/pdebench",
    reynolds_number=100,
    epochs=100,
    run_discovery=True,
    n_rollout_steps=10,  # Multi-step prediction
)

benchmark = PDEBenchBenchmark(config)
results = benchmark.run_benchmark()
```

## Architecture

### Data Flow

```
PDEBench HDF5 Data
       ↓
PDEBenchDataset (normalization, splitting)
       ↓
DataLoader (B, C, H, W)
       ↓
┌─────────────────────────────────────────┐
│  FNO (baseline)                         │
│  U-Net (baseline)                       │
│  RCLN (Axiom-OS)                        │
│    ├── Hard Core (physics formula)      │
│    └── Soft Shell (FNO)                 │
└─────────────────────────────────────────┘
       ↓
Prediction (B, C, H, W)
       ↓
Discovery Engine (extract formula from Soft Shell)
       ↓
Hippocampus.crystallize() (update Hard Core)
```

### Cross-Reynolds Test Protocol

```
Phase 1: Training
  - Load Re=100 data (train/val/test splits)
  - Train RCLN with FNO Soft Shell
  - Monitor Soft Shell activity

Phase 2: Discovery
  - Collect Soft Shell outputs: y_soft = F_soft(x)
  - Run DiscoveryEngine: π_out = f(π_in)
  - Crystallize discovered formula into Hard Core

Phase 3: Generalization Test
  - Load Re=1000 data (NO retraining)
  - Test pure FNO (trained on Re=100)
  - Test RCLN with crystallized formula
  - Compare: Does discovered physics generalize?
```

## Key Metrics

### 1. Single-Step Error
```python
MSE = ||u_pred - u_true||²  # One time step ahead
```

### 2. Rollout Error (Autoregressive)
```python
# Multi-step prediction
u_0 → u_1_pred → u_2_pred → ... → u_T_pred
MSE_rollout(t) = ||u_t_pred - u_t_true||²
```

### 3. Energy Spectrum
```python
E(k) = |FFT(u)|²  # Fourier energy at wavenumber k
```

### 4. Discovery Success
- Formula complexity (number of terms)
- Reynolds invariance (does formula work for Re=1000?)
- Physical interpretability

## Expected Results

### Hypothesis

| Model | Re=100 (Train) | Re=1000 (Test) | Notes |
|-------|---------------|----------------|-------|
| FNO | Low MSE | **High MSE** | Overfits to Re=100 |
| U-Net | Low MSE | **High MSE** | Overfits to Re=100 |
| RCLN (Soft only) | Low MSE | High MSE | Same as FNO |
| **RCLN (Crystallized)** | Low MSE | **Lower MSE** | Discovered physics generalizes! |

### Success Criteria

Axiom-OS successfully discovers physics if:

1. **Discovery extracts a formula** from Soft Shell
2. **Formula is Reynolds-invariant** (works across Re)
3. **Crystallized RCLN outperforms** pure neural methods on Re=1000
4. **Formula is physically interpretable** (e.g., relates to vorticity, strain rate)

## API Reference

### PDEBenchConfig

```python
@dataclass
class PDEBenchConfig:
    data_dir: str              # Path to HDF5 files
    dataset_name: str          # "2D_NavierStokes", "1D_Burgers", etc.
    resolution: int            # Grid resolution (64, 128)
    reynolds_number: int       # Re number (100, 1000, etc.)
    normalize: bool            # Standard normalization
    train_ratio: float         # Train split (0.8)
    val_ratio: float           # Val split (0.1)
```

### CrossReynoldsDataset

```python
ds = CrossReynoldsDataset(
    data_dir="./data/pdebench",
    train_re=100,      # Training Reynolds
    test_re=1000,      # Testing Reynolds
    split="train",     # "train" or "test"
)

x, y, re = ds[0]  # Returns Reynolds number with each sample
```

### Discovery Integration

```python
from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.core.hippocampus import Hippocampus

# Initialize
discovery = DiscoveryEngine(use_pysr=True)
hippocampus = Hippocampus(dim=32)

# After training RCLN...
formula = discovery.distill(
    rcln_layer=rcln_model,
    data_buffer=soft_shell_data,
    input_units=[[0,0,0,0,0]] * n_inputs,  # Dimensionless
)

# Crystallize
hippocampus.crystallize(
    formula=formula,
    target_rcln=rcln_model,
    formula_id="ns2d_reynolds_invariant",
)
```

## Troubleshooting

### Data Not Found
```
FileNotFoundError: PDEBench data not found
```
**Solution**: Download data from [PDEBench repository](https://github.com/pdebench/PDEBench)

### Memory Issues
```python
# Reduce batch size
config.batch_size = 8

# Reduce resolution for testing
config.resolution = 32  # Instead of 64 or 128
```

### Discovery Fails
```python
# Use fallback polynomial regression
discovery = DiscoveryEngine(use_pysr=False)  # Skip PySR

# Or increase discovery samples
config.n_discovery_samples = 2000
```

## Citation

If using PDEBench integration, cite:

```bibtex
@article{takamoto2022pdebench,
  title={PDEBench: An extensive benchmark for scientific machine learning},
  author={Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and 
          MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and 
          Niepert, Mathias},
  journal={arXiv preprint arXiv:2210.07182},
  year={2022}
}

@software{axiom_os,
  title={Axiom-OS: Neuro-Symbolic Physics AI},
  author={yuzechen83-crypto},
  year={2026}
}
```

## Next Steps

1. **Download PDEBench data** for 2D Navier-Stokes
2. **Run cross-Reynolds test** to validate Discovery Engine
3. **Extend to other PDEs**: Burgers, Shallow Water, etc.
4. **Scale up**: Test on 128x128 resolution
5. **Publish results**: Compare with official PDEBench leaderboard
