"""
PDEBench Full Experiment: Download + Train + Evaluate + Discovery
Uses existing 1D Advection data
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from axiom_os.experiments.pdebench_1d_advection import (
    AdvectionExperimentConfig, AdvectionExperiment
)


def main():
    print("="*70)
    print("PDEBench Full Experiment Pipeline")
    print("="*70)
    print("\nStep 1: Using existing data")
    print("  File: 1D_Advection_Sols_beta0.1.hdf5")
    print("  Size: 7.8 GB")
    
    # Check if data exists
    data_path = Path(r"C:\Users\ASUS\Downloads\1D_Advection_Sols_beta0.1.hdf5")
    if not data_path.exists():
        print(f"\n❌ Data not found: {data_path}")
        return
    
    print(f"\n[OK] Data verified: {data_path}")
    
    # Configure experiment
    config = AdvectionExperimentConfig(
        hdf5_path=str(data_path),
        epochs=30,
        batch_size=32,
        run_discovery=True,
        output_dir="./outputs/pdebench_1d_advection_full",
    )
    
    # Run experiment
    print("\n" + "="*70)
    print("Step 2: Running Experiment")
    print("="*70)
    
    experiment = AdvectionExperiment(config)
    results = experiment.run_experiment()
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nResults:")
    print(f"  Test MSE:  {results['test_metrics']['mse']:.6f}")
    print(f"  Test MAE:  {results['test_metrics']['mae']:.6f}")
    
    if 'discovered_formula' in results:
        print(f"\n  Discovered Formula: {results['discovered_formula']}")
    
    print(f"\n  Output: {config.output_dir}")


if __name__ == "__main__":
    main()
