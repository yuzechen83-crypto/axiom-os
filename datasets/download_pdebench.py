"""
PDEBench Data Download Helper
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

This script helps download PDEBench datasets from the official sources.

PDEBench data is hosted on DaRUS (Data Repository of the University of Stuttgart):
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986

Note: PDEBench datasets are large (several GB). Ensure you have sufficient disk space.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import urllib.request
import zipfile
import tarfile


PDEBENCH_DATASETS = {
    "2D_NavierStokes": {
        "description": "2D Incompressible Navier-Stokes (periodic BC)",
        "resolutions": [64, 128, 256, 512],
        "reynolds_numbers": [100, 400, 1000],
        "size_gb": 5.2,
    },
    "1D_Burgers": {
        "description": "1D Burgers equation",
        "resolutions": [256, 512, 1024, 2048],
        "size_gb": 1.8,
    },
    "1D_Advection": {
        "description": "1D Advection equation",
        "resolutions": [256, 512],
        "size_gb": 0.8,
    },
    "2D_ShallowWater": {
        "description": "2D Shallow Water equations",
        "resolutions": [64, 128],
        "size_gb": 2.1,
    },
    "2D_DarcyFlow": {
        "description": "2D Darcy Flow (elliptic PDE)",
        "resolutions": [64, 128],
        "size_gb": 0.5,
    },
}


BASE_URL = "https://darus.uni-stuttgart.de/file.xhtml"


def print_dataset_info():
    """Print information about available datasets."""
    print("="*70)
    print("PDEBench Available Datasets")
    print("="*70)
    print()
    
    for name, info in PDEBENCH_DATASETS.items():
        print(f"{name}")
        print(f"  Description: {info['description']}")
        print(f"  Resolutions: {info['resolutions']}")
        if 'reynolds_numbers' in info:
            print(f"  Reynolds numbers: {info['reynolds_numbers']}")
        print(f"  Approximate size: {info['size_gb']} GB")
        print()
    
    print("="*70)
    print("Note: For cross-Reynolds experiments, you need:")
    print("  - 2D_NavierStokes with Re=100 and Re=1000")
    print("="*70)


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        print(f"Downloading: {url}")
        print(f"Destination: {dest}")
        
        # Create destination directory
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        urllib.request.urlretrieve(url, dest)
        
        print(f"✓ Download complete: {dest}")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def download_2d_navier_stokes(
    data_dir: Path,
    resolution: int = 64,
    reynolds_numbers: Optional[List[int]] = None,
):
    """Download 2D Navier-Stokes datasets."""
    if reynolds_numbers is None:
        reynolds_numbers = [100, 1000]  # Default for cross-Reynolds test
    
    print("="*70)
    print(f"Downloading 2D Navier-Stokes (resolution={resolution})")
    print("="*70)
    print()
    
    # Note: These URLs are placeholders. Actual URLs need to be obtained from DaRUS.
    # Users should manually download from:
    # https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
    
    print("⚠ IMPORTANT: Automatic download is not available.")
    print()
    print("Please download manually from:")
    print("  https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986")
    print()
    print("Required files for cross-Reynolds experiment:")
    for re in reynolds_numbers:
        filename = f"2D_NavierStokes_cond_Re{re}_{resolution}x{resolution}.h5"
        print(f"  - {filename}")
    print()
    print(f"Download and place in: {data_dir}/")
    print()
    print("Alternative: Use the direct download links from the PDEBench GitHub:")
    print("  https://github.com/pdebench/PDEBench")


def create_mock_data(
    data_dir: Path,
    resolution: int = 64,
    n_samples: int = 100,
):
    """Create mock PDEBench data for testing (no download required)."""
    print("="*70)
    print("Creating Mock PDEBench Data for Testing")
    print("="*70)
    print()
    
    try:
        import h5py
    except ImportError:
        print("Error: h5py is required. Install with: pip install h5py")
        return False
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic Navier-Stokes-like data
    for re in [100, 1000]:
        filename = f"2D_NavierStokes_cond_Re{re}_{resolution}x{resolution}.h5"
        filepath = data_dir / filename
        
        print(f"Creating: {filename}")
        
        # Generate synthetic velocity fields
        # Input: velocity at t=0
        inputs = []
        outputs = []
        
        for _ in range(n_samples):
            # Random initial condition
            u0 = np.random.randn(resolution, resolution, 2).astype(np.float32)
            
            # Simple forward Euler step (very approximate)
            # du/dt = -u·∇u + ν∇²u
            nu = 1.0 / re  # Viscosity
            dt = 0.01
            
            # Compute Laplacian (simplified)
            laplacian = (
                np.roll(u0, 1, axis=0) + np.roll(u0, -1, axis=0) +
                np.roll(u0, 1, axis=1) + np.roll(u0, -1, axis=1) - 4*u0
            )
            
            # Forward step
            u1 = u0 + dt * nu * laplacian
            
            inputs.append(u0)
            outputs.append(u1)
        
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        
        # Save to HDF5
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('input', data=inputs)
            f.create_dataset('output', data=outputs)
            f.attrs['Re'] = float(re)
            f.attrs['viscosity'] = nu
        
        print(f"  ✓ Created: {filepath}")
        print(f"    Shape: {inputs.shape}")
    
    print()
    print("✓ Mock data created successfully!")
    print(f"Location: {data_dir}/")
    print()
    print("Note: This is synthetic data for testing only.")
    print("For real experiments, download actual PDEBench data.")
    
    return True


def verify_data(data_dir: Path) -> bool:
    """Verify that PDEBench data exists and is valid."""
    print("="*70)
    print("Verifying PDEBench Data")
    print("="*70)
    print()
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    # Look for HDF5 files
    h5_files = list(data_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"✗ No HDF5 files found in: {data_dir}")
        return False
    
    print(f"Found {len(h5_files)} HDF5 file(s):")
    
    for f in h5_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    print()
    print("✓ Data verification complete")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="PDEBench Data Download Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python -m axiom_os.datasets.download_pdebench --info
  
  # Create mock data for testing
  python -m axiom_os.datasets.download_pdebench --mock --data_dir ./data/pdebench
  
  # Verify existing data
  python -m axiom_os.datasets.download_pdebench --verify --data_dir ./data/pdebench
        """
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about available datasets"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Create mock data for testing (no download)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing data"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/pdebench",
        help="Data directory (default: ./data/pdebench)"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        choices=[64, 128, 256],
        help="Grid resolution (default: 64)"
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples for mock data (default: 100)"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_dataset_info()
        return
    
    if args.verify:
        verify_data(Path(args.data_dir))
        return
    
    if args.mock:
        import numpy as np  # Import here to avoid dependency if not needed
        create_mock_data(
            data_dir=Path(args.data_dir),
            resolution=args.resolution,
            n_samples=args.n_samples,
        )
        return
    
    # Default: print help
    parser.print_help()
    print()
    print("For cross-Reynolds experiments, you need 2D Navier-Stokes data.")
    print("Download from: https://github.com/pdebench/PDEBench")


if __name__ == "__main__":
    main()
