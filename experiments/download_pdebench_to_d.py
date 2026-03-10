"""
Download PDEBench Real Data to D: Drive
========================================
Downloads high-quality turbulence simulation data from PDEBench.
This is REAL computational fluid dynamics data (not synthetic).

Available datasets:
- 2D_CFD: 2D compressible Navier-Stokes (~2.3 GB) ⭐ RECOMMENDED
- 3D_CFD: 3D compressible Navier-Stokes (~30 GB)
- 2D_Darcy: Darcy flow (~500 MB)
- 2D_Diffusion: Diffusion equation (~200 MB)

Data source: https://doi.org/10.5281/zenodo.6993294
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
import time


# PDEBench dataset URLs from Zenodo
DATASETS = {
    '2D_CFD': {
        'url': 'https://zenodo.org/record/6993294/files/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5',
        'size_gb': 2.3,
        'description': '2D Compressible Navier-Stokes (turbulence)',
        'recommended': True
    },
    '2D_Darcy': {
        'url': 'https://zenodo.org/record/6993294/files/2D_DarcyFlow_beta1.0_Train.hdf5',
        'size_gb': 0.5,
        'description': '2D Darcy Flow (porous media)',
        'recommended': False
    },
    '2D_Diffusion': {
        'url': 'https://zenodo.org/record/6993294/files/2D_Diffusion_Re1000_Train.hdf5',
        'size_gb': 0.2,
        'description': '2D Diffusion Equation',
        'recommended': False
    },
    '1D_Burgers': {
        'url': 'https://zenodo.org/record/6993294/files/1D_Burgers_Sols_Nu0.001.hdf5',
        'size_gb': 0.8,
        'description': '1D Burgers Equation (shock waves)',
        'recommended': False
    },
    '3D_CFD': {
        'url': 'https://zenodo.org/record/6993294/files/3D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5',
        'size_gb': 30.0,
        'description': '3D Compressible Navier-Stokes (large!)',
        'recommended': False
    }
}


class DownloadProgress:
    """Track download progress"""
    def __init__(self, total_size):
        self.total_size = total_size
        self.downloaded = 0
        self.start_time = time.time()
        
    def __call__(self, block_num, block_size, total_size):
        self.downloaded = block_num * block_size
        percent = min(100, self.downloaded * 100 / self.total_size)
        elapsed = time.time() - self.start_time
        speed = self.downloaded / (elapsed + 1) / 1024 / 1024  # MB/s
        
        sys.stdout.write(f"\r  Progress: {percent:.1f}% ({self.downloaded/1024/1024:.1f} MB / "
                        f"{self.total_size/1024/1024:.1f} MB) @ {speed:.1f} MB/s")
        sys.stdout.flush()


def download_file(url: str, dest_path: Path, resume: bool = True) -> bool:
    """
    Download file with resume support.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        resume: Whether to resume partial downloads
        
    Returns:
        True if successful
    """
    dest_path = Path(dest_path)
    temp_path = dest_path.with_suffix('.tmp')
    
    # Check if already downloaded
    if dest_path.exists():
        print(f"  File already exists: {dest_path}")
        return True
    
    # Get file size
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            total_size = int(response.headers.get('Content-Length', 0))
    except Exception as e:
        print(f"  Warning: Could not get file size: {e}")
        total_size = 0
    
    # Resume partial download
    existing_size = 0
    if resume and temp_path.exists():
        existing_size = temp_path.stat().st_size
        print(f"  Resuming from {existing_size/1024/1024:.1f} MB")
    
    # Download
    try:
        headers = {}
        if existing_size > 0:
            headers['Range'] = f'bytes={existing_size}-'
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            if total_size == 0:
                total_size = int(response.headers.get('Content-Length', 0)) + existing_size
            
            mode = 'ab' if existing_size > 0 else 'wb'
            
            with open(temp_path, mode) as f:
                progress = DownloadProgress(total_size)
                block_size = 8192 * 1024  # 8 MB blocks
                
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress(progress.downloaded // block_size + 1, block_size, total_size)
        
        print()  # New line after progress
        
        # Rename temp file to final name
        temp_path.rename(dest_path)
        print(f"  Saved to: {dest_path}")
        return True
        
    except urllib.error.URLError as e:
        print(f"\n  Network error: {e}")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def verify_hdf5(filepath: Path) -> bool:
    """Verify HDF5 file integrity"""
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            # Try to read a small amount of data
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][0:1]
                    break
        print(f"  Verification: OK")
        return True
    except Exception as e:
        print(f"  Verification FAILED: {e}")
        return False


def main():
    """Main download interface"""
    print("="*70)
    print("PDEBench Real Data Downloader")
    print("="*70)
    print("\nAvailable datasets:")
    
    for i, (name, info) in enumerate(DATASETS.items(), 1):
        marker = "⭐ " if info['recommended'] else "   "
        print(f"{marker}{i}. {name}")
        print(f"      {info['description']}")
        print(f"      Size: {info['size_gb']:.1f} GB")
        print()
    
    # Ask user for selection
    print("Enter dataset number to download (or 'all' for all recommended):")
    choice = input("> ").strip()
    
    # Determine which datasets to download
    if choice.lower() == 'all':
        to_download = [name for name, info in DATASETS.items() if info['recommended']]
    else:
        try:
            idx = int(choice) - 1
            to_download = [list(DATASETS.keys())[idx]]
        except (ValueError, IndexError):
            print("Invalid selection!")
            return
    
    # Set download directory to D:
    download_dir = Path('D:/PDEBench_Data')
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownload directory: {download_dir}")
    print(f"Datasets to download: {', '.join(to_download)}")
    print()
    
    # Download each dataset
    for dataset_name in to_download:
        info = DATASETS[dataset_name]
        dest_file = download_dir / f"{dataset_name}.hdf5"
        
        print(f"\n{'='*70}")
        print(f"Downloading: {dataset_name}")
        print(f"{'='*70}")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size_gb']:.1f} GB")
        print(f"  Destination: {dest_file}")
        
        # Check disk space
        disk = os.statvfs(download_dir.anchor)
        free_gb = disk.f_frsize * disk.f_bavail / (1024**3)
        required_gb = info['size_gb'] * 1.5  # Add 50% buffer
        
        if free_gb < required_gb:
            print(f"\n  ERROR: Insufficient disk space!")
            print(f"  Required: {required_gb:.1f} GB")
            print(f"  Available: {free_gb:.1f} GB")
            continue
        
        # Download
        success = download_file(info['url'], dest_file)
        
        if success:
            # Verify
            print("\n  Verifying file integrity...")
            if verify_hdf5(dest_file):
                print(f"\n  ✅ {dataset_name} ready to use!")
                print(f"     Run: python experiments/axiom_pdebench_real.py")
            else:
                print(f"\n  ❌ File verification failed. Please re-download.")
        else:
            print(f"\n  ❌ Download failed!")
    
    print(f"\n{'='*70}")
    print("Download session complete!")
    print(f"{'='*70}")
    print(f"\nData location: {download_dir}")
    print("\nNext steps:")
    print("  1. Edit experiments/axiom_pdebench_real.py")
    print(f"  2. Set DATA_PATH = r'{download_dir}\\2D_CFD.hdf5'")
    print("  3. Run: python experiments/axiom_pdebench_real.py")


if __name__ == "__main__":
    main()
