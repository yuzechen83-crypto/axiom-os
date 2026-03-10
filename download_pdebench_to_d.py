"""
PDEBench Auto-Downloader for D: Drive
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Downloads PDEBench 2D Navier-Stokes data to D: drive for cross-Reynolds experiments.

Usage:
    python download_pdebench_to_d.py --resolution 64
    python download_pdebench_to_d.py --resolution 128 --re 100,400,1000
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import urllib.request
import urllib.error
import time
import hashlib


# PDEBench data URLs from DaRUS
# Note: These are placeholder URLs. Actual URLs may change.
# Check: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986

PDEBENCH_URLS = {
    "2D_NavierStokes": {
        64: {
            100: "https://darus.uni-stuttgart.de/api/access/datafile/107721",
            400: "https://darus.uni-stuttgart.de/api/access/datafile/107722",
            1000: "https://darus.uni-stuttgart.de/api/access/datafile/107723",
        },
        128: {
            100: "https://darus.uni-stuttgart.de/api/access/datafile/107724",
            400: "https://darus.uni-stuttgart.de/api/access/datafile/107725",
            1000: "https://darus.uni-stuttgart.de/api/access/datafile/107726",
        },
    },
    "1D_Burgers": {
        256: "https://darus.uni-stuttgart.de/api/access/datafile/107727",
        512: "https://darus.uni-stuttgart.de/api/access/datafile/107728",
    },
}

# Alternative: Direct links from PDEBench GitHub releases
GITHUB_MIRROR_URLS = {
    "2D_NavierStokes": {
        64: {
            100: "https://github.com/pdebench/PDEBench/raw/main/data/2D_NavierStokes_cond_Re100_64x64.h5",
            1000: "https://github.com/pdebench/PDEBench/raw/main/data/2D_NavierStokes_cond_Re1000_64x64.h5",
        },
    },
}


def get_d_drive_path() -> Path:
    """Get D: drive path, fallback to current directory if not available."""
    d_drive = Path("D:/")
    if d_drive.exists():
        return d_drive / "PDEBench_Data"
    else:
        print("⚠️ D: drive not found, using current directory...")
        return Path("./PDEBench_Data")


def download_with_progress(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress bar."""
    try:
        print(f"\n📥 Downloading:")
        print(f"   URL: {url}")
        print(f"   Dest: {dest}")
        
        # Create destination directory
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Open connection
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        })
        
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress bar
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r   Progress: {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end='')
        
        print(f"\n   ✅ Download complete: {dest.name}")
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n   ❌ HTTP Error {e.code}: {e.reason}")
        if e.code == 404:
            print("   File not found on server.")
        elif e.code == 403:
            print("   Access forbidden. May require authentication.")
        return False
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def verify_file(filepath: Path, expected_size_mb: Optional[float] = None) -> bool:
    """Verify downloaded file."""
    if not filepath.exists():
        return False
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"   File size: {size_mb:.1f} MB")
    
    if expected_size_mb and size_mb < expected_size_mb * 0.9:
        print(f"   ⚠️ File seems smaller than expected ({expected_size_mb:.1f} MB)")
        return False
    
    # Try to open as HDF5
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            datasets = list(f.keys())
            print(f"   HDF5 datasets: {datasets}")
        return True
    except Exception as e:
        print(f"   ⚠️ Could not verify as HDF5: {e}")
        return False


def download_2d_navier_stokes(
    data_dir: Path,
    resolution: int = 64,
    reynolds_numbers: List[int] = None,
    use_mirror: bool = False,
):
    """Download 2D Navier-Stokes datasets."""
    if reynolds_numbers is None:
        reynolds_numbers = [100, 1000]  # Default for cross-Reynolds
    
    print("="*70)
    print("PDEBench 2D Navier-Stokes Downloader")
    print("="*70)
    print(f"Target directory: {data_dir}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Reynolds numbers: {reynolds_numbers}")
    print()
    
    success_count = 0
    
    for re in reynolds_numbers:
        filename = f"2D_NavierStokes_cond_Re{re}_{resolution}x{resolution}.h5"
        filepath = data_dir / filename
        
        # Check if already exists
        if filepath.exists():
            print(f"\n📄 {filename} already exists.")
            if verify_file(filepath):
                print("   ✅ File verified, skipping download.")
                success_count += 1
                continue
            else:
                print("   ⚠️ File may be corrupted, re-downloading...")
        
        # Get URL
        if use_mirror and resolution in GITHUB_MIRROR_URLS.get("2D_NavierStokes", {}):
            url = GITHUB_MIRROR_URLS["2D_NavierStokes"][resolution].get(re)
        else:
            url = PDEBENCH_URLS.get("2D_NavierStokes", {}).get(resolution, {}).get(re)
        
        if not url:
            print(f"\n❌ No URL available for Re={re}, resolution={resolution}")
            continue
        
        # Download
        if download_with_progress(url, filepath):
            if verify_file(filepath):
                success_count += 1
        
        # Be nice to the server
        time.sleep(1)
    
    print("\n" + "="*70)
    print(f"Download summary: {success_count}/{len(reynolds_numbers)} files successful")
    print("="*70)
    
    return success_count == len(reynolds_numbers)


def create_torrent_download_script(data_dir: Path):
    """Create a script for torrent-based download (if available)."""
    script_path = data_dir / "download_torrent.sh"
    
    script_content = """#!/bin/bash
# PDEBench Torrent Download Script
# Use this if direct download is slow or fails

echo "PDEBench torrent download (if available)"
echo "Check: https://github.com/pdebench/PDEBench for alternative download methods"

# aria2c can be used for faster downloads
# aria2c -x 4 -s 4 <URL>

"""
    
    script_path.write_text(script_content)
    print(f"Created: {script_path}")


def manual_download_instructions(data_dir: Path):
    """Print manual download instructions."""
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("""
If automatic download fails, please download manually:

1. Visit the PDEBench GitHub:
   https://github.com/pdebench/PDEBench

2. Navigate to the data section or releases page

3. Download the following files for cross-Reynolds experiments:
   - 2D_NavierStokes_cond_Re100_64x64.h5
   - 2D_NavierStokes_cond_Re1000_64x64.h5
   
   (or 128x128 for higher resolution)

4. Save to:
   """)
    print(f"   {data_dir}/")
    print("""
5. Alternative sources:
   - DaRUS: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
   - Kaggle: Search for "PDEBench"
   - HuggingFace: https://huggingface.co/datasets/pdebench

""")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Download PDEBench data to D: drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Re=100 and Re=1000 (for cross-Reynolds test) to D: drive
  python download_pdebench_to_d.py --resolution 64 --re 100 1000
  
  # Download all available Re
  python download_pdebench_to_d.py --resolution 64 --re 100 400 1000
  
  # Use custom directory (if D: not available)
  python download_pdebench_to_d.py --data_dir E:/PDEBench_Data
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (default: D:/PDEBench_Data or ./PDEBench_Data)"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[64, 128, 256],
        default=64,
        help="Grid resolution (default: 64)"
    )
    
    parser.add_argument(
        "--re",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="Reynolds numbers to download (default: 100 1000)"
    )
    
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Use GitHub mirror instead of DaRUS"
    )
    
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Show manual download instructions only"
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = get_d_drive_path()
    
    if args.manual:
        manual_download_instructions(data_dir)
        return
    
    print("\n" + "="*70)
    print("Axiom-OS PDEBench Downloader")
    print("="*70)
    print()
    
    # Check if h5py is available
    try:
        import h5py
        print("✓ h5py is installed")
    except ImportError:
        print("⚠️ h5py not installed. Install with: pip install h5py")
        print("  (Download will continue but files cannot be verified)")
    
    print(f"✓ Target directory: {data_dir}")
    print()
    
    # Attempt download
    success = download_2d_navier_stokes(
        data_dir=data_dir,
        resolution=args.resolution,
        reynolds_numbers=args.re,
        use_mirror=args.mirror,
    )
    
    if not success:
        print("\n⚠️ Some downloads failed.")
        manual_download_instructions(data_dir)
        
        # Create alternative download script
        create_torrent_download_script(data_dir)
    else:
        print("\n✅ All downloads successful!")
        print(f"\nYou can now run experiments:")
        print(f"  python run_pdebench.py --mode cross_re --data_dir {data_dir}")


if __name__ == "__main__":
    main()
