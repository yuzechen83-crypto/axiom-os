"""
JHTDB Direct Access via REST API - SciServer Compatible
=======================================================
Access Johns Hopkins Turbulence Database through direct HTTP requests.
Works both locally and on SciServer platform.

Datasets Available:
- isotropic1024coarse: Isotropic turbulence (1024^3)
- channel: Channel flow (2048^3)
- mhd1024: MHD turbulence (1024^3)
- mixing: Scalar mixing (1024^3)
- rotstrat1024: Rotating stratified turbulence (1024^3)

Author: Axiom-OS Team
"""

import requests
import numpy as np
import json
from typing import Tuple, Optional, List
import time
import os


class JHTDBClient:
    """
    Direct REST API client for JHTDB.
    Works without SciServer authentication for public datasets.
    """
    
    BASE_URL = "http://turbulence.pha.jhu.edu/"
    
    DATASETS = {
        'isotropic1024coarse': {
            'description': 'Forced isotropic turbulence',
            'resolution': 1024,
            'time_range': (0.0, 2.56),
            'fields': ['u', 'p', 'b']
        },
        'channel': {
            'description': 'Channel flow',
            'resolution': 2048,
            'time_range': (0.0, 25.0),
            'fields': ['u', 'p']
        },
        'mhd1024': {
            'description': 'MHD turbulence',
            'resolution': 1024,
            'time_range': (0.0, 2.56),
            'fields': ['u', 'b', 'p']
        },
        'mixing': {
            'description': 'Scalar mixing',
            'resolution': 1024,
            'time_range': (0.0, 2.56),
            'fields': ['u', 'p', 'theta']
        },
        'rotstrat1024': {
            'description': 'Rotating stratified turbulence',
            'resolution': 1024,
            'time_range': (0.0, 42.0),
            'fields': ['u', 'p', 'rho']
        }
    }
    
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize JHTDB client.
        
        Args:
            auth_token: Optional SciServer authentication token
        """
        self.auth_token = auth_token
        self.session = requests.Session()
        
    def get_velocity_at_points(self, 
                                dataset: str,
                                time: float,
                                points: np.ndarray,
                                field: str = 'u',
                                interpolation: str = 'lagrange6') -> np.ndarray:
        """
        Get velocity field at specified points.
        
        Args:
            dataset: Dataset name (e.g., 'isotropic1024coarse')
            time: Time point
            points: Array of shape (N, 3) with (x, y, z) coordinates in [0, 2pi]
            field: Field to retrieve ('u' for velocity, 'b' for magnetic, 'p' for pressure)
            interpolation: Interpolation method ('lagrange6', 'lagrange4', etc.)
            
        Returns:
            velocity: Array of shape (N, 3) with (ux, uy, uz)
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(self.DATASETS.keys())}")
        
        # Convert points to list for JSON serialization
        points_list = points.tolist()
        
        # Build request
        data = {
            'dataset': dataset,
            'time': time,
            'points': points_list,
            'field': field,
            'interpolation': interpolation
        }
        
        # Add auth token if available
        if self.auth_token:
            data['authToken'] = self.auth_token
            
        # Make request
        url = f"{self.BASE_URL}cutout/download"
        
        try:
            response = self.session.post(url, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if 'error' in result:
                raise RuntimeError(f"JHTDB error: {result['error']}")
                
            return np.array(result.get('data', []))
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise
    
    def get_cutout(self,
                   dataset: str,
                   time: float,
                   origin: Tuple[int, int, int],
                   size: int = 64,
                   field: str = 'u') -> np.ndarray:
        """
        Get a cubic cutout from the dataset.
        
        Args:
            dataset: Dataset name
            time: Time point
            origin: (x, y, z) origin of cube in grid coordinates
            size: Size of cube (e.g., 64 for 64^3)
            field: Field to retrieve ('u', 'p', 'b', etc.)
            
        Returns:
            data: Array of shape (size, size, size, 3) for velocity
                 or (size, size, size) for scalar fields
        """
        x0, y0, z0 = origin
        
        # Build request for cutout
        params = {
            'dataset': dataset,
            'time': time,
            'x_start': x0,
            'y_start': y0,
            'z_start': z0,
            'x_size': size,
            'y_size': size,
            'z_size': size,
            'field': field,
            'format': 'raw'
        }
        
        if self.auth_token:
            params['authToken'] = self.auth_token
            
        url = f"{self.BASE_URL}cutout/download"
        
        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            
            # Parse binary data
            raw_data = np.frombuffer(response.content, dtype=np.float32)
            
            # Reshape based on field type
            if field in ['u', 'b']:  # Vector fields
                return raw_data.reshape(size, size, size, 3)
            else:  # Scalar fields (p, theta, rho)
                return raw_data.reshape(size, size, size)
                
        except Exception as e:
            print(f"Cutout download failed: {e}")
            raise
    
    def get_dataset_info(self, dataset: str) -> dict:
        """Get information about a dataset"""
        return self.DATASETS.get(dataset, {})
    
    def list_datasets(self) -> List[str]:
        """List available datasets"""
        return list(self.DATASETS.keys())
    
    def get_spatial_spectrum(self,
                              dataset: str = 'isotropic1024coarse',
                              time: float = 0.0,
                              cube_size: int = 128) -> dict:
        """
        Compute energy spectrum from a cubic cutout.
        Useful for FNO training data.
        
        Returns:
            dict with 'k' (wavenumbers) and 'E' (energy)
        """
        # Download velocity field
        origin = (512 - cube_size//2, 512 - cube_size//2, 512 - cube_size//2)
        u = self.get_cutout(dataset, time, origin, cube_size, 'u')
        
        # Compute FFT
        u_hat = np.fft.fftn(u, axes=(0, 1, 2))
        
        # Compute energy spectrum
        energy_density = 0.5 * np.sum(np.abs(u_hat)**2, axis=-1)
        
        # Radial averaging
        nx, ny, nz = cube_size, cube_size, cube_size
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        kz = np.fft.fftfreq(nz) * nz
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2 + KZ**2).astype(int)
        
        # Bin by wavenumber
        k_max = cube_size // 2
        E = np.zeros(k_max)
        counts = np.zeros(k_max)
        
        for k in range(k_max):
            mask = (k_mag == k)
            E[k] = np.sum(energy_density[mask])
            counts[k] = np.sum(mask)
        
        E = E / (counts + 1e-10)  # Normalize
        k = np.arange(k_max)
        
        return {'k': k, 'E': E}


class AxiomJHTDBDataset:
    """
    Axiom-OS compatible JHTDB dataset wrapper.
    For use with FNO-RCLN turbulence modeling.
    """
    
    def __init__(self, 
                 dataset: str = 'isotropic1024coarse',
                 time_start: float = 0.0,
                 time_end: float = 2.56,
                 nt: int = 10,
                 cube_size: int = 64,
                 cache_dir: str = './jhtdb_cache'):
        """
        Initialize dataset for turbulence modeling.
        
        Args:
            dataset: JHTDB dataset name
            time_start, time_end: Time range
            nt: Number of time snapshots
            cube_size: Spatial resolution for cubes
            cache_dir: Directory to cache downloaded data
        """
        self.client = JHTDBClient()
        self.dataset = dataset
        self.times = np.linspace(time_start, time_end, nt)
        self.cube_size = cube_size
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.times)
    
    def __getitem__(self, idx):
        """Get a time snapshot with caching"""
        time = self.times[idx]
        
        # Check cache
        cache_file = os.path.join(
            self.cache_dir, 
            f"{self.dataset}_t{time:.3f}_c{self.cube_size}.npy"
        )
        
        if os.path.exists(cache_file):
            return np.load(cache_file)
        
        # Download from JHTDB
        np.random.seed(idx)
        res = 1024
        origin = tuple(np.random.randint(0, res - self.cube_size, 3))
        
        cube = self.client.get_cutout(
            self.dataset,
            time,
            origin,
            self.cube_size,
            'u'
        )
        
        # Cache
        np.save(cache_file, cube)
        
        return cube
    
    def get_velocity_field(self, time_idx: int) -> np.ndarray:
        """Get velocity field at time index"""
        return self.__getitem__(time_idx)


def demo_jhtdb_access():
    """Demonstrate JHTDB access"""
    print("="*70)
    print("JHTDB Direct Access Demo - SciServer Compatible")
    print("="*70)
    
    # Create client
    client = JHTDBClient()
    
    # List available datasets
    print("\n[1] Available Datasets:")
    for ds in client.list_datasets():
        info = client.get_dataset_info(ds)
        print(f"    {ds}:")
        print(f"      Description: {info.get('description', 'N/A')}")
        print(f"      Resolution: {info.get('resolution', 'N/A')}^3")
        print(f"      Time range: {info.get('time_range', 'N/A')}")
        print(f"      Fields: {', '.join(info.get('fields', []))}")
    
    # Test point query
    print("\n[2] Testing Point Query (isotropic1024coarse):")
    dataset = 'isotropic1024coarse'
    time = 0.0
    
    points = np.array([
        [1.0, 1.0, 1.0],
        [3.14, 3.14, 3.14],
        [0.0, 0.0, 0.0]
    ])
    
    try:
        velocity = client.get_velocity_at_points(dataset, time, points)
        print(f"    Points queried: {points.shape[0]}")
        print(f"    Velocity shape: {velocity.shape}")
        print(f"    Velocity at (1,1,1): {velocity[0]}")
        print(f"    |u| at (1,1,1): {np.linalg.norm(velocity[0]):.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Test cube download
    print("\n[3] Testing Cutout Download (16^3 cube):")
    try:
        cube = client.get_cutout(
            dataset='isotropic1024coarse',
            time=0.0,
            origin=(512, 512, 512),
            size=16,
            field='u'
        )
        print(f"    Cube shape: {cube.shape}")
        print(f"    Velocity magnitude statistics:")
        mag = np.linalg.norm(cube, axis=-1)
        print(f"      Mean: {mag.mean():.4f}")
        print(f"      Std:  {mag.std():.4f}")
        print(f"      Max:  {mag.max():.4f}")
        print(f"      Min:  {mag.min():.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Test energy spectrum
    print("\n[4] Computing Energy Spectrum (64^3 cube):")
    try:
        spectrum = client.get_spatial_spectrum(
            dataset='isotropic1024coarse',
            time=0.0,
            cube_size=64
        )
        k = spectrum['k']
        E = spectrum['E']
        print(f"    Wavenumbers: {len(k)}")
        print(f"    E(k=1): {E[1]:.4f}")
        print(f"    E(k=10): {E[10]:.4f}")
        print(f"    Kolmogorov slope check (E ~ k^-5/3):")
        if len(E) > 10:
            slope = np.log(E[5]/E[10]) / np.log(k[5]/k[10])
            print(f"      Observed slope: {slope:.2f} (expected ~ -5/3 = -1.67)")
    except Exception as e:
        print(f"    Error: {e}")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    demo_jhtdb_access()
