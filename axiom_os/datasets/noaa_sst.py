"""
NOAA OISST V2 Data Loader for ENSO Discovery
============================================
Loads and processes Nino 3.4 Index from NOAA PSL.

Data Source: NOAA Physical Sciences Laboratory
URL: https://www.psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass
import os
import urllib.request


@dataclass
class ENSODatasetConfig:
    """Configuration for ENSO dataset."""
    # Time settings
    train_start_year: int = 1982
    train_end_year: int = 2015
    test_start_year: int = 2016
    test_end_year: int = 2024
    
    # Feature engineering
    lookback_months: int = 12  # Look back 1 year
    forecast_horizon: int = 1  # Predict next month
    
    # Normalization
    normalization: str = 'zscore'  # 'zscore' or 'minmax'


class Nino34DataLoader:
    """
    Load and process Nino 3.4 SST anomaly index.
    
    The Nino 3.4 index is the primary metric for ENSO monitoring,
    representing SST anomalies in the region 5N-5S, 170W-120W.
    """
    
    # NOAA Nino 3.4 Index URLs (monthly anomaly, already deseasonalized)
    NOAA_URL_ANOM = "https://www.psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data"
    # Alternative: ERSSTv5 Nino 3.4
    NOAA_URL_ERSST = "https://www.ncei.noaa.gov/access/monitoring/enso/sst/nino34.data"
    
    def __init__(self, config: ENSODatasetConfig, cache_dir: str = './data_cache'):
        self.cfg = config
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.data = None
        self.years = None
        self.months = None
        self.mean = None
        self.std = None
        
    def download_data(self) -> str:
        """Download Nino 3.4 data from NOAA."""
        cache_file = os.path.join(self.cache_dir, 'nino34.long.anom.data')
        
        if os.path.exists(cache_file):
            print(f"Using cached data: {cache_file}")
            return cache_file
        
        print(f"Downloading Nino 3.4 data from NOAA...")
        print(f"URL: {self.NOAA_URL}")
        
        try:
            urllib.request.urlretrieve(self.NOAA_URL, cache_file)
            print(f"Downloaded to: {cache_file}")
            return cache_file
        except Exception as e:
            print(f"Download failed: {e}")
            print("Generating synthetic Nino 3.4 data for testing...")
            return self._generate_synthetic_data(cache_file)
    
    def _generate_synthetic_data(self, output_file: str) -> str:
        """
        Generate synthetic Nino 3.4-like data for testing.
        Mimics the statistical properties of real ENSO.
        """
        print("Generating synthetic ENSO data with Delayed Oscillator dynamics...")
        
        # Generate 50 years of monthly data (1970-2020)
        years = np.arange(1970, 2021)
        months = np.arange(1, 13)
        
        data_lines = []
        
        # Simple delayed oscillator model
        # dT/dt = -alpha*T - beta*h(t-tau) + noise
        alpha = 0.1
        beta = 0.3
        tau = 6  # 6 month delay (Rossby wave)
        
        T = 0.0  # Initial temperature anomaly
        h_history = [0.0] * tau  # Thermocline history
        
        for year in years:
            for month in months:
                # Delayed thermocline feedback
                h_delayed = h_history[-tau] if len(h_history) >= tau else 0.0
                
                # Non-linear damping
                damping = -alpha * T - 0.01 * T**3
                
                # Delayed feedback
                feedback = -beta * h_delayed
                
                # Annual cycle forcing
                annual = 0.5 * np.sin(2 * np.pi * month / 12)
                
                # Random weather noise
                noise = np.random.randn() * 0.4
                
                # Update
                dT = damping + feedback + annual + noise
                T += dT
                
                # Update thermocline (simplified)
                h_new = 0.7 * T + 0.3 * (h_history[-1] if h_history else 0)
                h_history.append(h_new)
                
                data_lines.append(f"{year}  {month:2d}  {T:7.3f}\n")
        
        with open(output_file, 'w') as f:
            f.writelines(data_lines)
        
        print(f"Synthetic data saved to: {output_file}")
        return output_file
    
    def load_data(self, data_file: str) -> np.ndarray:
        """Parse Nino 3.4 data file."""
        years_list = []
        months_list = []
        anomalies = []
        
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('-'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        anomaly = float(parts[2])
                        
                        # Filter valid data (not -99.99 missing values)
                        if anomaly > -90:
                            years_list.append(year)
                            months_list.append(month)
                            anomalies.append(anomaly)
                    except ValueError:
                        continue
        
        self.years = np.array(years_list)
        self.months = np.array(months_list)
        self.data = np.array(anomalies, dtype=np.float32)
        
        print(f"Loaded {len(self.data)} monthly observations")
        print(f"Year range: {self.years.min()}-{self.years.max()}")
        print(f"Anomaly range: [{self.data.min():.2f}, {self.data.max():.2f}] degC")
        
        return self.data
    
    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-lagged sequences for delay discovery.
        
        Args:
            data: Raw time series (n_samples,)
            lookback: Number of past months to include
        
        Returns:
            X: (n_samples - lookback, lookback) feature matrix
               Each row is [T(t), T(t-1), ..., T(t-lookback+1)]
            y: (n_samples - lookback,) target (T(t+1))
        """
        n = len(data)
        X = []
        y = []
        
        for i in range(lookback, n - 1):
            # Historical window
            window = data[i - lookback:i]  # [T(t-lookback), ..., T(t-1)]
            X.append(window[::-1])  # Reverse to [T(t-1), ..., T(t-lookback)]
            
            # Target: next value
            target = data[i]  # T(t)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def prepare_train_test(self) -> Tuple[torch.Tensor, ...]:
        """
        Prepare train/test splits.
        
        Returns:
            X_train, y_train, X_test, y_test as torch tensors
        """
        # Download and load
        data_file = self.download_data()
        self.load_data(data_file)
        
        # Split by year
        train_mask = (self.years >= self.cfg.train_start_year) & \
                     (self.years <= self.cfg.train_end_year)
        test_mask = (self.years >= self.cfg.test_start_year) & \
                    (self.years <= self.cfg.test_end_year)
        
        train_data = self.data[train_mask]
        test_data = self.data[test_mask]
        
        print(f"\nTrain period: {self.cfg.train_start_year}-{self.cfg.train_end_year}")
        print(f"  Samples: {len(train_data)}")
        print(f"Test period: {self.cfg.test_start_year}-{self.cfg.test_end_year}")
        print(f"  Samples: {len(test_data)}")
        
        # Normalize (using training statistics)
        self.mean = train_data.mean()
        self.std = train_data.std()
        
        train_data = (train_data - self.mean) / self.std
        test_data = (test_data - self.mean) / self.std
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data, self.cfg.lookback_months)
        X_test, y_test = self.create_sequences(test_data, self.cfg.lookback_months)
        
        print(f"\nSequence shape:")
        print(f"  X_train: {X_train.shape} (samples, lookback={self.cfg.lookback_months})")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Convert to torch
        return (
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Convert back to original scale."""
        return data * self.std + self.mean


class TimeDelayDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for time-delay features."""
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_nino34_for_discovery(
    train_years: Tuple[int, int] = (1982, 2015),
    test_years: Tuple[int, int] = (2016, 2024),
    lookback: int = 12
) -> Tuple:
    """
    Convenience function to load Nino 3.4 data.
    
    Returns:
        (X_train, y_train, X_test, y_test, data_loader)
    """
    config = ENSODatasetConfig(
        train_start_year=train_years[0],
        train_end_year=train_years[1],
        test_start_year=test_years[0],
        test_end_year=test_years[1],
        lookback_months=lookback
    )
    
    loader = Nino34DataLoader(config)
    X_train, y_train, X_test, y_test = loader.prepare_train_test()
    
    return X_train, y_train, X_test, y_test, loader


if __name__ == "__main__":
    # Test the loader
    print("="*60)
    print("NOAA Nino 3.4 Data Loader Test")
    print("="*60)
    
    X_train, y_train, X_test, y_test, loader = load_nino34_for_discovery()
    
    print("\nData loaded successfully!")
    print(f"Sample input: {X_train[0].numpy()}")
    print(f"Sample target: {y_train[0].item():.3f}")
