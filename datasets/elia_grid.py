"""
Elia (Belgium TSO) Power Grid Data Loader
Operation "Grid Pulse" - System Imbalance vs Frequency Deviation.

Source: Elia Open Data (Frequency + System Imbalance)
- Frequency: Hz (target: deviation from 50Hz)
- System Imbalance: MW (Net Regulation Volume)

Physics: Swing Equation  df/dt = (1/M)(P_imbalance - D·f)

Usage:
  - Provide elia_2022.csv or elia_2023.csv (see 数据获取指南 below)
  - Or use synthetic data for demo: load_elia_grid(use_synthetic=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

ROOT = Path(__file__).resolve().parents[2]

# Expected CSV columns (user-provided or synthetic)
FREQ_COL = "frequency_Hz"
IMBALANCE_COL = "imbalance_MW"
TIME_COL = "timestamp"

# Belgian grid typical values (for synthetic)
F_NOMINAL = 50.0  # Hz
M_BASE_BELGIUM = 2.5e4  # MW·s/Hz (approx from ~6 GW synchronous capacity)
D_BASE = 500.0    # MW/Hz (damping)


def _generate_synthetic_elia(
    n_minutes: int = 7 * 24 * 60,  # 7 days
    dt_min: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic Elia-style data for testing.
    Simulates: daily load pattern, solar/wind variability, frequency events.
    """
    np.random.seed(seed)
    n = n_minutes
    t = np.arange(n, dtype=np.float64) * dt_min  # minutes

    # Daily load pattern (GW) - peak ~8 GW, trough ~5 GW
    load_GW = 6.5 + 1.5 * np.sin(2 * np.pi * t / (24 * 60) - np.pi / 2)
    load_GW += 0.3 * np.random.randn(n).cumsum() * 0.01
    load_GW = np.clip(load_GW, 4.0, 10.0)

    # Solar (0 at night, peak midday)
    hour_frac = (t % (24 * 60)) / (24 * 60)
    solar_GW = 3.0 * np.exp(-((hour_frac - 0.5) ** 2) / 0.08)
    solar_GW += 0.1 * np.random.randn(n)

    # Wind variability
    wind_GW = 2.0 + 1.5 * np.sin(2 * np.pi * t / (6 * 60)) + 0.5 * np.random.randn(n).cumsum() * 0.02
    wind_GW = np.clip(wind_GW, 0.2, 5.0)

    # Imbalance: load - (solar + wind + base_gen), with noise (scale down for realistic f_dev)
    base_gen = 4.0
    imbalance_MW = (load_GW - solar_GW - wind_GW - base_gen / 10) * 200  # scale for ~0.1 Hz range
    imbalance_MW += 20 * np.random.randn(n)
    # Add occasional large events (simulate plant trip)
    event_times = np.random.choice(n, size=5, replace=False)
    for et in event_times:
        imbalance_MW[et : et + 10] += np.random.uniform(50, 150)

    # Integrate Swing Equation to get frequency
    M, D = M_BASE_BELGIUM, D_BASE
    f_dev = np.zeros(n)
    for i in range(1, n):
        dt_s = dt_min * 60  # seconds
        df_dt = (1.0 / M) * (imbalance_MW[i - 1] - D * f_dev[i - 1])
        f_dev[i] = f_dev[i - 1] + dt_s * df_dt
        f_dev[i] = np.clip(f_dev[i], -1.0, 1.0)  # realistic bounds

    # Add measurement noise
    f_dev += 0.005 * np.random.randn(n)

    df = pd.DataFrame({
        TIME_COL: pd.to_datetime("2022-01-01") + pd.to_timedelta(t, unit="m"),
        FREQ_COL: F_NOMINAL + f_dev,
        IMBALANCE_COL: imbalance_MW,
        "load_GW": load_GW,
        "solar_GW": solar_GW,
        "wind_GW": wind_GW,
    })
    return df


def load_elia_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Load Elia data from CSV.
    Expected columns: timestamp (or datetime), frequency_Hz, imbalance_MW.
    Flexible: also accepts 'Frequency', 'Imbalance', 'Time' etc.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, nrows=50000)
    except Exception:
        return None

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "freq" in cl or "hz" in cl:
            col_map[c] = FREQ_COL
        elif "imbal" in cl or "regul" in cl or "mw" in cl:
            col_map[c] = IMBALANCE_COL
        elif "time" in cl or "date" in cl:
            col_map[c] = TIME_COL
    df = df.rename(columns=col_map)

    if FREQ_COL not in df.columns or IMBALANCE_COL not in df.columns:
        return None
    if TIME_COL not in df.columns:
        df[TIME_COL] = pd.to_datetime(df.index) if hasattr(df.index, "to_datetime") else pd.date_range(start="2022-01-01", periods=len(df), freq="1min")
    return df


def load_elia_grid(
    csv_path: Optional[Path] = None,
    use_synthetic_if_fail: bool = True,
    resample_min: int = 1,
    n_days: int = 7,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Elia grid data: System Imbalance (u) and Frequency (x).

    Returns:
        t: (N,) normalized time [0,1]
        u: (N,) System Imbalance (MW)
        x: (N,) Frequency deviation from 50Hz (Hz)
        df_dt: (N,) Frequency gradient df/dt (computed via finite diff)
        meta: dict with data_source, dt_sec, etc.

    Input (u): System Imbalance (MW)
    State (x): Frequency deviation from 50Hz (Hz)
    """
    meta: Dict[str, Any] = {"data_source": "synthetic", "dt_sec": resample_min * 60}

    if csv_path is not None:
        csv_path = Path(csv_path)
    else:
        for name in ["elia_2022.csv", "elia_2023.csv", "elia_grid.csv"]:
            p = ROOT / "axiom_os" / "data" / name
            if p.exists():
                csv_path = p
                break
        else:
            csv_path = None

    if csv_path is not None:
        df_raw = load_elia_csv(csv_path)
        if df_raw is not None and len(df_raw) > 10:
            meta["data_source"] = "real"
            df = df_raw.copy()
            df[TIME_COL] = pd.to_datetime(df[TIME_COL])
            df = df.sort_values(TIME_COL).dropna(subset=[FREQ_COL, IMBALANCE_COL])
            if resample_min > 1:
                df = df.set_index(TIME_COL).resample(f"{resample_min}min").mean().dropna().reset_index()
            f_Hz = df[FREQ_COL].values.astype(np.float64)
            u_MW = df[IMBALANCE_COL].values.astype(np.float64)
            x = f_Hz - F_NOMINAL  # deviation
            t_norm = np.linspace(0, 1, len(x))
            # df/dt via finite differences
            dt_s = (df[TIME_COL].iloc[-1] - df[TIME_COL].iloc[0]).total_seconds() / max(1, len(df) - 1)
            df_dt = np.gradient(x, dt_s) if len(x) > 1 else np.zeros_like(x)
            meta["dt_sec"] = dt_s
            return t_norm, u_MW, x, df_dt, meta

    if not use_synthetic_if_fail:
        raise FileNotFoundError("No Elia CSV found. Place elia_2022.csv in axiom_os/data/ or use use_synthetic_if_fail=True.")

    # Synthetic
    n_min = n_days * 24 * 60
    df = _generate_synthetic_elia(n_minutes=n_min, dt_min=float(resample_min), seed=seed)
    f_Hz = df[FREQ_COL].values.astype(np.float64)
    u_MW = df[IMBALANCE_COL].values.astype(np.float64)
    x = f_Hz - F_NOMINAL
    t_norm = np.linspace(0, 1, len(x))
    dt_s = resample_min * 60
    df_dt = np.gradient(x, dt_s)
    meta["load_GW"] = df["load_GW"].values
    meta["solar_GW"] = df["solar_GW"].values
    meta["wind_GW"] = df["wind_GW"].values
    return t_norm, u_MW, x, df_dt, meta


def build_grid_sequences(
    t: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    seq_len: int,
    forecast_horizon: int = 1,
    df_dt: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_seq, Y, coords) for training.
    X_seq: (N, seq_len, 4) = [t, u, x, df_dt] per step
    Y: (N, 1) = next x (frequency deviation)
    coords: (N, 4) = last step coords for hard core
    """
    if df_dt is None:
        dt = (t[-1] - t[0]) / max(1, len(t) - 1) if len(t) > 1 else 1.0
        df_dt = np.gradient(x, dt) if len(x) > 1 else np.zeros_like(x)
    n = len(t)
    if n < seq_len + forecast_horizon:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    seqs, ys, coords = [], [], []
    for i in range(n - seq_len - forecast_horizon + 1):
        last_idx = i + seq_len - 1  # last step of input sequence
        target_idx = i + seq_len + forecast_horizon - 1
        seq = np.column_stack([
            t[i : i + seq_len],
            u[i : i + seq_len],
            x[i : i + seq_len],
            df_dt[i : i + seq_len],
        ]).astype(np.float32)
        seqs.append(seq)
        ys.append(x[target_idx])
        # coords = last step (for hard core: predict next from f_prev, u)
        coords.append([t[last_idx], u[last_idx], x[last_idx], df_dt[last_idx]])
    return np.stack(seqs), np.array(ys, dtype=np.float32).reshape(-1, 1), np.array(coords, dtype=np.float32)
