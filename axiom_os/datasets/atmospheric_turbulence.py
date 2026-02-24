"""
3D Atmospheric Turbulence - Real Data Pipeline
Open-Meteo API: wind at multiple heights (10m, 80m, 120m, 180m) over lat/lon grid.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import json
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Height levels (m) for 3D structure
HEIGHT_LEVELS = [10, 80, 120, 180]
HEIGHT_PARAMS = [
    "wind_speed_10m", "wind_direction_10m",
    "wind_speed_80m", "wind_direction_80m",
    "wind_speed_120m", "wind_direction_120m",
    "wind_speed_180m", "wind_direction_180m",
]
# Physics params for ageostrophic discovery: pressure gradient, thermal wind
PHYSICS_PARAMS = ["surface_pressure", "temperature_2m"]
OMEGA_EARTH = 7.292e-5  # rad/s


def _speed_dir_to_uv(speed: np.ndarray, direction_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert wind speed (m/s) and direction (deg, from N) to u (east), v (north)."""
    rad = np.deg2rad(direction_deg)
    # Meteorological: direction = from where wind comes. 0=N, 90=E.
    # u_east = speed * sin(rad), v_north = speed * cos(rad) for wind "to" direction
    u = speed * np.sin(rad)
    v = speed * np.cos(rad)
    return u.astype(np.float32), v.astype(np.float32)


def _fetch_single_location(
    lat: float, lon: float,
    hourly: str,
    forecast_days: int = 3,
) -> Optional[Dict]:
    """Fetch hourly data for one location (urllib, no API key)."""
    try:
        url = (
            f"{OPEN_METEO_URL}?latitude={lat}&longitude={lon}"
            f"&hourly={hourly}&forecast_days={forecast_days}&wind_speed_unit=ms"
        )
        req = Request(url, headers={"User-Agent": "AxiomOS/1.0"})
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except (URLError, HTTPError, json.JSONDecodeError, OSError):
        return None


def load_atmospheric_turbulence_3d(
    lat_center: float = 39.9,
    lon_center: float = 116.4,
    n_lat: int = 3,
    n_lon: int = 3,
    delta_deg: float = 0.1,
    forecast_days: int = 2,
    use_synthetic_if_fail: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load 3D atmospheric wind field from Open-Meteo (real data).
    Returns coords (t, x, y, z) and targets (u, v) for turbulence modeling.

    Args:
        lat_center, lon_center: Center of grid (e.g., Beijing 39.9, 116.4)
        n_lat, n_lon: Grid size
        delta_deg: Spacing in degrees
        forecast_days: Hours = 24 * forecast_days

    Returns:
        coords: (N, 4) = (t_norm, x_norm, y_norm, z_norm) in [0,1]
        targets: (N, 2) = (u, v) wind components in m/s
        meta: dict with lat, lon, times, data_source
    """
    hourly = ",".join(HEIGHT_PARAMS)
    lats = lat_center + np.linspace(-(n_lat - 1) / 2, (n_lat - 1) / 2, n_lat) * delta_deg
    lons = lon_center + np.linspace(-(n_lon - 1) / 2, (n_lon - 1) / 2, n_lon) * delta_deg

    all_t = []
    all_u = []
    all_v = []
    all_lat = []
    all_lon = []
    all_z = []
    lats_used, lons_used = [], []

    for lat in lats:
        for lon in lons:
            data = _fetch_single_location(lat, lon, hourly, forecast_days)
            if data is None or "hourly" not in data:
                continue
            lats_used.append(lat)
            lons_used.append(lon)
            h = data["hourly"]
            times = h.get("time", [])
            n_t = len(times)
            if n_t < 2:
                continue
            for iz, (sp_key, di_key) in enumerate([
                ("wind_speed_10m", "wind_direction_10m"),
                ("wind_speed_80m", "wind_direction_80m"),
                ("wind_speed_120m", "wind_direction_120m"),
                ("wind_speed_180m", "wind_direction_180m"),
            ]):
                speed = np.array(h.get(sp_key, [0] * n_t), dtype=np.float64)
                direction = np.array(h.get(di_key, [0] * n_t), dtype=np.float64)
                if len(speed) != n_t:
                    speed = np.full(n_t, np.nan)
                if len(direction) != n_t:
                    direction = np.full(n_t, 0.0)
                u, v = _speed_dir_to_uv(speed, direction)
                for i in range(n_t):
                    all_t.append(i / max(1, n_t - 1))
                    all_u.append(u[i])
                    all_v.append(v[i])
                    all_lat.append(lat)
                    all_lon.append(lon)
                    all_z.append(HEIGHT_LEVELS[iz])

    if len(all_t) < 10:
        if use_synthetic_if_fail:
            print("Open-Meteo fetch failed or insufficient data, using synthetic 3D turbulence.")
            return _generate_synthetic_3d_turbulence()
        raise RuntimeError(
            "Open-Meteo API returned insufficient data (<10 points). "
            "Check network connection or use use_synthetic_if_fail=True."
        )

    lat_arr = np.array(all_lat)
    lon_arr = np.array(all_lon)
    lat_min, lat_max = lat_arr.min(), lat_arr.max()
    lon_min, lon_max = lon_arr.min(), lon_arr.max()
    coords = np.column_stack([
        np.array(all_t, dtype=np.float32),
        (lat_arr - lat_min) / (lat_max - lat_min + 1e-12),
        (lon_arr - lon_min) / (lon_max - lon_min + 1e-12),
        (np.array(all_z) - HEIGHT_LEVELS[0]) / (HEIGHT_LEVELS[-1] - HEIGHT_LEVELS[0] + 1e-12),
    ])
    targets = np.column_stack([np.array(all_u), np.array(all_v)]).astype(np.float32)
    meta = {
        "data_source": "real",
        "lat_range": (float(lats.min()), float(lats.max())),
        "lon_range": (float(lons.min()), float(lons.max())),
        "n_points": len(all_t),
    }
    return coords, targets, meta


def load_atmospheric_turbulence_3d_with_physics(
    lat_center: float = 39.9,
    lon_center: float = 116.4,
    n_lat: int = 3,
    n_lon: int = 3,
    delta_deg: float = 0.15,
    forecast_days: int = 3,
    use_synthetic_if_fail: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load 3D wind + physics features for ageostrophic discovery.
    Returns coords, targets, physics_features, meta.
    physics_features: (N, 7) = [gradP_x, gradP_y, gradT_x, gradT_y, inv_f, gradP_dt, isallobaric]
    - gradP_x, gradP_y: pressure gradient (hPa/deg)
    - gradT_x, gradT_y: temperature gradient (°C/deg)
    - inv_f: 1/f (s), f = 2Ωsin(φ) Coriolis parameter
    - gradP_dt: ∂P/∂t (hPa/h) pressure tendency
    - isallobaric: -(1/f²)*gradP_dt, Isallobaric wind driver
    """
    hourly_all = ",".join(HEIGHT_PARAMS + PHYSICS_PARAMS)
    lats = lat_center + np.linspace(-(n_lat - 1) / 2, (n_lat - 1) / 2, n_lat) * delta_deg
    lons = lon_center + np.linspace(-(n_lon - 1) / 2, (n_lon - 1) / 2, n_lon) * delta_deg

    # Per (lat, lon): P(t), T(t), u(t,z), v(t,z)
    data_by_loc: Dict[Tuple[float, float], Dict] = {}
    for lat in lats:
        for lon in lons:
            data = _fetch_single_location(lat, lon, hourly_all, forecast_days)
            if data is None or "hourly" not in data:
                continue
            h = data["hourly"]
            times = h.get("time", [])
            n_t = len(times)
            if n_t < 2:
                continue
            P = np.array(h.get("surface_pressure", [1013.0] * n_t), dtype=np.float64)
            T = np.array(h.get("temperature_2m", [15.0] * n_t), dtype=np.float64)
            if len(P) != n_t:
                P = np.full(n_t, 1013.0)
            if len(T) != n_t:
                T = np.full(n_t, 15.0)
            uv_by_z = []
            for sp_key, di_key in [
                ("wind_speed_10m", "wind_direction_10m"),
                ("wind_speed_80m", "wind_direction_80m"),
                ("wind_speed_120m", "wind_direction_120m"),
                ("wind_speed_180m", "wind_direction_180m"),
            ]:
                sp = np.array(h.get(sp_key, [0] * n_t), dtype=np.float64)
                di = np.array(h.get(di_key, [0] * n_t), dtype=np.float64)
                u, v = _speed_dir_to_uv(sp, di)
                uv_by_z.append((u, v))
            data_by_loc[(lat, lon)] = {"P": P, "T": T, "uv": uv_by_z, "n_t": n_t}

    if len(data_by_loc) < 4:
        if use_synthetic_if_fail:
            print("Insufficient data for physics gradients, using synthetic.")
            coords, targets, meta = _generate_synthetic_3d_turbulence(
                n_t=24 * forecast_days, n_x=n_lat, n_y=n_lon, n_z=4
            )
            n = len(coords)
            np.random.seed(42)
            physics = np.column_stack([
                np.random.randn(n).astype(np.float32) * 2,
                np.random.randn(n).astype(np.float32) * 2,
                np.random.randn(n).astype(np.float32) * 0.5,
                np.random.randn(n).astype(np.float32) * 0.5,
                np.full(n, 1e4, dtype=np.float32),
                np.random.randn(n).astype(np.float32) * 0.5,
                np.random.randn(n).astype(np.float32) * 1e6,
            ])
            return coords, targets, physics, {**meta, "data_source": "synthetic"}
        raise RuntimeError(
            "Insufficient locations (<4) for physics gradients. "
            "Need 4+ grid points for gradP/gradT. Use use_synthetic_if_fail=True."
        )

    # Build P, T grids: (n_lat, n_lon, n_t)
    lat_list = sorted(set(k[0] for k in data_by_loc))
    lon_list = sorted(set(k[1] for k in data_by_loc))
    n_t = next(iter(data_by_loc.values()))["n_t"]
    P_grid = np.full((len(lat_list), len(lon_list), n_t), np.nan)
    T_grid = np.full((len(lat_list), len(lon_list), n_t), np.nan)
    for i, lat in enumerate(lat_list):
        for j, lon in enumerate(lon_list):
            if (lat, lon) in data_by_loc:
                P_grid[i, j, :] = data_by_loc[(lat, lon)]["P"]
                T_grid[i, j, :] = data_by_loc[(lat, lon)]["T"]

    # Gradients: central difference, Pa/deg (1 deg ≈ 111 km)
    delta = delta_deg
    gradP_x = np.zeros_like(P_grid)
    gradP_y = np.zeros_like(P_grid)
    gradT_x = np.zeros_like(T_grid)
    gradT_y = np.zeros_like(T_grid)
    for i in range(len(lat_list)):
        for j in range(len(lon_list)):
            for t in range(n_t):
                if i > 0 and i < len(lat_list) - 1:
                    gradP_y[i, j, t] = (P_grid[i + 1, j, t] - P_grid[i - 1, j, t]) / (2 * delta)
                    gradT_y[i, j, t] = (T_grid[i + 1, j, t] - T_grid[i - 1, j, t]) / (2 * delta)
                elif i > 0:
                    gradP_y[i, j, t] = (P_grid[i, j, t] - P_grid[i - 1, j, t]) / delta
                    gradT_y[i, j, t] = (T_grid[i, j, t] - T_grid[i - 1, j, t]) / delta
                else:
                    gradP_y[i, j, t] = (P_grid[i + 1, j, t] - P_grid[i, j, t]) / delta
                    gradT_y[i, j, t] = (T_grid[i + 1, j, t] - T_grid[i, j, t]) / delta
                if j > 0 and j < len(lon_list) - 1:
                    gradP_x[i, j, t] = (P_grid[i, j + 1, t] - P_grid[i, j - 1, t]) / (2 * delta)
                    gradT_x[i, j, t] = (T_grid[i, j + 1, t] - T_grid[i, j - 1, t]) / (2 * delta)
                elif j > 0:
                    gradP_x[i, j, t] = (P_grid[i, j, t] - P_grid[i, j - 1, t]) / delta
                    gradT_x[i, j, t] = (T_grid[i, j, t] - T_grid[i, j - 1, t]) / delta
                else:
                    gradP_x[i, j, t] = (P_grid[i, j + 1, t] - P_grid[i, j, t]) / delta
                    gradT_x[i, j, t] = (T_grid[i, j + 1, t] - T_grid[i, j, t]) / delta

    # Coriolis: f = 2Ω sin(φ), inv_f = 1/f (s), inv_f2 = 1/f² (s²)
    lat_rad = np.deg2rad(lat_list)
    f_vals = 2 * OMEGA_EARTH * np.sin(np.abs(lat_rad))
    f_vals = np.clip(f_vals, 1e-10, 1.0)
    inv_f_grid = np.zeros((len(lat_list), len(lon_list)))
    inv_f2_grid = np.zeros((len(lat_list), len(lon_list)))
    for i in range(len(lat_list)):
        inv_f_grid[i, :] = 1.0 / f_vals[i]
        inv_f2_grid[i, :] = 1.0 / (f_vals[i] ** 2)

    # Temporal derivative: gradP_dt = ∂P/∂t (hPa/h), Isallobaric wind driver
    dt_h = 1.0  # hourly data
    gradP_dt = np.zeros_like(P_grid)
    for i in range(len(lat_list)):
        for j in range(len(lon_list)):
            for t in range(n_t):
                if t > 0 and t < n_t - 1:
                    gradP_dt[i, j, t] = (P_grid[i, j, t + 1] - P_grid[i, j, t - 1]) / (2 * dt_h)
                elif t > 0:
                    gradP_dt[i, j, t] = (P_grid[i, j, t] - P_grid[i, j, t - 1]) / dt_h
                else:
                    gradP_dt[i, j, t] = (P_grid[i, j, t + 1] - P_grid[i, j, t]) / dt_h

    # Isallobaric term: -(1/f²) * gradP_dt (ageostrophic from pressure tendency)
    isallobaric = -inv_f2_grid[:, :, np.newaxis] * gradP_dt

    # Flatten to match (t, lat, lon, z) ordering
    all_t, all_u, all_v, all_lat_idx, all_lon_idx, all_z = [], [], [], [], [], []
    for i, lat in enumerate(lat_list):
        for j, lon in enumerate(lon_list):
            if (lat, lon) not in data_by_loc:
                continue
            for t in range(n_t):
                for iz in range(4):
                    u, v = data_by_loc[(lat, lon)]["uv"][iz]
                    all_t.append(t / max(1, n_t - 1))
                    all_u.append(u[t])
                    all_v.append(v[t])
                    all_lat_idx.append(i)
                    all_lon_idx.append(j)
                    all_z.append(HEIGHT_LEVELS[iz])

    all_t = np.array(all_t)
    all_lat_idx = np.array(all_lat_idx, dtype=int)
    all_lon_idx = np.array(all_lon_idx, dtype=int)
    t_idx = (all_t * (n_t - 1)).astype(int).clip(0, n_t - 1)

    gradP_x_flat = gradP_x[all_lat_idx, all_lon_idx, t_idx]
    gradP_y_flat = gradP_y[all_lat_idx, all_lon_idx, t_idx]
    gradT_x_flat = gradT_x[all_lat_idx, all_lon_idx, t_idx]
    gradT_y_flat = gradT_y[all_lat_idx, all_lon_idx, t_idx]
    inv_f_flat = inv_f_grid[all_lat_idx, all_lon_idx]
    gradP_dt_flat = gradP_dt[all_lat_idx, all_lon_idx, t_idx]
    isallobaric_flat = isallobaric[all_lat_idx, all_lon_idx, t_idx]

    lat_min, lat_max = min(lat_list), max(lat_list)
    lon_min, lon_max = min(lon_list), max(lon_list)
    coords = np.column_stack([
        all_t.astype(np.float32),
        (np.array([lat_list[i] for i in all_lat_idx]) - lat_min) / (lat_max - lat_min + 1e-12),
        (np.array([lon_list[j] for j in all_lon_idx]) - lon_min) / (lon_max - lon_min + 1e-12),
        (np.array(all_z) - HEIGHT_LEVELS[0]) / (HEIGHT_LEVELS[-1] - HEIGHT_LEVELS[0] + 1e-12),
    ])
    targets = np.column_stack([np.array(all_u), np.array(all_v)]).astype(np.float32)
    physics = np.column_stack([
        gradP_x_flat.astype(np.float32),
        gradP_y_flat.astype(np.float32),
        gradT_x_flat.astype(np.float32),
        gradT_y_flat.astype(np.float32),
        inv_f_flat.astype(np.float32),
        gradP_dt_flat.astype(np.float32),
        isallobaric_flat.astype(np.float32),
    ])
    meta = {
        "data_source": "real",
        "lat_range": (float(lat_min), float(lat_max)),
        "lon_range": (float(lon_min), float(lon_max)),
        "n_points": len(all_t),
    }
    return coords, targets, physics, meta


def _generate_synthetic_3d_turbulence(
    n_t: int = 24,
    n_x: int = 4,
    n_y: int = 4,
    n_z: int = 4,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Synthetic 3D turbulence when API unavailable."""
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, 1, n_t)
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    z = np.linspace(0, 1, n_z)
    tt, xx, yy, zz = np.meshgrid(t, x, y, z, indexing="ij")
    tt = tt.ravel()
    xx = xx.ravel()
    yy = yy.ravel()
    zz = zz.ravel()

    u = 2.0 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) * np.cos(2 * np.pi * tt)
    v = -2.0 * np.cos(2 * np.pi * xx) * np.sin(2 * np.pi * yy) * np.cos(2 * np.pi * tt)
    u += 0.5 * np.sin(4 * np.pi * zz) * np.cos(2 * np.pi * tt)
    v += 0.3 * np.cos(4 * np.pi * zz) * np.sin(2 * np.pi * tt)
    u += 0.1 * np.random.randn(len(tt))
    v += 0.1 * np.random.randn(len(tt))

    coords = np.column_stack([tt, xx, yy, zz]).astype(np.float32)
    targets = np.column_stack([u, v]).astype(np.float32)
    meta = {"data_source": "synthetic", "n_points": len(tt)}
    return coords, targets, meta
