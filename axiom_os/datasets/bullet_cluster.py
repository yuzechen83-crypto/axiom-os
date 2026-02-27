"""
Bullet Cluster (1E 0657-56) - Clowe et al. 2006, ApJ 648, L109

Data for testing fiber-bundle inertia mechanism: gravitational vs baryonic center offset.
Sources: weak lensing mass map, Chandra X-ray (gas), optical (stellar).

FITS: astropy.io.fits. Fallback: published parameters + synthetic spatial grid.

NOTE (single-system limitation): v_collision and t_since_collision are constants for the
entire Bullet system → symbol regression cannot discover τ_response. Use
axiom_os.datasets.merging_clusters for multi-system discovery (x_offset = f(v,t,M)).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Clowe 2006 published values (1E 0657-56)
BULLET_Z = 0.296
BULLET_D = 1180.0  # Mpc (angular diameter distance)
BULLET_V_COLL = 3000.0  # km/s (collision velocity)
BULLET_M_TOTAL = 1e15  # M_sun (total mass)
BULLET_T_Myr = 100.0  # Myr since collision (typical)
BULLET_X_OFFSET = 720.0  # kpc (gas-mass center separation, Clowe 2006)
ARCSEC_PER_KPC = 0.25  # at z=0.296, ~4 kpc/arcsec


def _compute_center(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Barycenter from 2D array."""
    w = np.maximum(arr, 0)
    s = w.sum()
    if s == 0:
        return float(np.mean(x)), float(np.mean(y))
    xc = np.average(x, weights=w)
    yc = np.average(y, weights=w)
    return float(xc), float(yc)


def load_bullet_cluster_fits(
    mass_map_path: Optional[Path] = None,
    gas_map_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load FITS mass and gas maps. Returns (x_kpc, y_kpc, rho_bar, g_obs_x, g_obs_y).
    g_obs from mass map via gradient of potential (surface density -> acceleration proxy).
    """
    try:
        from astropy.io import fits
    except ImportError:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), "astropy not installed"

    mass_data = None
    gas_data = None

    if mass_map_path and Path(mass_map_path).exists():
        with fits.open(mass_map_path) as hdul:
            mass_data = hdul[0].data
            if mass_data is None and len(hdul) > 1:
                mass_data = hdul[1].data
    if gas_map_path and Path(gas_map_path).exists():
        with fits.open(gas_map_path) as hdul:
            gas_data = hdul[0].data
            if gas_data is None and len(hdul) > 1:
                gas_data = hdul[1].data

    if mass_data is None and gas_data is None:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), "no FITS loaded"

    if mass_data is None:
        mass_data = np.ones_like(gas_data)
    if gas_data is None:
        gas_data = np.ones_like(mass_data)

    ny, nx = mass_data.shape
    scale_kpc = 4.0  # kpc per pixel (typical for Bullet cluster)
    x = (np.arange(nx) - nx / 2) * scale_kpc
    y = (np.arange(ny) - ny / 2) * scale_kpc
    xx, yy = np.meshgrid(x, y)

    x_flat = xx.ravel()
    y_flat = yy.ravel()
    mass_flat = np.nan_to_num(mass_data.ravel(), nan=0, posinf=0, neginf=0)
    gas_flat = np.nan_to_num(gas_data.ravel(), nan=0, posinf=0, neginf=0)

    x_g, y_g = _compute_center(mass_flat, x_flat, y_flat)
    x_b, y_b = _compute_center(gas_flat, x_flat, y_flat)

    rho_bar = gas_flat + 1e-10
    g_obs_x = np.gradient(mass_data.reshape(ny, nx), axis=1).ravel()
    g_obs_y = np.gradient(mass_data.reshape(ny, nx), axis=0).ravel()
    v_coll = np.full(len(x_flat), BULLET_V_COLL)
    t_coll = np.full(len(x_flat), BULLET_T_Myr)

    return x_flat, y_flat, rho_bar, g_obs_x, g_obs_y, v_coll, t_coll


def load_bullet_cluster_published(
    n_spatial: int = 200,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Build synthetic spatial grid from Clowe 2006 published parameters.
    Returns (x_kpc, y_kpc, rho_bar, g_obs_x, g_obs_y, v_collision, t_since_collision).
    """
    rng = np.random.default_rng(seed)
    x_offset = BULLET_X_OFFSET
    v_coll = BULLET_V_COLL
    t_myr = BULLET_T_Myr

    n_side = max(10, int(np.sqrt(n_spatial)))
    n_spatial = n_side * n_side
    x = np.linspace(-800, 800, n_side)
    y = np.linspace(-600, 600, n_side)
    xx, yy = np.meshgrid(x, y)

    x_flat = xx.ravel()
    y_flat = yy.ravel()

    x_g, y_g = 0, 0
    x_b = x_offset / 2
    y_b = 0

    sigma_g = 150
    sigma_b = 120
    rho_g = np.exp(-((x_flat - x_g) ** 2 + (y_flat - y_g) ** 2) / (2 * sigma_g**2))
    rho_b = np.exp(-((x_flat - x_b) ** 2 + (y_flat - y_b) ** 2) / (2 * sigma_b**2))
    rho_bar = rho_g * 0.3 + rho_b * 0.7 + 1e-6

    g_mag = rho_g * 2 + rho_b * 1.5 + 1e-6
    dx = x_flat - x_g
    dy = y_flat - y_g
    r = np.sqrt(dx**2 + dy**2) + 1
    g_obs_x = -g_mag * dx / r
    g_obs_y = -g_mag * dy / r

    v_coll_arr = np.full(n_spatial, v_coll)
    t_arr = np.full(n_spatial, t_myr)

    return x_flat, y_flat, rho_bar, g_obs_x, g_obs_y, v_coll_arr, t_arr


def compute_centers(
    rho_bar: np.ndarray,
    g_obs_mag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return (x_g, y_g, x_b, y_b) for gravitational and baryonic centers."""
    x_g, y_g = _compute_center(g_obs_mag, x, y)
    x_b, y_b = _compute_center(rho_bar, x, y)
    return x_g, y_g, x_b, y_b


def to_discovery_format(
    x_kpc: np.ndarray,
    y_kpc: np.ndarray,
    rho_bar: np.ndarray,
    g_obs_x: np.ndarray,
    g_obs_y: np.ndarray,
    v_collision: Optional[np.ndarray] = None,
    t_since_collision: Optional[np.ndarray] = None,
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    normalize: bool = True,
    causal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[Dict[str, float]]]:
    """
    Convert to per-pixel rows for Discovery Engine.
    Returns (X, y, var_names, scale_dict).

    When normalize=True: dimensionless ratios (rho/rho_mean, g/g_mean, v/v_ref, t/t_ref).
    When causal=True: X excludes x_offset (use rho, v, t to predict g; no circular input).
    """
    g_obs_mag = np.sqrt(g_obs_x**2 + g_obs_y**2 + 1e-20)

    if x_offset is None or y_offset is None:
        x_g, y_g, x_b, y_b = compute_centers(rho_bar, g_obs_mag, x_kpc, y_kpc)
    else:
        x_b, y_b = 0, 0
        x_g, y_g = x_offset, y_offset

    x_offset_local = np.sqrt((x_kpc - x_b) ** 2 + (y_kpc - y_b) ** 2)
    R_cluster = max(np.max(np.abs(x_kpc)), np.max(np.abs(y_kpc)), 1.0)

    rho_mean = float(np.mean(rho_bar) + 1e-20)
    g_mean = float(np.mean(g_obs_mag) + 1e-20)
    v_ref = BULLET_V_COLL
    t_ref = BULLET_T_Myr
    scale_dict = {"rho_mean": rho_mean, "g_mean": g_mean, "R_cluster": R_cluster, "v_ref": v_ref, "t_ref": t_ref}

    if normalize:
        rho_n = rho_bar / rho_mean
        g_n = g_obs_mag / g_mean
        s_n = x_offset_local / R_cluster
        v_n = v_collision / v_ref if v_collision is not None else None
        t_n = t_since_collision / t_ref if t_since_collision is not None else None
    else:
        rho_n, g_n, s_n = rho_bar, g_obs_mag, x_offset_local
        v_n, t_n = v_collision, t_since_collision

    if causal:
        cols = [rho_n]
        names = ["rho_bar_norm"]
        if v_n is not None:
            cols.append(v_n)
            names.append("v_collision_norm")
        if t_n is not None:
            cols.append(t_n)
            names.append("t_since_collision_norm")
        y = g_n
    else:
        cols = [rho_n, s_n]
        names = ["rho_bar_norm", "x_offset_local_norm"]
        if v_n is not None:
            cols.append(v_n)
            names.append("v_collision_norm")
        if t_n is not None:
            cols.append(t_n)
            names.append("t_since_collision_norm")
        y = g_n

    X = np.column_stack(cols)
    return X, y, names, scale_dict


def load_bullet_cluster_mvp(
    use_fits: bool = False,
    mass_path: Optional[Path] = None,
    gas_path: Optional[Path] = None,
    n_synthetic: int = 200,
    normalize: bool = True,
    causal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, Optional[Dict[str, float]]]:
    """
    MVP: Load Bullet Cluster data.
    When causal=True: g_obs/g_mean = f(rho_bar/rho_mean, v/v_ref, t/t_ref), no x_offset input.
    Returns (X, y, var_names, source, scale_dict).
    """
    if use_fits and (mass_path or gas_path):
        x, y, rho, gx, gy, v, t = load_bullet_cluster_fits(mass_path, gas_path)
        if len(x) > 0:
            X, y_out, names, scale = to_discovery_format(x, y, rho, gx, gy, v, t, normalize=normalize, causal=causal)
            return X, y_out, names, "FITS", scale

    x, y, rho, gx, gy, v, t = load_bullet_cluster_published(n_spatial=n_synthetic)
    X, y_out, names, scale = to_discovery_format(x, y, rho, gx, gy, v, t, normalize=normalize, causal=causal)
    return X, y_out, names, "published (Clowe 2006)", scale


def save_bullet_csv(
    out_path: Path,
    x_kpc: np.ndarray,
    y_kpc: np.ndarray,
    rho_bar: np.ndarray,
    g_obs_x: np.ndarray,
    g_obs_y: np.ndarray,
    v_collision: np.ndarray,
    t_since_collision: np.ndarray,
) -> None:
    """Save to CSV for Axiom input."""
    g_mag = np.sqrt(g_obs_x**2 + g_obs_y**2 + 1e-20)
    rows = np.column_stack([x_kpc, y_kpc, rho_bar, g_obs_x, g_obs_y, v_collision, t_since_collision])
    header = "x_kpc,y_kpc,rho_bar,g_obs_x,g_obs_y,v_collision,t_since_collision"
    np.savetxt(out_path, rows, delimiter=",", header=header, comments="")
