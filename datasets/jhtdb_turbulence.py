"""
JHTDB Turbulence - Real DNS Data (Johns Hopkins Turbulence Database)

REAL DATA ONLY. No synthetic fallback.
Uses givernylocal to fetch isotropic turbulence velocity from JHTDB web service.

Auth token: Set JHTDB_AUTH_TOKEN env for larger cutouts. Default test token limits to 4096 points.
Get token: https://turbulence.idies.jhu.edu/staging/database

Run: python -m axiom_os.experiments.jhtdb_les_sgs
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "jhtdb"
JHTDB_DATASETS = ["isotropic8192", "isotropic32768", "sabl2048low", "sabl2048high"]


def _get_auth_token() -> str:
    """JHTDB auth token. Default test token limits to 4096 points."""
    return os.environ.get("JHTDB_AUTH_TOKEN", "")


def load_jhtdb_velocity_cutout(
    size: int = 16,
    dataset: str = "isotropic8192",
    timepoint: int = 1,
    auth_token: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load real JHTDB velocity cutout. Shape (size, size, size, 3) for ux, uy, uz.

    Args:
        size: Box size per axis (e.g. 16 -> 16^3 = 4096 points). Test token max 4096.
        dataset: isotropic8192 or isotropic32768
        timepoint: Time index (isotropic8192: 1-6)
        auth_token: Override env. Empty = use default test token.

    Returns:
        ux, uy, uz: (N,N,N) velocity components in m/s
        meta: dict with data_source, shape, dataset
    """
    try:
        from givernylocal.turbulence_dataset import turb_dataset
        from givernylocal.turbulence_toolkit import getCutout
    except ImportError as e:
        raise RuntimeError(
            "JHTDB requires givernylocal: pip install givernylocal. "
            "Real data only - no synthetic fallback."
        ) from e

    token = auth_token if auth_token is not None else _get_auth_token()
    n_points = size**3
    if n_points > 4096 and not token:
        raise RuntimeError(
            f"JHTDB test token limits to 4096 points. Requested {n_points} (size={size}). "
            "Set JHTDB_AUTH_TOKEN env or get token from https://turbulence.idies.jhu.edu/staging/database"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(DATA_DIR / "cutout_cache")

    cube = turb_dataset(dataset_title=dataset, output_path=out_path, auth_token=token)
    axes = np.array([[0, size - 1], [0, size - 1], [0, size - 1], [timepoint, timepoint]])
    strides = np.array([1, 1, 1, 1])

    result = getCutout(cube, "velocity", axes, strides, verbose=False)
    key = [k for k in result.data_vars if "velocity" in k][0]
    vel = result[key].values  # (16,16,16,3) or (N,N,N,3)
    ux = vel[..., 0].astype(np.float64)
    uy = vel[..., 1].astype(np.float64)
    uz = vel[..., 2].astype(np.float64)

    meta = {
        "data_source": "JHTDB",
        "dataset": dataset,
        "shape": ux.shape,
        "n_points": int(np.prod(ux.shape)),
        "timepoint": timepoint,
    }
    return ux, uy, uz, meta


def load_jhtdb_for_les_sgs(
    fine_size: int = 16,
    coarse_ratio: int = 2,
    dataset: str = "isotropic8192",
    timepoint: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load JHTDB and prepare for LES-SGS: fine (DNS) and coarse (filtered) views.

    Returns:
        u_fine: (N,N,N,3) velocity in m/s
        u_coarse: (N/c, N/c, N/c, 3) box-filtered
        tau_target: (N/c, N/c, N/c, 3,3) Reynolds stress tensor for Soft Shell target
        meta: dict
    """
    ux, uy, uz, meta = load_jhtdb_velocity_cutout(
        size=fine_size, dataset=dataset, timepoint=timepoint
    )
    u_fine = np.stack([ux, uy, uz], axis=-1)

    # Box filter: average over coarse_ratio^3 cells
    r = coarse_ratio
    n_c = fine_size // r
    if n_c < 2:
        raise ValueError(f"coarse_ratio={r} too large for fine_size={fine_size}")
    u_coarse = np.zeros((n_c, n_c, n_c, 3))
    for i in range(n_c):
        for j in range(n_c):
            for k in range(n_c):
                slab = u_fine[i * r : (i + 1) * r, j * r : (j + 1) * r, k * r : (k + 1) * r, :]
                u_coarse[i, j, k, :] = np.mean(slab, axis=(0, 1, 2))

    # Reynolds stress: tau_ij = <u_i u_j> - <u_i><u_j> (Leonard + cross + SGS)
    # Simplified: tau_ij = filtered(u_i u_j) - filtered(u_i)*filtered(u_j)
    uu_fine = np.zeros((fine_size, fine_size, fine_size, 3, 3))
    for i in range(3):
        for j in range(3):
            uu_fine[..., i, j] = u_fine[..., i] * u_fine[..., j]
    uu_coarse = np.zeros((n_c, n_c, n_c, 3, 3))
    for i in range(n_c):
        for j in range(n_c):
            for k in range(n_c):
                for pi in range(3):
                    for pj in range(3):
                        slab = uu_fine[i * r : (i + 1) * r, j * r : (j + 1) * r, k * r : (k + 1) * r, pi, pj]
                        uu_coarse[i, j, k, pi, pj] = np.mean(slab)
    tau_target = np.zeros_like(uu_coarse)
    for i in range(3):
        for j in range(3):
            tau_target[..., i, j] = uu_coarse[..., i, j] - u_coarse[..., i] * u_coarse[..., j]

    meta["coarse_shape"] = u_coarse.shape
    meta["coarse_ratio"] = r
    return u_fine, u_coarse, tau_target, meta
