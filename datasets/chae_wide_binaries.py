"""
Chae 2023/2024 Wide Binary Data - Gaia DR3

Independent test of gravity at low acceleration.
Data: Zenodo 8065875 (Chae 2023) - gaia_dr3_MSMS_d200pc.csv
Ref: Chae 2023 ApJ 952, Chae 2024 ApJ 960

Theory prediction: sigma_v_obs / sigma_v_newton = sqrt(nu(g_newton/a0))
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHAE_CSV = DATA_DIR / "gaia_dr3_MSMS_d200pc.csv"
CHAE_ZENODO = "https://zenodo.org/records/8065875/files/gaia_dr3_MSMS_d200pc.csv?download=1"

# Physical constants
G_SI = 6.67430e-11
M_SUN = 1.9885e30
AU_M = 1.495978707e11
KAU_M = 1000 * AU_M


def _fetch_chae_csv() -> bool:
    """Download Chae CSV from Zenodo. Returns True if success."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CHAE_CSV.exists() and CHAE_CSV.stat().st_size > 1e6:
        return True
    try:
        import urllib.request
        req = urllib.request.Request(CHAE_ZENODO, headers={"User-Agent": "AxiomOS/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) > 1e6:
            CHAE_CSV.write_bytes(data)
            return True
    except Exception:
        pass
    try:
        import requests
        r = requests.get(CHAE_ZENODO, timeout=120)
        if r.status_code == 200 and len(r.content) > 1e6:
            CHAE_CSV.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def _parse_chae_csv(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse Chae CSV (gaia_dr3_MSMS_d200pc.csv).
    Columns: s[kau], M1[Msun], M2[Msun], ...
    """
    import csv
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < 100:
        return None
    sep_col = "s[kau]"
    m1_col, m2_col = "M1[Msun]", "M2[Msun]"
    cols = list(rows[0].keys())
    if sep_col not in cols:
        sep_col = next((c for c in cols if "s" in c.lower() and "kau" in c.lower()), cols[0])
    if m1_col not in cols:
        m1_col = next((c for c in cols if "M1" in c or "m1" in c), None)
    if m2_col not in cols:
        m2_col = next((c for c in cols if "M2" in c or "m2" in c), None)
    sep_list, mass_list = [], []
    for row in rows:
        try:
            s = float(row.get(sep_col, 0))
            m1 = float(row.get(m1_col, 1.0)) if m1_col else 1.0
            m2 = float(row.get(m2_col, 1.0)) if m2_col else 1.0
            m = m1 + m2
            if s > 0.1 and s < 1e5 and m > 0.1:
                sep_list.append(s)
                mass_list.append(m)
        except (ValueError, KeyError, TypeError):
            continue
    if len(sep_list) < 100:
        return None
    out = {}
    out["separation_kau"] = np.array(sep_list)
    out["mass_sun"] = np.array(mass_list)
    return out


def load_chae_wide_binaries() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Chae wide binary data.
    Returns: separation_kau, mass_sun (or empty arrays if unavailable).
    """
    if not _fetch_chae_csv():
        return np.array([]), np.array([])
    parsed = _parse_chae_csv(CHAE_CSV)
    if parsed is None:
        return np.array([]), np.array([])
    return parsed["separation_kau"], parsed["mass_sun"]


def g_newton_from_separation(s_kau: float, M_sun: float = 2.0) -> float:
    """Newtonian g = GM/r^2 [m/s^2]. s_kau = separation in kilo-AU."""
    r_m = s_kau * KAU_M
    M_kg = M_sun * M_SUN
    return G_SI * M_kg / (r_m**2 + 1e-30)


def get_chae_published_bins() -> List[Dict]:
    """
    Chae 2024 published binned results (when raw CSV unavailable).
    g_newton in m/s^2, gamma_g = g_obs/g_newton observed.
    """
    return [
        {"g_newton": 1e-10, "gamma_g": 1.49, "gamma_g_lo": 0.19, "gamma_g_hi": 0.21},
        {"g_newton": 10**-10.15, "gamma_g": 1.43, "gamma_g_lo": 0.06, "gamma_g_hi": 0.06},
        {"s_kau_min": 5, "gamma_vp": 1.20, "gamma_vp_stat": 0.06, "gamma_vp_sys": 0.05},
    ]
