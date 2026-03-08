"""
RC100: Rotation Curves of 100 Massive Star-Forming Galaxies at z=0.6-2.5

Real data from Shachar et al. 2023, ApJ 944, 78.
Source: https://iopscience.iop.org/0004-637X/944/1/78/suppdata/apjaca9cft3_ascii.txt
Table B1: Galaxy best-fit parameters (z, R_e, V_circ, M_baryon).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RC100_URL = (
    "https://iopscience.iop.org/0004-637X/944/1/78/suppdata/apjaca9cft3_ascii.txt"
    "?doi=10.3847/1538-4357/aca9cf"
)


def _parse_value(s: str) -> float:
    """Extract first number from string like '8.10 (1.53)' or '0.61'."""
    s = s.strip()
    # Take part before first space or parenthesis
    for sep in (" ", "(", "\t"):
        if sep in s:
            s = s.split(sep)[0]
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_rc100_from_url() -> Optional[List[Dict]]:
    """
    Fetch and parse RC100 Table B1 from ApJ supplementary data.
    Returns list of dicts: {z, R, V_rot, M_bary} with R in kpc, V_rot in km/s, M_bary in M_sun.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(RC100_URL, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    lines = raw.strip().split("\n")
    if len(lines) < 3:
        return None

    # Header: Index, Galaxy, z, FWHM, T_int, ..., log(M_baryon), ..., R_e, f_DM, V_circ, ...
    # Columns: 0=Index, 1=Galaxy, 2=z, 3=FWHM, 4=T_int, 5=delta_log_SFR, 6=log_M_star,
    #          7=log_M_baryon, 8=log_M_bulge, 9=R_e, 10=f_DM, 11=V_circ, ...
    # Tab-separated
    data: List[Dict] = []
    for line in lines[2:]:  # skip header lines
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            z = _parse_value(parts[2])
            log_m_baryon = _parse_value(parts[7])
            r_e = _parse_value(parts[9])
            v_circ = _parse_value(parts[11])
            if np.isfinite(z) and np.isfinite(log_m_baryon) and np.isfinite(r_e) and np.isfinite(v_circ):
                m_bary = 10.0 ** log_m_baryon  # M_sun
                data.append({
                    "z": z,
                    "R": r_e,
                    "V_rot": v_circ,
                    "M_bary": m_bary,
                })
        except (IndexError, ValueError):
            continue
    return data if data else None


def load_rc100_cached(cache_dir: Optional[Path] = None) -> Optional[List[Dict]]:
    """
    Load RC100 data, using cache if available.
    Cache file: cache_dir/rc100_table_b1.json
    """
    cache_dir = cache_dir or (ROOT / "axiom_os" / "datasets" / "cache")
    cache_path = cache_dir / "rc100_table_b1.json"

    if cache_path.exists():
        try:
            import json
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, list) and len(raw) > 0:
                return raw
        except Exception:
            pass

    data = load_rc100_from_url()
    if data:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import json
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
    return data
