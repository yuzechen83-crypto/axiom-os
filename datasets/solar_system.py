"""
Solar System Orbital Data - NASA Planetary Fact Sheets

Semi-major axes and distances for falsification tests.
Primary: astroquery.jplhorizons (JPL Horizons API, live). Fallback: NASA JPL Fact Sheets.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AU_M = 1.495978707e11  # 1 AU in m
LY_M = 9.4607e15       # 1 light-year in m

# NASA JPL Planetary Fact Sheets - Semi-major axis (AU)
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/
SOLAR_SYSTEM_TARGETS = {
    "Mercury": {"r_au": 0.387, "source": "NASA JPL"},
    "Venus": {"r_au": 0.723, "source": "NASA JPL"},
    "Earth": {"r_au": 1.000, "source": "NASA JPL"},
    "Mars": {"r_au": 1.524, "source": "NASA JPL"},
    "Jupiter": {"r_au": 5.203, "source": "NASA JPL"},
    "Saturn": {"r_au": 9.537, "source": "NASA JPL"},
    "Uranus": {"r_au": 19.191, "source": "NASA JPL"},
    "Neptune": {"r_au": 30.069, "source": "NASA JPL"},
    "Pluto": {"r_au": 39.482, "source": "NASA JPL"},
}

# Extrasolar: distance from Sun (JPL Horizons does not provide stellar distances)
EXTRASOLAR_TARGETS = {
    "Proxima_Centauri": {"r_ly": 4.246, "source": "Gaia DR3"},
}


# JPL Horizons IDs (avoid ambiguous names: Saturn->699, Pluto->999)
_HORIZONS_IDS = {"Saturn": "699", "Pluto": "999"}


def _fetch_horizons_semi_major_axis(body_name: str) -> Optional[float]:
    """
    Fetch semi-major axis (AU) from JPL Horizons for a solar system body.
    body_name: 'Saturn', 'Pluto'.
    Returns None on failure.
    """
    body_id = _HORIZONS_IDS.get(body_name, body_name)
    try:
        from astroquery.jplhorizons import Horizons
        import astropy.time
        t = astropy.time.Time.now()
        obj = Horizons(id=body_id, location="@0", epochs=t.jd)
        el = obj.elements()
        if el is not None and len(el) > 0:
            a_au = float(el["a"][0])  # semi-major axis in AU
            return a_au
    except Exception:
        pass
    return None


def load_solar_system_targets(
    include_extrasolar: bool = True,
    selection: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Load orbital parameters for solar system falsification.
    Returns dict: name -> {r_au, r_m, source}.
    """
    out = {}
    for name, cfg in SOLAR_SYSTEM_TARGETS.items():
        if selection and name not in selection:
            continue
        r_au = cfg["r_au"]
        out[name] = {
            "r_au": r_au,
            "r_m": r_au * AU_M,
            "source": cfg.get("source", "NASA JPL"),
        }
    if include_extrasolar:
        for name, cfg in EXTRASOLAR_TARGETS.items():
            if selection and name not in selection:
                continue
            r_ly = cfg["r_ly"]
            r_au = r_ly * (LY_M / AU_M)
            out[name] = {
                "r_au": r_au,
                "r_ly": r_ly,
                "r_m": r_ly * LY_M,
                "source": cfg.get("source", "Gaia"),
            }
    return out


def get_default_falsification_targets(use_real_data: bool = True) -> Dict[str, Dict]:
    """
    Default targets for falsify_solar: Saturn, Pluto, Proxima.
    When use_real_data=True, fetches Saturn and Pluto semi-major axes from JPL Horizons
    at runtime. Proxima uses Gaia DR3 distance (Horizons does not provide stellar distances).
    """
    selection = ["Saturn", "Pluto", "Proxima_Centauri"]
    out: Dict[str, Dict] = {}

    if use_real_data:
        for body in ["Saturn", "Pluto"]:
            a_au = _fetch_horizons_semi_major_axis(body)
            if a_au is not None:
                out[body] = {
                    "r_au": a_au,
                    "r_m": a_au * AU_M,
                    "source": "JPL Horizons (live)",
                }

    # Fill any missing or add Proxima from static tables
    fallback = load_solar_system_targets(
        include_extrasolar=True,
        selection=selection,
    )
    for name in selection:
        if name not in out:
            out[name] = fallback[name]

    return out
