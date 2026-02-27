"""
Golovich et al. 2019 - Merging Cluster Collaboration Radio-Selected Sample

29 merging galaxy clusters, z=0.07-0.55, from ApJS 240, 39.
VizieR J/ApJS/240/39, CDS ftp://cdsarc.cds.unistra.fr/ftp/J/ApJS/240/39/

Table1: Name, AName, RA, Dec, z, Band (discovery band: Radio/Optical/X-ray/SZ)
Table7: 4431 spectroscopic galaxies (DEIMOS)

NOTE: Table1 does NOT contain v_collision, t_since_collision, M_total, x_offset.
These must be supplemented from:
- Individual papers (weak lensing, dynamics)
- MCMAC runs (MCC has Bullet, Musket Ball only)
- Golovich 2019 ApJ 882 follow-up (dynamics analysis)
"""

from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "axiom_os" / "datasets" / "cache"
GOLOVICH_TABLE1 = CACHE / "golovich_2019_table1.dat"

# Name mapping: Golovich name -> our curated catalog id
GOLOVICH_TO_CURATED = {
    "CIZA J2242.8+5301": "CIZA_J2242",
    "ZwCl 2341+0000": "ZwCl_2341",
    "1RXS J0603.3+4212": "1RXSJ0603",  # Toothbrush
}

# Clusters in Golovich that we already have in MERGING_CLUSTER_CATALOG
GOLOVICH_IN_CURATED = {"CIZA J2242.8+5301", "ZwCl 2341+0000"}


def load_golovich_table1(path: Optional[Path] = None) -> List[Dict]:
    """
    Load Golovich 2019 table1. Returns list of {name, aname, z, ra, dec, band}.
    If path not given, uses cache. Returns [] if file missing.
    """
    p = path or GOLOVICH_TABLE1
    if not p.exists():
        return []

    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            name = line[0:19].strip()
            aname = line[20:29].strip() if len(line) > 29 else ""
            ra = line[30:40].strip() if len(line) > 40 else ""
            dec = line[41:50].strip() if len(line) > 50 else ""
            try:
                z = float(line[51:56]) if len(line) > 56 else None
            except ValueError:
                z = None
            band = line[57:64].strip() if len(line) > 64 else ""
            rows.append({"name": name, "aname": aname, "ra": ra, "dec": dec, "z": z, "band": band})
    return rows


def golovich_clusters_needing_params() -> List[str]:
    """
    Return Golovich cluster names that are NOT in our curated catalog.
    These need v, t, M, x_offset from papers.
    """
    from .merging_clusters import MERGING_CLUSTER_CATALOG

    curated_ids = {r["id"] for r in MERGING_CLUSTER_CATALOG}
    curated_names = {r["name"] for r in MERGING_CLUSTER_CATALOG}
    # Also match by common aliases
    alias_map = {
        "1E0657-56": "Bullet",
        "1RXS J0603.3+4212": "1RXSJ0603",
        "Abell 115": "A115",
        "Abell 521": "A521",
        "Abell 523": "A523",
        "Abell 746": "A746",
        "Abell 781": "A781",
        "Abell 1240": "A1240",
        "Abell 1300": "A1300",
        "Abell 1612": "A1612",
        "Abell 2034": "A2034",
        "Abell 2061": "A2061",
        "Abell 2163": "A2163",
        "Abell 2255": "A2255",
        "Abell 2345": "A2345",
        "Abell 2443": "A2443",
        "Abell 2744": "A2744",
        "Abell 3365": "A3365",
        "Abell 3411": "A3411",
        "CIZA J2242.8+5301": "CIZA_J2242",
        "MACS J1149.5+2223": "MACSJ1149",
        "MACS J1752.0+4440": "MACSJ1752",
        "PLCKESZ G287.0+32.9": "PLCKG287",
        "PSZ1 G108.18-11.53": "PSZ1G108",
        "RXC J1053.7+5452": "RXCJ1053",
        "RXC J1314.4-2515": "RXCJ1314",
        "ZwCl 0008.8+5215": "ZwCl0008",
        "ZwCl 1447+2619": "ZwCl1447",
        "ZwCl 1856.8+6616": "ZwCl1856",
        "ZwCl 2341+0000": "ZwCl_2341",
    }

    rows = load_golovich_table1()
    missing = []
    for r in rows:
        name = r["name"]
        aid = alias_map.get(name, name.replace(" ", "_").replace(".", ""))
        if aid not in curated_ids and name not in curated_names:
            # Check if any curated id matches
            matched = any(
                aid in c["id"] or name in c.get("id", "")
                for c in MERGING_CLUSTER_CATALOG
            )
            if not matched:
                missing.append(name)
    return missing
