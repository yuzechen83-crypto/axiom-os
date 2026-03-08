"""
Multi-System Merging Cluster Catalog - for τ_response discovery

Single-system data (e.g. Bullet only) has v_collision and t_since_collision as constants
→ symbol regression cannot find dynamics. Need multiple systems with varying v, t.

Target: x_offset/R_cluster = f(v_collision, t_since_collision, M_total)

Sources:
- Bullet (1E 0657-56): Clowe 2006, Dawson 2013 ApJ 772
- Musket Ball (DLSCL J0916.2+2951): Dawson 2012 ApJ 747L, Dawson 2013
- MACS J0025.4-1222: Bradač 2008 ApJ 687
- Abell 520: Mahdavi 2007, Clowe 2012
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Merging cluster catalog: system-level parameters
# Columns: name, v_collision_km_s, t_since_collision_Myr, x_offset_kpc, M_total_Msun, R_cluster_kpc, ref
MERGING_CLUSTER_CATALOG = [
    # Bullet Cluster - Clowe 2006, Dawson 2013
    {
        "name": "Bullet",
        "id": "1E0657-56",
        "v_collision_km_s": 3000.0,
        "t_since_collision_Myr": 100.0,
        "x_offset_kpc": 720.0,
        "M_total_Msun": 1e15,
        "R_cluster_kpc": 800.0,
        "ref": "Clowe 2006, Dawson 2013 ApJ 772",
    },
    # Musket Ball - Dawson 2012, 2013; ~0.7 Gyr since collision, slower v
    {
        "name": "MusketBall",
        "id": "DLSCL_J0916.2+2951",
        "v_collision_km_s": 600.0,
        "t_since_collision_Myr": 700.0,
        "x_offset_kpc": 500.0,  # ~1 Mpc projected separation / 2
        "M_total_Msun": 5e14,
        "R_cluster_kpc": 500.0,
        "ref": "Dawson 2012 ApJ 747L, 2013 ApJ 772",
    },
    # MACS J0025 - Bradač 2008; equal-mass merger
    {
        "name": "MACS_J0025",
        "id": "MACS_J0025.4-1222",
        "v_collision_km_s": 2000.0,
        "t_since_collision_Myr": 150.0,
        "x_offset_kpc": 300.0,
        "M_total_Msun": 5e14,
        "R_cluster_kpc": 400.0,
        "ref": "Bradač 2008 ApJ 687",
    },
    # Abell 520 - Mahdavi 2007, Clowe 2012; dark core
    {
        "name": "Abell_520",
        "id": "A520",
        "v_collision_km_s": 2300.0,
        "t_since_collision_Myr": 200.0,
        "x_offset_kpc": 500.0,
        "M_total_Msun": 8e14,
        "R_cluster_kpc": 500.0,
        "ref": "Mahdavi 2007, Clowe 2012 ApJ 758",
    },
    # Abell 56 - binary merger, 535 kpc BCG separation, 60-271 Myr since pericenter
    {
        "name": "Abell_56",
        "id": "A56",
        "v_collision_km_s": 400.0,
        "t_since_collision_Myr": 165.0,
        "x_offset_kpc": 270.0,
        "M_total_Msun": 7e14,
        "R_cluster_kpc": 400.0,
        "ref": "Pearson 2022 AAS240, 2306.01715",
    },
    # Abell 1758N - NW/NE subclusters, ~270 Myr since pericenter
    {
        "name": "Abell_1758N",
        "id": "A1758N",
        "v_collision_km_s": 1800.0,
        "t_since_collision_Myr": 270.0,
        "x_offset_kpc": 400.0,
        "M_total_Msun": 6e14,
        "R_cluster_kpc": 500.0,
        "ref": "Boschin 2012 A&A 540, Ragozzine 2012",
    },
    # MACS J0018.5+1626 - 0-60 Myr post-pericenter, v~1700-3000 km/s
    {
        "name": "MACS_J0018",
        "id": "MACS_J0018.5+1626",
        "v_collision_km_s": 2300.0,
        "t_since_collision_Myr": 30.0,
        "x_offset_kpc": 200.0,
        "M_total_Msun": 9e14,
        "R_cluster_kpc": 450.0,
        "ref": "Sayers 2024 ApJ 968",
    },
    # ZwCl 2341.1+0000 - v~1900 km/s, radio relics
    {
        "name": "ZwCl_2341",
        "id": "ZwCl_2341.1+0000",
        "v_collision_km_s": 1900.0,
        "t_since_collision_Myr": 250.0,
        "x_offset_kpc": 450.0,
        "M_total_Msun": 9e14,
        "R_cluster_kpc": 600.0,
        "ref": "Botteon 2017 ApJ 841",
    },
    # CIZA J2242.8+5301 (Sausage) - radio relic, ~1 Gyr since merger
    {
        "name": "CIZA_J2242",
        "id": "CIZA_J2242.8+5301",
        "v_collision_km_s": 2500.0,
        "t_since_collision_Myr": 1000.0,
        "x_offset_kpc": 600.0,
        "M_total_Msun": 1e15,
        "R_cluster_kpc": 700.0,
        "ref": "van Weeren 2011 A&A, Stroe 2015",
    },
]


def load_merging_clusters_multi_system(
    catalog: Optional[List[Dict]] = None,
    normalize: bool = True,
    exclude_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, Optional[Dict[str, float]], List[Dict]]:
    """
    Load multi-system merging cluster data for x_offset = f(v, t, M) discovery.

    Each row = one merging cluster system.
    X: [v_collision, t_since_collision, M_total] (or normalized)
    y: x_offset / R_cluster (dimensionless offset)

    exclude_names: skip these systems (for sensitivity analysis).
    Returns (X, y, var_names, source, scale_dict, filtered_catalog).
    """
    cat = catalog or MERGING_CLUSTER_CATALOG
    if exclude_names:
        cat = [r for r in cat if r["name"] not in exclude_names]
    v = np.array([r["v_collision_km_s"] for r in cat])
    t = np.array([r["t_since_collision_Myr"] for r in cat])
    M = np.array([r["M_total_Msun"] for r in cat])
    x_off = np.array([r["x_offset_kpc"] for r in cat])
    R = np.array([r["R_cluster_kpc"] for r in cat])

    y = x_off / R  # target: dimensionless offset

    if normalize:
        v_ref = 3000.0  # km/s
        t_ref = 100.0   # Myr
        M_ref = 1e15    # M_sun
        X = np.column_stack([
            v / v_ref,
            t / t_ref,
            M / M_ref,
        ])
        var_names = ["v_collision_norm", "t_since_collision_norm", "M_total_norm"]
        scale_dict = {"v_ref": v_ref, "t_ref": t_ref, "M_ref": M_ref}
    else:
        X = np.column_stack([v, t, M])
        var_names = ["v_collision_km_s", "t_since_collision_Myr", "M_total_Msun"]
        scale_dict = None

    return X, y, var_names, "merging_cluster_catalog", scale_dict, cat


def save_merging_cluster_catalog_csv(
    out_path: Path, catalog: Optional[List[Dict]] = None
) -> None:
    """Save catalog to CSV for inspection and external use."""
    import csv
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cat = catalog or MERGING_CLUSTER_CATALOG
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "v_collision_km_s", "t_since_collision_Myr",
            "x_offset_kpc", "M_total_Msun", "R_cluster_kpc", "x_offset_norm",
        ])
        for r in cat:
            w.writerow([
                r["name"],
                r["v_collision_km_s"],
                r["t_since_collision_Myr"],
                r["x_offset_kpc"],
                r["M_total_Msun"],
                r["R_cluster_kpc"],
                r["x_offset_kpc"] / r["R_cluster_kpc"],
            ])


COLLECTION_29_PATH = ROOT / "axiom_os" / "datasets" / "merging_clusters_29_collection.json"


def load_merging_clusters_29_extended(
    use_collection: bool = True,
    exclude_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, Optional[Dict[str, float]], List[Dict]]:
    """
    Load merged catalog: curated 9 + filled systems from 29-collection JSON.
    When use_collection=True, merges MERGING_CLUSTER_CATALOG with filled rows
    from merging_clusters_29_collection.json (avoids duplicates by name).
    """
    cat = list(MERGING_CLUSTER_CATALOG)
    seen = {r["name"] for r in cat}

    if use_collection and COLLECTION_29_PATH.exists():
        import json
        with open(COLLECTION_29_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("systems", []):
            if s.get("status") != "filled" or s["name"] in seen:
                continue
            v, t, M = s.get("v_collision_km_s"), s.get("t_since_collision_Myr"), s.get("M_total_Msun")
            x, R = s.get("x_offset_kpc"), s.get("R_cluster_kpc")
            if v is None or t is None or M is None or x is None or R is None:
                continue
            cat.append({
                "name": s["name"],
                "id": s.get("golovich_name", s["name"]),
                "v_collision_km_s": float(v),
                "t_since_collision_Myr": float(t),
                "x_offset_kpc": float(x),
                "M_total_Msun": float(M),
                "R_cluster_kpc": float(R),
                "ref": s.get("ref", ""),
            })
            seen.add(s["name"])
    return load_merging_clusters_multi_system(
        catalog=cat, normalize=True, exclude_names=exclude_names
    )


def get_catalog_table() -> str:
    """Return a formatted table of the catalog for display."""
    lines = [
        "| System        | v (km/s) | t (Myr) | x_offset (kpc) | M (10^14 M_sun) | x/R |",
        "|---------------|----------|---------|----------------|-----------------|-----|",
    ]
    for r in MERGING_CLUSTER_CATALOG:
        M14 = r["M_total_Msun"] / 1e14
        xR = r["x_offset_kpc"] / r["R_cluster_kpc"]
        lines.append(
            f"| {r['name']:13} | {r['v_collision_km_s']:7.0f} | {r['t_since_collision_Myr']:7.0f} | "
            f"{r['x_offset_kpc']:14.0f} | {M14:15.1f} | {xR:.3f} |"
        )
    return "\n".join(lines)
