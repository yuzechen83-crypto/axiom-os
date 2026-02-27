"""
Extract MCMAC posterior medians from MCC FITS for Bullet and Musket Ball.

MCC FITS: ~2M samples, 13 params. Units from Dawson 2013:
- v_col: km/s (collision velocity at pericenter)
- TSC0, TSC1: Gyr (time since collision, outgoing/return)
- d_proj: kpc (projected separation)
- M1, M2: M_sun (M_200)

Run after download: python scripts/download_merging_clusters.py --mcc
Then: python scripts/extract_mcc_mcmac_medians.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "axiom_os" / "datasets" / "cache"


def list_fits_columns(fits_path: Path) -> list:
    """List column names and sample stats from MCMAC FITS."""
    try:
        from astropy.io import fits
    except ImportError:
        return []

    if not fits_path.exists():
        return []

    with fits.open(fits_path) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        cols = list(data.columns.names) if hasattr(data, "columns") else []
        return cols


def extract_mcmac_medians(fits_path: Path) -> dict:
    """
    Extract median v_col, TSC0, TSC1, d_proj, M1, M2 from MCMAC FITS.
    TSC in Gyr -> convert to Myr (*1000).
    Returns dict with: v_col_km_s, t_since_Myr_t0, t_since_Myr_t1, d_proj_kpc, M1_Msun, M2_Msun.
    """
    try:
        from astropy.io import fits
    except ImportError:
        return {}

    if not fits_path or not fits_path.exists():
        return {}

    with fits.open(fits_path) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        cols = {c.lower(): c for c in (data.columns.names if hasattr(data, "columns") else [])}

        out = {}
        # v_3d_col: km/s (MCMAC output)
        for k in ["v_3d_col", "v_col", "vcol", "v3d_tcol"]:
            if k in cols:
                out["v_col_km_s"] = float(np.median(data[cols[k]]))
                break
        # TSM_0, TSM_1: Gyr -> Myr (MCMAC: Time Since Merger)
        for k in ["tsm_0", "tsc0", "t_0"]:
            if k in cols:
                out["t_since_Myr_t0"] = float(np.median(data[cols[k]])) * 1000.0
                break
        for k in ["tsm_1", "tsc1", "t_1"]:
            if k in cols:
                out["t_since_Myr_t1"] = float(np.median(data[cols[k]])) * 1000.0
                break
        # d_proj: Mpc (MCMAC) -> kpc
        for k in ["d_proj", "dproj", "d_projected"]:
            if k in cols:
                val = float(np.median(data[cols[k]]))
                out["d_proj_kpc"] = val * 1000.0 if val < 10 else val  # Mpc if <10
                break
        # M1, M2: M_sun
        for k in ["m_1", "m1", "m_200_1"]:
            if k in cols:
                out["M1_Msun"] = float(np.median(data[cols[k]]))
                break
        for k in ["m_2", "m2", "m_200_2"]:
            if k in cols:
                out["M2_Msun"] = float(np.median(data[cols[k]]))
                break

        return out


def main():
    print("=" * 60)
    print("MCC MCMAC Median Extraction")
    print("=" * 60)

    for name, fname in [("Bullet", "mcc_bullet_mc_samples.fits"), ("Musket Ball", "mcc_musketball_mc_samples.fits")]:
        path = CACHE / fname
        print(f"\n{name}: {path}")
        if not path.exists():
            print("  [Not found] Run: python scripts/download_merging_clusters.py --mcc")
            continue

        cols = list_fits_columns(path)
        print(f"  Columns ({len(cols)}): {cols[:8]}..." if len(cols) > 8 else f"  Columns: {cols}")

        med = extract_mcmac_medians(path)
        if med:
            print("  Medians:")
            for k, v in med.items():
                if "Myr" in k or "kpc" in k:
                    print(f"    {k}: {v:.1f}")
                else:
                    print(f"    {k}: {v:.3g}")
        else:
            print("  [No params extracted]")

    print("\n" + "=" * 60)
    print("Suggested catalog update (merge with curated):")
    print("  Bullet: use v_col_km_s, t_since_Myr_t0, d_proj_kpc/2 or x_offset from Clowe")
    print("  Musket Ball: use v_col_km_s, t_since_Myr_t0, d_proj_kpc/2")
    print("=" * 60)


if __name__ == "__main__":
    main()
