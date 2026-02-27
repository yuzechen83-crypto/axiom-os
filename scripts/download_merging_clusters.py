"""
Download Merging Cluster Data for Axiom Discovery

Sources:
1. Golovich et al. 2019 (ApJS 240, 39) - 29 radio-relic merging clusters
   VizieR J/ApJS/240/39, table1: Name, z, RA, Dec, Band
   NOTE: table1 does NOT contain v_collision, t_since_collision, M_total.
   These must be supplemented from individual papers or MCMAC runs.

2. Merging Cluster Collaboration - MCMAC posterior samples
   http://www.mergingclustercollaboration.org/merger-mc-samples.html
   Bullet, Musket Ball: FITS with ~2M MC samples, 13 params including
   v_col, TSC0, TSC1, d_proj, M1, M2, etc.

Run: python scripts/download_merging_clusters.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "axiom_os" / "datasets" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

GOLOVICH_TABLE1_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/240/39/table1.dat"
MCC_BULLET_FITS_URL = "http://www.mergingclustercollaboration.org/uploads/1/1/7/3/11736016/bullet_mc_samples.fits"
MCC_MUSKETBALL_FITS_URL = "http://www.mergingclustercollaboration.org/uploads/1/1/7/3/11736016/musketball_mc_samples.fits"


# Fallback: table1 content if download fails (e.g. SSL on Windows)
GOLOVICH_TABLE1_FALLBACK = """1RXS J0603.3+4212   1RXSJ0603 06 03 13.4 +42 12 31 0.226 Radio
Abell 115           A115      00 55 59.5 +26 19 14 0.193 Optical
Abell 521           A521      04 54 08.6 -10 14 39 0.247 Optical
Abell 523           A523      04 59 01.0 +08 46 30 0.104 Optical
Abell 746           A746      09 09 37.0 +51 32 48 0.214 Optical
Abell 781           A781      09 20 23.2 +30 26 15 0.297 Optical
Abell 1240          A1240     11 23 31.9 +43 06 29 0.195 Optical
Abell 1300          A1300     11 32 00.7 -19 53 34 0.306 Optical
Abell 1612          A1612     12 47 43.2 -02 47 32 0.182 Optical
Abell 2034          A2034     15 10 10.8 +33 30 22 0.114 Optical
Abell 2061          A2061     15 21 20.6 +30 40 15 0.078 Optical
Abell 2163          A2163     16 15 34.1 -06 07 26 0.201 Optical
Abell 2255          A2255     17 12 50.0 +64 03 11 0.080 Optical
Abell 2345          A2345     21 27 09.8 -12 09 59 0.179 Optical
Abell 2443          A2443     22 26 02.6 +17 22 41 0.110 Optical
Abell 2744          A2744     00 14 18.9 -30 23 22 0.306 Optical
Abell 3365          A3365     05 48 12.0 -21 56 06 0.093 Optical
Abell 3411          A3411     08 41 54.7 -17 29 05 0.163 Optical
CIZA J2242.8+5301   CIZAJ2242 22 42 51.0 +53 01 24 0.189 X-ray
MACS J1149.5+2223   MACSJ1149 11 49 35.8 +22 23 55 0.544 X-ray
MACS J1752.0+4440   MACSJ1752 17 52 01.6 +44 40 46 0.365 X-ray
PLCKESZ G287.0+32.9 PLCKG287  11 50 49.2 -28 04 37 0.383 SZ
PSZ1 G108.18-11.53  PSZ1G108  23 22 29.7 +48 46 30 0.335 SZ
RXC J1053.7+5452    RXCJ1053  10 53 44.4 +54 52 21 0.072 X-ray
RXC J1314.4-2515    RXCJ1314  13 14 23.7 -25 15 21 0.247 X-ray
ZwCl 0008.8+5215    ZwCl0008  00 08 25.6 +52 31 41 0.104 Optical
ZwCl 1447+2619      ZwCl1447  14 49 28.2 +26 07 57 0.376 Optical
ZwCl 1856.8+6616    ZwCl1856  18 56 41.3 +66 21 56 0.304 Optical
ZwCl 2341+0000      ZwCl2341  23 43 39.7 +00 16 39 0.270 Optical
"""


def download_golovich_table1() -> Path:
    """Download Golovich 2019 table1 (29 clusters). Fallback to embedded if SSL fails."""
    import urllib.request

    out = CACHE / "golovich_2019_table1.dat"
    print(f"Downloading Golovich 2019 table1 -> {out}")
    try:
        req = urllib.request.Request(GOLOVICH_TABLE1_URL)
        with urllib.request.urlopen(req) as resp:
            with open(out, "wb") as f:
                f.write(resp.read())
    except Exception as e:
        print(f"  Download failed ({e}), using embedded fallback")
        with open(out, "w", encoding="utf-8") as f:
            f.write(GOLOVICH_TABLE1_FALLBACK)
    return out


def parse_golovich_table1(path: Path) -> list:
    """Parse table1.dat. Returns list of dicts: name, aname, ra, dec, z, band."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            # Fixed-width: Name 1-19, AName 21-29, RA 31-40, Dec 42-50, z 52-56, Band 58-64
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


def download_mcc_fits(name: str, url: str) -> Path:
    """Download MCC MCMAC FITS file."""
    import urllib.request

    out = CACHE / f"mcc_{name}_mc_samples.fits"
    print(f"Downloading MCC {name} FITS -> {out}")
    try:
        urllib.request.urlretrieve(url, out)
        return out
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def extract_mcmac_medians(fits_path: Path) -> dict:
    """
    Extract median v_col, TSC0, TSC1, d_proj, M1, M2 from MCMAC FITS.
    Returns dict with keys: v_col_km_s, t_since_Myr_t0, t_since_Myr_t1, d_proj_kpc, M1, M2.
    """
    import numpy as np
    try:
        from astropy.io import fits
    except ImportError:
        print("  astropy required for FITS")
        return {}

    if not fits_path or not fits_path.exists():
        return {}

    with fits.open(fits_path) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        cols = data.columns.names if hasattr(data, "columns") else []

        out = {}
        # MCMAC param names may vary; try common ones
        for c in cols:
            c_lower = c.lower()
            if "v_col" in c_lower or "vcol" in c_lower:
                out["v_col_km_s"] = float(np.median(data[c]))
            elif "tsc0" in c_lower or "t_0" in c_lower:
                val = np.median(data[c])
                out["t_since_Myr_t0"] = float(val)  # Gyr? Myr? Check units
            elif "tsc1" in c_lower or "t_1" in c_lower:
                val = np.median(data[c])
                out["t_since_Myr_t1"] = float(val)
            elif "d_proj" in c_lower or "dproj" in c_lower:
                out["d_proj_kpc"] = float(np.median(data[c]))
            elif c_lower == "m1" or "m_200_1" in c_lower:
                out["M1_Msun"] = float(np.median(data[c]))
            elif c_lower == "m2" or "m_200_2" in c_lower:
                out["M2_Msun"] = float(np.median(data[c]))

        return out


def main():
    print("=" * 60)
    print("Download Merging Cluster Data")
    print("=" * 60)

    # 1. Golovich 2019
    try:
        path = download_golovich_table1()
        rows = parse_golovich_table1(path)
        print(f"  Parsed {len(rows)} clusters from Golovich 2019")
        for r in rows[:5]:
            print(f"    {r['name']:25} z={r['z']}")
        print("    ...")
    except Exception as e:
        print(f"  Golovich download failed: {e}")

    # 2. MCC FITS (optional, large files ~112 MB each)
    print("\nMCC MCMAC FITS (optional, ~112 MB each):")
    print("  Run with --mcc to download Bullet and Musket Ball posteriors")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcc", action="store_true", help="Download MCC FITS files")
    args = parser.parse_args()

    if args.mcc:
        for name, url in [("bullet", MCC_BULLET_FITS_URL), ("musketball", MCC_MUSKETBALL_FITS_URL)]:
            p = download_mcc_fits(name, url)
            if p and p.exists():
                import numpy as np
                med = extract_mcmac_medians(p)
                if med:
                    print(f"  {name} medians: {med}")

    print("\nDone. Cache:", CACHE)
    print("=" * 60)


if __name__ == "__main__":
    main()
