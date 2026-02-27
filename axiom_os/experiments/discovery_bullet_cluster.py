"""
Bullet Cluster Discovery - Symbolic Regression on Gravitational vs Baryonic Offset

Clowe et al. 2006, ApJ 648, L109. Test: g_obs = f(rho_bar, x_offset) and
|x_offset| = f(v_collision, M_total, tau).

Phase 1 (static): g_obs = f(rho_bar, x_offset_local)
Phase 2 (MVP): Use published Bullet Cluster parameters, run discovery.

Run: python -m axiom_os.experiments.discovery_bullet_cluster
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.bullet_cluster import (
    load_bullet_cluster_mvp,
    load_bullet_cluster_published,
    save_bullet_csv,
    BULLET_X_OFFSET,
    BULLET_V_COLL,
    BULLET_M_TOTAL,
    BULLET_T_Myr,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bullet Cluster symbolic regression")
    parser.add_argument("--mass", type=str, default=None, help="Path to mass map FITS")
    parser.add_argument("--gas", type=str, default=None, help="Path to gas map FITS")
    parser.add_argument("--n", type=int, default=200, help="Synthetic grid size if no FITS")
    parser.add_argument("--no-normalize", action="store_true", help="Use raw values (no dimensionless ratios)")
    parser.add_argument("--no-causal", action="store_true", help="Include x_offset as input (circular)")
    args = parser.parse_args()

    from axiom_os.engine import DiscoveryEngine

    print("=" * 60)
    print("Bullet Cluster Discovery (1E 0657-56)")
    print("=" * 60)

    mass_path = Path(args.mass) if args.mass else None
    gas_path = Path(args.gas) if args.gas else None
    if mass_path is None and gas_path is None:
        cache_dir = ROOT / "axiom_os" / "datasets" / "cache"
        mass_path = cache_dir / "bullet_mass_test.fits" if (cache_dir / "bullet_mass_test.fits").exists() else None
        gas_path = cache_dir / "bullet_gas_test.fits" if (cache_dir / "bullet_gas_test.fits").exists() else None
    X, y, var_names, source, scale_dict = load_bullet_cluster_mvp(
        use_fits=(mass_path is not None and gas_path is not None),
        mass_path=mass_path,
        gas_path=gas_path,
        n_synthetic=args.n,
        normalize=not args.no_normalize,
        causal=not args.no_causal,
    )
    print(f"\n[1] Data: {len(y)} points (source: {source})")
    print(f"    Features: {var_names}")
    print(f"    Target: g_obs/g_mean (dimensionless)" if not args.no_normalize else "    Target: |g_obs|")
    if args.no_causal:
        print("    [Warning: x_offset in inputs - circular]")
    else:
        print("    [Causal: g = f(rho, v, t), no x_offset]")

    if len(y) < 10:
        print("\n[!] Insufficient data. Need FITS or n_synthetic >= 10.")
        return

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if not args.no_normalize:
        X_fit = X
        y_fit = y
    else:
        X_fit = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        y_fit = (y - y.mean()) / (y.std() + 1e-10)

    print(f"\n[2] Running symbolic regression...")
    engine = DiscoveryEngine(use_pysr=False)
    formula, pred, coefs = engine.discover_multivariate(
        X_fit, y_fit, var_names=var_names
    )

    if formula:
        print(f"    Formula: {formula}")
        mse = float(np.mean((pred - y_fit) ** 2))
        print(f"    MSE: {mse:.6f}")
    else:
        print("    [No formula found]")

    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "FITS" and mass_path and gas_path:
        from axiom_os.datasets.bullet_cluster import load_bullet_cluster_fits
        x, y_xy, rho, gx, gy, v, t = load_bullet_cluster_fits(mass_path, gas_path)
    else:
        x, y_xy, rho, gx, gy, v, t = load_bullet_cluster_published(n_spatial=args.n)
    csv_path = out_dir / "bullet_cluster_data.csv"
    save_bullet_csv(csv_path, x, y_xy, rho, gx, gy, v, t)
    print(f"\n[3] CSV saved: {csv_path}")

    import json
    from datetime import datetime
    out = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "n_points": len(y),
        "formula": formula,
        "var_names": var_names,
        "normalize": not args.no_normalize,
        "causal": not args.no_causal,
        "scale_dict": scale_dict if scale_dict else {},
        "bullet_params": {
            "x_offset_kpc": BULLET_X_OFFSET,
            "v_collision_km_s": BULLET_V_COLL,
            "M_total_Msun": BULLET_M_TOTAL,
            "t_since_collision_Myr": BULLET_T_Myr,
        },
    }
    with open(out_dir / "discovery_bullet_cluster.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"    Results saved: {out_dir / 'discovery_bullet_cluster.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
