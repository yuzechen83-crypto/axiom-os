"""
Multi-System Merging Cluster Discovery - x_offset = f(v, t, M)

Single Bullet data: v,t are constants → symbol regression finds only g ∝ ρ.
Multi-system: each cluster has different v_collision, t_since_collision → can discover τ_response.

Target: x_offset/R_cluster = f(v_collision, t_since_collision, M_total)

Run: python -m axiom_os.experiments.discovery_merging_clusters
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.merging_clusters import (
    load_merging_clusters_multi_system,
    load_merging_clusters_29_extended,
    save_merging_cluster_catalog_csv,
    get_catalog_table,
    MERGING_CLUSTER_CATALOG,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-system merging cluster symbolic regression"
    )
    parser.add_argument("--no-normalize", action="store_true", help="Use raw values")
    parser.add_argument("--extended", action="store_true", help="Use 29-collection (curated + filled from JSON)")
    args = parser.parse_args()

    from axiom_os.engine import DiscoveryEngine

    print("=" * 60)
    print("Multi-System Merging Cluster Discovery")
    print("Target: x_offset/R_cluster = f(v_collision, t_since_collision, M_total)")
    print("=" * 60)

    if args.extended:
        X, y, var_names, source, scale_dict, cat = load_merging_clusters_29_extended(use_collection=True)
        print("\n[0] Catalog (extended 29-collection):")
        for r in cat:
            xR = r["x_offset_kpc"] / r["R_cluster_kpc"]
            print(f"  {r['name']:14} v={r['v_collision_km_s']:6.0f} t={r['t_since_collision_Myr']:6.0f} x/R={xR:.3f}")
    else:
        print("\n[0] Catalog:")
        print(get_catalog_table())
        X, y, var_names, source, scale_dict, cat = load_merging_clusters_multi_system(
            normalize=not args.no_normalize,
        )
    print(f"\n[1] Data: {len(y)} systems (source: {source})")
    print(f"    Features: {var_names}")
    print(f"    Target: x_offset/R_cluster (dimensionless)")

    if len(y) < 4:
        print("\n[!] Need at least 4 systems for meaningful regression. Add more from literature.")
        # Still run, but warn

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
        print("    [No formula found - try PySR: use_pysr=True]")

    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / ("merging_cluster_catalog_extended.csv" if args.extended else "merging_cluster_catalog.csv")
    save_merging_cluster_catalog_csv(csv_path, catalog=cat)
    print(f"\n[3] Catalog CSV: {csv_path}")

    import json
    from datetime import datetime
    out = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "extended": args.extended,
        "n_systems": len(y),
        "formula": formula,
        "var_names": var_names,
        "target": "x_offset_norm",
        "scale_dict": scale_dict or {},
        "catalog": [
            {k: v for k, v in r.items() if k != "ref"}
            for r in cat
        ],
    }
    out_json = "discovery_merging_clusters_extended.json" if args.extended else "discovery_merging_clusters.json"
    with open(out_dir / out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"    Results: {out_dir / out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
