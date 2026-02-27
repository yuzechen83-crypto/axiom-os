"""
Sensitivity Analysis: Exclude outlier systems and compare LOOCV.

Outliers (from validation): MusketBall, CIZA_J2242, MACS_J0018
- MusketBall: v=600 (slowest), t=700 (long), x/R=1.0
- CIZA_J2242: t=1000 (longest), extreme extrapolation
- MACS_J0018: t=30 (shortest), early merger

Run: python -m axiom_os.experiments.sensitivity_merging_clusters
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.merging_clusters import (
    load_merging_clusters_multi_system,
    load_merging_clusters_29_extended,
)

# Outlier subsets to test
OUTLIER_SETS = [
    [],
    ["MusketBall"],
    ["CIZA_J2242"],
    ["MACS_J0018"],
    ["MusketBall", "CIZA_J2242"],
    ["MusketBall", "MACS_J0018"],
    ["CIZA_J2242", "MACS_J0018"],
    ["MusketBall", "CIZA_J2242", "MACS_J0018"],
]


def _build_poly_features(X):
    from axiom_os.engine.forms import _build_multivariate_features
    F, _ = _build_multivariate_features(X, degree=2, include_interactions=True)
    return F


def _fit_poly_lasso(X_train, y_train):
    from sklearn.linear_model import Lasso
    F = _build_poly_features(X_train)
    model = Lasso(alpha=1e-3, max_iter=5000)
    model.fit(F, y_train)
    return model.coef_, model.intercept_


def _predict_poly(X, coef, intercept):
    F = _build_poly_features(X)
    return F @ coef + intercept


def leave_one_out_cv(X, y, names):
    n = len(y)
    pred = np.full(n, np.nan)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        coef, intercept = _fit_poly_lasso(X[train_idx], y[train_idx])
        if coef is not None:
            pred[i] = _predict_poly(X[i : i + 1], coef, intercept)[0]
    valid = ~np.isnan(pred)
    mse = float(np.mean((pred[valid] - y[valid]) ** 2))
    mae = float(np.mean(np.abs(pred[valid] - y[valid])))
    return mse, mae


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true", help="Use 29-collection")
    args = parser.parse_args()

    print("=" * 70)
    print("Sensitivity Analysis: Exclude Outliers")
    print("=" * 70)
    if args.extended:
        print("  [Extended] 29-collection (curated + filled)")

    results = []
    for exclude in OUTLIER_SETS:
        if args.extended:
            X, y, _, _, _, cat = load_merging_clusters_29_extended(
                use_collection=True, exclude_names=exclude if exclude else None
            )
        else:
            X, y, _, _, _, cat = load_merging_clusters_multi_system(
                normalize=True, exclude_names=exclude if exclude else None
            )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        names = [r["name"] for r in cat]

        if len(y) < 5:
            print(f"  Exclude {exclude}: n={len(y)} (skip, too few)")
            continue

        mse, mae = leave_one_out_cv(X, y, names)
        label = "all" if not exclude else ",".join(exclude)
        results.append({"exclude": label, "n": len(y), "loo_mse": mse, "loo_mae": mae})
        print(f"  Exclude [{label:35}] n={len(y)}  LOOCV MSE={mse:.4f}  MAE={mae:.4f}")

    print("\n" + "=" * 70)
    best = min(results, key=lambda r: r["loo_mse"])
    print(f"Best: exclude [{best['exclude']}] -> MSE={best['loo_mse']:.4f}")
    print("=" * 70)

    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    out_json = "sensitivity_merging_clusters_extended.json" if args.extended else "sensitivity_merging_clusters.json"
    with open(out_dir / out_json, "w", encoding="utf-8") as f:
        json.dump({"extended": args.extended, "outlier_sets": OUTLIER_SETS, "results": results}, f, indent=2)
    print(f"\nSaved: {out_dir / out_json}")


if __name__ == "__main__":
    main()
