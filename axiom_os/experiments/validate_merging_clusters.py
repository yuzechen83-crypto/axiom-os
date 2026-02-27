"""
Merging Cluster Validation: Leave-One-Out CV + Physical Form

1. Leave-one-out: fit on 8 systems, predict the 9th. Cycle 9 times.
2. Physical form: x/R = α·(v·t/R)/(1+β·M/M_ref) + γ
   Compare MSE with 6-parameter polynomial.

Run: python -m axiom_os.experiments.validate_merging_clusters
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.merging_clusters import (
    MERGING_CLUSTER_CATALOG,
    load_merging_clusters_multi_system,
    load_merging_clusters_29_extended,
)

# Reference scales
V_REF = 3000.0   # km/s
T_REF = 100.0    # Myr
M_REF = 1e15     # M_sun
R_REF = 500.0    # kpc (for physical form)


def _build_poly_features(X: np.ndarray) -> np.ndarray:
    """Same as MultivariatePolyForm: linear, quadratic, interactions."""
    from axiom_os.engine.forms import _build_multivariate_features
    F, _ = _build_multivariate_features(X, degree=2, include_interactions=True)
    return F


def _fit_poly_lasso(X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray = None) -> tuple:
    """Fit Lasso on polynomial features. Returns (coef, intercept)."""
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        return None, None
    F = _build_poly_features(X_train)
    model = Lasso(alpha=1e-3, max_iter=5000)
    model.fit(F, y_train, sample_weight=sample_weight)
    return model.coef_, model.intercept_


def _predict_poly(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Predict using polynomial features and fitted coef/intercept."""
    F = _build_poly_features(X)
    return F @ coef + intercept


def leave_one_out_cv(X: np.ndarray, y: np.ndarray, names: list, sample_weight: np.ndarray = None) -> dict:
    """
    Leave-one-out cross-validation.
    Returns dict with pred, actual, mae, mse, per-system table.
    """
    n = len(y)
    pred_holdout = np.full(n, np.nan)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train = X[train_idx]
        y_train = y[train_idx]
        sw = sample_weight[train_idx] if sample_weight is not None else None

        coef, intercept = _fit_poly_lasso(X_train, y_train, sample_weight=sw)
        if coef is None:
            continue
        p = _predict_poly(X[i : i + 1], coef, intercept)
        pred_holdout[i] = float(p[0])


    valid = ~np.isnan(pred_holdout)
    mse = float(np.mean((pred_holdout[valid] - y[valid]) ** 2))
    mae = float(np.mean(np.abs(pred_holdout[valid] - y[valid])))

    rows = []
    for i in range(n):
        rows.append({
            "system": names[i],
            "actual": float(y[i]),
            "predicted": float(pred_holdout[i]) if valid[i] else np.nan,
            "error": float(pred_holdout[i] - y[i]) if valid[i] else np.nan,
        })

    return {
        "pred_holdout": pred_holdout,
        "actual": y,
        "mse": mse,
        "mae": mae,
        "rows": rows,
    }


def fit_physical_form(X_raw: np.ndarray, y: np.ndarray, force_origin: bool = False) -> tuple:
    """
    Fit x/R = α·(v·t/R)/(1+β·M/M_ref) + γ.
    If force_origin: γ=0, x/R = α·(v·t/R) (through origin; t=0 or v=0 => x/R=0).
    X_raw: (n, 4) = [v_km_s, t_Myr, M_Msun, R_kpc]
    """
    from scipy.optimize import minimize

    def model_full(params, X):
        alpha, beta, gamma = params
        v, t, M, R = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        term = (v * t / R) / (1 + beta * M / M_REF)
        return alpha * term + gamma

    def model_origin(params, X):
        alpha = params[0]
        v, t, M, R = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        return alpha * (v * t / R)

    if force_origin:
        def loss(p):
            return np.sum((model_origin(p, X_raw) - y) ** 2)
        res = minimize(loss, x0=[0.001], method="L-BFGS-B", bounds=[(1e-10, 0.1)])
        alpha = float(res.x[0])
        beta, gamma = 0.0, 0.0
        pred = model_origin(res.x, X_raw)
    else:
        def loss(p):
            return np.sum((model_full(p, X_raw) - y) ** 2)
        res = minimize(loss, x0=[0.002, 0.5, 0.4], method="L-BFGS-B",
                      bounds=[(1e-8, 0.1), (0, 10), (-1, 2)])
        alpha, beta, gamma = res.x
        pred = model_full(res.x, X_raw)
    mse = float(np.mean((pred - y) ** 2))
    return alpha, beta, gamma, pred, mse


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", nargs="+", default=[], help="Exclude systems (e.g. MusketBall CIZA_J2242)")
    parser.add_argument("--weighted", action="store_true", help="Use inverse-variance weights (placeholder sigma)")
    parser.add_argument("--force-origin", action="store_true", help="Physical form: x/R = alpha*(v*t/R), no intercept")
    parser.add_argument("--extended", action="store_true", help="Use 29-collection (curated + filled from JSON)")
    args = parser.parse_args()

    print("=" * 70)
    print("Merging Cluster Validation: Leave-One-Out + Physical Form")
    print("=" * 70)

    if args.extended:
        X, y, var_names, _, scale_dict, cat = load_merging_clusters_29_extended(
            use_collection=True, exclude_names=args.exclude if args.exclude else None
        )
        print(f"\n  [Extended] {len(cat)} systems from 29-collection")
    else:
        X, y, var_names, _, scale_dict, cat = load_merging_clusters_multi_system(
            normalize=True, exclude_names=args.exclude if args.exclude else None
        )
    names = [r["name"] for r in cat]
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    sample_weight = None
    if args.weighted:
        # Placeholder: sigma_x_offset ~ 0.1 * x_offset for each system
        sigma = np.array([r.get("sigma_x_offset_frac", 0.1) * (r["x_offset_kpc"] / r["R_cluster_kpc"]) for r in cat])
        sigma = np.maximum(sigma, 0.05)
        sample_weight = 1.0 / (sigma ** 2)
        print(f"  [Weighted] sigma_frac=0.1")

    # --- 1. Full-data polynomial (baseline) ---
    from axiom_os.engine import DiscoveryEngine
    engine = DiscoveryEngine(use_pysr=False)
    formula_full, pred_full, _ = engine.discover_multivariate(
        X, y, var_names=var_names, sample_weight=sample_weight
    )
    mse_full = float(np.mean((pred_full - y) ** 2))
    print(f"\n[1] Full-data polynomial (6-param):")
    print(f"    Formula: {formula_full}")
    print(f"    MSE (in-sample): {mse_full:.6f}")

    # --- 2. Leave-one-out CV ---
    print(f"\n[2] Leave-one-out cross-validation:")
    loo = leave_one_out_cv(X, y, names, sample_weight=sample_weight)
    print(f"    MSE (holdout): {loo['mse']:.6f}")
    print(f"    MAE (holdout): {loo['mae']:.6f}")
    print(f"\n    Per-system: predicted vs actual")
    print(f"    {'System':<14} {'Actual':>8} {'Predicted':>10} {'Error':>10}")
    print("    " + "-" * 46)
    for r in loo["rows"]:
        pred_str = f"{r['predicted']:.4f}" if not np.isnan(r['predicted']) else "N/A"
        err_str = f"{r['error']:+.4f}" if not np.isnan(r.get('error', np.nan)) else "N/A"
        print(f"    {r['system']:<14} {r['actual']:>8.4f} {pred_str:>10} {err_str:>10}")

    # --- 3. Physical form ---
    v = np.array([r["v_collision_km_s"] for r in cat])
    t = np.array([r["t_since_collision_Myr"] for r in cat])
    M = np.array([r["M_total_Msun"] for r in cat])
    R = np.array([r["R_cluster_kpc"] for r in cat])
    X_raw = np.column_stack([v, t, M, R])

    alpha, beta, gamma, pred_phys, mse_phys = fit_physical_form(X_raw, y, force_origin=args.force_origin)
    if args.force_origin:
        print(f"\n[3] Physical form (force origin): x/R = alpha*(v*t/R)")
    else:
        print(f"\n[3] Physical form: x/R = alpha*(v*t/R)/(1+beta*M/M_ref) + gamma")
    if args.force_origin:
        print(f"    alpha = {alpha:.4g}")
        print(f"    MSE (in-sample): {mse_phys:.6f}")
        print(f"    Formula: x/R = {alpha:.4g}*(v*t/R)")
    else:
        print(f"    alpha = {alpha:.4g}, beta = {beta:.4g}, gamma = {gamma:.4g}")
        print(f"    MSE (in-sample): {mse_phys:.6f}")
        print(f"    Formula: x/R ~ {alpha:.4g}*(v*t/R)/(1+{beta:.4g}*M/M_ref) + {gamma:.4g}")

    # --- 4. Physical form LOOCV ---
    pred_phys_loo = np.full(len(y), np.nan)
    for i in range(len(y)):
        train_idx = [j for j in range(len(y)) if j != i]
        X_raw_train = X_raw[train_idx]
        y_train = y[train_idx]
        a, b, g, _, _ = fit_physical_form(X_raw_train, y_train, force_origin=args.force_origin)
        v, t, M, R = X_raw[i, 0], X_raw[i, 1], X_raw[i, 2], X_raw[i, 3]
        if args.force_origin:
            pred_phys_loo[i] = a * (v * t / R)
        else:
            term = (v * t / R) / (1 + b * M / M_REF)
            pred_phys_loo[i] = a * term + g
    mse_phys_loo = float(np.mean((pred_phys_loo - y) ** 2))
    mae_phys_loo = float(np.mean(np.abs(pred_phys_loo - y)))

    # --- 5. Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Full polynomial MSE:      {mse_full:.6f} (in-sample)")
    print(f"  Poly LOOCV holdout MSE:   {loo['mse']:.6f}  <-- prediction test")
    print(f"  Physical form MSE:       {mse_phys:.6f} (in-sample)")
    print(f"  Physical form LOOCV MSE: {mse_phys_loo:.6f}")
    print(f"\n  Poly LOOCV / Phys LOOCV = {loo['mse']/mse_phys_loo:.2f}x")
    if loo["mse"] > mse_phys_loo * 1.5:
        print("  -> Polynomial overfits; physical form generalizes better.")
    else:
        print("  -> Both similar; need more data to distinguish.")
    print("=" * 70)

    # Save results
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    out = {
        "loo_mse": loo["mse"],
        "loo_mae": loo["mae"],
        "loo_rows": loo["rows"],
        "full_mse": mse_full,
        "physical_mse": mse_phys,
        "physical_loo_mse": mse_phys_loo,
        "physical_loo_mae": mae_phys_loo,
        "physical_params": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "formula_full": formula_full,
        "exclude": args.exclude,
        "extended": args.extended,
        "weighted": args.weighted,
        "force_origin": args.force_origin,
    }
    out_json = "validate_merging_clusters_extended.json" if args.extended else "validate_merging_clusters.json"
    with open(out_dir / out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_dir / out_json}")


if __name__ == "__main__":
    main()
