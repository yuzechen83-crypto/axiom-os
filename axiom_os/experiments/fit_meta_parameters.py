"""
Fit Meta-Axis Coupling Parameters (ξ, λ) from SPARC

Theory: L-field action has U(L) = (λ/4)(L²-L₀²)² and non-minimal coupling ξRL².
The correction strength in ν(g) = 1 + α·(ν_McGaugh - 1) is encoded by α,
which depends on ξ and λ. Fitting α from SPARC constrains the theory.

Run: python -m axiom_os.experiments.fit_meta_parameters
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import (
    load_sparc_rar,
    load_sparc_rar_real_only,
    load_sparc_rar_mrt,
    load_sparc_rar_real_only_per_galaxy,
)
from axiom_os.layers.meta_kernel import (
    KM_S_SQ_PER_KPC_TO_MS2,
    a0_si_from_g0,
    nu_mcgaugh_parametrized,
)

# RAR model: g_obs = g_bar * nu(g_bar), nu = 1 + α*(1/(1-exp(-√(g_bar/g0))) - 1)
def _rar_model_parametrized(g_bar: np.ndarray, g0: float, alpha: float) -> np.ndarray:
    """g_obs = g_bar * nu(g_bar; g0, α). g0 in (km/s)²/kpc."""
    nu = nu_mcgaugh_parametrized(g_bar, g0, alpha=alpha)
    return g_bar * nu


def fit_g0_only(g_bar: np.ndarray, g_obs: np.ndarray, g0_bounds: Tuple[float, float] = (100.0, 50000.0)) -> Dict:
    """Fit g0 with alpha=1 fixed (standard McGaugh form, linear MSE)."""
    eps = 1e-14
    x = np.maximum(g_bar, eps)
    y = np.asarray(g_obs, dtype=np.float64)

    def loss(g0_val: float) -> float:
        g0_val = np.clip(g0_val, g0_bounds[0], g0_bounds[1])
        pred = _rar_model_parametrized(x, g0_val, 1.0)
        return np.mean((pred - y) ** 2)

    from scipy.optimize import minimize
    res = minimize(loss, [3700.0], method="L-BFGS-B", bounds=[g0_bounds], options={"maxiter": 500})
    g0_fit = float(np.clip(res.x[0], g0_bounds[0], g0_bounds[1]))
    return {"g0": g0_fit, "a0_si": float(a0_si_from_g0(g0_fit)), "n_points": len(g_bar)}


def fit_g0_log_loss(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0_bounds: Tuple[float, float] = (100.0, 50000.0),
) -> Dict:
    """Fit g0 with alpha=1, but minimize MSE in log-space (like McGaugh binned fit)."""
    eps = 1e-14
    x = np.maximum(g_bar, eps)
    y = np.asarray(g_obs, dtype=np.float64)
    log_y = np.log10(y + eps)

    def loss_log(g0_val: float) -> float:
        g0_val = np.clip(g0_val, g0_bounds[0], g0_bounds[1])
        pred = _rar_model_parametrized(x, g0_val, 1.0)
        log_pred = np.log10(pred + eps)
        return np.mean((log_pred - log_y) ** 2)

    from scipy.optimize import minimize
    res = minimize(loss_log, [3700.0], method="L-BFGS-B", bounds=[g0_bounds], options={"maxiter": 500})
    g0_fit = float(np.clip(res.x[0], g0_bounds[0], g0_bounds[1]))
    return {"g0": g0_fit, "a0_si": float(a0_si_from_g0(g0_fit)), "n_points": len(g_bar)}


def fit_g0_alpha(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0_bounds: Tuple[float, float] = (100.0, 50000.0),
    alpha_bounds: Tuple[float, float] = (0.1, 10.0),
    two_step: bool = True,
) -> Dict:
    """
    Joint fit of (g0, α) to SPARC RAR.
    Returns dict with g0, alpha, a0_si, rmse, r2, cov (if available).
    """
    eps = 1e-14
    x = np.maximum(g_bar, eps)
    y = np.asarray(g_obs, dtype=np.float64)

    def loss(params: np.ndarray) -> float:
        g0, alpha = params[0], params[1]
        g0 = np.clip(g0, g0_bounds[0], g0_bounds[1])
        alpha = np.clip(alpha, alpha_bounds[0], alpha_bounds[1])
        pred = _rar_model_parametrized(x, g0, alpha)
        return np.mean((pred - y) ** 2)

    from scipy.optimize import minimize

    if two_step:
        # Step 1: fit g0 with alpha=1 (standard McGaugh)
        def loss_g0(g0_val: float) -> float:
            g0_val = np.clip(g0_val, g0_bounds[0], g0_bounds[1])
            pred = _rar_model_parametrized(x, g0_val, 1.0)
            return np.mean((pred - y) ** 2)
        res_g0 = minimize(loss_g0, [3700.0], method="L-BFGS-B", bounds=[g0_bounds], options={"maxiter": 500})
        g0_fit = float(np.clip(res_g0.x[0], g0_bounds[0], g0_bounds[1]))
        # Step 2: fit alpha with g0 fixed
        def loss_alpha(alpha_val: float) -> float:
            alpha_val = np.clip(alpha_val, alpha_bounds[0], alpha_bounds[1])
            pred = _rar_model_parametrized(x, g0_fit, alpha_val)
            return np.mean((pred - y) ** 2)
        res_alpha = minimize(loss_alpha, [1.0], method="L-BFGS-B", bounds=[alpha_bounds], options={"maxiter": 500})
        alpha_fit = float(np.clip(res_alpha.x[0], alpha_bounds[0], alpha_bounds[1]))
    else:
        # Joint fit
        res = minimize(
            loss,
            x0=[3700.0, 1.0],
            method="L-BFGS-B",
            bounds=[g0_bounds, alpha_bounds],
            options={"maxiter": 1000},
        )
        g0_fit = float(np.clip(res.x[0], g0_bounds[0], g0_bounds[1]))
        alpha_fit = float(np.clip(res.x[1], alpha_bounds[0], alpha_bounds[1]))

    pred = _rar_model_parametrized(x, g0_fit, alpha_fit)
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    ss_res = np.sum((pred - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + eps
    r2 = 1.0 - ss_res / ss_tot

    # Log-space R² (RAR is often plotted in log)
    log_y = np.log10(y + 1e-14)
    log_pred = np.log10(pred + 1e-14)
    ss_res_log = np.sum((log_pred - log_y) ** 2)
    ss_tot_log = np.sum((log_y - np.mean(log_y)) ** 2) + eps
    r2_log = 1.0 - ss_res_log / ss_tot_log

    # Compare to α=1 (standard McGaugh) for theory check
    pred_alpha1 = _rar_model_parametrized(x, g0_fit, 1.0)
    rmse_alpha1 = np.sqrt(np.mean((pred_alpha1 - y) ** 2))

    return {
        "g0": float(g0_fit),
        "alpha": float(alpha_fit),
        "a0_si": float(a0_si_from_g0(g0_fit)),
        "rmse": float(rmse),
        "r2": float(r2),
        "r2_log": float(r2_log),
        "rmse_alpha1": float(rmse_alpha1),
        "alpha_prefers_1": bool(rmse < rmse_alpha1 * 1.01),
        "n_points": int(len(g_bar)),
    }


def bootstrap_confidence_log(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
    subsample_frac: float = 0.5,
) -> Dict:
    """
    Bootstrap 1σ CI for a0 using log MSE fit (fit_g0_log_loss).
    Matches McGaugh binned-fit methodology.
    subsample_frac: fraction of data per sample (0.5 = 50%) for non-degenerate CI.
    """
    rng = np.random.default_rng(seed)
    n = len(g_bar)
    n_sub = max(200, int(n * subsample_frac))
    g0_list, a0_list = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n_sub)  # resample with replacement
        gb, go = g_bar[idx], g_obs[idx]
        try:
            res = fit_g0_log_loss(gb, go)
            g0_list.append(res["g0"])
            a0_list.append(res["a0_si"])
        except Exception:
            continue
    g0_arr = np.array(g0_list)
    a0_arr = np.array(a0_list)
    return {
        "g0": {
            "mean": float(np.mean(g0_arr)),
            "std": float(np.std(g0_arr)),
            "ci68_lo": float(np.percentile(g0_arr, 16)),
            "ci68_hi": float(np.percentile(g0_arr, 84)),
            "ci95_lo": float(np.percentile(g0_arr, 2.5)),
            "ci95_hi": float(np.percentile(g0_arr, 97.5)),
        },
        "a0_si": {
            "mean": float(np.mean(a0_arr)),
            "std": float(np.std(a0_arr)),
            "ci68_lo": float(np.percentile(a0_arr, 16)),
            "ci68_hi": float(np.percentile(a0_arr, 84)),
            "ci95_lo": float(np.percentile(a0_arr, 2.5)),
            "ci95_hi": float(np.percentile(a0_arr, 97.5)),
        },
        "n_bootstrap": len(g0_list),
        "method": "log_mse",
    }


def bootstrap_confidence(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap confidence intervals for g0, alpha, a0_si.
    Returns 1-sigma (16th, 84th percentile) and 2-sigma (2.5th, 97.5th).
    """
    rng = np.random.default_rng(seed)
    n = len(g_bar)
    g0_list, alpha_list, a0_list = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gb, go = g_bar[idx], g_obs[idx]
        try:
            res = fit_g0_alpha(gb, go)
            g0_list.append(res["g0"])
            alpha_list.append(res["alpha"])
            a0_list.append(res["a0_si"])
        except Exception:
            continue
    g0_arr = np.array(g0_list)
    alpha_arr = np.array(alpha_list)
    a0_arr = np.array(a0_list)
    return {
        "g0": {
            "mean": float(np.mean(g0_arr)),
            "std": float(np.std(g0_arr)),
            "ci68_lo": float(np.percentile(g0_arr, 16)),
            "ci68_hi": float(np.percentile(g0_arr, 84)),
            "ci95_lo": float(np.percentile(g0_arr, 2.5)),
            "ci95_hi": float(np.percentile(g0_arr, 97.5)),
        },
        "alpha": {
            "mean": float(np.mean(alpha_arr)),
            "std": float(np.std(alpha_arr)),
            "ci68_lo": float(np.percentile(alpha_arr, 16)),
            "ci68_hi": float(np.percentile(alpha_arr, 84)),
        },
        "a0_si": {
            "mean": float(np.mean(a0_arr)),
            "std": float(np.std(a0_arr)),
            "ci68_lo": float(np.percentile(a0_arr, 16)),
            "ci68_hi": float(np.percentile(a0_arr, 84)),
            "ci95_lo": float(np.percentile(a0_arr, 2.5)),
            "ci95_hi": float(np.percentile(a0_arr, 97.5)),
        },
        "n_bootstrap": len(g0_list),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fit Meta-Axis (xi,lambda) from SPARC")
    parser.add_argument("--n-galaxies", type=int, default=50, help="Number of SPARC galaxies")
    parser.add_argument("--real-only", action="store_true", default=True,
                        help="Use REAL SPARC data only (no mock). Default: True.")
    parser.add_argument("--allow-mock", action="store_true",
                        help="Allow mock fallback if real data insufficient")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Bootstrap samples for confidence intervals (0=skip)")
    parser.add_argument("--log-mse", action="store_true",
                        help="Use log MSE fit for bootstrap (McGaugh-style, compare a0 to [1.1,1.3]e-10)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run diagnostic: g_bar segments, alpha=1 fit, log-space fit")
    args = parser.parse_args()

    print("=" * 60)
    print("Fit Meta-Axis Coupling (xi,lambda) from SPARC")
    print("=" * 60)
    print("\nModel: nu(g) = 1 + alpha*(nu_McGaugh(g) - 1)")
    print("  alpha encodes correction strength from L-field action (xi, lambda).")
    print("  Theory predicts alpha ~ 1 if derivation is correct.")

    min_points = 50
    g_bar, g_obs, names = np.array([]), np.array([]), []

    if args.real_only and not args.allow_mock:
        # Prefer RAR.mrt (canonical McGaugh+2016, correct M/L) over Rotmod
        g_bar, g_obs = load_sparc_rar_mrt()
        if len(g_bar) >= min_points:
            data_source = "SPARC RAR.mrt (McGaugh+2016 canonical)"
            names = [f"RAR_{i}" for i in range(len(g_bar))]
        else:
            for attempt in range(1, 4):
                g_bar, g_obs, names = load_sparc_rar_real_only(n_galaxies=args.n_galaxies)
                if len(g_bar) >= min_points:
                    break
                print(f"\n[!] Attempt {attempt}/3: Got {len(g_bar)} points from {len(names)} galaxies.")
                if attempt < 3:
                    print("    Retrying SPARC download...")
            data_source = "SPARC REAL (Rotmod_LTG)"
    else:
        g_bar, g_obs, names = load_sparc_rar(
            n_galaxies=args.n_galaxies,
            use_mock_if_fail=args.allow_mock,
            use_real=True,
        )
        data_source = "SPARC (mock fallback)" if args.allow_mock else "SPARC REAL"

    if len(g_bar) < min_points:
        print(f"\n[!] FAILED: Insufficient real data. Got {len(g_bar)} points, need >= {min_points}.")
        print("    SPARC requires network to download Rotmod_LTG.zip from Zenodo/CWRU.")
        return

    print(f"\n[1] Data: {len(g_bar)} points from {len(names)} galaxies")
    print(f"    Source: {data_source}")

    result = fit_g0_alpha(g_bar, g_obs)

    # Diagnostic: segment by g_bar, alpha=1 fit, log-space fit
    if args.diagnose:
        print("\n[DIAGNOSE] Bias attribution tests")
        # 1. Alpha=1 fixed (no alpha parameter)
        res_a1 = fit_g0_only(g_bar, g_obs)
        print(f"  (1) alpha=1 fixed (linear MSE): a0 = {res_a1['a0_si']:.4e} m/s^2")
        # 2. Log-space loss
        res_log = fit_g0_log_loss(g_bar, g_obs)
        print(f"  (2) alpha=1 fixed (log MSE):    a0 = {res_log['a0_si']:.4e} m/s^2")
        # 3. Segment by g_bar (terciles)
        lg_bar = np.log10(g_bar + 1e-20)
        p33, p67 = np.percentile(lg_bar, [33.3, 66.7])
        mask_lo = lg_bar <= p33
        mask_mid = (lg_bar > p33) & (lg_bar <= p67)
        mask_hi = lg_bar > p67
        segs = [("low g_bar", g_bar[mask_lo], g_obs[mask_lo]), ("mid g_bar", g_bar[mask_mid], g_obs[mask_mid]), ("high g_bar", g_bar[mask_hi], g_obs[mask_hi])]
        for seg_name, gb, go in segs:
            if len(gb) >= 50:
                r = fit_g0_only(gb, go)
                print(f"  (3) {seg_name} (n={len(gb)}): a0 = {r['a0_si']:.4e} m/s^2")
        # 4. Rotmod per-galaxy median (McGaugh-style: fit each galaxy, median g0)
        per_gal, gal_names = load_sparc_rar_real_only_per_galaxy(n_galaxies=175)
        if len(per_gal) >= 10:
            g0_per_gal = []
            for gb, go in per_gal:
                try:
                    r = fit_g0_only(gb, go)
                    g0_per_gal.append(r["g0"])
                except Exception:
                    continue
            if g0_per_gal:
                g0_med = float(np.median(g0_per_gal))
                a0_med = float(a0_si_from_g0(g0_med))
                print(f"  (4) Rotmod per-galaxy median (n={len(g0_per_gal)} gal): a0 = {a0_med:.4e} m/s^2")
        print(f"  -> If (4)~1.2e-10: diff from RAR.mrt is sample/weighting. If (1)(2)~1.55e-10: method diff.")

    # Bootstrap confidence intervals
    A0_MCGAUGH = 1.2e-10  # m/s^2
    MCGAUGH_RANGE = (1.1e-10, 1.3e-10)  # McGaugh 2016 range
    boot = None
    if args.bootstrap > 0:
        if args.log_mse:
            print(f"\n[1b] Bootstrap log MSE ({args.bootstrap} samples)...")
            boot = bootstrap_confidence_log(g_bar, g_obs, n_bootstrap=args.bootstrap)
            res_log = fit_g0_log_loss(g_bar, g_obs)
            print(f"    Point estimate (log MSE): a0 = {res_log['a0_si']:.4e} m/s^2")
        else:
            print(f"\n[1b] Bootstrap ({args.bootstrap} samples)...")
            boot = bootstrap_confidence(g_bar, g_obs, n_bootstrap=args.bootstrap)
        a0_lo, a0_hi = boot["a0_si"]["ci68_lo"], boot["a0_si"]["ci68_hi"]
        mcgaugh_in_1sigma = a0_lo <= A0_MCGAUGH <= a0_hi
        mcgaugh_range_overlap = a0_lo <= MCGAUGH_RANGE[1] and a0_hi >= MCGAUGH_RANGE[0]
        print(f"    a0: {boot['a0_si']['mean']:.4e} +- {boot['a0_si']['std']:.4e} m/s^2")
        print(f"    1-sigma CI: [{a0_lo:.4e}, {a0_hi:.4e}]")
        if boot["a0_si"]["std"] < 1e-12:
            print(f"    (Bootstrap CI degenerate: RAR fit very stable)")
        print(f"    McGaugh [1.1, 1.3]e-10 overlap? {'YES' if mcgaugh_range_overlap else 'NO'}")

    print(f"\n[2] Best fit:")
    print(f"    g0 (g-dagger) = {result['g0']:.2f} (km/s)^2/kpc")
    print(f"    a0 = {result['a0_si']:.4e} m/s^2")
    print(f"    alpha = {result['alpha']:.4f}  (strength parameter)")
    print(f"    RMSE = {result['rmse']:.4f}")
    print(f"    R2 (linear) = {result['r2']:.4f}")
    print(f"    R2 (log)   = {result['r2_log']:.4f}")

    print(f"\n[3] Theory check (alpha=1):")
    print(f"    RMSE(alpha free)   = {result['rmse']:.4f}")
    print(f"    RMSE(alpha=1 fix)  = {result['rmse_alpha1']:.4f}")
    if result["alpha_prefers_1"]:
        print(f"    -> alpha~1 is consistent (free alpha does not significantly improve fit)")
    else:
        print(f"    -> Free alpha improves fit; alpha={result['alpha']:.3f} deviates from 1")

    if 0.8 <= result["alpha"] <= 1.2:
        verdict = "CONSISTENT"
        msg = "alpha within ~20% of 1. Theory (xi,lambda) compatible with SPARC."
    elif 0.5 <= result["alpha"] <= 1.5:
        verdict = "MARGINAL"
        msg = "alpha within ~50% of 1. Further data or systematics may refine."
    else:
        verdict = "TENSION"
        msg = "alpha deviates from 1. Revisit L-field coupling or data systematics."

    print(f"\n[4] Verdict: {verdict}")
    print(f"    -> {msg}")

    # McGaugh consistency (if bootstrap ran)
    if boot is not None:
        a0_lo, a0_hi = boot["a0_si"]["ci68_lo"], boot["a0_si"]["ci68_hi"]
        mcgaugh_in_1sigma = a0_lo <= A0_MCGAUGH <= a0_hi
        print(f"\n[5] McGaugh a0 consistency:")
        print(f"    Fitted a0: {result['a0_si']:.4e} m/s^2  (1-sigma: [{a0_lo:.4e}, {a0_hi:.4e}])")
        print(f"    McGaugh:   {A0_MCGAUGH:.4e} m/s^2")
        print(f"    -> {'Consistent (1.2e-10 within 1-sigma)' if mcgaugh_in_1sigma else '~30% offset: may need explanation'}")

    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime
    n_gal = len(names) if isinstance(names, list) and names and not names[0].startswith("RAR_") else 0
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_galaxies": n_gal, "n_points": result["n_points"], "data_source": data_source},
        "fit": result,
        "verdict": verdict,
    }
    def _to_json(v):
        if isinstance(v, (np.integer, np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    out_clean = {
        "timestamp": out["timestamp"],
        "config": {**out["config"], "log_mse_bootstrap": args.log_mse},
        "fit": {k: _to_json(v) for k, v in out["fit"].items()},
        "verdict": out["verdict"],
    }
    if args.diagnose:
        diag = {}
        res_a1 = fit_g0_only(g_bar, g_obs)
        res_log = fit_g0_log_loss(g_bar, g_obs)
        diag["alpha1_linear_a0"] = res_a1["a0_si"]
        diag["alpha1_log_a0"] = res_log["a0_si"]
        lg_bar = np.log10(g_bar + 1e-20)
        p33, p67 = np.percentile(lg_bar, [33.3, 66.7])
        for seg_name, mask in [("low", lg_bar <= p33), ("mid", (lg_bar > p33) & (lg_bar <= p67)), ("high", lg_bar > p67)]:
            gb, go = g_bar[mask], g_obs[mask]
            if len(gb) >= 50:
                diag[f"segment_{seg_name}_a0"] = fit_g0_only(gb, go)["a0_si"]
        per_gal, _ = load_sparc_rar_real_only_per_galaxy(n_galaxies=175)
        g0_list = []
        for gb, go in per_gal:
            try:
                g0_list.append(fit_g0_only(gb, go)["g0"])
            except Exception:
                pass
        if g0_list:
            diag["rotmod_per_galaxy_median_a0"] = float(a0_si_from_g0(np.median(g0_list)))
        out_clean["diagnose"] = diag
    if boot is not None:
        out_clean["bootstrap"] = boot
        out_clean["mcgaugh_in_1sigma"] = boot["a0_si"]["ci68_lo"] <= A0_MCGAUGH <= boot["a0_si"]["ci68_hi"]
        a0_lo, a0_hi = boot["a0_si"]["ci68_lo"], boot["a0_si"]["ci68_hi"]
        out_clean["mcgaugh_range_overlap"] = a0_lo <= MCGAUGH_RANGE[1] and a0_hi >= MCGAUGH_RANGE[0]
    if args.log_mse and boot is not None:
        res_log = fit_g0_log_loss(g_bar, g_obs)
        out_clean["log_mse_point_estimate"] = {"g0": res_log["g0"], "a0_si": res_log["a0_si"]}
    with open(out_dir / "fit_meta_parameters.json", "w", encoding="utf-8") as f:
        json.dump(out_clean, f, indent=2, ensure_ascii=False)
    print(f"\n    Results saved: {out_dir / 'fit_meta_parameters.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
