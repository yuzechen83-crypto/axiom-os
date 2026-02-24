"""
RAR Discovery (Radial Acceleration Relation) - Acceleration Space

Hypothesis: Meta-Axis modifies the gravitational field, not matter density.
Manifests as tight correlation: g_obs vs g_bar.

g_bar = V_bary^2 / r  (Newtonian from visible matter)
g_obs = V_obs^2 / r   (observed)

Target: g_obs = g_bar * nu(g_bar). Find nu (the "boost" factor).
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import (
    load_sparc_rar,
    load_sparc_galaxy,
    get_available_sparc_galaxies,
    SPARC_GALAXIES,
)
from axiom_os.engine import DiscoveryEngine
from axiom_os.layers.meta_kernel import (
    KM_S_SQ_PER_KPC_TO_MS2,
    compute_meta_length,
    nu_mcgaugh,
)


def _build_rar_data(n_galaxies: int = 175) -> tuple:
    """Aggregate (g_bar, g_obs) from all galaxies."""
    avail = get_available_sparc_galaxies()
    n = min(n_galaxies, len(avail) if avail else len(SPARC_GALAXIES))
    g_bar, g_obs, names = load_sparc_rar(n_galaxies=n, use_mock_if_fail=True, use_real=True)
    if len(g_bar) < 10:
        return None, None, []
    return g_bar, g_obs, names


def _build_rar_data_with_components(n_galaxies: int = 175) -> tuple:
    """
    Load (g_gas, g_disk, g_bulge, g_obs, galaxy_id) for Bayesian Mass Calibration.
    Returns: g_gas, g_disk, g_bulge, g_obs, galaxy_id, names
    g_bar = g_gas + Upsilon * (g_disk + g_bulge), Upsilon = 0.5 * exp(log_Upsilon[galaxy_id])
    """
    avail = get_available_sparc_galaxies()
    n = min(n_galaxies, len(avail) if avail else len(SPARC_GALAXIES))
    names = avail[:n] if avail else SPARC_GALAXIES[:n]

    all_g_gas, all_g_disk, all_g_bulge, all_g_obs, all_galaxy_id = [], [], [], [], []
    for gid, name in enumerate(names):
        d = load_sparc_galaxy(name, use_mock_if_fail=True, use_real=True)
        R = np.asarray(d["R"], dtype=np.float64)
        r_safe = np.maximum(R, 0.1)
        g_gas = (d["V_gas"] ** 2) / r_safe
        g_disk = (d["V_disk"] ** 2) / r_safe
        g_bulge = (d["V_bulge"] ** 2) / r_safe
        g_obs = (d["V_obs"] ** 2) / r_safe
        mask = (
            (R > 0.1) & (d["V_obs"] > 0) &
            np.isfinite(g_gas) & np.isfinite(g_disk) & np.isfinite(g_bulge) & np.isfinite(g_obs) &
            (g_obs > 1e-12)
        )
        if mask.sum() < 3:
            continue
        all_g_gas.append(g_gas[mask])
        all_g_disk.append(g_disk[mask])
        all_g_bulge.append(g_bulge[mask])
        all_g_obs.append(g_obs[mask])
        all_galaxy_id.append(np.full(mask.sum(), gid, dtype=np.int64))

    if not all_g_gas:
        return None
    return (
        np.concatenate(all_g_gas),
        np.concatenate(all_g_disk),
        np.concatenate(all_g_bulge),
        np.concatenate(all_g_obs),
        np.concatenate(all_galaxy_id),
        names[: len(all_g_gas)],
    )


class GalaxyCalibrator(nn.Module):
    """
    Bayesian Mass Calibration: per-galaxy log_Upsilon.
    Upsilon = 0.5 * exp(log_Upsilon)  (base assumption 0.5)
    g_bar_new = g_gas + Upsilon * (g_disk + g_bulge)
    """

    def __init__(self, n_galaxies: int):
        super().__init__()
        self.log_Upsilon = nn.Parameter(torch.zeros(n_galaxies))

    def forward(
        self,
        g_gas: torch.Tensor,
        g_disk: torch.Tensor,
        g_bulge: torch.Tensor,
        galaxy_id: torch.Tensor,
    ) -> torch.Tensor:
        Upsilon = 0.5 * torch.exp(self.log_Upsilon[galaxy_id])
        g_bar_new = g_gas + Upsilon * (g_disk + g_bulge)
        return g_bar_new.clamp(min=1e-14)


def _rar_formula_torch(g_bar: torch.Tensor, g0: float, eps: float = 1e-14) -> torch.Tensor:
    """RAR McGaugh: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g0)))."""
    x = (g_bar / (g0 + eps)).clamp(min=eps)
    denom = (1.0 - torch.exp(-torch.sqrt(x))).clamp(min=0.01)
    return g_bar / denom


def _mcgowan_pred_log(g_bar: np.ndarray, g0: float) -> np.ndarray:
    """McGaugh formula: g_obs = g_bar/(1-exp(-sqrt(g_bar/g0))). Returns log10(g_obs_pred)."""
    x = np.maximum(g_bar, 1e-14)
    denom = 1.0 - np.exp(-np.sqrt(x / (g0 + 1e-14)))
    denom = np.maximum(denom, 0.01)
    g_obs_pred = x / denom
    return np.log10(g_obs_pred + 1e-14)


def run_rar_g0_diagnostic(g_bar: np.ndarray, g_obs: np.ndarray, g0_free: float = 321.0, g0_prior: float = 3700.0) -> dict:
    """
    Step 2: Residual analysis for why free fit gives g0_free instead of g0_prior.
    Returns residuals (log space), binned stats, and interpretation.
    """
    log_g_obs = np.log10(g_obs + 1e-14)
    log_pred_free = _mcgowan_pred_log(g_bar, g0_free)
    log_pred_prior = _mcgowan_pred_log(g_bar, g0_prior)
    residual_free = log_g_obs - log_pred_free
    residual_prior = log_g_obs - log_pred_prior

    # Bins by log10(g_bar): low (<2.5), mid (2.5-3.5), high (>3.5) -> roughly <316, 316-3162, >3162
    lg = np.log10(g_bar + 1e-14)
    bins = [lg.min() - 0.01, 2.5, 3.5, lg.max() + 0.01]
    out = {
        "residual_free": residual_free,
        "residual_prior": residual_prior,
        "log_g_bar": lg,
        "g_bar": g_bar,
        "log_g_obs": log_g_obs,
    }
    bin_stats = []
    for i in range(len(bins) - 1):
        mask = (lg >= bins[i]) & (lg < bins[i + 1])
        if mask.sum() == 0:
            continue
        rf, rp = residual_free[mask], residual_prior[mask]
        bin_stats.append({
            "log_g_bar_lo": bins[i], "log_g_bar_hi": bins[i + 1],
            "n": int(mask.sum()),
            "mean_resid_free": float(np.mean(rf)), "std_resid_free": float(np.std(rf)),
            "mean_resid_prior": float(np.mean(rp)), "std_resid_prior": float(np.std(rp)),
        })
    out["bin_stats"] = bin_stats
    return out


class MonotonicRARMLP(nn.Module):
    """
    log(g_bar) -> log(g_obs). Monotonicity via positive weights in final layer.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.fc3.weight, 0.01, 0.5)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, log_g_bar: torch.Tensor) -> torch.Tensor:
        x = log_g_bar.unsqueeze(-1) if log_g_bar.dim() == 1 else log_g_bar
        h = torch.nn.functional.silu(self.fc1(x))
        h = torch.nn.functional.silu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


def run_rar_discovery(n_galaxies: int = 175, epochs: int = 800) -> dict:
    """
    Task 1: Build (g_bar, g_obs) dataset
    Task 2: Train MLP log(g_bar) -> log(g_obs)
    Task 3: Discovery: fit g_obs = g_bar * nu(g_bar)
    Task 4: Meta-Axis check: g_obs > g_bar at low g_bar?
    """
    g_bar, g_obs, names = _build_rar_data(n_galaxies)
    if g_bar is None:
        return {"error": "Insufficient data"}

    # Step 1: Data coverage (why free fit may give g0=321 vs 3700)
    n_pts = len(g_bar)
    n_lo = int((g_bar < 1000).sum())
    n_hi = int((g_bar > 10000).sum())
    print(f"\n[0] DATA COVERAGE: g_bar range {g_bar.min():.1f} – {g_bar.max():.1f} (km/s)^2/kpc, n={n_pts}")
    print(f"    g_bar < 1000: {n_lo} ({100*n_lo/n_pts:.1f}%),  g_bar > 10000: {n_hi} ({100*n_hi/n_pts:.1f}%)")

    log_g_bar = np.log10(g_bar + 1e-14)
    log_g_obs = np.log10(g_obs + 1e-14)

    X_t = torch.from_numpy(log_g_bar).float().unsqueeze(-1)
    y_t = torch.from_numpy(log_g_obs).float()

    # Asymptotic limits (physical constraints beyond data range)
    LIMIT_HIGH = 1000.0  # Way above data. Target: g_obs = g_bar (Newtonian)
    LIMIT_LOW = 0.0001   # Way below data. Target: g_obs = sqrt(a0 * g_bar) (deep MOND)
    A0_PRIOR = 3700.0   # (km/s)^2/kpc for low-acc limit target

    model = MonotonicRARMLP(hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    log_high = torch.tensor([np.log10(LIMIT_HIGH)], dtype=torch.float32)
    log_low = torch.tensor([np.log10(LIMIT_LOW)], dtype=torch.float32)
    # Targets in log space (consistent with data loss)
    target_log_high = np.log10(LIMIT_HIGH)  # g_obs = g_bar
    target_log_low = 0.5 * (np.log10(A0_PRIOR) + np.log10(LIMIT_LOW))  # log10(sqrt(a0 * g_bar))

    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t.squeeze(-1))
        loss_data = ((pred - y_t) ** 2).mean()
        # Constraint: high g_bar -> g_obs = g_bar; low g_bar -> g_obs = sqrt(a0 * g_bar)
        pred_log_high = model(log_high).squeeze()
        pred_log_low = model(log_low).squeeze()
        loss_high = (pred_log_high - target_log_high) ** 2
        loss_low = (pred_log_low - target_log_low) ** 2
        loss = loss_data + 0.1 * (loss_high + loss_low)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

    with torch.no_grad():
        log_g_obs_pred = model(X_t.squeeze(-1)).numpy()
    g_obs_pred = 10**log_g_obs_pred

    mse = np.mean((g_obs_pred - g_obs) ** 2)
    r2 = 1 - np.sum((g_obs_pred - g_obs) ** 2) / (np.sum((g_obs - g_obs.mean()) ** 2) + 1e-12)

    # Discovery: fit nu = g_obs / g_bar as function of g_bar
    nu = g_obs / (g_bar + 1e-14)
    X_nu = np.column_stack([np.log10(g_bar + 1e-14), g_bar])
    engine = DiscoveryEngine(use_pysr=False)
    formula_nu, pred_nu, _ = engine.discover_multivariate(
        X_nu[:, :1], nu, var_names=["log_g_bar"], selector="bic",
    )

    # Candidate forms: MOND, Power law
    def fit_mond(g_bar: np.ndarray, g_obs: np.ndarray) -> tuple:
        from scipy.optimize import minimize
        def loss(p):
            g0 = 10**p[0]
            nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(g_bar / g0, 1e-10))))
            pred = g_bar * np.clip(nu, 1.0, 100.0)
            return np.mean((pred - g_obs) ** 2)
        res = minimize(loss, [-10.0], method="Nelder-Mead", options={"maxiter": 200})
        g0 = 10**res.x[0]
        nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(g_bar / g0, 1e-10))))
        pred = g_bar * np.clip(nu, 1.0, 100.0)
        r2 = 1 - np.sum((pred - g_obs) ** 2) / (np.sum((g_obs - g_obs.mean()) ** 2) + 1e-12)
        return f"MOND g0={g0:.2e}", pred, r2

    def fit_power(g_bar: np.ndarray, g_obs: np.ndarray) -> tuple:
        log_gb = np.log10(g_bar + 1e-14)
        log_go = np.log10(g_obs + 1e-14)
        alpha = np.polyfit(log_gb, log_go, 1)[0]
        pred = g_bar**alpha
        r2 = 1 - np.sum((pred - g_obs) ** 2) / (np.sum((g_obs - g_obs.mean()) ** 2) + 1e-12)
        return f"Power g_obs=g_bar^{alpha:.3f}", pred, r2

    try:
        form_mond, pred_mond, r2_mond = fit_mond(g_bar, g_obs)
    except Exception:
        form_mond, pred_mond, r2_mond = "MOND (fit failed)", np.zeros_like(g_obs), 0.0
    form_power, pred_power, r2_power = fit_power(g_bar, g_obs)

    # Meta-Axis check: g_obs > g_bar at low g_bar?
    low_acc = g_bar < np.percentile(g_bar, 25)
    boost_at_low = np.mean(g_obs[low_acc] / (g_bar[low_acc] + 1e-14)) if np.any(low_acc) else 1.0

    # Crystallize: symbolic extraction of the Red Line
    crystallized = crystallize_rar_law(g_bar, g_obs)

    # Bayesian Mass Calibration: fine-tune Upsilon per galaxy
    g_bar_calibrated = None
    calibration_info = None
    comp_data = _build_rar_data_with_components(n_galaxies)
    if comp_data is not None:
        g_gas, g_disk, g_bulge, g_obs_comp, galaxy_id, comp_names = comp_data
        g0_cal = crystallized.get("g_dagger") or crystallized.get("a0_or_gdagger") or 3700.0
        if g0_cal is not None and len(g_gas) >= 50:
            try:
                g_bar_calibrated, calibrator, loss_hist = run_bayesian_calibration(
                    g_gas, g_disk, g_bulge, g_obs_comp, galaxy_id,
                    g0=float(g0_cal), lambda_prior=0.1, n_steps=1000,
                )
                upsilon_vals = (0.5 * torch.exp(calibrator.log_Upsilon.detach())).numpy()
                g_bar_orig = g_gas + g_disk + g_bulge
                calibration_info = {
                    "g_bar_calibrated": g_bar_calibrated,
                    "g_bar_original": g_bar_orig,
                    "g_obs_calibrated": g_obs_comp,
                    "log_g_bar_calibrated": np.log10(g_bar_calibrated + 1e-14),
                    "log_g_obs_calibrated": np.log10(g_obs_comp + 1e-14),
                    "Upsilon_per_galaxy": upsilon_vals,
                    "loss_history": loss_hist[-10:],
                }
                print(f"    [Calibration] Optimized {len(upsilon_vals)} galaxies, final loss={loss_hist[-1]:.4e}")
            except Exception as e:
                print(f"    [Calibration] Skip: {e}")

    return {
        "g_bar": g_bar,
        "g_obs": g_obs,
        "g_bar_calibrated": g_bar_calibrated,
        "calibration_info": calibration_info,
        "g_obs_pred": g_obs_pred,
        "log_g_bar": log_g_bar,
        "log_g_obs": log_g_obs,
        "r2": float(r2),
        "mse": float(mse),
        "formula_nu": formula_nu,
        "form_mond": form_mond,
        "r2_mond": r2_mond,
        "form_power": form_power,
        "r2_power": r2_power,
        "boost_at_low": float(boost_at_low),
        "n_galaxies": len(names),
        "n_samples": len(g_bar),
        "crystallized": crystallized,
    }


def run_bayesian_calibration(
    g_gas: np.ndarray,
    g_disk: np.ndarray,
    g_bulge: np.ndarray,
    g_obs: np.ndarray,
    galaxy_id: np.ndarray,
    g0: float,
    lambda_prior: float = 0.1,
    n_steps: int = 1000,
) -> tuple:
    """
    Calibration Loop: optimize log_Upsilon to minimize scatter around RAR.
    Freeze RAR formula (g0), optimize only GalaxyCalibrator params.
    Returns: (g_bar_new, calibrator, loss_history)
    """
    n_galaxies = int(galaxy_id.max()) + 1
    calibrator = GalaxyCalibrator(n_galaxies=n_galaxies)
    opt = torch.optim.Adam(calibrator.parameters(), lr=1e-2)

    g_gas_t = torch.from_numpy(g_gas).float()
    g_disk_t = torch.from_numpy(g_disk).float()
    g_bulge_t = torch.from_numpy(g_bulge).float()
    g_obs_t = torch.from_numpy(g_obs).float()
    gal_id_t = torch.from_numpy(galaxy_id).long()

    loss_hist = []
    for step in range(n_steps):
        opt.zero_grad()
        g_bar_new = calibrator(g_gas_t, g_disk_t, g_bulge_t, gal_id_t)
        g_obs_pred = _rar_formula_torch(g_bar_new, g0)
        fit_loss = ((g_obs_pred - g_obs_t) ** 2).mean()
        prior_loss = lambda_prior * (calibrator.log_Upsilon ** 2).mean()
        loss = fit_loss + prior_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 2.0)
        opt.step()
        loss_hist.append(loss.item())

    with torch.no_grad():
        g_bar_new = calibrator(g_gas_t, g_disk_t, g_bulge_t, gal_id_t).numpy()

    return g_bar_new, calibrator, loss_hist


def crystallize_rar_law(g_bar: np.ndarray, g_obs: np.ndarray) -> dict:
    """
    Crystallize the Law of Gravity: find exact equation of the RAR (Red Line).
    Target: y = x * nu(x), nu = interpolation function.
    Search: x, sqrt(x), 1/x, exp(-x), tanh(x).
    Extract a0 or g† from MOND-like fits.
    """
    x = np.asarray(g_bar, dtype=np.float64)
    y = np.asarray(g_obs, dtype=np.float64)
    eps = 1e-12
    x_safe = np.maximum(x, eps)

    # Build RAR feature library
    F = np.column_stack([
        x_safe,
        np.sqrt(x_safe),
        1.0 / x_safe,
        np.exp(-x_safe),
        np.tanh(x_safe),
        x_safe * np.exp(-x_safe),
        np.sqrt(x_safe) * np.exp(-np.sqrt(x_safe)),
    ])
    fnames = ["x", "sqrt(x)", "1/x", "exp(-x)", "tanh(x)", "x*exp(-x)", "sqrt(x)*exp(-sqrt(x))"]

    results = []

    # Fit 1: y against X (Lasso)
    try:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=1e-4, max_iter=5000, positive=False)
        model.fit(F, y)
        pred = model.predict(F)
        r2 = 1 - np.sum((pred - y) ** 2) / (np.sum((y - y.mean()) ** 2) + eps)
        terms = [f"{model.coef_[i]:.4g}*{fnames[i]}" for i in range(len(fnames)) if abs(model.coef_[i]) > 1e-6]
        if abs(model.intercept_) > 1e-6:
            terms.append(f"{model.intercept_:.4g}")
        formula = " + ".join(terms) if terms else "0"
        results.append(("y vs X (Lasso)", formula, r2, pred))
    except Exception as e:
        results.append(("y vs X (Lasso)", f"failed: {e}", -1, np.zeros_like(y)))

    # Fit 2: nu = y/x against X
    nu = y / x_safe
    nu = np.clip(nu, 0.1, 100.0)
    try:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=1e-4, max_iter=5000, positive=False)
        model.fit(F, nu)
        pred_nu = model.predict(F)
        pred_y = pred_nu * x_safe
        r2 = 1 - np.sum((pred_y - y) ** 2) / (np.sum((y - y.mean()) ** 2) + eps)
        terms = [f"{model.coef_[i]:.4g}*{fnames[i]}" for i in range(len(fnames)) if abs(model.coef_[i]) > 1e-6]
        if abs(model.intercept_) > 1e-6:
            terms.append(f"{model.intercept_:.4g}")
        formula_nu = " + ".join(terms) if terms else "1"
        results.append(("nu=y/x vs X (Lasso)", f"nu={formula_nu}, y=x*nu", r2, pred_y))
    except Exception as e:
        results.append(("nu vs X (Lasso)", f"failed: {e}", -1, np.zeros_like(y)))

    # Fit 3: RAR (McGaugh) y = x / (1 - exp(-sqrt(x/g†))); g0 in [100, 50000] (km/s)^2/kpc
    def _rar_model(x_arr: np.ndarray, g0: float) -> np.ndarray:
        xa = np.maximum(x_arr, 1e-14)
        denom = 1.0 - np.exp(-np.sqrt(xa / (g0 + 1e-14)))
        return xa / np.maximum(denom, 0.01)

    a0_extracted = None
    g_dagger = None
    a0_si = None
    r2_linear = None
    r2_log = None
    try:
        from scipy.optimize import curve_fit, minimize
        g0_bounds = (100.0, 50000.0)  # (km/s)^2/kpc

        def loss_for_min(g0_val: float) -> float:
            g0_val = np.clip(g0_val, g0_bounds[0], g0_bounds[1])
            pred_c = _rar_model(x_safe, g0_val)
            return np.mean((pred_c - y) ** 2)

        best_loss, best_g0, best_pred = np.inf, None, None

        # Weighted curve_fit: equal contribution per log10(g_bar) bin (avoid mid-range dominance)
        log_g_bar_fit = np.log10(x_safe + 1e-14)
        log_y_fit = np.log10(y + 1e-14)
        bin_edges = np.linspace(log_g_bar_fit.min(), log_g_bar_fit.max(), 11)  # 10 bins
        bin_idx = np.digitize(log_g_bar_fit, bin_edges)
        bin_counts = np.bincount(bin_idx, minlength=bin_idx.max() + 1)
        # weight_i = 1 / count_in_bin -> sigma = 1/weight = count_in_bin (curve_fit minimizes sum((y-f)/sigma)^2)
        sigma_fit = np.maximum(bin_counts[bin_idx].astype(np.float64), 1.0)

        def _rar_model_log(g_bar_arr: np.ndarray, g0: float) -> np.ndarray:
            return np.log10(_rar_model(g_bar_arr, g0) + 1e-14)

        try:
            popt, _ = curve_fit(
                _rar_model_log, x_safe, log_y_fit,
                p0=[3700.0],
                sigma=sigma_fit,
                bounds=([100.0], [50000.0]),
                maxfev=5000,
            )
            g0_cf = float(np.clip(popt[0], g0_bounds[0], g0_bounds[1]))
            pred_cf = _rar_model(x_safe, g0_cf)
            loss_cf = np.mean((pred_cf - y) ** 2)
            best_loss, best_g0, best_pred = loss_cf, g0_cf, pred_cf
        except Exception:
            pass

        # Multiple starting points with bounds [100, 50000] — only when curve_fit failed
        if best_g0 is None:
            for p0 in [300.0, 1000.0, 3000.0, 5000.0, 10000.0]:
                res = minimize(
                    lambda p: loss_for_min(p[0]),
                    [p0],
                    method="L-BFGS-B",
                    bounds=[(g0_bounds[0], g0_bounds[1])],
                    options={"maxiter": 500},
                )
                g0_cand = float(np.clip(res.x[0], g0_bounds[0], g0_bounds[1]))
                pred_c = _rar_model(x_safe, g0_cand)
                loss_c = np.mean((pred_c - y) ** 2)
                if loss_c < best_loss:
                    best_loss, best_g0, best_pred = loss_c, g0_cand, pred_c

        if best_g0 is None:
            raise RuntimeError("RAR fit did not converge")
        g0 = best_g0
        pred = best_pred
        # 1. g0 in data units (printed in main before SI)
        ss_res = np.sum((pred - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + eps
        r2_linear = 1.0 - (ss_res / ss_tot)
        log_y = np.log10(y + 1e-14)
        log_pred = np.log10(pred + 1e-14)
        ss_res_log = np.sum((log_pred - log_y) ** 2)
        ss_tot_log = np.sum((log_y - np.mean(log_y)) ** 2) + eps
        r2_log = 1.0 - (ss_res_log / ss_tot_log)
        r2 = r2_log
        a0_extracted = g0
        g_dagger = g0
        formula = f"y = x/(1-exp(-sqrt(x/g0))), g0={g0:.4e}"
        results.append(("RAR McGaugh", formula, r2, pred))
        a0_si = g0 * KM_S_SQ_PER_KPC_TO_MS2
        # Physical prior: g0 = 3700 (km/s)^2/kpc → a0 ≈ 1.2e-10 m/s^2; plot both to show divergence
        g_dagger_prior = 3700.0
        pred_prior = _rar_model(x_safe, g_dagger_prior)
    except Exception as e:
        results.append(("RAR McGaugh", f"failed: {e}", -1, np.zeros_like(y)))
        g_dagger_prior = 3700.0
        pred_prior = _rar_model(x_safe, g_dagger_prior)

    # Sort by R2, take best 3
    results.sort(key=lambda r: r[2], reverse=True)
    best3 = results[:3]

    return {
        "best3": best3,
        "a0_or_gdagger": a0_extracted,
        "g_dagger": g_dagger,
        "g_dagger_prior": g_dagger_prior,
        "pred_prior": pred_prior,
        "a0_si": a0_si,
        "r2_linear_rar": r2_linear,
        "r2_log_rar": r2_log,
        "formulas": [r[1] for r in best3],
        "r2_scores": [r[2] for r in best3],
    }


def _plot_rar(res: dict, out_path: Path) -> None:
    """RAR scatter + discovered curve + residual. Includes calibrated scatter if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_panels = 3 if res.get("calibration_info") else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 2:
            axes = [axes[0], axes[1]]
        lg_bar = res["log_g_bar"]
        lg_obs = res["log_g_obs"]
        g_obs_pred = res["g_obs_pred"]

        # Sort for curve overlay
        idx = np.argsort(lg_bar)
        lg_bar_s = lg_bar[idx]
        lg_obs_pred_s = np.log10(g_obs_pred[idx] + 1e-14)

        # Panel 0: Original RAR
        axes[0].scatter(lg_bar, lg_obs, alpha=0.4, s=8, c="gray", label="SPARC (raw)")
        axes[0].plot(lg_bar_s, lg_obs_pred_s, "b-", lw=2, label="MLP")
        axes[0].plot(lg_bar_s, lg_bar_s, "k--", lw=1, alpha=0.7, label="1:1")
        cry = res.get("crystallized", {})
        pred_prior = cry.get("pred_prior")
        if pred_prior is not None:
            lg_prior_s = np.log10(pred_prior[idx] + 1e-14)
            axes[0].plot(lg_bar_s, lg_prior_s, "r-", lw=2, alpha=0.9, label="RAR (Red Line)")
        pred_free = None
        for r in cry.get("best3", []):
            if len(r) >= 4 and r[0] == "RAR McGaugh":
                pred_free = r[3]
                break
        if pred_free is not None:
            lg_free_s = np.log10(pred_free[idx] + 1e-14)
            axes[0].plot(lg_bar_s, lg_free_s, "g-", lw=1.5, alpha=0.9, label="McGaugh (free fit)")
        axes[0].set_xlabel("log10(g_bar) [(km/s)^2/kpc]")
        axes[0].set_ylabel("log10(g_obs) [(km/s)^2/kpc]")
        axes[0].set_title("RAR: Before Calibration")
        axes[0].legend(fontsize=8)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].grid(True, alpha=0.3)

        # Panel 1: Residual (original)
        resid = lg_obs - np.log10(g_obs_pred + 1e-14)
        axes[1].scatter(lg_bar, resid, alpha=0.4, s=8, c="steelblue")
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("log10(g_bar)")
        axes[1].set_ylabel("Residual")
        axes[1].set_title(f"Residual: std={np.std(resid):.3f}")
        axes[1].grid(True, alpha=0.3)

        # Panel 2: After Bayesian Calibration (gray cloud shrinks)
        if res.get("calibration_info"):
            cal = res["calibration_info"]
            lg_bar_cal = cal["log_g_bar_calibrated"]
            lg_obs_cal = cal["log_g_obs_calibrated"]
            g0 = cry.get("g_dagger") or cry.get("a0_or_gdagger") or 3700.0
            g_bar_cal = cal["g_bar_calibrated"]
            pred_cal = g_bar_cal / np.maximum(1.0 - np.exp(-np.sqrt(g_bar_cal / (g0 + 1e-14))), 0.01)
            lg_pred_cal = np.log10(pred_cal + 1e-14)
            idx_cal = np.argsort(lg_bar_cal)
            axes[2].scatter(lg_bar_cal, lg_obs_cal, alpha=0.5, s=10, c="darkgreen", label="Calibrated")
            axes[2].plot(lg_bar_cal[idx_cal], lg_pred_cal[idx_cal], "r-", lw=2, label="RAR (Red Line)")
            axes[2].plot(lg_bar_cal[idx_cal], lg_bar_cal[idx_cal], "k--", lw=1, alpha=0.5, label="1:1")
            resid_cal = lg_obs_cal - lg_pred_cal
            axes[2].set_xlabel("log10(g_bar_new) [(km/s)^2/kpc]")
            axes[2].set_ylabel("log10(g_obs) [(km/s)^2/kpc]")
            axes[2].set_title(f"After Calibration: std={np.std(resid_cal):.3f}")
            axes[2].legend(fontsize=8)
            axes[2].set_aspect("equal", adjustable="box")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot skip: {e}")


def main():
    print("=" * 70)
    print("RAR Discovery - Acceleration Space")
    print("=" * 70)
    print("\nHypothesis: g_obs = g_bar * nu(g_bar). Find nu (Meta-Axis boost).")
    print("-" * 70)

    res = run_rar_discovery(n_galaxies=175, epochs=800)
    if "error" in res:
        print(f"Error: {res['error']}")
        return

    print(f"\n[1] Data: {res['n_galaxies']} galaxies, {res['n_samples']} samples")
    print(f"    MLP R2: {res['r2']:.4f}, MSE: {res['mse']:.2e}")

    print(f"\n[2] Discovery (nu = g_obs/g_bar): {res['formula_nu'] or 'None'}")
    print(f"    MOND: {res['form_mond']} R2={res['r2_mond']:.4f}")
    print(f"    Power: {res['form_power']} R2={res['r2_power']:.4f}")

    print(f"\n[3] Meta-Axis check (g_obs > g_bar at low g_bar?):")
    print(f"    Mean boost at low g_bar: {res['boost_at_low']:.2f}")
    if res["boost_at_low"] > 1.2:
        print("    -> YES: Boost factor nu > 1 at low acceleration (Meta-Axis projection)")
    else:
        print("    -> Weak or no boost at low g_bar")

    # Crystallized Law
    cry = res.get("crystallized", {})
    if cry:
        print(f"\n[4] CRYSTALLIZED LAW - Best 3 Discovered Formulas:")
        for i, (name, formula, r2, _) in enumerate(cry.get("best3", [])[:3], 1):
            print(f"    #{i} ({name}) R2={r2:.4f}:")
            print(f"       {formula}")
        a0 = cry.get("a0_or_gdagger")
        a0_si = cry.get("a0_si")
        g0_prior = cry.get("g_dagger_prior")
        if a0 is not None:
            # Free fit (data's answer)
            print(f"\n[5] FREE FIT (data's answer): g0 = {a0:.2f} [(km/s)^2/kpc]")
            if cry.get("r2_log_rar") is not None:
                print(f"    R2 (log space) = {cry['r2_log_rar']:.4f}")
            if a0_si is not None:
                print(f"    a0 = {a0_si:.4e} m/s^2 (SI)")
            if g0_prior is not None:
                print(f"    PHYSICAL PRIOR: g0 = {g0_prior:.0f} [(km/s)^2/kpc] (a0 = 1.2e-10 m/s^2)")
                if abs(a0 - g0_prior) > 100:
                    print(f"    -> Divergence: McGaugh formula does not perfectly fit this dataset (e.g. overshoot at high g_bar).")
            if a0_si is not None:
                meta = compute_meta_length(a0_si)
                if np.isfinite(meta["L_m"]):
                    print(f"    Meta-Axis L = c^2/a0 = {meta['L_m']:.4e} m = {meta['L_Gly']:.4e} Gly")

    out = ROOT / "axiom_os" / "discovery_rar_plot.png"
    _plot_rar(res, out)
    print(f"\n[6] Plot saved: {out}")

    cal = res.get("calibration_info")
    if cal:
        g0 = cry.get("g_dagger") or 3700.0
        g_bar_orig = cal["g_bar_original"]
        g_bar_cal = cal["g_bar_calibrated"]
        g_obs_cal = cal["g_obs_calibrated"]
        pred_orig = g_bar_orig / np.maximum(
            1.0 - np.exp(-np.sqrt(g_bar_orig / (g0 + 1e-14))), 0.01
        )
        pred_cal = g_bar_cal / np.maximum(
            1.0 - np.exp(-np.sqrt(g_bar_cal / (g0 + 1e-14))), 0.01
        )
        resid_before = np.log10(g_obs_cal + 1e-14) - np.log10(pred_orig + 1e-14)
        resid_after = np.log10(g_obs_cal + 1e-14) - np.log10(pred_cal + 1e-14)
        std_before = np.std(resid_before)
        std_after = np.std(resid_after)
        print(f"\n[7] Bayesian Mass Calibration:")
        print(f"    Residual std: before={std_before:.4f} -> after={std_after:.4f}")
        if std_before > 1e-10:
            print(f"    Scatter reduction: {(1 - std_after/std_before)*100:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()
