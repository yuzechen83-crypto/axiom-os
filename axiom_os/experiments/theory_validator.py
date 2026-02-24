"""
Theory Validator - Axiom-OS as "Theory Verification Machine"

Paper prediction: Dark matter effect = integral projection on Meta-Axis
  ρ_eff = ρ_baryon + ∫ K(z) ρ(z) dz
  K(z) ∝ 1/√(1 - (z/L)²)

Pipeline:
  1. Hard Core: Newtonian (baryons only) → V_def² = 0
  2. Soft Shell: Learns "missing gravity" from SPARC (r, V_bary²) → V_def²
  3. Discovery: Symbolic regression on soft shell output
  4. Theory Match: Compare discovered formula with K(z) integral form
     (predicts V_def² ∝ ρ ∝ V_bary²/r² or similar)
"""

import sys
from pathlib import Path
import re
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.layers.rcln import RCLNLayer
from axiom_os.engine import DiscoveryEngine
from axiom_os.datasets.sparc import load_sparc_galaxy, load_mock_batch, get_available_sparc_galaxies, SPARC_GALAXIES


def galaxy_hard_core(_x):
    """Newtonian: no dark matter → V_def² = 0."""
    if isinstance(_x, torch.Tensor):
        n = _x.shape[0] if _x.dim() > 1 else 1
        dev = _x.device
    elif hasattr(_x, "values"):
        v = _x.values
        n = v.shape[0] if hasattr(v, "shape") and len(v.shape) > 1 else 1
        dev = v.device if isinstance(v, torch.Tensor) else None
    else:
        n, dev = 1, None
    return torch.zeros(n, 1, dtype=torch.float32, device=dev)


def run_theory_validator(
    n_galaxies: int = 20,
    epochs_per_galaxy: int = 300,
    data_source: str = "real",
) -> dict:
    """
    Run full theory validation pipeline.
    Returns: discovered formula, theory_match_score, match_details.
    """
    # 1. Build RCLN: Hard Core = 0, Soft Shell = MLP(r, V_bary_sq) → V_def²
    rcln = RCLNLayer(
        input_dim=2,
        hidden_dim=64,
        output_dim=1,
        hard_core_func=galaxy_hard_core,
        lambda_res=1.0,
        net_type="mlp",
    )

    # 2. Load data
    if data_source == "real":
        avail = get_available_sparc_galaxies()
        names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]
        galaxies = [load_sparc_galaxy(n, use_mock_if_fail=True, use_real=True) for n in names]
    else:
        galaxies = load_mock_batch(n_galaxies, data_source=data_source, seed=42)

    # 3. Train on pooled (r, V_bary_sq) → V_def²
    all_R, all_Vb, all_Vd = [], [], []
    for d in galaxies:
        R = d["R"]
        V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
        V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1.0)
        V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0.0)
        all_R.append(R)
        all_Vb.append(V_bary_sq)
        all_Vd.append(V_def_sq)

    R_all = np.concatenate(all_R)
    Vb_all = np.concatenate(all_Vb)
    Vd_all = np.concatenate(all_Vd)
    X = np.column_stack([R_all, Vb_all]).astype(np.float32)
    y = Vd_all.astype(np.float32)
    scale_y = float(np.max(y) + 1e-8)
    y_norm = y / scale_y

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y_norm).float().unsqueeze(1)
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-2)
    for _ in range(epochs_per_galaxy):
        opt.zero_grad()
        pred = rcln(X_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rcln.parameters(), 2.0)
        opt.step()

    # 4. Collect (x, y_soft) for Discovery
    with torch.no_grad():
        _ = rcln(X_t)
    y_soft = rcln._last_y_soft.cpu().numpy()
    data_buffer = list(zip(X.tolist(), y_soft.tolist()))
    x_arr = np.array([p[0] for p in data_buffer])
    y_soft_arr = np.array([p[1] for p in data_buffer]).ravel()

    # 5. Discovery: distill + try theory form explicitly
    engine = DiscoveryEngine(use_pysr=False)
    formula_theory, r2_theory, bic_theory = _fit_theory_form(X, y)
    formula_distill = engine.distill(
        rcln_layer=rcln,
        data_buffer=data_buffer,
        input_units=[[0, 1, 0, 0, 0], [0, 2, -2, 0, 0]],  # r [L], V_bary² [L²/T²]
    )
    formula_direct, pred_direct, _ = engine.discover_multivariate(
        X, y, var_names=["r", "V_bary_sq"], selector="bic",
    )
    formula = formula_direct or formula_distill

    # 6. Theory match: K(z) integral predicts V_def² ∝ ρ ∝ V_bary²/r^α
    match_result = theory_match_kernel_integral(formula, X, y)

    bic_direct = len(y) * np.log(np.mean((pred_direct - y) ** 2) + 1e-12) + 6 * np.log(len(y)) if pred_direct is not None else np.inf
    return {
        "formula": formula,
        "formula_distill": formula_distill,
        "formula_direct": formula_direct,
        "formula_theory": formula_theory,
        "r2_theory": r2_theory,
        "bic_theory": bic_theory,
        "bic_direct": bic_direct,
        "theory_match": match_result,
        "data_source": "REAL" if data_source == "real" else f"MOCK_{data_source.upper()}",
        "n_samples": len(y),
        "r2_direct": float(1 - np.sum((pred_direct - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)) if pred_direct is not None else 0.0,
    }


def _fit_theory_form(X: np.ndarray, y: np.ndarray) -> tuple:
    """Fit theory form: V_def^2 = C * V_bary_sq / (r^alpha + eps). Returns (formula, r2, bic)."""
    r, Vb = X[:, 0], X[:, 1]
    r_safe = np.maximum(r, 0.1)
    best_r2, best_bic, best_form = -1e9, np.inf, "None"
    for alpha in [2.0, 1.5, 2.5, 1.0, 3.0]:
        rho = Vb / (r_safe**alpha + 1e-6)
        scale = np.sum(y * rho) / (np.sum(rho**2) + 1e-12)
        pred = scale * rho
        mse = np.mean((y - pred) ** 2)
        r2 = 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        bic = len(y) * np.log(mse + 1e-12) + 2 * np.log(len(y))
        if bic < best_bic:
            best_bic, best_r2, best_form = bic, r2, f"C*V_bary_sq/r^{alpha:.1f}"
    return best_form, float(best_r2), float(best_bic)


def theory_match_kernel_integral(formula: str, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Check if discovered formula matches K(z) integral form.
    Theory: V_def² ∝ ∫K(z)ρ(z)dz ∝ ρ(r) ∝ V_bary²/r² (or r^α).
    """
    result = {
        "score": 0.0,
        "has_V_bary_sq": False,
        "has_r": False,
        "has_rho_form": False,
        "rho_form_r2": 0.0,
        "details": "",
    }
    if not formula or len(formula) < 2:
        return result

    f_lower = formula.lower()
    result["has_V_bary_sq"] = "v_bary" in f_lower or "x1" in f_lower
    result["has_r"] = "r" in f_lower or "x0" in f_lower
    result["has_rho_form"] = ("/" in formula and ("r" in f_lower or "x0" in f_lower)) or "**-" in formula

    # Fit theoretical form: y ≈ C * V_bary_sq / (r^α + ε)
    r, Vb = X[:, 0], X[:, 1]
    r_safe = np.maximum(r, 0.1)
    for alpha in [2.0, 1.5, 2.5, 1.0]:
        rho = Vb / (r_safe**alpha + 1e-6)
        rho = rho / (np.max(rho) + 1e-8)
        scale = np.sum(y * rho) / (np.sum(rho**2) + 1e-12)
        pred_rho = scale * rho
        ss_res = np.sum((y - pred_rho) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        if r2 > result["rho_form_r2"]:
            result["rho_form_r2"] = float(r2)

    result["score"] = (
        (1.0 if result["has_V_bary_sq"] else 0.0) +
        (1.0 if result["has_r"] else 0.0) +
        (1.0 if result["has_rho_form"] else 0.0) +
        min(2.0, result["rho_form_r2"] * 2)
    ) / 5.0
    result["details"] = (
        f"V_bary_sq={result['has_V_bary_sq']}, r={result['has_r']}, "
        f"rho_form={result['has_rho_form']}, rho_R2={result['rho_form_r2']:.3f}"
    )
    return result


def main():
    print("=" * 70)
    print("Theory Validator - Meta-Axis K(z) Integral Verification")
    print("=" * 70)
    print("\nPipeline: Hard Core (Newtonian=0) -> Soft Shell (learns V_def^2) -> Discovery -> Theory Match")
    print("Theory: V_def^2 ~ integral K(z)rho(z)dz, K(z)~1/sqrt(1-(z/L)^2), rho~V_bary^2/r^2")
    print("-" * 70)

    for src in ["meta", "real"]:
        print(f"\n--- Data source: {src.upper()} ---")
        res = run_theory_validator(n_galaxies=15, epochs_per_galaxy=400, data_source=src)

        print(f"\n[1] Data: {res['data_source']}, n_samples={res['n_samples']}")
        print(f"\n[2] Discovered formula (distill): {res['formula_distill'] or 'None'}")
        print(f"    Discovered formula (direct):  {res['formula_direct'] or 'None'}")
        print(f"    Theory form (K integral):     {res['formula_theory']} R2={res['r2_theory']:.4f} BIC={res['bic_theory']:.1f}")
        print(f"    Direct poly BIC={res['bic_direct']:.1f}, R2={res['r2_direct']:.4f}")

        tm = res["theory_match"]
        print(f"\n[3] Theory Match (K(z) integral form)")
        print(f"    Score: {tm['score']:.2f} / 1.0")
        print(f"    {tm['details']}")
        theory_wins_bic = res.get("bic_theory", np.inf) < res.get("bic_direct", np.inf)
        verdict = "THEORY CONSISTENT" if (tm["score"] >= 0.5 or theory_wins_bic) else "THEORY INCONSISTENT"
        print(f"\n[4] Verdict: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
