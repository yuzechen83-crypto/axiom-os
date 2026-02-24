"""
Inverse Projection (Theory Repair) - Discovery Mode

Instead of fixed models, train a Free-Form Projection MLP to learn:
  Sigma_halo = f(Sigma_baryon, r, ...)

Then use Discovery Engine to find the symbolic form of f.
Compare discovered K_eff with theoretical K(z) ~ 1/sqrt(1-(r/L)^2).
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import (
    load_sparc_galaxy,
    load_sparc_multi,
    get_available_sparc_galaxies,
    SPARC_GALAXIES,
)
from axiom_os.engine import DiscoveryEngine


def _sigma_proxy(V_sq: np.ndarray, r: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Surface density proxy: Sigma ~ V^2 / (r^2 + eps)"""
    r_safe = np.maximum(r, 0.1)
    return V_sq / (r_safe**2 + eps)


def _build_training_data(n_galaxies: int = 175, use_components: bool = False) -> tuple:
    """
    Build (Sigma_baryon, r, Sigma_halo) from SPARC.
    use_components: if True, also return Sigma_gas, Sigma_disk, Sigma_bulge for Discovery.
    """
    avail = get_available_sparc_galaxies()
    names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]
    all_R, all_Sb, all_Sh = [], [], []
    all_Sg, all_Sd, all_Sb_comp = [], [], []
    for name in names:
        d = load_sparc_galaxy(name, use_mock_if_fail=True, use_real=True)
        R = d["R"]
        V_gas_sq = d["V_gas"]**2
        V_disk_sq = d["V_disk"]**2
        V_bulge_sq = d["V_bulge"]**2
        V_bary_sq = V_gas_sq + V_disk_sq + V_bulge_sq
        V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1.0)
        V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0.0)
        Sigma_baryon = _sigma_proxy(V_bary_sq, R)
        Sigma_halo = _sigma_proxy(V_def_sq, R)
        all_R.append(R)
        all_Sb.append(Sigma_baryon)
        all_Sh.append(Sigma_halo)
        if use_components:
            all_Sg.append(_sigma_proxy(V_gas_sq, R))
            all_Sd.append(_sigma_proxy(V_disk_sq, R))
            all_Sb_comp.append(_sigma_proxy(V_bulge_sq, R))
    R_all = np.concatenate(all_R)
    Sb_all = np.concatenate(all_Sb)
    Sh_all = np.concatenate(all_Sh)
    if use_components:
        return R_all, Sb_all, Sh_all, names, np.concatenate(all_Sg), np.concatenate(all_Sd), np.concatenate(all_Sb_comp)
    return R_all, Sb_all, Sh_all, names


class FreeFormProjectionMLP(nn.Module):
    """
    Black-box kernel: [Sigma_baryon, r] -> Sigma_halo.
    Output: softplus for positivity (monotonic mass).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r: torch.Tensor, sigma_baryon: torch.Tensor) -> torch.Tensor:
        x = torch.stack([r, sigma_baryon], dim=-1)
        out = self.mlp(x)
        return torch.nn.functional.softplus(out).squeeze(-1)


def run_discovery_mode(
    n_galaxies: int = 50,
    epochs: int = 500,
    hidden_dim: int = 64,
) -> dict:
    """
    Task 1: Train Free-Form Projection MLP.
    Task 2: Discovery on learned (Sigma_baryon, Sigma_halo) mapping.
    Task 3: K_eff plot and analysis.
    """
    out = _build_training_data(n_galaxies)
    R, Sb, Sh, names = out[0], out[1], out[2], out[3]
    n = len(R)
    scale_sb = np.max(Sb) + 1e-8
    scale_sh = np.max(Sh) + 1e-8
    Sb_norm = Sb / scale_sb
    Sh_norm = Sh / scale_sh

    R_t = torch.from_numpy(R).float()
    Sb_t = torch.from_numpy(Sb_norm).float()
    Sh_t = torch.from_numpy(Sh_norm).float()

    model = FreeFormProjectionMLP(hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(R_t, Sb_t)
        loss = ((pred - Sh_t) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

    with torch.no_grad():
        Sh_pred = model(R_t, Sb_t).numpy() * scale_sh

    mse = float(np.mean((Sh_pred - Sh) ** 2))
    r2_fit = 1 - np.sum((Sh_pred - Sh) ** 2) / (np.sum((Sh - Sh.mean()) ** 2) + 1e-12)

    # Task 2: Discovery on (Sigma_baryon, r) -> Sigma_halo
    X = np.column_stack([Sb, R])
    y = Sh_pred
    engine = DiscoveryEngine(use_pysr=False)
    formula, pred_disc, _ = engine.discover_multivariate(
        X, y, var_names=["Sigma_baryon", "r"], selector="bic",
    )

    # Theory forms: Sigma_halo = K * Sigma_baryon / r^alpha, etc.
    formula_theory, r2_theory = _fit_theory_forms(Sb, R, Sh_pred)

    # Task 3: K_eff = Sigma_halo / Sigma_baryon
    eps = 1e-6
    K_eff = Sh_pred / (Sb + eps)
    p99 = np.nanpercentile(K_eff[np.isfinite(K_eff)], 99) if np.any(np.isfinite(K_eff)) else 1.0
    K_eff = np.clip(K_eff, 0, max(p99, 1e-6))

    L_theory = 10.2
    theory_curve = 1.0 / np.sqrt(np.maximum(1 - (R / L_theory) ** 2, 0.01))

    return {
        "R": R,
        "Sb": Sb,
        "Sh": Sh,
        "Sh_pred": Sh_pred,
        "K_eff": K_eff,
        "formula": formula,
        "formula_theory": formula_theory,
        "r2_theory": r2_theory,
        "theory_curve": theory_curve,
        "n_galaxies": len(names),
        "n_samples": n,
        "mse": mse,
        "r2_fit": r2_fit,
    }


def _fit_theory_forms(Sb: np.ndarray, R: np.ndarray, Sh: np.ndarray) -> tuple:
    """Fit Sigma_halo = C * Sigma_baryon / r^alpha (projection form)."""
    r_safe = np.maximum(R, 0.1)
    best_r2, best_form = -1e9, "None"
    for alpha in [0, 1, 2]:
        if alpha == 0:
            rho = Sb
        else:
            rho = Sb / (r_safe**alpha + 1e-6)
        scale = np.sum(Sh * rho) / (np.sum(rho**2) + 1e-12)
        pred = scale * rho
        mse = np.mean((Sh - pred) ** 2)
        r2 = 1 - np.sum((Sh - pred) ** 2) / (np.sum((Sh - Sh.mean()) ** 2) + 1e-12)
        if r2 > best_r2:
            best_r2, best_form = r2, f"C*Sigma_baryon/r^{alpha}" if alpha > 0 else "C*Sigma_baryon"
    return best_form, float(best_r2)


def _plot_and_save(res: dict, out_path: Path) -> None:
    """K_eff vs r plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        R, K_eff = res["R"], res["K_eff"]
        theory = res["theory_curve"]

        axes[0].scatter(R, K_eff, alpha=0.3, s=5, c="blue", label="K_eff (learned)")
        r_bins = np.linspace(0, R.max(), 20)
        k_binned = []
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
            if mask.sum() > 0:
                k_binned.append(np.median(K_eff[mask]))
            else:
                k_binned.append(np.nan)
        r_mid = (r_bins[:-1] + r_bins[1:]) / 2
        axes[0].plot(r_mid, k_binned, "r-", lw=2, label="K_eff median")
        axes[0].plot(r_mid, np.interp(r_mid, np.linspace(0, R.max(), len(theory)), theory), "g--", lw=1.5, label="Theory 1/sqrt(1-(r/L)^2)")
        axes[0].set_xlabel("r (kpc)")
        axes[0].set_ylabel("K_eff")
        axes[0].set_title("Effective Kernel: Sigma_halo = K_eff * Sigma_baryon")
        axes[0].legend()
        axes[0].set_ylim(0, min(np.percentile(K_eff, 99) * 1.2, 50))

        axes[1].hist(K_eff, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        axes[1].axvline(np.median(K_eff), color="red", linestyle="--", label=f"median={np.median(K_eff):.2f}")
        axes[1].set_xlabel("K_eff")
        axes[1].set_ylabel("Count")
        axes[1].set_title("K_eff distribution")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot skip: {e}")


def main():
    print("=" * 70)
    print("Inverse Projection (Theory Repair) - Discovery Mode")
    print("=" * 70)
    print("\nTask 1: Free-Form MLP [Sigma_baryon, r] -> Sigma_halo")
    print("Task 2: Symbolic Discovery on learned mapping")
    print("Task 3: K_eff plot vs theoretical K(z)")
    print("-" * 70)

    res = run_discovery_mode(n_galaxies=50, epochs=500)

    print(f"\n[1] Data: {res['n_galaxies']} galaxies, {res['n_samples']} samples")
    print(f"    MLP fit: MSE={res.get('mse',0):.2e}, R2={res.get('r2_fit',0):.4f}")
    print(f"\n[2] Discovered formula: {res['formula'] or 'None'}")
    print(f"    Theory form (K*Sigma_baryon/r^a): {res['formula_theory']} R2={res['r2_theory']:.4f}")

    K_eff = res["K_eff"]
    print(f"\n[3] K_eff analysis:")
    print(f"    K_eff = Sigma_halo / Sigma_baryon")
    print(f"    median(K_eff) = {np.median(K_eff):.2f}")
    print(f"    mean(K_eff) = {np.mean(K_eff):.2f}")
    print(f"    std(K_eff) = {np.std(K_eff):.2f}")
    if np.std(K_eff) < 0.5 * np.mean(K_eff):
        print("    -> K_eff ~ constant: hidden scaling (unlikely)")
    else:
        print("    -> K_eff varies with r: projection/spread")

    out = ROOT / "axiom_os" / "discovery_K_eff_plot.png"
    _plot_and_save(res, out)
    print(f"\n[4] Plot saved: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
