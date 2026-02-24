"""
Publication-quality RAR (Radial Acceleration Relation) Trophy Plot.

Plots BOTH:
- Free fit: McGaugh formula with g0 from optimization (data's answer).
- Physical prior: McGaugh with g0=3700 (a0 = 1.2e-10 m/s^2).
The divergence between them is the science — where the formula doesn't perfectly fit the dataset.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.experiments.discovery_rar import run_rar_discovery
from axiom_os.layers.meta_kernel import KM_S_SQ_PER_KPC_TO_MS2


def _formula_curve(g_bar: np.ndarray, g0: float) -> np.ndarray:
    """g_obs = g_bar / (1 - exp(-sqrt(g_bar/g0))) in linear space."""
    x = np.maximum(g_bar, 1e-14)
    denom = 1.0 - np.exp(-np.sqrt(x / (g0 + 1e-14)))
    denom = np.maximum(denom, 0.01)
    return x / denom


def main():
    out_path = ROOT / "axiom_os" / "paper_rar_trophy_plot.png"

    print("Running RAR discovery...")
    res = run_rar_discovery(n_galaxies=175, epochs=800)
    if "error" in res:
        print(f"Error: {res['error']}")
        return

    g_bar = res["g_bar"]
    log_g_bar = res["log_g_bar"]
    g_obs_pred_mlp = res["g_obs_pred"]

    cry = res.get("crystallized", {})
    g0_free = cry.get("g_dagger") or cry.get("a0_or_gdagger")
    g0_prior = cry.get("g_dagger_prior", 3700.0)
    pred_prior = cry.get("pred_prior")
    pred_free = None
    for r in cry.get("best3", []):
        if len(r) >= 4 and r[0] == "RAR McGaugh":
            pred_free = r[3]
            break

    if g0_free is None:
        g0_free = 500.0
    if pred_prior is None:
        pred_prior = _formula_curve(g_bar, g0_prior)
    if pred_free is None:
        pred_free = _formula_curve(g_bar, g0_free)

    idx = np.argsort(log_g_bar)
    lg_bar_s = log_g_bar[idx]
    lg_obs_mlp_s = np.log10(g_obs_pred_mlp[idx] + 1e-14)
    lg_free_s = np.log10(pred_free[idx] + 1e-14)
    lg_prior_s = np.log10(pred_prior[idx] + 1e-14)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(log_g_bar, res["log_g_obs"], s=12, c="0.5", alpha=0.6, label="SPARC", zorder=1)
        ax.plot(lg_bar_s, lg_obs_mlp_s, "b-", lw=2.5, label="MLP prediction", zorder=2)
        ax.plot(lg_bar_s, lg_free_s, "g-", lw=2, alpha=0.9, label=f"McGaugh (free fit, g0={g0_free:.0f})", zorder=3)
        ax.plot(lg_bar_s, lg_prior_s, "r--", lw=2, label=f"McGaugh (prior g0={g0_prior:.0f})", zorder=4)
        ax.plot(lg_bar_s, lg_bar_s, "k:", lw=1, alpha=0.7, label="1:1 (g_obs = g_bar)", zorder=0)

        eq_text = r"$g_{\mathrm{obs}} = \frac{g_{\mathrm{bar}}}{1 - e^{-\sqrt{g_{\mathrm{bar}}/g_0}}}$"
        lines = [eq_text, f"Free fit: $g_0$ = {g0_free:.0f} (data)", f"Prior: $g_0$ = {g0_prior:.0f} ($a_0 \\approx 1.2\\times 10^{-10}$ m/s$^2$)"]
        if abs(g0_free - g0_prior) > 100:
            lines.append("Divergence = where McGaugh does not perfectly fit.")
        ax.text(0.05, 0.95, "\n".join(lines),
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

        ax.set_xlabel(r"$\log_{10}(g_{\mathrm{bar}})$ [(km/s)$^2$/kpc]", fontsize=12)
        ax.set_ylabel(r"$\log_{10}(g_{\mathrm{obs}})$ [(km/s)$^2$/kpc]", fontsize=12)
        ax.set_title("RAR — Free fit vs physical prior (divergence = science)", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Trophy plot saved: {out_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
