"""
Diagnose why free fit gives g0=321 instead of 3700.

Step 1: Data coverage (g_bar range, counts in low/mid/high).
Step 2: Residual plot and binned stats for g0=321 vs g0=3700.
Step 3: Compare with McGaugh 2016 — see docstring at end.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import load_sparc_rar, get_available_sparc_galaxies, SPARC_GALAXIES
from axiom_os.experiments.discovery_rar import run_rar_g0_diagnostic


def main():
    n_galaxies = 50
    avail = get_available_sparc_galaxies()
    n = min(n_galaxies, len(avail) if avail else len(SPARC_GALAXIES))
    g_bar, g_obs, names = load_sparc_rar(n_galaxies=n, use_mock_if_fail=True, use_real=True)
    if g_bar is None or len(g_bar) < 10:
        print("Insufficient RAR data.")
        return

    # Step 1: Data coverage
    n_pts = len(g_bar)
    n_lo = int((g_bar < 1000).sum())
    n_hi = int((g_bar > 10000).sum())
    print("=" * 60)
    print("STEP 1: DATA COVERAGE")
    print("=" * 60)
    print(f"g_bar range: {g_bar.min():.1f} – {g_bar.max():.1f} (km/s)^2/kpc")
    print(f"Data points: {n_pts}")
    print(f"g_bar < 1000:  {n_lo} ({100*n_lo/n_pts:.1f}%)  [low acceleration, most sensitive to g0]")
    print(f"g_bar > 10000: {n_hi} ({100*n_hi/n_pts:.1f}%)")
    if n_lo < 0.1 * n_pts:
        print("-> Few low-g_bar points: optimizer may push g0 down to fit the majority (mid/high) region.")
    print()

    # Step 2: Residual analysis (g0=321 vs g0=3700)
    diag = run_rar_g0_diagnostic(g_bar, g_obs, g0_free=321.0, g0_prior=3700.0)
    print("STEP 2: RESIDUALS BY g_bar BIN (log10(g_bar) bins)")
    print("-" * 60)
    print(f"{'log10(g_bar)':<20} {'n':>6} {'res_free (mean±std)':<22} {'res_prior (mean±std)':<22}")
    print("-" * 60)
    for b in diag["bin_stats"]:
        rf = f"{b['mean_resid_free']:+.3f}±{b['std_resid_free']:.3f}"
        rp = f"{b['mean_resid_prior']:+.3f}±{b['std_resid_prior']:.3f}"
        print(f"[{b['log_g_bar_lo']:.2f}, {b['log_g_bar_hi']:.2f})  {b['n']:>6}   {rf:<22}   {rp:<22}")
    print("-" * 60)
    rf_all = diag["residual_free"]
    rp_all = diag["residual_prior"]
    print(f"Overall:  g0=321  mean_resid={np.mean(rf_all):+.4f}, std={np.std(rf_all):.4f}")
    print(f"          g0=3700 mean_resid={np.mean(rp_all):+.4f}, std={np.std(rp_all):.4f}")
    print("-> If g0=321 residuals small at low g_bar and large at high: data range / sampling.")
    print("-> If both similar: possible V_bary or units issue.")
    print()

    # Residual plot
    out_plot = ROOT / "axiom_os" / "rar_g0_diagnostic_residuals.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        lg = diag["log_g_bar"]
        ax.scatter(lg, diag["residual_free"], s=8, alpha=0.5, c="green", label="Residual (g0=321 free fit)")
        ax.scatter(lg, diag["residual_prior"], s=8, alpha=0.5, c="red", label="Residual (g0=3700 prior)")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.axvline(2.5, color="gray", ls=":", alpha=0.7)
        ax.axvline(3.5, color="gray", ls=":", alpha=0.7)
        ax.set_xlabel(r"$\log_{10}(g_{\mathrm{bar}})$ [(km/s)$^2$/kpc]")
        ax.set_ylabel("Residual (log10 g_obs - log10 g_pred)")
        ax.set_title("Why g0=321 vs 3700? Residuals by acceleration")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Residual plot saved: {out_plot}")
    except Exception as e:
        print(f"Plot skip: {e}")

    # Step 3: McGaugh 2016 comparison
    print()
    print("STEP 3: MCGAUGH 2016 COMPARISON")
    print("-" * 60)
    print("McGaugh 2016 data are public. To check if the difference is sampling:")
    print("  1. Download the same RAR table (e.g. SPARC/LTG catalog or McGaugh et al. 2016).")
    print("  2. Run this script and discovery_rar.py on that file (same g_bar, g_obs columns).")
    print("  3. If McGaugh data give g0 ≈ 3700 and our subset gives 321, the cause is sampling.")
    print("=" * 60)


if __name__ == "__main__":
    main()
