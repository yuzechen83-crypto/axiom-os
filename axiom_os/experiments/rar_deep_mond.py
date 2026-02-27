"""
RAR Deep MOND Limit - Extreme Low Acceleration Test

Theory prediction in deep MOND regime (g_bar << a₀):
  g_obs = sqrt(a₀ · g_bar)

Filter SPARC data: g_bar < 0.01×a₀ (strongest regime).
Check if points fall on the sqrt line. Large scatter → ν(y) may need revision at low g.

Data: RAR.mrt (canonical) or Rotmod_LTG (per-galaxy).

Run: python -m axiom_os.experiments.rar_deep_mond
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import load_sparc_rar_mrt, KM_S_SQ_PER_KPC_TO_MS2
from axiom_os.layers.meta_kernel import a0_si_from_g0
from axiom_os.experiments.fit_meta_parameters import fit_g0_log_loss

A0_SI = 1.2e-10  # m/s²
G_BAR_THRESH = 0.01 * A0_SI  # g_bar < this → deep MOND


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAR deep MOND limit test")
    parser.add_argument("--use-rotmod", action="store_true",
                        help="Use Rotmod per-galaxy (fallback if RAR.mrt empty)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="g_bar < threshold×a₀ (default 0.01)")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Save scatter plot")
    args = parser.parse_args()

    thresh = args.threshold * A0_SI

    print("=" * 60)
    print("RAR Deep MOND Limit: g_obs = sqrt(a0*g_bar)")
    print("=" * 60)
    print(f"\nRegime: g_bar < {args.threshold}*a0 = {thresh:.2e} m/s^2")

    g_bar, g_obs = load_sparc_rar_mrt()
    data_src = "RAR.mrt"

    if len(g_bar) < 50 and args.use_rotmod:
        from axiom_os.datasets.sparc import load_sparc_rar_real_only
        g_bar, g_obs, _ = load_sparc_rar_real_only(n_galaxies=175)
        data_src = "Rotmod_LTG"

    if len(g_bar) < 50:
        print(f"\n[!] Insufficient data: {len(g_bar)} points. Need RAR.mrt or Rotmod.")
        return

    # Convert to m/s² for threshold
    g_bar_si = g_bar * KM_S_SQ_PER_KPC_TO_MS2
    g_obs_si = g_obs * KM_S_SQ_PER_KPC_TO_MS2

    mask = g_bar_si < thresh
    n_deep = int(np.sum(mask))

    if n_deep < 10:
        print(f"\n[!] Only {n_deep} points in deep MOND regime. Try --threshold 0.05")
        thresh = 0.05 * A0_SI
        mask = g_bar_si < thresh
        n_deep = int(np.sum(mask))
        if n_deep < 10:
            return

    gb = g_bar_si[mask]
    go = g_obs_si[mask]

    # Fit a₀ from full RAR for consistency
    res = fit_g0_log_loss(g_bar, g_obs)
    a0_fit = res["a0_si"]
    g0_fit = res["g0"]

    # Deep MOND prediction: g_obs_pred = sqrt(a₀ · g_bar)
    g_obs_pred = np.sqrt(a0_fit * gb)
    residual = np.log10(go + 1e-20) - np.log10(g_obs_pred + 1e-20)
    mae_log = float(np.mean(np.abs(residual)))
    rmse_log = float(np.sqrt(np.mean(residual**2)))

    print(f"\n[1] Data: {len(g_bar)} points ({data_src})")
    print(f"    Deep MOND (g_bar < {thresh:.2e} m/s^2): {n_deep} points")
    print(f"    a0 (from full RAR log MSE): {a0_fit:.4e} m/s^2")

    print(f"\n[2] Deep MOND fit: g_obs_pred = sqrt(a0*g_bar)")
    print(f"    Residual (log10): mean={np.mean(residual):.4f}, std={np.std(residual):.4f}")
    print(f"    MAE(log) = {mae_log:.4f}, RMSE(log) = {rmse_log:.4f}")

    # Correlation
    r = np.corrcoef(np.log10(go + 1e-20), np.log10(g_obs_pred + 1e-20))[0, 1]
    print(f"    Corr(log g_obs, log g_pred) = {r:.4f}")

    # Verdict
    if mae_log < 0.15 and r > 0.9:
        verdict = "PASS"
        msg = "Deep MOND limit consistent: points follow sqrt(a0*g_bar)."
    elif mae_log < 0.25:
        verdict = "MARGINAL"
        msg = "Moderate scatter; nu(y) at low g may need minor revision."
    else:
        verdict = "TENSION"
        msg = "Large scatter; nu(y) in deep MOND regime may need revision."

    print(f"\n[3] Verdict: {verdict}")
    print(f"    -> {msg}")

    # Save results
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_source": data_src,
            "n_total": int(len(g_bar)),
            "n_deep_mond": n_deep,
            "threshold": args.threshold,
            "a0_si": a0_fit,
        },
        "metrics": {
            "mae_log": mae_log,
            "rmse_log": rmse_log,
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual)),
            "correlation": float(r),
        },
        "verdict": verdict,
    }
    with open(out_dir / "rar_deep_mond.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n    Saved: {out_dir / 'rar_deep_mond.json'}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

            # Left: log g_obs vs log g_pred (1:1 line = deep MOND)
            ax = axes[0]
            log_go = np.log10(go + 1e-20)
            log_gp = np.log10(g_obs_pred + 1e-20)
            ax.scatter(log_gp, log_go, s=8, alpha=0.6, c="steelblue", edgecolors="none")
            lims = [min(log_go.min(), log_gp.min()) - 0.2, max(log_go.max(), log_gp.max()) + 0.2]
            ax.plot(lims, lims, "k--", lw=1.5, label="g_obs = sqrt(a0*g_bar)")
            ax.set_xlabel(r"$\log_{10} g_{pred}$ (m/s^2)")
            ax.set_ylabel(r"$\log_{10} g_{obs}$ (m/s^2)")
            ax.set_title("Deep MOND: g_obs vs sqrt(a₀·g_bar)")
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Right: residual histogram
            ax = axes[1]
            ax.hist(residual, bins=min(30, n_deep // 5), color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.set_xlabel(r"$\log_{10}(g_{obs}/g_{pred})$")
            ax.set_ylabel("Count")
            ax.set_title(f"Residual (n={n_deep})")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(out_dir / "rar_deep_mond.png", dpi=120, bbox_inches="tight")
            plt.close()
            print(f"    Plot: {out_dir / 'rar_deep_mond.png'}")
        except ImportError:
            pass

    print("=" * 60)


if __name__ == "__main__":
    main()
