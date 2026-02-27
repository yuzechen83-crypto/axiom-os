"""
RAR Residual Analysis - log(g_obs) - log(g_pred) by Galaxy Group

Per-galaxy residuals from McGaugh RAR fit. Group by:
  - Gas vs Star (M_gas > M_star vs M_star > M_gas)
  - HighSB vs LowSB (median split of central surface density)

Output: per-galaxy residuals, 4-group box plot.

Run: python -m axiom_os.experiments.rar_residuals
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import (
    get_available_sparc_galaxies,
    SPARC_GALAXIES,
    compute_accelerations,
    _load_real_sparc,
)
from axiom_os.experiments.fit_meta_parameters import fit_g0_log_loss
from axiom_os.layers.meta_kernel import nu_mcgaugh, a0_si_from_g0


def _mass_proxy(d: dict) -> Tuple[float, float]:
    """M_gas proxy: sum(V_gas^2 * R), M_star proxy: sum((V_disk^2 + V_bulge^2) * R)."""
    R = np.asarray(d["R"], dtype=np.float64)
    r_safe = np.maximum(R, 0.1)
    M_gas = np.sum(d["V_gas"] ** 2 * r_safe)
    M_star = np.sum(d["V_disk"] ** 2 * r_safe + d["V_bulge"] ** 2 * r_safe)
    return float(M_gas), float(M_star)


def _central_surface_density(d: dict) -> float:
    """Central surface density proxy: V_bary^2/R^2 at inner points."""
    R = np.asarray(d["R"], dtype=np.float64)
    V_bary_sq = d["V_gas"] ** 2 + d["V_disk"] ** 2 + d["V_bulge"] ** 2
    r_safe = np.maximum(R, 0.1)
    sigma = V_bary_sq / (r_safe**2 + 1e-12)
    n_inner = max(1, len(R) // 4)
    return float(np.median(np.sort(sigma)[:n_inner]))


def _g_pred_mcgaugh(g_bar: np.ndarray, g0: float) -> np.ndarray:
    """g_obs = g_bar * nu(g_bar/g0). g0 in (km/s)^2/kpc."""
    nu = nu_mcgaugh(g_bar, g0)
    return g_bar * nu


def load_per_galaxy_with_groups(
    n_galaxies: int = 175,
    use_mock_if_fail: bool = True,
    use_real: bool = True,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, str, str, str]], np.ndarray, np.ndarray, str]:
    """
    Load per-galaxy (g_bar, g_obs) with group labels.
    Returns: list of (g_bar, g_obs, gas_star, high_low_sb, name), all_g_bar, all_g_obs
    gas_star: "Gas" | "Star"
    high_low_sb: "HighSB" | "LowSB"
    """
    avail = get_available_sparc_galaxies()
    names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]

    per_gal = []
    all_g_bar, all_g_obs = [], []

    for name in names:
        d = _load_real_sparc(name) if use_real else None
        if d is None and use_mock_if_fail:
            from axiom_os.datasets.sparc import load_sparc_galaxy
            d = load_sparc_galaxy(name, use_mock_if_fail=True, use_real=False)
        if d is None:
            continue
        g_bar, g_obs, _ = compute_accelerations(d)
        if len(g_bar) < 5:
            continue

        M_gas, M_star = _mass_proxy(d)
        sb = _central_surface_density(d)
        gas_star = "Gas" if M_gas > M_star else "Star"
        per_gal.append((sb, gas_star, name, g_bar, g_obs))
        all_g_bar.append(g_bar)
        all_g_obs.append(g_obs)

    if not per_gal:
        return [], np.array([]), np.array([]), "NONE"

    # Data source: REAL if we used _load_real_sparc (no mock fallback)
    data_source = "REAL (Rotmod_LTG)" if use_real and not use_mock_if_fail else (
        "REAL (Rotmod_LTG)" if per_gal and _load_real_sparc(per_gal[0][2]) is not None else "MOCK"
    )

    # Median split for HighSB vs LowSB
    sbs = [x[0] for x in per_gal]
    median_sb = np.median(sbs)

    out = []
    for sb, gas_star, name, g_bar, g_obs in per_gal:
        high_low = "HighSB" if sb >= median_sb else "LowSB"
        out.append((g_bar, g_obs, gas_star, high_low, name))

    g_bar_all = np.concatenate(all_g_bar)
    g_obs_all = np.concatenate(all_g_obs)
    return out, g_bar_all, g_obs_all, data_source


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAR residual analysis by Gas/Star, HighSB/LowSB")
    parser.add_argument("--n-galaxies", type=int, default=175, help="Max galaxies to load")
    parser.add_argument("--allow-mock", action="store_true", help="Fall back to mock if real SPARC unavailable (default: fail)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("RAR Residual Analysis")
    print("residual = log10(g_obs) - log10(g_pred)")
    print("Groups: Gas/Star x HighSB/LowSB")
    print("=" * 60)

    # Default: real data only. Use axiom_os/data/Rotmod_LTG.zip if present.
    per_gal, g_bar_all, g_obs_all, data_source = load_per_galaxy_with_groups(
        n_galaxies=args.n_galaxies,
        use_mock_if_fail=args.allow_mock,
        use_real=True,
    )

    if not per_gal:
        print("\n[!] No galaxies loaded.")
        print("    Ensure axiom_os/data/Rotmod_LTG.zip exists, or run with --allow-mock for mock fallback.")
        return

    print(f"\n[0] Data: {len(per_gal)} galaxies, {len(g_bar_all)} points — {data_source}")
    if data_source == "MOCK":
        print("    [!] Using MOCK data — results have no scientific meaning. Use real Rotmod for validation.")

    # Fit g0 (log MSE) on full data
    res = fit_g0_log_loss(g_bar_all, g_obs_all)
    g0 = res["g0"]
    a0_si = res.get("a0_si", a0_si_from_g0(g0))
    print(f"\n[1] Fitted g0 = {g0:.1f} (km/s)^2/kpc")
    print(f"    a0 = {a0_si:.2e} m/s^2")

    eps = 1e-14
    log_g_obs_all = np.log10(g_obs_all + eps)
    g_pred_all = _g_pred_mcgaugh(g_bar_all, g0)
    log_g_pred_all = np.log10(g_pred_all + eps)
    residual_all = log_g_obs_all - log_g_pred_all
    print(f"\n[2] Global: mean residual = {np.mean(residual_all):.4f}, std = {np.std(residual_all):.4f}")

    # Per-galaxy residuals, grouped
    groups: Dict[str, List[float]] = {
        "Gas+HighSB": [],
        "Gas+LowSB": [],
        "Star+HighSB": [],
        "Star+LowSB": [],
    }

    for g_bar, g_obs, gas_star, high_low, name in per_gal:
        g_pred = _g_pred_mcgaugh(g_bar, g0)
        log_g_obs = np.log10(g_obs + eps)
        log_g_pred = np.log10(g_pred + eps)
        resids = (log_g_obs - log_g_pred).tolist()
        key = f"{gas_star}+{high_low}"
        groups[key].extend(resids)

    print(f"\n[3] Per-group residual counts:")
    for k, v in groups.items():
        print(f"    {k}: n={len(v)}")

    # Save residuals per group
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    summary = {
        "data_source": data_source,
        "g0": g0,
        "a0_si": float(a0_si),
        "n_galaxies": len(per_gal),
        "n_points": len(g_bar_all),
        "mean_residual": float(np.mean(residual_all)),
        "std_residual": float(np.std(residual_all)),
        "groups": {k: {"n": len(v), "mean": float(np.mean(v)) if v else np.nan, "std": float(np.std(v)) if v else np.nan} for k, v in groups.items()},
    }
    with open(out_dir / "rar_residuals.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[4] Saved: {out_dir / 'rar_residuals.json'}")

    # Box plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(groups.keys())
        data = [groups[k] for k in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        ax.axhline(0, color="gray", ls="--", alpha=0.7)
        ax.set_ylabel("log10(g_obs) - log10(g_pred)")
        ax.set_title("RAR Residuals by Galaxy Group")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / "rar_residuals_boxplot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Plot: {plot_path}")
    except ImportError:
        print("    [matplotlib not available, skipping box plot]")

    print("=" * 60)


if __name__ == "__main__":
    main()
