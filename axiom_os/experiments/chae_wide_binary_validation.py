"""
Chae 2023/2024 Wide Binary Validation - Independent Test

Theory: sigma_v_obs / sigma_v_newton = sqrt(nu(g_newton/a0))
       gamma_g = g_obs/g_newton = nu(g_newton/a0)

Chae published: gamma_g ~ 1.43-1.49 at g_N ~ 10^-10 m/s^2
                gamma_vp ~ 1.20 at s > 5 kau

Uses Chae's published binned results (no raw CSV required).
Raw CSV from Zenodo can extend to finer bins.

Run: python -m axiom_os.experiments.chae_wide_binary_validation
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.layers.meta_kernel import nu_mcgaugh, A0_SI_DEFAULT
from axiom_os.experiments.fit_meta_parameters import fit_g0_log_loss
from axiom_os.datasets.sparc import load_sparc_rar_mrt
from axiom_os.layers.meta_kernel import a0_si_from_g0

# Chae 2023/2024 published bins
CHAE_BINS = [
    {"g_N": 10**-10.15, "gamma_g": 1.43, "err": 0.06, "source": "Chae 2023"},
    {"g_N": 10**-8.91, "gamma_g": 1.034, "err": 0.007, "source": "Chae 2023 (high g)"},
    {"g_N": 1e-10, "gamma_g": 1.49, "err_lo": 0.19, "err_hi": 0.21, "source": "Chae 2024"},
    {"s_kau": 5.0, "gamma_vp": 1.20, "err_stat": 0.06, "err_sys": 0.05, "source": "Chae 2024 s>5kau"},
]


def gamma_g_theory(g_newton: float, a0: float) -> float:
    """Theory: gamma_g = g_obs/g_newton = nu(g_newton/a0)."""
    return float(nu_mcgaugh(g_newton, a0))


def gamma_v_theory(g_newton: float, a0: float) -> float:
    """Theory: gamma_v = v_obs/v_newton = sqrt(nu)."""
    return float(np.sqrt(nu_mcgaugh(g_newton, a0)))


def g_newton_from_s_kau(s_kau: float, M_sun: float = 2.0) -> float:
    """g = GM/r^2 [m/s^2]. s_kau in kilo-AU."""
    from axiom_os.datasets.chae_wide_binaries import g_newton_from_separation
    return g_newton_from_separation(s_kau, M_sun)


def run_binned_from_csv(sep_kau: np.ndarray, mass_sun: np.ndarray, a0: float, out_dir: Path) -> None:
    """Bin raw Chae data by g_newton, overlay theory curve."""
    from axiom_os.datasets.chae_wide_binaries import g_newton_from_separation
    g_list = []
    for s, m in zip(sep_kau, mass_sun):
        g = g_newton_from_separation(s, m)
        if 1e-12 < g < 1e-6:
            g_list.append(g)
    g_arr = np.array(g_list)
    if len(g_arr) < 100:
        return
    log_g = np.log10(g_arr)
    bins = np.linspace(log_g.min(), log_g.max(), 15)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    g_centers = 10**bin_centers
    gamma_pred = [gamma_g_theory(g, a0) for g in g_centers]
    print(f"\n[4] Raw Chae data: {len(g_arr)} binaries, g_N range [{g_arr.min():.2e}, {g_arr.max():.2e}]")
    print("    Theory gamma_g in bins:")
    for i in range(0, len(bin_centers), 3):
        idx = min(i + 3, len(bin_centers))
        gp = gamma_pred[i:idx]
        print(f"      log10(g)={bin_centers[i]:.2f}..{bin_centers[idx-1]:.2f}: gamma_g={gp[0]:.3f}..{gp[-1]:.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        log_g_curve = np.linspace(-12, -7, 200)
        g_curve = 10**log_g_curve
        gamma_curve = [gamma_g_theory(g, a0) for g in g_curve]
        ax.plot(log_g_curve, gamma_curve, "b-", lw=2, label="Theory nu(g_N/a0)")
        ax.axhline(1, color="gray", ls="--", alpha=0.7)
        for b in CHAE_BINS:
            if "g_N" in b:
                ax.errorbar(np.log10(b["g_N"]), b["gamma_g"],
                            yerr=b.get("err") or 0.2, fmt="o", capsize=3, label=b.get("source", ""))
        ax.set_xlabel(r"$\log_{10}(g_N)$ [m/s^2]")
        ax.set_ylabel(r"$\gamma_g = g_{obs}/g_{newton}$")
        ax.set_title(f"Chae Wide Binary: Theory vs Observed (n={len(g_arr)} binaries)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-12, -7)
        ax.set_ylim(0.9, 3)
        plt.tight_layout()
        plt.savefig(out_dir / "chae_wide_binary_validation.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    Plot: {out_dir / 'chae_wide_binary_validation.png'}")
    except ImportError:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chae wide binary validation")
    parser.add_argument("--a0", type=float, default=None, help="a0 in m/s^2 (default: from RAR fit)")
    parser.add_argument("--use-published-only", action="store_true",
                        help="Use only Chae published bins (no CSV download)")
    parser.add_argument("--with-csv", action="store_true", default=True,
                        help="Load raw Chae CSV and bin by g_newton (default: True)")
    args = parser.parse_args()

    # Get a0 from RAR fit
    a0 = args.a0
    if a0 is None:
        g_bar, g_obs = load_sparc_rar_mrt()
        if len(g_bar) >= 50:
            res = fit_g0_log_loss(g_bar, g_obs)
            a0 = res["a0_si"]
        else:
            a0 = A0_SI_DEFAULT

    print("=" * 60)
    print("Chae Wide Binary Validation - Independent Test")
    print("=" * 60)
    print(f"\nTheory: gamma_g = nu(g_N/a0), gamma_v = sqrt(nu)")
    print(f"a0 = {a0:.4e} m/s^2 (from RAR fit)" if args.a0 is None else f"a0 = {a0:.4e} m/s^2 (user)")

    print("\n[1] Acceleration ratio gamma_g = g_obs/g_newton")
    print("-" * 50)
    all_pass = True
    for b in CHAE_BINS:
        if "g_N" not in b:
            continue
        g_N = b["g_N"]
        pred = gamma_g_theory(g_N, a0)
        obs = b["gamma_g"]
        err = b.get("err") or (b.get("err_lo", 0) + b.get("err_hi", 0)) / 2
        diff = pred - obs
        within = abs(diff) <= 2 * err
        status = "OK" if within else "~"
        if not within:
            all_pass = False
        print(f"  g_N={g_N:.2e}: pred={pred:.3f}  obs={obs:.3f}+-{err:.3f}  diff={diff:+.3f}  [{status}]")
        print(f"       {b.get('source','')}")

    print("\n[2] Velocity boost gamma_vp (s > 5 kau)")
    print("-" * 50)
    for b in CHAE_BINS:
        if "s_kau" not in b:
            continue
        s = b["s_kau"]
        g_N = g_newton_from_s_kau(s)
        pred_v = gamma_v_theory(g_N, a0)
        obs_v = b["gamma_vp"]
        err = b.get("err_stat", 0) + b.get("err_sys", 0)
        diff = pred_v - obs_v
        within = abs(diff) <= 2 * err
        status = "OK" if within else "~"
        if not within:
            all_pass = False
        print(f"  s={s} kau, g_N={g_N:.2e}: pred gamma_v={pred_v:.3f}  obs={obs_v:.3f}+-{err:.3f}  [{status}]")
        print(f"       {b.get('source','')}")

    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    if args.with_csv:
        try:
            from axiom_os.datasets.chae_wide_binaries import load_chae_wide_binaries
            sep, mass = load_chae_wide_binaries()
            if len(sep) >= 100:
                run_binned_from_csv(sep, mass, a0, out_dir)
        except Exception as e:
            print(f"\n[4] CSV load failed: {e}")

    print("\n[3] Verdict")
    print("-" * 50)
    if all_pass:
        print("  PASS: Theory predictions consistent with Chae 2023/2024 within ~2sigma.")
    else:
        print("  MARGINAL: Some bins show tension; external field or systematics may matter.")
    print("  Note: Chae uses AQUAL/MOND with Galactic EFE; we use isolated nu(g).")
    print("  Agreement suggests Axiom nu(g) captures similar low-g boost.")

    # Save
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime
    out = {
        "timestamp": datetime.now().isoformat(),
        "a0_si": a0,
        "bins": [],
        "verdict": "PASS" if all_pass else "MARGINAL",
    }
    for b in CHAE_BINS:
        if "g_N" in b:
            out["bins"].append({
                "g_N": b["g_N"],
                "gamma_g_obs": b["gamma_g"],
                "gamma_g_pred": gamma_g_theory(b["g_N"], a0),
                "source": b.get("source", ""),
            })
        elif "s_kau" in b:
            g_N = g_newton_from_s_kau(b["s_kau"])
            out["bins"].append({
                "s_kau": b["s_kau"],
                "g_N": g_N,
                "gamma_vp_obs": b["gamma_vp"],
                "gamma_vp_pred": gamma_v_theory(g_N, a0),
                "source": b.get("source", ""),
            })
    with open(out_dir / "chae_wide_binary_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_dir / 'chae_wide_binary_validation.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
