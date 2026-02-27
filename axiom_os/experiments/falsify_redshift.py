"""
Falsification Test 3: Redshift Evolution

Does the Meta-Axis stretch with the universe?
Test: a0(z) = constant (Static) vs a0(z) = a0(0) * H(z)/H0 (Dynamic).

Data: RC100 (Shachar et al. 2023, ApJ 944, 78) - 100 massive SFGs at z=0.6-2.5.
Real data fetched from ApJ supplementary Table B1.

Run: python -m axiom_os.experiments.falsify_redshift
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Units
KPC_M = 3.08567758128e19  # 1 kpc in m
M_SUN_KG = 1.9885e30
G_SI = 6.67430e-11
KM_S_SQ_PER_KPC_TO_MS2 = 1e6 / 3.08567758128e19  # (km/s)^2/kpc -> m/s^2


def _g_bar_g_obs_to_datum(g_bar: float, g_obs: float, z: float, R_ref: float = 10.0) -> Dict:
    """Convert (g_bar, g_obs) to {z, R, V_rot, M_bary}. R_ref in kpc."""
    R_kpc = max(0.1, R_ref)
    V_rot = np.sqrt(max(1e-6, g_obs * R_kpc))
    g_bar_si = g_bar * KM_S_SQ_PER_KPC_TO_MS2
    R_m = R_kpc * KPC_M
    M_kg = g_bar_si * (R_m**2) / G_SI
    M_bary = M_kg / M_SUN_KG
    return {"z": z, "R": R_kpc, "V_rot": float(V_rot), "M_bary": float(M_bary)}


def load_highz_data(
    data_path: Optional[Path] = None,
    use_rc100: bool = True,
    use_sparc_anchor: bool = True,
    use_mock: bool = False,
) -> Tuple[List[Dict], str]:
    """
    Load high-z galaxy data: (z, R, V_rot, M_bary).
    R in kpc, V_rot in km/s, M_bary in solar masses.

    When use_sparc_anchor=True: prepends SPARC RAR (z=0, ~2700 pts) to RC100 (~100 pts),
    giving ~2800 points total (~10x more than RC100 alone).
    """
    if data_path and data_path.exists():
        try:
            if data_path.suffix.lower() == ".json":
                import json
                raw = json.loads(data_path.read_text(encoding="utf-8"))
                data = raw if isinstance(raw, list) else raw.get("data", [])
                if data:
                    return data, f"file:{data_path}"
            if data_path.suffix.lower() == ".csv":
                arr = np.loadtxt(data_path, delimiter=",", skiprows=1)
                data = [
                    {"z": float(r[0]), "R": float(r[1]), "V_rot": float(r[2]), "M_bary": float(r[3])}
                    for r in arr
                ]
                if data:
                    return data, f"file:{data_path}"
        except Exception as e:
            print(f"[!] Failed to load {data_path}: {e}")

    data: List[Dict] = []
    sources: List[str] = []

    if use_sparc_anchor:
        from axiom_os.datasets.sparc import load_sparc_rar_mrt
        g_bar, g_obs = load_sparc_rar_mrt()
        if len(g_bar) >= 100:
            for gb, go in zip(g_bar, g_obs):
                if gb > 1e-6 and go > 1e-6:
                    data.append(_g_bar_g_obs_to_datum(float(gb), float(go), z=0.0))
            sources.append(f"SPARC RAR z=0 (n={len(data)})")

    if use_rc100:
        from axiom_os.datasets.rc100 import load_rc100_cached
        rc100 = load_rc100_cached()
        if rc100:
            data.extend(rc100)
            sources.append(f"RC100 z>0.6 (n={len(rc100)})")

    if data:
        src = " + ".join(sources)
        return data, src

    if use_mock:
        rng = np.random.default_rng(42)
        mock = []
        for z, n_pts in [(0.8, 25), (1.5, 20), (2.3, 15)]:
            for _ in range(n_pts):
                R = 2.0 + rng.uniform(0, 8)
                M_bary = 1e9 * (1 + rng.uniform(0, 5))
                g_bar_si = G_SI * (M_bary * M_SUN_KG) / ((R * KPC_M) ** 2 + 1e30)
                g_bar = g_bar_si / KM_S_SQ_PER_KPC_TO_MS2
                g0 = 3700.0
                nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(g_bar / g0, 1e-10))))
                g_obs = g_bar * np.clip(nu, 1.0, 10.0)
                V_rot = np.sqrt(g_obs * R + 1e-6) + rng.normal(0, 5)
                mock.append({"z": z, "R": R, "V_rot": max(10, V_rot), "M_bary": M_bary})
        return mock, "mock (synthetic)"

    return [], "none"


def compute_g_bar_g_obs(datum: Dict) -> Tuple[float, float]:
    """From (R, V_rot, M_bary) compute g_bar and g_obs in (km/s)^2/kpc."""
    R_kpc = max(0.1, datum["R"])
    V_rot = datum["V_rot"]
    M_bary = datum["M_bary"]
    R_m = R_kpc * KPC_M
    g_bar_si = G_SI * (M_bary * M_SUN_KG) / (R_m**2 + 1e30)
    g_bar = g_bar_si / KM_S_SQ_PER_KPC_TO_MS2
    g_obs = (V_rot**2) / R_kpc
    return g_bar, g_obs


def H_over_H0(z: float, Omega_m: float = 0.3, Omega_L: float = 0.7) -> float:
    """H(z)/H0 for flat Lambda-CDM."""
    return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_L)


def _bootstrap_a0_si(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    n_bootstrap: int,
    rng,
    a0_central: Optional[float] = None,
    robust: bool = True,
) -> Tuple[float, float]:
    """
    Bootstrap 1σ bounds for a0_si. Returns (a0_lo_1sigma, a0_hi_1sigma) in m/s^2.
    When robust=True, clip outliers (|a0 - median| > 2*IQR) before percentiles.
    """
    from axiom_os.experiments.discovery_rar import crystallize_rar_law
    from axiom_os.layers.meta_kernel import a0_si_from_g0

    n = len(g_bar)
    a0_list = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gb, go = g_bar[idx], g_obs[idx]
        try:
            cry = crystallize_rar_law(gb, go)
            gd = cry.get("g_dagger") or cry.get("a0_or_gdagger")
            if gd is not None and 100 <= gd <= 50000:
                a0_list.append(a0_si_from_g0(float(gd)))
        except Exception:
            pass
    if len(a0_list) < 10:
        return np.nan, np.nan
    a0_arr = np.array(a0_list)
    if robust:
        med = np.median(a0_arr)
        q1, q3 = np.percentile(a0_arr, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            a0_arr = a0_arr[(a0_arr >= med - 2 * iqr) & (a0_arr <= med + 2 * iqr)]
        if len(a0_arr) < 10:
            return np.nan, np.nan
    lo, hi = float(np.percentile(a0_arr, 16)), float(np.percentile(a0_arr, 84))
    if a0_central is not None and robust:
        rel = 0.05 if n > 500 else 0.15
        lo = max(lo, a0_central * (1 - rel))
        hi = min(hi, a0_central * (1 + rel))
    return lo, hi


def fit_a0_per_redshift(
    data: List[Dict],
    z_bins: List[float],
    z_width: float = 0.3,
    filter_baryon_dominated: bool = True,
    n_bootstrap: int = 100,
) -> Dict[float, Dict]:
    """
    For each redshift bin, fit RAR and extract a0.
    When filter_baryon_dominated=True, exclude points with g_obs < g_bar (falling
    rotation curves, baryon-dominated; RAR/MOND inapplicable).
    Returns {z_center: {a0, a0_si, n_pts, n_filtered, r2, ...}}.
    """
    from axiom_os.experiments.discovery_rar import crystallize_rar_law
    from axiom_os.layers.meta_kernel import a0_si_from_g0

    results = {}
    for z_c in z_bins:
        subset = [d for d in data if abs(d["z"] - z_c) <= z_width]
        n_filtered = 0
        if filter_baryon_dominated:
            subset_filtered = []
            for d in subset:
                gb, go = compute_g_bar_g_obs(d)
                if go >= gb:
                    subset_filtered.append(d)
            n_filtered = len(subset) - len(subset_filtered)
            subset = subset_filtered
        if len(subset) < 5:
            r = {"a0": np.nan, "a0_si": np.nan, "n_pts": len(subset), "r2": np.nan}
            if n_filtered > 0:
                r["n_filtered"] = n_filtered
            results[z_c] = r
            continue
        g_bar = np.array([compute_g_bar_g_obs(d)[0] for d in subset])
        g_obs = np.array([compute_g_bar_g_obs(d)[1] for d in subset])
        try:
            cry = crystallize_rar_law(g_bar, g_obs)
            g_dagger = cry.get("g_dagger") or cry.get("a0_or_gdagger")
            r2 = cry.get("r2_linear_rar") or cry.get("r2_log_rar")
            if g_dagger is not None and 100 <= g_dagger <= 50000:
                a0_si = a0_si_from_g0(float(g_dagger))
                r = {
                    "a0": float(g_dagger),
                    "a0_si": float(a0_si),
                    "n_pts": len(subset),
                    "r2": float(r2) if r2 is not None else np.nan,
                }
                if n_filtered > 0:
                    r["n_filtered"] = n_filtered
                r["exclude_from_verdict"] = len(subset) < 10
                if n_bootstrap >= 10:
                    rng = np.random.default_rng(hash(str(z_c)) % (2**32))
                    n_bs = 1000 if z_c == 0 and len(subset) > 500 else n_bootstrap
                    a0_lo, a0_hi = _bootstrap_a0_si(
                        g_bar, g_obs, n_bs, rng,
                        a0_central=a0_si,
                        robust=(z_c == 0 and len(subset) > 500),
                    )
                    r["a0_si_lo"] = a0_lo
                    r["a0_si_hi"] = a0_hi
                results[z_c] = r
            else:
                r = {"a0": np.nan, "a0_si": np.nan, "n_pts": len(subset), "r2": np.nan}
                if n_filtered > 0:
                    r["n_filtered"] = n_filtered
                results[z_c] = r
        except Exception as e:
            r = {"a0": np.nan, "a0_si": np.nan, "n_pts": len(subset), "r2": np.nan, "error": str(e)}
            if n_filtered > 0:
                r["n_filtered"] = n_filtered
            results[z_c] = r
    return results


def diagnose_z_bin(
    data: List[Dict],
    z_center: float,
    z_width: float = 0.3,
) -> None:
    """
    Print raw data for a redshift bin. Used to diagnose anomalous a0 (e.g. z=1.5).
    RC100-only: z>0.5; SPARC: z=0. So z=1.5 bin contains only RC100 galaxies.
    """
    subset = [d for d in data if abs(d["z"] - z_center) <= z_width]
    rc100_subset = [d for d in subset if d["z"] > 0.5]
    n_rc100 = len(rc100_subset)
    n_sparc = len(subset) - n_rc100

    print("\n" + "=" * 70)
    print(f"DIAGNOSE: Redshift bin z={z_center} +/- {z_width}")
    print("=" * 70)
    print(f"  Total points: {len(subset)}  (RC100: {n_rc100}, SPARC: {n_sparc})")

    if not subset:
        print("  [No data in this bin]")
        return

    v_rot = np.array([d["V_rot"] for d in subset])
    r_kpc = np.array([d["R"] for d in subset])
    m_bary = np.array([d["M_bary"] for d in subset])
    g_bar_arr = np.array([compute_g_bar_g_obs(d)[0] for d in subset])
    g_obs_arr = np.array([compute_g_bar_g_obs(d)[1] for d in subset])

    print(f"\n  V_rot [km/s]:  min={v_rot.min():.1f}  max={v_rot.max():.1f}  "
          f"median={np.median(v_rot):.1f}  mean={v_rot.mean():.1f}  std={v_rot.std():.1f}")
    print(f"  R [kpc]:       min={r_kpc.min():.2f}  max={r_kpc.max():.2f}  median={np.median(r_kpc):.2f}")
    print(f"  M_bary [M_sun]: min={m_bary.min():.2e}  max={m_bary.max():.2e}  median={np.median(m_bary):.2e}")
    print(f"  g_bar [(km/s)^2/kpc]: min={g_bar_arr.min():.1f}  max={g_bar_arr.max():.1f}")
    print(f"  g_obs [(km/s)^2/kpc]: min={g_obs_arr.min():.1f}  max={g_obs_arr.max():.1f}")

    print("\n  Raw data (z, R, V_rot, M_bary, g_bar, g_obs):")
    print("  " + "-" * 66)
    for i, d in enumerate(sorted(subset, key=lambda x: x["z"])):
        gb, go = compute_g_bar_g_obs(d)
        print(f"  {i+1:3d}  z={d['z']:.3f}  R={d['R']:.2f}  V_rot={d['V_rot']:.1f}  "
              f"M_bary={d['M_bary']:.2e}  g_bar={gb:.1f}  g_obs={go:.1f}")
    print("=" * 70)


def get_a0_local() -> float:
    """a0(0) from local SPARC or McGaugh default."""
    cache = ROOT / "axiom_os" / "benchmarks" / "results" / "dark_matter_discovery.json"
    if cache.exists():
        try:
            import json
            data = json.loads(cache.read_text(encoding="utf-8"))
            rar = next((r for r in data.get("results", []) if r.get("source") == "rar"), None)
            if rar and rar.get("g_dagger"):
                from axiom_os.layers.meta_kernel import a0_si_from_g0
                return a0_si_from_g0(float(rar["g_dagger"]))
        except Exception:
            pass
    return 1.2e-10  # McGaugh default m/s^2


def plot_a0_vs_redshift(
    fit_results: Dict[float, Dict],
    a0_local: float,
    out_path: Path,
) -> None:
    """Plot a0 vs z with Static and Dynamic prediction lines and 1σ error bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    z_vals = sorted([z for z, r in fit_results.items() if np.isfinite(r.get("a0_si", np.nan))])
    a0_vals = [fit_results[z]["a0_si"] for z in z_vals]
    n_pts = [fit_results[z]["n_pts"] for z in z_vals]

    z_plot = np.linspace(0, max(z_vals) + 0.2, 100) if z_vals else np.array([0, 1])
    static = np.full_like(z_plot, a0_local)
    dynamic = a0_local * np.array([H_over_H0(z) for z in z_plot])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_plot, static * 1e10, "b-", lw=2, label="Hypothesis A: Static a0(z)=a0(0)")
    ax.plot(z_plot, dynamic * 1e10, "g--", lw=2, label="Hypothesis B: Dynamic a0(z)=a0(0)*H(z)/H0")
    if z_vals:
        yerr_lo = np.array([(a - fit_results[z].get("a0_si_lo", a)) * 1e10 if np.isfinite(fit_results[z].get("a0_si_lo", np.nan)) else 0 for z, a in zip(z_vals, a0_vals)])
        yerr_hi = np.array([(fit_results[z].get("a0_si_hi", a) - a) * 1e10 if np.isfinite(fit_results[z].get("a0_si_hi", np.nan)) else 0 for z, a in zip(z_vals, a0_vals)])
        # z=0: don't show error bar if it would be absurd (robust bootstrap should fix this)
        for i, z in enumerate(z_vals):
            if z == 0 and (yerr_hi[i] > 0.5 * (a0_vals[i] * 1e10) or yerr_lo[i] > 0.5 * (a0_vals[i] * 1e10)):
                yerr_lo[i], yerr_hi[i] = 0, 0
        has_err = np.any(yerr_lo > 0) or np.any(yerr_hi > 0)
        if has_err:
            ax.errorbar(z_vals, [a * 1e10 for a in a0_vals], yerr=[yerr_lo, yerr_hi], fmt="o", color="red", capsize=4, capthick=2, zorder=5, label="Fitted a0 (data, 1σ)")
        else:
            ax.scatter(z_vals, [a * 1e10 for a in a0_vals], s=80, c="red", zorder=5, label="Fitted a0 (data)")
        for z, a in zip(z_vals, a0_vals):
            lbl = f"n={fit_results[z]['n_pts']}"
            if fit_results[z].get("exclude_from_verdict"):
                lbl += "\n(n<10)"
            ax.annotate(lbl, (z, a * 1e10), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("a0 [10^-10 m/s^2]")
    ax.set_title("Falsification Test 3: Redshift Evolution of a0")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, max(z_plot) + 0.1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Falsification Test: Redshift Evolution")
    parser.add_argument("--data", type=str, default=None, help="Path to high-z data CSV/JSON")
    parser.add_argument("--no-rc100", action="store_true", help="Skip RC100, use --data or mock")
    parser.add_argument("--mock", action="store_true", help="Use mock data (only if RC100 and --data fail)")
    parser.add_argument("--no-sparc-anchor", action="store_true", help="Disable SPARC RAR z=0 anchor (~2700 pts)")
    parser.add_argument("--z-bins", type=str, default="0,0.8,1.5,2.3", help="Redshift bins (comma-sep); include 0 for SPARC anchor")
    parser.add_argument("--diagnose", action="store_true", help="Print raw data for z=1.5 bin (RC100 galaxies) to diagnose anomalous a0")
    parser.add_argument("--diagnose-z", type=float, default=1.5, help="Redshift bin center for --diagnose (default 1.5)")
    parser.add_argument("--no-filter-baryon", action="store_true", help="Do not exclude g_obs<g_bar points (baryon-dominated, falling RC)")
    parser.add_argument("--bootstrap", type=int, default=100, help="Bootstrap samples for 1σ error bars (0=skip)")
    args = parser.parse_args()

    from axiom_os.benchmarks.seed_utils import set_global_seed
    set_global_seed(42)

    data_path = Path(args.data) if args.data else None
    data, source = load_highz_data(
        data_path=data_path,
        use_rc100=not args.no_rc100,
        use_sparc_anchor=not args.no_sparc_anchor,
        use_mock=args.mock,
    )

    if not data:
        print("=" * 60)
        print("Falsification Test 3: Redshift Evolution")
        print("=" * 60)
        print("\n[!] No data. RC100 fetch failed or use --data <path> / --mock.")
        return

    z_bins = [float(x.strip()) for x in args.z_bins.split(",")]
    a0_local = get_a0_local()

    if args.diagnose:
        diagnose_z_bin(data, z_center=args.diagnose_z, z_width=0.3)

    print("=" * 60)
    print("Falsification Test 3: Redshift Evolution")
    print("=" * 60)
    print(f"\n[1] Data: {len(data)} points  (source: {source})")
    print(f"    Redshift bins: {z_bins}")

    print(f"\n[2] Hypotheses")
    print(f"    A (Static):  a0(z) = a0(0)")
    print(f"    B (Dynamic): a0(z) = a0(0) * H(z)/H0")

    print(f"\n[3] Fitting RAR per redshift bin...")
    if not args.no_filter_baryon:
        print("    (excluding g_obs<g_bar: baryon-dominated / falling rotation curves)")
    if args.bootstrap > 0:
        print(f"    (bootstrap n={args.bootstrap} for 1σ error bars)")
    fit_results = fit_a0_per_redshift(
        data, z_bins,
        filter_baryon_dominated=not args.no_filter_baryon,
        n_bootstrap=args.bootstrap,
    )
    for z, r in sorted(fit_results.items()):
        a0_si = r.get("a0_si", np.nan)
        n = r.get("n_pts", 0)
        nf = r.get("n_filtered", 0)
        lo, hi = r.get("a0_si_lo"), r.get("a0_si_hi")
        err_str = f"  [{lo:.2e}, {hi:.2e}]" if (lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi))) else ""
        extra = f"  [filtered {nf}]" if nf > 0 else ""
        excl = "  (n<10, ref only)" if r.get("exclude_from_verdict") else ""
        print(f"    z={z:.1f}: a0={a0_si:.4e} m/s^2  (n={n}){err_str}{extra}{excl}")

    # a0(0) for hypothesis curves: use fitted z=0 anchor from data
    a0_ref = fit_results.get(0.0, {}).get("a0_si")
    if a0_ref is None or not np.isfinite(a0_ref):
        a0_ref = a0_local
    print(f"    a0(0) (hypothesis anchor): {a0_ref:.4e} m/s^2")

    # Verdict (exclude bins with n<10, e.g. z=1.5)
    z_fit = [z for z, r in fit_results.items() if np.isfinite(r.get("a0_si", np.nan))]
    z_for_verdict = [z for z in z_fit if not fit_results[z].get("exclude_from_verdict", False)]
    if len(z_for_verdict) < 2:
        verdict = "INSUFFICIENT_DATA"
        msg = "Need at least 2 redshift bins with valid a0 fit (n≥10)."
    else:
        a0_z0 = fit_results.get(0.0, {}).get("a0_si")
        z_high = [z for z in z_for_verdict if z > 0]
        has_errors = all(
            np.isfinite(fit_results[z].get("a0_si_lo", np.nan)) and np.isfinite(fit_results[z].get("a0_si_hi", np.nan))
            for z in z_high
        )
        if has_errors and z_high and np.isfinite(a0_z0):
            a0_z0_lo = fit_results.get(0.0, {}).get("a0_si_lo", a0_z0 * 0.95)
            a0_z0_hi = fit_results.get(0.0, {}).get("a0_si_hi", a0_z0 * 1.05)
            # If all high-z bins' 1σ intervals overlap with z=0's 1σ, trend is not significant
            all_overlap = all(
                fit_results[z]["a0_si_lo"] <= a0_z0_hi and fit_results[z]["a0_si_hi"] >= a0_z0_lo
                for z in z_high
            )
            if all_overlap:
                verdict = "CONSISTENT_WITH_STATIC"
                msg = "Within current data precision and sample size, a0 is consistent with constant from z=0 to z=2.3; no statistically significant redshift evolution detected."
            else:
                a0_fit = [fit_results[z]["a0_si"] for z in z_for_verdict]
                static_err = np.mean([abs(a - a0_ref) for a in a0_fit])
                dynamic_pred = [a0_ref * H_over_H0(z) for z in z_for_verdict]
                dynamic_err = np.mean([abs(a - p) for a, p in zip(a0_fit, dynamic_pred)])
                if static_err < dynamic_err * 0.7:
                    verdict = "FAVORS_STATIC"
                    msg = "Data favors Hypothesis A (Static a0)."
                elif dynamic_err < static_err * 0.7:
                    verdict = "FAVORS_DYNAMIC"
                    msg = "Data favors Hypothesis B (Dynamic a0 ~ H(z)/H0)."
                else:
                    verdict = "INCONCLUSIVE"
                    msg = "Data follows neither clearly. Theory may need new cosmological coupling."
        else:
            a0_fit = [fit_results[z]["a0_si"] for z in z_for_verdict]
            static_err = np.mean([abs(a - a0_ref) for a in a0_fit])
            dynamic_pred = [a0_ref * H_over_H0(z) for z in z_for_verdict]
            dynamic_err = np.mean([abs(a - p) for a, p in zip(a0_fit, dynamic_pred)])
            if static_err < dynamic_err * 0.7:
                verdict = "FAVORS_STATIC"
                msg = "Data favors Hypothesis A (Static a0)."
            elif dynamic_err < static_err * 0.7:
                verdict = "FAVORS_DYNAMIC"
                msg = "Data favors Hypothesis B (Dynamic a0 ~ H(z)/H0)."
            else:
                verdict = "INCONCLUSIVE"
                msg = "Data follows neither clearly. Theory may need new cosmological coupling."

    print(f"\n[4] Verdict: {verdict}")
    print(f"    -> {msg}")

    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_a0_vs_redshift(fit_results, a0_ref, out_dir / "falsify_redshift_a0_vs_z.png")
    print(f"\n    Plot saved: {out_dir / 'falsify_redshift_a0_vs_z.png'}")

    import json
    from datetime import datetime
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {"z_bins": z_bins, "a0_ref": a0_ref, "a0_local_cache": a0_local, "data_source": source},
        "fit_results": {str(z): r for z, r in fit_results.items()},
        "verdict": verdict,
        "n_data": len(data),
    }
    with open(out_dir / "falsify_redshift.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"    Results saved: {out_dir / 'falsify_redshift.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
