"""
RAR Per-Group a0 Fit: Gas+HighSB vs Star+LowSB

Test universality of a0: fit a0 separately for two extreme subgroups.
If a0 differs significantly, it suggests a0 is not a universal constant.

Run: python -m axiom_os.experiments.rar_group_a0_fit
     python -m axiom_os.experiments.rar_group_a0_fit --use-rar-mrt  # ~2700 pts, g_bar segments
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.experiments.rar_residuals import load_per_galaxy_with_groups
from axiom_os.experiments.fit_meta_parameters import fit_g0_log_loss
from axiom_os.layers.meta_kernel import a0_si_from_g0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-group a0 fit: Gas+HighSB vs Star+LowSB")
    parser.add_argument("--n-galaxies", type=int, default=999, help="Max galaxies (Rotmod)")
    parser.add_argument("--allow-mock", action="store_true", help="Allow mock fallback")
    parser.add_argument("--bootstrap", type=int, default=100, help="Bootstrap samples for CI (0=skip)")
    parser.add_argument("--split", choices=["extreme", "gas_star"], default="gas_star",
                        help="extreme: Gas+HighSB vs Star+LowSB; gas_star: Gas(all) vs Star(all)")
    parser.add_argument("--use-rar-mrt", action="store_true",
                        help="Use RAR.mrt (~2700 pts), split by g_bar tertiles (low vs high)")
    args = parser.parse_args()

    print("=" * 65)
    print("RAR Per-Group a0 Fit: Test Universality")
    print("=" * 65)

    if args.use_rar_mrt:
        from axiom_os.datasets.sparc import load_sparc_rar_mrt
        g_bar_all, g_obs_all = load_sparc_rar_mrt()
        if len(g_bar_all) < 100:
            print("\n[!] RAR.mrt load failed or insufficient data.")
            return
        data_source = "RAR.mrt (McGaugh+2016, ~2700 pts)"
        p33, p67 = np.percentile(np.log10(g_bar_all + 1e-20), [33.3, 66.7])
        mask_lo = np.log10(g_bar_all + 1e-20) <= p33
        mask_hi = np.log10(g_bar_all + 1e-20) > p67
        g_bar_a = g_bar_all[mask_lo]
        g_obs_a = g_obs_all[mask_lo]
        g_bar_b = g_bar_all[mask_hi]
        g_obs_b = g_obs_all[mask_hi]
        label_a, label_b = "low_g_bar (bottom 33%)", "high_g_bar (top 33%)"
        n_a, n_b = len(g_bar_a), len(g_bar_b)
        n_gal_a, n_gal_b = None, None
        print(f"\n[0] Data: {len(g_bar_all)} points — {data_source}")
        print(f"\n[1] Group sizes (g_bar tertiles):")
        print(f"    {label_a}:  {n_a} points")
        print(f"    {label_b}:  {n_b} points")
    else:
        print("\nHypothesis: If a0 is universal, subgroups should yield the same optimal a0.")
        per_gal, g_bar_all, g_obs_all, data_source = load_per_galaxy_with_groups(
            n_galaxies=args.n_galaxies,
            use_mock_if_fail=args.allow_mock,
            use_real=True,
        )
        if not per_gal:
            print("\n[!] No galaxies loaded.")
            return
        print(f"\n[0] Data: {len(per_gal)} galaxies — {data_source}")

        # Split into groups. per_gal: (g_bar, g_obs, gas_star, high_low, name)
        gas_high, star_low = [], []
        gas_all, star_all = [], []
        for g_bar, g_obs, gas_star, high_low, name in per_gal:
            if gas_star == "Gas":
                gas_all.append((g_bar, g_obs))
                if high_low == "HighSB":
                    gas_high.append((g_bar, g_obs))
            else:
                star_all.append((g_bar, g_obs))
                if high_low == "LowSB":
                    star_low.append((g_bar, g_obs))

        if args.split == "extreme":
            group_a, group_b = gas_high, star_low
            label_a, label_b = "Gas+HighSB", "Star+LowSB"
        else:
            group_a, group_b = gas_all, star_all
            label_a, label_b = "Gas (all)", "Star (all)"

        if not group_a or not group_b:
            print(f"\n[!] Need both {label_a} and {label_b}. Check grouping.")
            return

        g_bar_a = np.concatenate([x[0] for x in group_a])
        g_obs_a = np.concatenate([x[1] for x in group_a])
        g_bar_b = np.concatenate([x[0] for x in group_b])
        g_obs_b = np.concatenate([x[1] for x in group_b])

        n_a, n_b = len(g_bar_a), len(g_bar_b)
        print(f"\n[1] Group sizes ({args.split}):")
        print(f"    {label_a}:  {len(group_a)} galaxies, {n_a} points")
        print(f"    {label_b}:  {len(group_b)} galaxies, {n_b} points")
        if len(group_a) < 5:
            print(f"    [!] {label_a} has few galaxies — limited power to detect a0 difference.")
        n_gal_a, n_gal_b = len(group_a), len(group_b)

    # Fit a0 per group
    res_a = fit_g0_log_loss(g_bar_a, g_obs_a)
    res_b = fit_g0_log_loss(g_bar_b, g_obs_b)

    a0_a, a0_b = res_a["a0_si"], res_b["a0_si"]
    g0_a, g0_b = res_a["g0"], res_b["g0"]

    print(f"\n[2] Per-group fit (log MSE):")
    print(f"    {label_a}:  g0 = {g0_a:.1f} (km/s)^2/kpc  ->  a0 = {a0_a:.4e} m/s^2")
    print(f"    {label_b}:  g0 = {g0_b:.1f} (km/s)^2/kpc  ->  a0 = {a0_b:.4e} m/s^2")

    ratio = a0_a / a0_b if a0_b > 0 else np.nan
    print(f"\n[3] Ratio a0({label_a}) / a0({label_b}) = {ratio:.3f}")

    if abs(ratio - 1.0) > 0.15:
        print("    -> a0 differs by >15%: suggests a0 may NOT be universal.")
    else:
        print("    -> a0 within ~15%: consistent with universal a0.")

    ci_a, ci_b = None, None
    if args.bootstrap > 0:
        rng = np.random.default_rng(42)
        a0_a_list, a0_b_list = [], []
        for _ in range(args.bootstrap):
            idx_a = rng.integers(0, n_a, size=n_a)
            idx_b = rng.integers(0, n_b, size=n_b)
            try:
                r_a = fit_g0_log_loss(g_bar_a[idx_a], g_obs_a[idx_a])
                r_b = fit_g0_log_loss(g_bar_b[idx_b], g_obs_b[idx_b])
                a0_a_list.append(r_a["a0_si"])
                a0_b_list.append(r_b["a0_si"])
            except Exception:
                pass
        if a0_a_list and a0_b_list:
            ci_a = (np.percentile(a0_a_list, 16), np.percentile(a0_a_list, 84))
            ci_b = (np.percentile(a0_b_list, 16), np.percentile(a0_b_list, 84))
            print(f"\n[4] Bootstrap 1σ CI (n={len(a0_a_list)}):")
            print(f"    {label_a}:  [{ci_a[0]:.4e}, {ci_a[1]:.4e}]")
            print(f"    {label_b}:  [{ci_b[0]:.4e}, {ci_b[1]:.4e}]")
            overlap = (ci_a[0] <= ci_b[1] + 1e-15) and (ci_b[0] <= ci_a[1] + 1e-15)
            print(f"    CIs overlap? {'YES' if overlap else 'NO'}")

    # Save
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    out = {
        "data_source": data_source,
        "split": args.split if not args.use_rar_mrt else "g_bar_tertiles",
        "group_a": {"label": label_a, "g0": g0_a, "a0_si": a0_a, "n_galaxies": n_gal_a, "n_points": n_a},
        "group_b": {"label": label_b, "g0": g0_b, "a0_si": a0_b, "n_galaxies": n_gal_b, "n_points": n_b},
        "ratio_a0_a_over_b": float(ratio),
    }
    if args.bootstrap > 0 and ci_a is not None and ci_b is not None:
        out["bootstrap"] = {
            f"{label_a.replace(' ', '_')}_ci68": list(ci_a),
            f"{label_b.replace(' ', '_')}_ci68": list(ci_b),
        }
    with open(out_dir / "rar_group_a0_fit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n    Saved: {out_dir / 'rar_group_a0_fit.json'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
