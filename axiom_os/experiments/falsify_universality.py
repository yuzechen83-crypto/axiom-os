"""
Falsification Test 1: Universality of a0

Verify if the discovered Meta-Axis constant a0 (g†) is truly universal
or just an average artifact. Split galaxies into extreme subgroups and
check if a0 drifts.

Data Source: SPARC Dataset
Logic: Gas-dominated vs Star-dominated vs Surface-brightness split.

Run: python -m axiom_os.experiments.falsify_universality
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import (
    load_sparc_galaxy,
    get_available_sparc_galaxies,
    SPARC_GALAXIES,
    compute_accelerations,
)
from axiom_os.experiments.discovery_rar import crystallize_rar_law
from axiom_os.benchmarks.seed_utils import set_global_seed


def _mass_proxy(d: dict) -> Tuple[float, float]:
    """
    M_gas proxy: sum(V_gas^2 * R), M_star proxy: sum((V_disk^2 + V_bulge^2) * R).
    Units arbitrary; used for relative comparison.
    """
    R = np.asarray(d["R"], dtype=np.float64)
    r_safe = np.maximum(R, 0.1)
    M_gas = np.sum(d["V_gas"] ** 2 * r_safe)
    M_star = np.sum(d["V_disk"] ** 2 * r_safe + d["V_bulge"] ** 2 * r_safe)
    return float(M_gas), float(M_star)


def _central_surface_density(d: dict) -> float:
    """Central surface density proxy: V^2/R^2 at smallest R, or median in inner 25%."""
    R = np.asarray(d["R"], dtype=np.float64)
    V_bary_sq = d["V_gas"] ** 2 + d["V_disk"] ** 2 + d["V_bulge"] ** 2
    r_safe = np.maximum(R, 0.1)
    sigma = V_bary_sq / (r_safe**2 + 1e-12)
    n_inner = max(1, len(R) // 4)
    return float(np.median(np.sort(sigma)[:n_inner]))


def split_galaxies(
    n_galaxies: int = 50,
    use_mock_if_fail: bool = True,
    use_real: bool = True,
) -> Dict[str, List[str]]:
    """
    Split galaxies into:
    - Group A: Gas Dominated (M_gas > M_star)
    - Group B: Star Dominated (M_star > M_gas)
    - Group C: High vs Low Surface Brightness (median split)
    """
    avail = get_available_sparc_galaxies()
    names = (avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies])

    gas_dominated = []
    star_dominated = []
    all_with_sb = []

    for name in names:
        d = load_sparc_galaxy(name, use_mock_if_fail=use_mock_if_fail, use_real=use_real)
        M_gas, M_star = _mass_proxy(d)
        sb = _central_surface_density(d)
        g_bar, g_obs, mask = compute_accelerations(d)
        if len(g_bar) < 10:
            continue
        all_with_sb.append((name, sb, M_gas, M_star))

    if not all_with_sb:
        return {"gas": [], "star": [], "high_sb": [], "low_sb": []}

    # Gas vs Star
    for name, sb, M_gas, M_star in all_with_sb:
        if M_gas > M_star:
            gas_dominated.append(name)
        else:
            star_dominated.append(name)

    # Surface brightness
    sbs = [x[1] for x in all_with_sb]
    median_sb = np.median(sbs)
    high_sb = [x[0] for x in all_with_sb if x[1] >= median_sb]
    low_sb = [x[0] for x in all_with_sb if x[1] < median_sb]

    return {
        "gas": gas_dominated,
        "star": star_dominated,
        "high_sb": high_sb,
        "low_sb": low_sb,
    }


def run_rar_discovery_on_group(
    galaxy_names: List[str],
    n_bootstrap: int = 5,
    use_mock_if_fail: bool = True,
    use_real: bool = True,
) -> Tuple[float, float]:
    """
    Run RAR discovery on a group, extract a0 (g†). Bootstrap for sigma.
    Returns (a0_mean, a0_std).
    """
    from axiom_os.datasets.sparc import load_sparc_rar

    g_bar, g_obs, _ = load_sparc_rar(
        n_galaxies=999,
        galaxy_list=galaxy_names,
        use_mock_if_fail=use_mock_if_fail,
        use_real=use_real,
    )
    if len(g_bar) < 20:
        return np.nan, np.nan

    a0_list = []
    n = len(g_bar)
    rng = np.random.default_rng(42)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=min(n, int(0.9 * n)), replace=True)
        gb, go = g_bar[idx], g_obs[idx]
        try:
            cry = crystallize_rar_law(gb, go)
            a0 = cry.get("g_dagger") or cry.get("a0_or_gdagger")
            if a0 is not None and 100 <= a0 <= 50000:
                a0_list.append(float(a0))
        except Exception:
            pass

    if not a0_list:
        try:
            cry = crystallize_rar_law(g_bar, g_obs)
            a0 = cry.get("g_dagger") or cry.get("a0_or_gdagger")
            if a0 is not None:
                return float(a0), 0.0
        except Exception:
            pass
        return np.nan, np.nan

    return float(np.mean(a0_list)), float(np.std(a0_list)) if len(a0_list) > 1 else 0.0


def falsification_criteria(
    a0_gas: float, sigma_gas: float,
    a0_star: float, sigma_star: float,
) -> Tuple[float, str]:
    """
    Z = |a0_gas - a0_star| / sqrt(sigma_gas^2 + sigma_star^2)
    Verdict: Z < 2 Passed, Z > 5 Falsified, else Inconclusive.
    """
    denom = np.sqrt(sigma_gas**2 + sigma_star**2 + 1e-12)
    Z = abs(a0_gas - a0_star) / denom
    if Z < 2:
        verdict = "Passed"
    elif Z > 5:
        verdict = "Falsified"
    else:
        verdict = "Inconclusive"
    return float(Z), verdict


def plot_rar_curves(
    groups: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    out_path: Path,
) -> None:
    """Plot RAR curves (g_bar vs g_obs) for different groups."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"gas": "#1f77b4", "star": "#ff7f0e", "high_sb": "#2ca02c", "low_sb": "#d62728"}
    labels = {"gas": "Gas Dominated", "star": "Star Dominated", "high_sb": "High SB", "low_sb": "Low SB"}

    for key, (g_bar, g_obs, names) in groups.items():
        if len(g_bar) < 5:
            continue
        c = colors.get(key, "gray")
        ax.scatter(g_bar, g_obs, alpha=0.4, s=8, c=c, label=f"{labels.get(key, key)} (n={len(g_bar)})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("g_bar [(km/s)^2/kpc]")
    ax.set_ylabel("g_obs [(km/s)^2/kpc]")
    ax.set_title("RAR: g_obs vs g_bar by Galaxy Group")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 3e5)
    ax.set_ylim(50, 3e5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Falsification Test: Universality of a0")
    parser.add_argument("--n-galaxies", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    set_global_seed(42)

    print("=" * 60)
    print("Falsification Test 1: Universality of a0")
    print("=" * 60)

    splits = split_galaxies(n_galaxies=args.n_galaxies)
    print(f"\n[1] Data splitting:")
    print(f"    Gas dominated: {len(splits['gas'])} galaxies")
    print(f"    Star dominated: {len(splits['star'])} galaxies")
    print(f"    High SB: {len(splits['high_sb'])}, Low SB: {len(splits['low_sb'])}")

    if len(splits["gas"]) < 5 or len(splits["star"]) < 5:
        print("\n[!] Insufficient galaxies in gas/star groups. Using mock data or reduce --n-galaxies.")
        if len(splits["gas"]) + len(splits["star"]) < 10:
            print("    Aborting.")
            return

    print("\n[2] Independent Discovery (extract a0 per group):")
    a0_gas, sigma_gas = run_rar_discovery_on_group(
        splits["gas"], n_bootstrap=args.n_bootstrap,
    )
    a0_star, sigma_star = run_rar_discovery_on_group(
        splits["star"], n_bootstrap=args.n_bootstrap,
    )
    a0_high, sigma_high = run_rar_discovery_on_group(
        splits["high_sb"], n_bootstrap=args.n_bootstrap,
    )
    a0_low, sigma_low = run_rar_discovery_on_group(
        splits["low_sb"], n_bootstrap=args.n_bootstrap,
    )

    print(f"    Gas dominated:  a0 = {a0_gas:.1f} +/- {sigma_gas:.1f} (km/s)^2/kpc")
    print(f"    Star dominated: a0 = {a0_star:.1f} +/- {sigma_star:.1f} (km/s)^2/kpc")
    print(f"    High SB:        a0 = {a0_high:.1f} +/- {sigma_high:.1f}")
    print(f"    Low SB:        a0 = {a0_low:.1f} +/- {sigma_low:.1f}")

    Z, verdict = falsification_criteria(a0_gas, sigma_gas, a0_star, sigma_star)
    print(f"\n[3] Falsification Criteria (Gas vs Star):")
    print(f"    Z-score = {Z:.2f}")
    print(f"    Verdict: {verdict}")
    if Z < 2:
        print("    -> Theory holds: a0 is universal across matter type.")
    elif Z > 5:
        print("    -> Falsified: Physics depends on matter type, not geometry.")
    else:
        print("    -> Inconclusive: More data or bootstrap needed.")

    if not args.no_plot:
        from axiom_os.datasets.sparc import load_sparc_rar
        groups_data = {}
        for key, names in [
            ("gas", splits["gas"]), ("star", splits["star"]),
            ("high_sb", splits["high_sb"]), ("low_sb", splits["low_sb"]),
        ]:
            if len(names) < 3:
                continue
            g_bar, g_obs, _ = load_sparc_rar(n_galaxies=999, galaxy_list=names)
            if len(g_bar) >= 5:
                groups_data[key] = (g_bar, g_obs, names)
        out_path = ROOT / "axiom_os" / "benchmarks" / "results" / "falsify_universality_rar.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_rar_curves(groups_data, out_path)
        print(f"\n[4] Plot saved: {out_path}")

    # Save JSON results
    import json
    from datetime import datetime
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_galaxies": args.n_galaxies, "n_bootstrap": args.n_bootstrap},
        "splits": {k: len(v) for k, v in splits.items()},
        "a0": {
            "gas": {"mean": a0_gas, "std": sigma_gas},
            "star": {"mean": a0_star, "std": sigma_star},
            "high_sb": {"mean": a0_high, "std": sigma_high},
            "low_sb": {"mean": a0_low, "std": sigma_low},
        },
        "falsification": {"Z_score": Z, "verdict": verdict},
    }
    json_path = ROOT / "axiom_os" / "benchmarks" / "results" / "falsify_universality.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"    Results saved: {json_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
