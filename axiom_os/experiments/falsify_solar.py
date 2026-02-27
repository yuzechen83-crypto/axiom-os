"""
Falsification Test 2: Solar System Constraints

Check if the Meta-Axis formula breaks Newtonian mechanics in the high
acceleration regime (solar system). Compare predicted g_meta vs g_newton
with Cassini ephemeris residuals.

Uses Axiom: a0 derived from RAR Discovery on SPARC real data.
Data: NASA JPL orbital parameters (Saturn, Pluto, Proxima).

Run: python -m axiom_os.experiments.falsify_solar
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Physical constants (SI)
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN = 1.9885e30   # kg
AU_M = 1.495978707e11  # 1 AU in m
LY_M = 9.4607e15   # 1 light-year in m
SEC_PER_YEAR = 3.15576e7
SEC_PER_CENTURY = 3.15576e9

# Fallback when Axiom not run
A0_MCGAUGH = 1.2e-10   # m/s^2 (McGaugh standard)


def g_newton(r_m: float, M_kg: float = M_SUN) -> float:
    """Newtonian: g = GM/r^2 [m/s^2]."""
    return G_SI * M_kg / (r_m**2 + 1e-30)


def g_meta(g_n: float, a0: float, eps: float = 1e-20) -> float:
    """
    Meta-Axis (McGaugh): g_meta = g_newton / (1 - exp(-sqrt(g_newton/a0))).
    g_obs = g_bar * nu(g_bar), nu = 1/(1-exp(-sqrt(g_bar/a0))).
    So g_meta = g_newton * nu = g_newton / (1 - exp(-sqrt(g_newton/a0))).
    """
    x = np.maximum(g_n / (a0 + eps), eps)
    denom = 1.0 - np.exp(-np.sqrt(x))
    denom = np.maximum(denom, eps)
    return g_n / denom


def delta_g(r_m: float, a0: float, M_kg: float = M_SUN) -> Tuple[float, float, float]:
    """Compute g_newton, g_meta, delta_g = g_meta - g_newton."""
    g_n = g_newton(r_m, M_kg)
    g_m = g_meta(g_n, a0)
    return g_n, g_m, g_m - g_n


def delta_g_to_orbital_shift_m_per_year(delta_g: float, r_m: float, g_n: float) -> float:
    """
    Convert delta_g (m/s^2) to approximate orbital drift (m/year).
    For circular orbit: v^2 = g*r. Perturbation delta_g causes radial drift.
    Rough estimate: delta_r_per_orbit ~ r * (delta_g / g_n) * (orbital_period_factor).
    Simpler: position error accumulates as (1/2)*delta_g*t^2.
    Over 1 year: drift ~ 0.5 * delta_g * (1 year)^2.
    """
    t_year = SEC_PER_YEAR
    return 0.5 * abs(delta_g) * (t_year**2)


def delta_g_to_precession_arcsec_per_century(delta_g: float, g_n: float) -> float:
    """
    Convert delta_g/g to approximate precession shift (arcsec/century).
    For a perturbing radial force delta_g, the precession per orbit is
    delta_omega ~ -2*pi * (delta_g/g) for small perturbation (rough).
    Per century: multiply by orbits per century.
    Mercury: ~415 orbits/century. Saturn: ~0.034 orbits/century.
    Returns order-of-magnitude estimate in arcsec/century.
    """
    if abs(g_n) < 1e-30:
        return 0.0
    ratio = delta_g / g_n
    # Precession per orbit ~ 2*pi * ratio (rad). 1 rad = 206265 arcsec.
    arcsec_per_orbit = 2 * np.pi * abs(ratio) * 206265
    # Saturn: ~0.034 orbits/century
    return arcsec_per_orbit * 0.034  # rough for Saturn



# Cassini ephemeris: Saturn orbit known to ~10 m level over ~13 years.
# Implied acceleration uncertainty: 2*10m / (13 yr)^2 ~ 4e-16 m/s^2
# Conservative: use 1e-15 m/s^2 as threshold for |delta_g|
CASSINI_DELTA_G_THRESHOLD = 1e-15  # m/s^2
CASSINI_POSITION_UNCERTAINTY_M = 10.0  # meters
CASSINI_MISSION_YEARS = 13.0


def implied_acceleration_uncertainty(delta_x_m: float, t_years: float) -> float:
    """From position uncertainty delta_x over t years: delta_g ~ 2*delta_x / t^2."""
    t_sec = t_years * SEC_PER_YEAR
    return 2.0 * delta_x_m / (t_sec**2)


def get_a0_from_axiom(
    n_galaxies: int = 25,
    epochs: int = 400,
    use_cached: bool = True,
) -> Tuple[float, str]:
    """
    Derive a0 from Axiom RAR Discovery on SPARC real data.
    Returns (a0_si, source_str).
    """
    from axiom_os.layers.meta_kernel import a0_si_from_g0

    # Try cached dark_matter_discovery.json first
    if use_cached:
        cache_path = ROOT / "axiom_os" / "benchmarks" / "results" / "dark_matter_discovery.json"
        if cache_path.exists():
            import json
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                results = data.get("results", [])
                rar = next((r for r in results if r.get("source") == "rar"), None)
                if rar and rar.get("ok") and rar.get("g_dagger") is not None:
                    g_dagger = float(rar["g_dagger"])
                    a0_si = a0_si_from_g0(g_dagger)
                    return a0_si, f"Axiom (cached, g_dagger={g_dagger:.1f} (km/s)^2/kpc)"
            except Exception:
                pass

    # Run RAR Discovery
    from axiom_os.benchmarks.seed_utils import set_global_seed
    from axiom_os.experiments.discovery_rar import run_rar_discovery

    set_global_seed(42)
    res = run_rar_discovery(n_galaxies=n_galaxies, epochs=epochs, use_pysr=True)
    if res.get("error"):
        raise RuntimeError(f"RAR Discovery failed: {res['error']}")
    cryst = res.get("crystallized") or {}
    g_dagger = cryst.get("g_dagger") or cryst.get("a0_or_gdagger")
    if g_dagger is None:
        raise RuntimeError("RAR Discovery did not extract g_dagger")
    a0_si = a0_si_from_g0(float(g_dagger))
    return a0_si, f"Axiom RAR Discovery (SPARC n={n_galaxies}, g_dagger={g_dagger:.1f})"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Falsification Test: Solar System Constraints")
    parser.add_argument("--a0", type=float, default=None,
                        help="Override a0 in m/s^2 (default: from Axiom RAR Discovery)")
    parser.add_argument("--skip-axiom", action="store_true",
                        help="Skip Axiom, use McGaugh a0=1.2e-10 fallback")
    parser.add_argument("--n-galaxies", type=int, default=25, help="Galaxies for RAR Discovery")
    parser.add_argument("--epochs", type=int, default=400, help="Epochs for RAR Discovery")
    args = parser.parse_args()

    a0_source = ""
    if args.a0 is not None:
        a0 = args.a0
        a0_source = "manual override"
    elif args.skip_axiom:
        a0 = A0_MCGAUGH
        a0_source = "McGaugh fallback (1.2e-10 m/s^2)"
    else:
        try:
            a0, a0_source = get_a0_from_axiom(
                n_galaxies=args.n_galaxies,
                epochs=args.epochs,
                use_cached=True,
            )
        except Exception as e:
            print(f"[!] Axiom RAR Discovery failed: {e}")
            print("    Using McGaugh fallback a0=1.2e-10")
            a0 = A0_MCGAUGH
            a0_source = "McGaugh fallback (Axiom failed)"

    from axiom_os.datasets.solar_system import get_default_falsification_targets
    TARGETS = get_default_falsification_targets()

    print("=" * 60)
    print("Falsification Test 2: Solar System Constraints")
    print("=" * 60)
    print(f"\n[1] Physics Kernel")
    print(f"    Newton:  g = GM/r^2")
    print(f"    Meta:    g_meta = g_newton / (1 - exp(-sqrt(g_newton/a0)))")
    print(f"    a0 = {a0:.4e} m/s^2  ({a0_source})")

    threshold = implied_acceleration_uncertainty(
        CASSINI_POSITION_UNCERTAINTY_M, CASSINI_MISSION_YEARS
    )
    print(f"\n[2] Cassini Ephemeris Reference")
    print(f"    Saturn position uncertainty: ~{CASSINI_POSITION_UNCERTAINTY_M} m over {CASSINI_MISSION_YEARS} yr")
    print(f"    Implied |delta_g| threshold: ~{threshold:.2e} m/s^2")

    sources = [TARGETS[n].get("source", "?") for n in TARGETS]
    print(f"\n[3] Targets: {list(TARGETS.keys())}")
    print(f"    Data sources: {sources}")
    print(f"    Calculation: delta_g = g_meta - g_newton")
    print("-" * 60)

    results = []
    for name, cfg in TARGETS.items():
        r_m = cfg["r_m"]
        g_n, g_m, dg = delta_g(r_m, a0)
        drift_myr = delta_g_to_orbital_shift_m_per_year(dg, r_m, g_n)
        precess = delta_g_to_precession_arcsec_per_century(dg, g_n)

        r_au = r_m / AU_M
        print(f"    {name}:")
        print(f"      r = {r_au:.2f} AU ({r_m:.2e} m)")
        print(f"      g_newton = {g_n:.4e} m/s^2")
        print(f"      g_meta   = {g_m:.4e} m/s^2")
        print(f"      delta_g  = {dg:.4e} m/s^2")
        print(f"      Orbital drift (est): ~{drift_myr:.2e} m/year")
        print(f"      Precession (est):    ~{precess:.4e} arcsec/century")

        results.append({
            "name": name,
            "r_au": r_au,
            "g_newton": g_n,
            "g_meta": g_m,
            "delta_g": dg,
            "drift_m_per_year": drift_myr,
            "precession_arcsec_century": precess,
        })

    print("-" * 60)
    print(f"\n[4] Falsification Criteria (Saturn vs Cassini)")
    saturn = next(r for r in results if r["name"] == "Saturn")
    dg_saturn = abs(saturn["delta_g"])
    if dg_saturn > threshold:
        verdict = "FALSIFIED"
        msg = "Predicted shift > Observation uncertainty. Meta-Axis nu(g) too strong at high g."
    else:
        verdict = "PASSED"
        msg = "Predicted shift within Cassini ephemeris uncertainty. Newtonian limit holds."

    print(f"    |delta_g| (Saturn) = {dg_saturn:.4e} m/s^2")
    print(f"    Threshold          = {threshold:.4e} m/s^2")
    print(f"    Verdict: {verdict}")
    print(f"    -> {msg}")

    # Save JSON
    import json
    from datetime import datetime
    data_sources = [TARGETS[n].get("source", "?") for n in TARGETS]
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {"a0_si": a0, "a0_source": a0_source, "data_sources": data_sources},
        "cassini": {
            "position_uncertainty_m": CASSINI_POSITION_UNCERTAINTY_M,
            "mission_years": CASSINI_MISSION_YEARS,
            "implied_delta_g_threshold": threshold,
        },
        "results": results,
        "falsification": {"verdict": verdict, "delta_g_saturn": dg_saturn},
    }
    out_path = ROOT / "axiom_os" / "benchmarks" / "results" / "falsify_solar.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n    Results saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
