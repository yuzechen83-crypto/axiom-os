"""
Test: Run complete Axiom system via UPI interface
- meta_validation: SPARC galaxies, Hippocampus, Discovery
- acrobot: Full lifecycle with external Hippocampus
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_meta_validation_upi():
    """Run Meta-Validation via UPI API (50 galaxies, full Axiom flow)."""
    from axiom_os.api import run_axiom

    print("\n" + "=" * 70)
    print("UPI API: meta_validation (SPARC + Hippocampus + Discovery)")
    print("=" * 70)

    res = run_axiom(mode="meta_validation", n_galaxies=10, meta_epochs=200)
    # Use 10 galaxies / 200 epochs for faster CI; increase for full run

    assert res["mode"] == "meta_validation"
    assert "nfw_wins" in res and "meta_wins" in res
    assert res["n_galaxies"] == 10
    assert len(res["L_values"]) == 10
    assert "knowledge" in res
    assert "upi_log" in res and len(res["upi_log"]) == 10

    print(f"\n  Data: {res['data_source']}, Galaxies: {res['n_galaxies']}")
    print(f"  NFW wins: {res['nfw_wins']}, Meta wins: {res['meta_wins']}")
    print(f"  L mean: {res['L_mean']:.2f} ± {res['L_std']:.2f}")
    print(f"  Knowledge keys: {list(res['knowledge'].keys())[:5]}...")
    print("=" * 70)


def test_acrobot_upi():
    """Run Acrobot via UPI API (full Axiom lifecycle)."""
    from axiom_os.api import run_axiom

    print("\n" + "=" * 70)
    print("UPI API: acrobot (full Axiom lifecycle)")
    print("=" * 70)

    res = run_axiom(mode="acrobot", n_steps=200)

    assert res["mode"] == "acrobot"
    assert "knowledge" in res
    assert "hippocampus" in res

    print(f"\n  Knowledge keys: {list(res['knowledge'].keys())}")
    print("=" * 70)


def test_observe_galaxy_upi():
    """Test observe_galaxy_upi returns UPIState."""
    from axiom_os.api import observe_galaxy_upi
    from axiom_os.core import UPIState

    upi = observe_galaxy_upi("NGC_6503")
    assert isinstance(upi, UPIState)
    assert upi.values.ndim == 2
    assert upi.values.shape[1] == 3  # R, V_def_sq, V_bary_sq
    assert "GalaxyRotation" in upi.semantics
    print(f"  observe_galaxy_upi: shape={upi.values.shape}, semantics={upi.semantics[:50]}...")


if __name__ == "__main__":
    test_observe_galaxy_upi()
    test_meta_validation_upi()
    test_acrobot_upi()
    print("\nAll UPI API tests passed.")
