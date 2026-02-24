"""
Axiom-OS Unified Runner - UPI Interface to Full System
Modes: acrobot, meta_validation (SPARC), turbulence_3d
"""

from typing import Optional, Dict, Any, Literal
import numpy as np

from axiom_os.core import UPIState, Units, Hippocampus
from axiom_os.engine import DiscoveryEngine


Mode = Literal["acrobot", "meta_validation", "turbulence_3d"]


def run_axiom(
    mode: Mode = "acrobot",
    n_steps: int = 200,
    n_galaxies: int = 50,
    meta_epochs: int = 500,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run complete Axiom system via UPI interface.
    Returns dict with results, hippocampus knowledge, discovery log.
    """
    hippocampus = Hippocampus(dim=32, capacity=5000)
    discovery_engine = DiscoveryEngine(use_pysr=False)
    results: Dict[str, Any] = dict(
        mode=mode,
        hippocampus=hippocampus,
        discovery_engine=discovery_engine,
        upi_log=[],
        knowledge={},
    )

    if mode == "acrobot":
        return _run_acrobot(
            hippocampus=hippocampus,
            discovery_engine=discovery_engine,
            n_steps=n_steps,
            results=results,
            **kwargs,
        )
    if mode == "meta_validation":
        return _run_meta_validation(
            hippocampus=hippocampus,
            discovery_engine=discovery_engine,
            n_galaxies=n_galaxies,
            meta_epochs=meta_epochs,
            results=results,
            **kwargs,
        )
    if mode == "turbulence_3d":
        return _run_turbulence_3d(
            hippocampus=hippocampus,
            discovery_engine=discovery_engine,
            results=results,
            **kwargs,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _run_acrobot(
    hippocampus: Hippocampus,
    discovery_engine: DiscoveryEngine,
    n_steps: int,
    results: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Acrobot mode: UPIState (q,p) -> MPC -> RCLN -> Discovery."""
    from axiom_os.main import main as main_acrobot
    main_acrobot(
        uncertainty_mode=kwargs.get("uncertainty_mode", False),
        hippocampus=hippocampus,
        discovery_engine=discovery_engine,
    )
    results["knowledge"] = dict(hippocampus.knowledge_base)
    return results


def _run_meta_validation(
    hippocampus: Hippocampus,
    discovery_engine: DiscoveryEngine,
    n_galaxies: int,
    meta_epochs: int,
    results: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Meta-Validation: SPARC galaxies with UPI units, Hippocampus, Discovery."""
    from axiom_os.datasets.sparc import load_sparc_galaxy, get_available_sparc_galaxies
    from axiom_os.layers.meta_kernel import MetaProjectionModel
    from axiom_os.engine.discovery_check import discovery_check_kernel_shape
    import torch

    avail = get_available_sparc_galaxies()
    names = avail[:n_galaxies] if avail else [
        "NGC_6503", "NGC_3198", "NGC_2403", "NGC_2841", "NGC_925",
        "NGC_2976", "NGC_7331", "NGC_5055", "NGC_6946", "NGC_628",
    ][:n_galaxies]

    LENGTH = Units.LENGTH

    nfw_wins = 0
    meta_wins = 0
    L_values = []
    upi_log = []

    for name in names:
        d = load_sparc_galaxy(name, use_mock_if_fail=True, use_real=True)
        R = d["R"]
        V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
        V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1.0)
        V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0.0)

        # UPIState for observation: (R, V_def_sq, V_bary_sq)
        # Units: R [L], V_def_sq/V_bary_sq [L²/T²]; use LENGTH as representative
        obs_vals = np.column_stack([R, V_def_sq, V_bary_sq]).astype(np.float64)
        upi_obs = UPIState(
            values=obs_vals,
            units=LENGTH,
            semantics=f"GalaxyRotation_{name}: R[kpc], V_def_sq, V_bary_sq [(km/s)^2]",
        )
        upi_log.append({"galaxy": name, "upi": upi_obs})

        # Fit models (from verify_mashd)
        from axiom_os.experiments.verify_mashd import fit_nfw, fit_meta, bic

        mse_nfw, k_nfw, _ = fit_nfw(R, V_def_sq, epochs=meta_epochs)
        mse_meta, k_meta, meta_model = fit_meta(R, V_def_sq, V_bary_sq, epochs=meta_epochs)
        bic_nfw = bic(len(R), mse_nfw, k_nfw)
        bic_meta = bic(len(R), mse_meta, k_meta)

        if bic_meta < bic_nfw:
            meta_wins += 1
        else:
            nfw_wins += 1
        L_val = float(meta_model.meta.L.detach().item())
        L_values.append(L_val)

        # Crystallize best formula to Hippocampus
        dc = discovery_check_kernel_shape(R, V_def_sq, V_bary_sq)
        if dc.get("formula") and dc.get("r2", 0) > 0.5:
            hippocampus.knowledge_base[f"sparc_{name}_kernel"] = {
                "formula": dc["formula"],
                "r2": dc["r2"],
                "L_learned": L_val,
            }

    results["data_source"] = "REAL" if avail else "MOCK"
    results["n_galaxies"] = len(names)
    results["nfw_wins"] = nfw_wins
    results["meta_wins"] = meta_wins
    results["L_values"] = L_values
    results["L_mean"] = float(np.mean(L_values))
    results["L_std"] = float(np.std(L_values))
    results["upi_log"] = upi_log
    results["knowledge"] = dict(hippocampus.knowledge_base)
    return results


def _run_turbulence_3d(
    hippocampus: Hippocampus,
    discovery_engine: DiscoveryEngine,
    results: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Turbulence 3D: delegate to test script, collect results."""
    import sys
    from pathlib import Path
    from io import StringIO

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    # Run the turbulence script and capture - for now just indicate it runs
    results["message"] = "Run: py tests/test_axiom_turbulence_3d_full.py for full pipeline"
    results["knowledge"] = dict(hippocampus.knowledge_base)
    return results


def observe_galaxy_upi(galaxy_name: str) -> UPIState:
    """
    UPI interface: Load one galaxy as UPIState.
    values: (R, V_def_sq, V_bary_sq) per data point.
    units: LENGTH (representative; columns have R[kpc], V²[(km/s)²]).
    """
    from axiom_os.datasets.sparc import load_sparc_galaxy

    d = load_sparc_galaxy(galaxy_name, use_mock_if_fail=True, use_real=True)
    R = d["R"]
    V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
    V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1.0)
    V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0.0)

    obs = np.column_stack([R, V_def_sq, V_bary_sq]).astype(np.float64)
    return UPIState(
        values=obs,
        units=Units.LENGTH,
        semantics=f"GalaxyRotation_{galaxy_name}: R[kpc], V_def_sq, V_bary_sq [(km/s)^2]",
    )
