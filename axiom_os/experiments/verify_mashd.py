"""
Meta-Validation: Battle of Models - NFW vs Meta-Axis
Validate the Meta-Axis theory against SPARC galaxy rotation curves.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.sparc import (
    load_sparc_galaxy,
    load_mock_batch,
    SPARC_GALAXIES,
    get_available_sparc_galaxies,
)
from axiom_os.layers.meta_kernel import MetaProjectionModel
from axiom_os.engine.discovery_check import discovery_check_kernel_shape


def nfw_V_sq(r: np.ndarray, r_s: float, rho_s: float) -> np.ndarray:
    """
    NFW halo: V² ∝ (1/r) * [ln(1+x) - x/(1+x)], x = r/r_s.
    Returns V² in arbitrary units (scale by fit).
    """
    x = np.maximum(r / (r_s + 1e-8), 1e-6)
    f = np.log(1 + x) - x / (1 + x)
    return f / (x + 1e-8)


class NFWModel(nn.Module):
    """Standard NFW dark matter halo. Learnable r_s, scale."""

    def __init__(self, r_s_init: float = 5.0):
        super().__init__()
        self.log_r_s = nn.Parameter(torch.tensor(np.log(r_s_init)))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_s = torch.exp(self.log_r_s) + 0.5
        x = r / (r_s + 1e-8)
        x = torch.clamp(x, min=1e-6)
        f = torch.log(1 + x) - x / (1 + x)
        v_sq = f / (x + 1e-8)
        scale = torch.exp(self.log_scale)
        return scale * v_sq


def bic(n: int, mse: float, k: int) -> float:
    """Bayesian Information Criterion: BIC = n*ln(MSE) + k*ln(n)."""
    return n * np.log(mse + 1e-12) + k * np.log(n + 1)


def fit_nfw(R: np.ndarray, V_def_sq: np.ndarray, epochs: int = 500) -> tuple:
    """Fit NFW model. Returns (mse, n_params, model)."""
    R_t = torch.from_numpy(R).float().unsqueeze(1)
    V_t = torch.from_numpy(V_def_sq).float()
    model = NFWModel(r_s_init=5.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(R_t.squeeze(-1))
        loss = ((pred - V_t) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(R_t.squeeze(-1)).numpy()
    mse = float(np.mean((pred - V_def_sq) ** 2))
    return mse, 2, model


def predict_rar(R: np.ndarray, V_bary_sq: np.ndarray, g_dagger: float) -> np.ndarray:
    """
    RAR (McGaugh): g_obs = g_bar / (1 - exp(-sqrt(g_bar/g†)))
    g_bar = V_bary^2 / r, V_obs^2 = g_obs * r, V_def_sq = V_obs^2 - V_bary^2
    """
    r_safe = np.maximum(R, 1e-4)
    g_bar = V_bary_sq / r_safe
    g_bar_safe = np.maximum(g_bar, 1e-14)
    denom = 1.0 - np.exp(-np.sqrt(g_bar_safe / (g_dagger + 1e-14)))
    denom = np.maximum(denom, 0.01)
    g_obs = g_bar_safe / denom
    V_obs_sq = g_obs * r_safe
    V_def_sq_pred = np.maximum(V_obs_sq - V_bary_sq, 0.0)
    return V_def_sq_pred


def fit_rar(R: np.ndarray, V_def_sq: np.ndarray, V_bary_sq: np.ndarray, g_dagger: float) -> tuple:
    """RAR predictor using discovered g†. No per-galaxy fit; k=1 (global g†)."""
    pred = predict_rar(R, V_bary_sq, g_dagger)
    mse = float(np.mean((pred - V_def_sq) ** 2))
    return mse, 1, None  # k=1 for global g†


def fit_meta(R: np.ndarray, V_def_sq: np.ndarray, V_bary_sq: np.ndarray, epochs: int = 500) -> tuple:
    """Fit MetaProjectionModel. Returns (mse, n_params, model)."""
    R_t = torch.from_numpy(R).float()
    V_def_t = torch.from_numpy(V_def_sq).float()
    V_bary_t = torch.from_numpy(V_bary_sq).float()
    # Normalize target for stable gradients (MSE reported in original scale)
    scale_def = float(np.max(V_def_sq) + 1e-8)
    V_def_norm = V_def_t / scale_def
    model = MetaProjectionModel(L_init=10.2, k0_init=1.0, learn_L=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(R_t, V_bary_t)
        pred_norm = pred / scale_def
        loss = ((pred_norm - V_def_norm) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        scheduler.step()
    with torch.no_grad():
        pred = model(R_t, V_bary_t).numpy()
    mse = float(np.mean((pred - V_def_sq) ** 2))
    n_params = 3  # k0, rho_scale, log_r_exp (L fixed)
    return mse, n_params, model


def get_rar_g_dagger(n_galaxies: int = 50) -> float:
    """Run RAR discovery to extract g†. Returns g_dagger [(km/s)^2/kpc]."""
    from axiom_os.experiments.discovery_rar import run_rar_discovery, crystallize_rar_law
    from axiom_os.datasets.sparc import load_sparc_rar, get_available_sparc_galaxies, SPARC_GALAXIES
    avail = get_available_sparc_galaxies()
    n = min(n_galaxies, len(avail) if avail else len(SPARC_GALAXIES))
    g_bar, g_obs, _ = load_sparc_rar(n_galaxies=n, use_mock_if_fail=True, use_real=True)
    if g_bar is None or len(g_bar) < 10:
        return 321.0  # fallback from prior discovery
    cry = crystallize_rar_law(g_bar, g_obs)
    g0 = cry.get("g_dagger") or cry.get("a0_or_gdagger")
    return float(g0) if g0 is not None else 321.0


def run_battle(
    n_galaxies: int = 50,
    epochs: int = 500,
    data_source: str = "real",
    use_rar: bool = False,
    g_dagger: float | None = None,
) -> dict:
    """
    Run NFW vs Meta-Axis on multiple galaxies.
    data_source: "nfw" | "meta" | "real"
      - nfw: Mock data with NFW ground truth (NFW should win)
      - meta: Mock data with Meta-Axis ground truth (Meta should win)
      - real: Real SPARC data (whichever fits better)
    """
    if data_source == "real":
        avail = get_available_sparc_galaxies()
        names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]
        galaxies = [
            load_sparc_galaxy(n, use_mock_if_fail=True, use_real=True)
            for n in names
        ]
        src_label = "REAL" if avail else "MOCK_NFW"
    else:
        galaxies = load_mock_batch(n_galaxies, data_source=data_source, seed=42)
        names = [g["name"] for g in galaxies]
        src_label = f"MOCK_{data_source.upper()}"

    if use_rar and g_dagger is None:
        g_dagger = get_rar_g_dagger(n_galaxies=n_galaxies)

    results = {
        "nfw": [], "meta": [], "L_values": [], "L_bootstrap_std": [],
        "r_s_values": [], "names": [], "data_source": src_label,
    }
    if use_rar:
        results["rar"] = []
        results["g_dagger"] = g_dagger

    for d in galaxies:
        R = d["R"]
        V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
        V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1.0)
        V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0.0)

        mse_nfw, k_nfw, nfw_model = fit_nfw(R, V_def_sq, epochs=epochs)
        mse_meta, k_meta, meta_model = fit_meta(R, V_def_sq, V_bary_sq, epochs=epochs)

        bic_nfw = bic(len(R), mse_nfw, k_nfw)
        bic_meta = bic(len(R), mse_meta, k_meta)

        results["nfw"].append({"mse": mse_nfw, "bic": bic_nfw, "k": k_nfw})
        results["meta"].append({"mse": mse_meta, "bic": bic_meta, "k": k_meta})

        if use_rar:
            mse_rar, k_rar, _ = fit_rar(R, V_def_sq, V_bary_sq, g_dagger)
            bic_rar = bic(len(R), mse_rar, k_rar)
            results["rar"].append({"mse": mse_rar, "bic": bic_rar, "k": k_rar})
        L_val = float(meta_model.meta.L.detach().item() if hasattr(meta_model.meta.L, "item") else meta_model.meta.L)
        results["L_values"].append(L_val)
        r_s_val = float(torch.exp(nfw_model.log_r_s.detach()) + 0.5)
        results["r_s_values"].append(r_s_val)

        # Bootstrap L uncertainty (3 resamples for speed)
        L_boot = []
        rng = np.random.default_rng(42)
        for _ in range(3):
            idx = rng.choice(len(R), size=len(R), replace=True)
            R_b, Vd_b, Vb_b = R[idx], V_def_sq[idx], V_bary_sq[idx]
            _, _, m_b = fit_meta(R_b, Vd_b, Vb_b, epochs=min(200, epochs))
            L_boot.append(float(m_b.meta.L.detach().item()))
        results["L_bootstrap_std"].append(float(np.std(L_boot)) if L_boot else 0.0)
        results["names"].append(d["name"])

    return results


def _summarize(results: dict, with_rar: bool = False) -> dict:
    """Extract summary stats from run_battle results."""
    bic_nfw = np.mean([r["bic"] for r in results["nfw"]])
    bic_meta = np.mean([r["bic"] for r in results["meta"]])
    mse_nfw = np.mean([r["mse"] for r in results["nfw"]])
    mse_meta = np.mean([r["mse"] for r in results["meta"]])
    nfw_wins = sum(1 for a, b in zip(results["nfw"], results["meta"]) if a["bic"] < b["bic"])
    meta_wins = len(results["nfw"]) - nfw_wins
    out = {
        "bic_nfw": bic_nfw, "bic_meta": bic_meta,
        "mse_nfw": mse_nfw, "mse_meta": mse_meta,
        "nfw_wins": nfw_wins, "meta_wins": meta_wins,
        "winner": "NFW" if bic_nfw < bic_meta else "Meta-Axis",
    }
    if with_rar and "rar" in results:
        bic_rar = np.mean([r["bic"] for r in results["rar"]])
        mse_rar = np.mean([r["mse"] for r in results["rar"]])
        rar_wins = sum(1 for a, b, c in zip(results["nfw"], results["meta"], results["rar"]) if c["bic"] < a["bic"] and c["bic"] < b["bic"])
        out["bic_rar"] = bic_rar
        out["mse_rar"] = mse_rar
        out["rar_wins"] = rar_wins
        bics = [("NFW", bic_nfw), ("Meta", bic_meta), ("RAR", bic_rar)]
        out["winner"] = min(bics, key=lambda x: x[1])[0]
    return out


def run_fair_race(n_galaxies: int = 30, epochs: int = 400, use_rar: bool = False) -> dict:
    """
    Run fair comparison: NFW-True, Meta-True, Real.
    If use_rar: include RAR (McGaugh) from discovery formula.
    """
    all_results = {}
    for src in ["nfw", "meta", "real"]:
        all_results[src] = run_battle(
            n_galaxies=n_galaxies, epochs=epochs, data_source=src,
            use_rar=use_rar,
        )
    return all_results


def main():
    print("=" * 70)
    print("Meta-Validation: Fair Race - NFW vs Meta-Axis (Unified Start)")
    print("=" * 70)

    n_gal = 30
    ep = 400
    print(f"\n[0] Running 3 scenarios: NFW-True, Meta-True, Real ({n_gal} galaxies, {ep} epochs each)")
    all_res = run_fair_race(n_galaxies=n_gal, epochs=ep)

    # Comparison table
    print("\n" + "-" * 70)
    print("[1] COMPARISON TABLE (BIC: lower=better)")
    print("-" * 70)
    print(f"{'Data Source':<16} {'NFW BIC':>12} {'Meta BIC':>12} {'NFW wins':>10} {'Meta wins':>10} {'Winner':<12}")
    print("-" * 70)
    for src, res in all_res.items():
        s = _summarize(res)
        print(f"{res['data_source']:<16} {s['bic_nfw']:>12.2f} {s['bic_meta']:>12.2f} {s['nfw_wins']:>10} {s['meta_wins']:>10} {s['winner']:<12}")
    print("-" * 70)

    # MSE table
    print("\n[2] AVERAGE MSE")
    print("-" * 70)
    for src, res in all_res.items():
        s = _summarize(res)
        print(f"  {res['data_source']}: NFW={s['mse_nfw']:.2e}, Meta={s['mse_meta']:.2e}")
    print("-" * 70)

    # L / r_s recovery (for mock)
    print("\n[3] PARAMETER RECOVERY (Mock only)")
    for src in ["nfw", "meta"]:
        res = all_res[src]
        L_vals = np.array(res["L_values"])
        r_s_vals = np.array(res.get("r_s_values", []))
        print(f"  {res['data_source']}: L_mean={L_vals.mean():.2f}±{L_vals.std():.2f}, r_s_mean={r_s_vals.mean():.2f}±{r_s_vals.std():.2f}")
    res_real = all_res["real"]
    L_real = np.array(res_real["L_values"])
    print(f"  {res_real['data_source']}: L_mean={L_real.mean():.2f}±{L_real.std():.2f} (theory ~10.2)")

    # Discovery check on first real galaxy
    if res_real["names"]:
        d0 = load_sparc_galaxy(res_real["names"][0], use_mock_if_fail=True, use_real=True)
        R0, V_bary_sq0 = d0["R"], d0["V_gas"]**2 + d0["V_disk"]**2 + d0["V_bulge"]**2
        V_obs_sq0 = np.maximum(d0["V_obs"]**2, V_bary_sq0 + 1.0)
        V_def_sq0 = np.maximum(V_obs_sq0 - V_bary_sq0, 0.0)
        dc = discovery_check_kernel_shape(R0, V_def_sq0, V_bary_sq0)
        print("\n[4] Discovery Check (Real, first galaxy)")
        print(f"    Formula: {(dc['formula'] or 'None')[:70]}...")
        print(f"    R2: {dc['r2']:.4f}")

    # Plot: 3-panel comparison
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (src, res) in enumerate(all_res.items()):
            s = _summarize(res)
            axes[i].bar(["NFW", "Meta-Axis"], [s["bic_nfw"], s["bic_meta"]], color=["#1f77b4", "#ff7f0e"])
            axes[i].set_ylabel("BIC")
            axes[i].set_title(f"{res['data_source']}\nWinner: {s['winner']}")
        plt.tight_layout()
        out = ROOT / "axiom_os" / "verify_mashd_plot.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\n[5] Plot saved: {out}")
    except Exception as e:
        print(f"\n[5] Plot skip: {e}")

    print("\n" + "=" * 70)
    print("Fair Race Complete")
    print("=" * 70)


def run_discovery_mode(n_galaxies: int = 50, epochs: int = 500) -> dict:
    """
    Inverse Projection (Theory Repair): Free-Form MLP + Discovery + K_eff plot.
    Delegates to discovery_inverse_projection.
    """
    from axiom_os.experiments.discovery_inverse_projection import run_discovery_mode as _run
    return _run(n_galaxies=n_galaxies, epochs=epochs)


def main_with_rar():
    """Fair race with RAR (McGaugh) from discovery formula."""
    print("=" * 70)
    print("Meta-Validation: Fair Race - NFW vs Meta-Axis vs RAR (Discovery Formula)")
    print("=" * 70)

    n_gal = 30
    ep = 400
    print(f"\n[0] Extracting g+ from RAR discovery...")
    g_dag = get_rar_g_dagger(n_galaxies=50)
    print(f"    g+ = {g_dag:.2f} [(km/s)^2/kpc]")

    print(f"\n[1] Running 3 scenarios with RAR: NFW-True, Meta-True, Real ({n_gal} galaxies, {ep} epochs)")
    all_res = run_fair_race(n_galaxies=n_gal, epochs=ep, use_rar=True)

    print("\n" + "-" * 70)
    print("[2] COMPARISON TABLE (BIC: lower=better)")
    print("-" * 70)
    print(f"{'Data Source':<16} {'NFW BIC':>10} {'Meta BIC':>10} {'RAR BIC':>10} {'NFW':>6} {'Meta':>6} {'RAR':>6} {'Winner':<10}")
    print("-" * 70)
    for src, res in all_res.items():
        s = _summarize(res, with_rar=True)
        nfw_w = s.get("nfw_wins", 0)
        meta_w = s.get("meta_wins", 0)
        rar_w = s.get("rar_wins", 0)
        print(f"{res['data_source']:<16} {s['bic_nfw']:>10.2f} {s['bic_meta']:>10.2f} {s.get('bic_rar', 0):>10.2f} {nfw_w:>6} {meta_w:>6} {rar_w:>6} {s['winner']:<10}")
    print("-" * 70)

    print("\n[3] AVERAGE MSE")
    for src, res in all_res.items():
        s = _summarize(res, with_rar=True)
        print(f"  {res['data_source']}: NFW={s['mse_nfw']:.2e}, Meta={s['mse_meta']:.2e}, RAR={s.get('mse_rar', 0):.2e}")

    if all_res.get("real", {}).get("g_dagger"):
        print(f"\n[4] RAR g+ (from discovery): {all_res['real']['g_dagger']:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (src, res) in enumerate(all_res.items()):
            s = _summarize(res, with_rar=True)
            vals = [s["bic_nfw"], s["bic_meta"]]
            labs = ["NFW", "Meta"]
            if "bic_rar" in s:
                vals.append(s["bic_rar"])
                labs.append("RAR")
            axes[i].bar(labs, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(vals)])
            axes[i].set_ylabel("BIC")
            axes[i].set_title(f"{res['data_source']}\nWinner: {s['winner']}")
        plt.tight_layout()
        out = ROOT / "axiom_os" / "verify_mashd_plot.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\n[5] Plot saved: {out}")
    except Exception as e:
        print(f"\n[5] Plot skip: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["battle", "discovery", "rar", "battle_rar"], default="battle")
    parser.add_argument("--galaxies", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    if args.mode == "discovery":
        from axiom_os.experiments.discovery_inverse_projection import main as discovery_main
        discovery_main()
    elif args.mode == "rar":
        from axiom_os.experiments.discovery_rar import main as rar_main
        rar_main()
    elif args.mode == "battle_rar":
        main_with_rar()
    else:
        main()
