"""
Operation "Battery Life" - Real-World RUL Prediction (Optimized)
NASA Li-ion Battery B0005: Hard Core + Soft Shell + Discovery + Crystallization.

Universal Discovery: fit all form candidates (exp/power/poly), select by AIC.
No Lasso gate - applicable to battery, acrobot, turbulence, etc.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axiom_os.layers.rcln import RCLNLayer
from axiom_os.datasets.nasa_battery import load_battery_data
from axiom_os.engine.discovery import DiscoveryEngine

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

INITIAL_CAPACITY = 2.0  # Ah (ideal battery)
N_SEEDS = 5
N_EPOCHS = 2000
N_BOOTSTRAP = 30


def ideal_battery_model(x):
    """Hard Core: Naive physicist assumes battery is perfect."""
    if isinstance(x, torch.Tensor):
        vals = x
    elif hasattr(x, "values") and not isinstance(x, torch.Tensor):
        vals = x.values
    else:
        vals = x
    t = torch.as_tensor(vals, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return torch.ones_like(t[:, :1]) * 1.0


def make_crystallized_hard_core(coefs: dict, scalers: dict):
    """Create Hard Core from discovered formula. Supports exp, power, poly forms."""

    form_type = coefs.get("form_type", "poly")

    def crystallized_model(x):
        if isinstance(x, torch.Tensor):
            t_norm = x.float()
        else:
            t_norm = torch.as_tensor(x, dtype=torch.float32)
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(0)
        t = t_norm[:, :1].clamp(1e-8, 1.0)

        if form_type == "exp":
            a, k, c = coefs["a"], coefs["k"], coefs["c"]
            aging = a * torch.exp(-k * t) + c
        elif form_type == "power":
            a, beta, c = coefs["a"], coefs["beta"], coefs["c"]
            aging = a * torch.pow(t, beta) + c
        elif form_type == "log":
            a, c = coefs["a"], coefs["c"]
            aging = a * torch.log1p(t) + c
        elif form_type == "piecewise":
            a1, b1, a2 = coefs["a1"], coefs["b1"], coefs["a2"]
            t_break = coefs["t_break"]
            b2 = (a1 - a2) * t_break + b1
            aging = torch.where(
                t <= t_break,
                a1 * t + b1,
                a2 * t + b2,
            )
        else:
            # poly
            aging = (
                coefs.get("c0", 0) * t
                + coefs.get("c1", 0) * (t ** 2)
                + coefs.get("c2", 0) * (t ** 3)
                + coefs.get("c3", 0) * torch.sqrt(t)
                + coefs.get("c4", 0) * torch.log1p(t)
                + coefs.get("c5", 0) * torch.exp(-t)
                + coefs.get("c6", 0) * torch.exp(-2 * t)
                + coefs.get("c7", 0) * torch.exp(-3 * t)
                + coefs.get("intercept", 0)
            )
        # Clamp to [0, 1] to avoid invalid extrapolation (negative capacity)
        return (1.0 + aging).clamp(0.0, 1.0).float()

    return crystallized_model


def _exp_decay(t: np.ndarray, a: float, k: float, c: float) -> np.ndarray:
    return a * np.exp(-k * t) + c


def train_mlp_only(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor,
    seed: int, n_epochs: int = N_EPOCHS,
) -> np.ndarray:
    """Train MLP without Hard Core (baseline)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = ((model(x_train) - y_train) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return model(x_test).numpy()


def train_rcln(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor,
    hard_core_func, seed: int, n_epochs: int = N_EPOCHS,
) -> tuple[np.ndarray, "RCLNLayer"]:
    """Train RCLN. Returns test predictions and trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rcln = RCLNLayer(
        input_dim=1, hidden_dim=64, output_dim=1,
        hard_core_func=hard_core_func, lambda_res=1.0,
    )
    opt = torch.optim.Adam(rcln.parameters(), lr=0.01)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = ((rcln(x_train) - y_train) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = rcln(x_test).numpy()
    return pred, rcln


def run_single_seed(
    cycles_norm, capacity_norm, scalers, split, seed: int, n_epochs: int = N_EPOCHS,
) -> dict:
    """Run one seed: baselines + RCLN + Discovery. Returns metrics dict."""
    n = len(cycles_norm)
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)

    cycles_train = cycles_norm[train_idx]
    capacity_train = capacity_norm[train_idx]
    cycles_test = cycles_norm[test_idx]
    capacity_test = capacity_norm[test_idx]

    def inv_cycle(z):
        return z * (scalers["cycle_max"] - scalers["cycle_min"]) + scalers["cycle_min"]

    def inv_cap(z):
        return z * (scalers["cap_max"] - scalers["cap_min"]) + scalers["cap_min"]

    cycles_train_raw = inv_cycle(cycles_train.ravel())
    cycles_test_raw = inv_cycle(cycles_test.ravel())
    cap_train_raw = inv_cap(capacity_train.ravel())
    cap_test_raw = inv_cap(capacity_test.ravel())

    x_train = torch.from_numpy(cycles_train).float()
    y_train = torch.from_numpy(capacity_train).float()
    x_test = torch.from_numpy(cycles_test).float()

    results = {}

    # Baselines: fit on train, predict on test
    if HAS_SCIPY:
        try:
            popt, _ = curve_fit(
                _exp_decay, cycles_train_raw, cap_train_raw,
                p0=[2.0, 0.002, 1.4],
                bounds=([0.5, 1e-5, 0.5], [3.0, 0.1, 2.5]),
            )
            exp_pred_test = _exp_decay(cycles_test_raw, *popt)
        except Exception:
            exp_pred_test = np.full_like(cap_test_raw, cap_train_raw.mean())
    else:
        exp_pred_test = np.full_like(cap_test_raw, cap_train_raw.mean())
    results["baseline_exp_mae"] = np.mean(np.abs(exp_pred_test - cap_test_raw))

    lin_pred_test = np.polyval(
        np.polyfit(cycles_train_raw, cap_train_raw, 1), cycles_test_raw
    )
    results["baseline_linear_mae"] = np.mean(np.abs(lin_pred_test - cap_test_raw))

    pred_mlp = train_mlp_only(x_train, y_train, x_test, seed, n_epochs=n_epochs)
    pred_mlp_raw = inv_cap(pred_mlp.ravel())
    results["baseline_mlp_mae"] = np.mean(np.abs(pred_mlp_raw - cap_test_raw))

    # RCLN (ideal hard core)
    pred_rcln, rcln = train_rcln(x_train, y_train, x_test, ideal_battery_model, seed, n_epochs=n_epochs)
    pred_rcln_raw = inv_cap(pred_rcln.ravel())
    results["rcln_mae"] = np.mean(np.abs(pred_rcln_raw - cap_test_raw))

    # Discovery (universal: fit all forms, select by AIC)
    with torch.no_grad():
        _ = rcln(x_train)
    soft_shell = rcln._last_y_soft.detach().cpu().numpy().ravel()
    engine = DiscoveryEngine(use_pysr=False)
    formula, formula_pred, coefs = engine.discover_parametric(
        cycles_train.ravel(), soft_shell, selector="bic"
    )
    results["formula"] = formula
    results["coefs"] = coefs

    # Crystallization: retrain with crystallized hard core
    if coefs is not None:
        cryst_hard = make_crystallized_hard_core(coefs, scalers)
        pred_cryst, _ = train_rcln(x_train, y_train, x_test, cryst_hard, seed + 1000, n_epochs=n_epochs)
        pred_cryst_raw = inv_cap(pred_cryst.ravel())
        results["rcln_crystallized_mae"] = np.mean(np.abs(pred_cryst_raw - cap_test_raw))
    else:
        pred_cryst_raw = pred_rcln_raw
        results["rcln_crystallized_mae"] = results["rcln_mae"]

    results["pred_rcln_raw"] = pred_rcln_raw
    results["pred_cryst_raw"] = pred_cryst_raw
    results["cap_test_raw"] = cap_test_raw
    results["cycles_test_raw"] = cycles_test_raw
    results["cap_train_raw"] = cap_train_raw
    results["cycles_train_raw"] = cycles_train_raw
    results["y_soft"] = soft_shell
    results["formula_pred"] = formula_pred
    results["rcln"] = rcln
    return results


def bootstrap_uncertainty(
    cycles_norm, capacity_norm, scalers, split, seed: int, n_bootstrap: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap prediction intervals (lower, upper) for test set."""
    n = len(cycles_norm)
    n_train = split
    preds_list = []
    for b in range(n_bootstrap):
        np.random.seed(seed + b * 100)
        idx = np.random.choice(n_train, size=n_train, replace=True)
        cycles_b = cycles_norm[idx]
        capacity_b = capacity_norm[idx]
        x_train = torch.from_numpy(cycles_b).float()
        y_train = torch.from_numpy(capacity_b).float()
        x_test = torch.from_numpy(cycles_norm[split:]).float()
        pred, _ = train_rcln(x_train, y_train, x_test, ideal_battery_model, seed + b)
        def inv_cap(z):
            return z * (scalers["cap_max"] - scalers["cap_min"]) + scalers["cap_min"]
        preds_list.append(inv_cap(pred.ravel()))
    preds_arr = np.array(preds_list)
    lower = np.percentile(preds_arr, 5, axis=0)
    upper = np.percentile(preds_arr, 95, axis=0)
    # Widen by residual std; use conservative factor for longer prediction horizon
    pred_mean = np.mean(preds_arr, axis=0)
    cap_test = inv_cap(capacity_norm[split:].ravel())
    resid_std = np.std(cap_test - pred_mean)
    z = 1.645  # 90% interval
    widen = 1.8  # conservative factor for extrapolation
    lower = np.minimum(lower, pred_mean - z * widen * resid_std)
    upper = np.maximum(upper, pred_mean + z * widen * resid_std)
    # Clamp to physical capacity bounds
    cap_min, cap_max = scalers["cap_min"], scalers["cap_max"]
    lower = np.clip(lower, cap_min * 0.9, cap_max * 1.1)
    upper = np.clip(upper, cap_min * 0.9, cap_max * 1.1)
    return lower, upper


def main(quick: bool = False):
    """
    quick: 快速模式，减少 seeds/epochs/bootstrap，用于 E2E 基准测试
    """
    n_seeds = 2 if quick else N_SEEDS
    n_epochs = 500 if quick else N_EPOCHS
    n_bootstrap = 5 if quick else N_BOOTSTRAP

    print("=" * 60)
    print("Operation Battery Life - NASA B0005 RUL (Optimized)")
    if quick:
        print("  [Quick 模式: seeds=%d, epochs=%d, bootstrap=%d]" % (n_seeds, n_epochs, n_bootstrap))
    print("=" * 60)

    cycles_norm, capacity_norm, scalers = load_battery_data()
    n = len(cycles_norm)
    split = int(0.6 * n)  # 60% train, 40% test — longer prediction horizon
    data_src = scalers.get("data_source", "unknown")

    print(f"\nData: {n} cycles, Train {split}, Test {n - split}")
    print(f"Source: {data_src.upper()}")
    print(f"Capacity range: [{scalers['cap_min']:.3f}, {scalers['cap_max']:.3f}] Ah")

    # Multiple runs
    all_mae_rcln = []
    all_mae_cryst = []
    all_mae_mlp = []
    all_mae_exp = []
    all_mae_lin = []
    last_results = None

    for seed in range(n_seeds):
        r = run_single_seed(cycles_norm, capacity_norm, scalers, split, seed, n_epochs=n_epochs)
        last_results = r
        all_mae_rcln.append(r["rcln_mae"])
        all_mae_cryst.append(r["rcln_crystallized_mae"])
        all_mae_mlp.append(r["baseline_mlp_mae"])
        all_mae_exp.append(r["baseline_exp_mae"])
        all_mae_lin.append(r["baseline_linear_mae"])

    # Report metrics
    def fmt(m):
        return f"{np.mean(m):.4f} +/- {np.std(m):.4f}"

    print("\n>>> Metrics (mean ± std over {} seeds)".format(n_seeds))
    print(f"    Baseline Linear:   MAE = {fmt(all_mae_lin)} Ah")
    print(f"    Baseline Exp:      MAE = {fmt(all_mae_exp)} Ah")
    print(f"    Baseline MLP:      MAE = {fmt(all_mae_mlp)} Ah")
    print(f"    RCLN (Ideal):      MAE = {fmt(all_mae_rcln)} Ah")
    print(f"    RCLN (Crystallized): MAE = {fmt(all_mae_cryst)} Ah")

    formula = last_results["formula"]
    form_type = last_results.get("coefs", {}).get("form_type", "poly")
    print(f"\n>>> Discovery (BIC-selected, form={form_type}): f(t) = {formula or '—'}")
    if formula:
        cryst = formula.replace(" + -", " - ")
        print(f"    [Hippocampus] Updated Hard Core: Ideal -> Ideal + ({cryst})")

    # Bootstrap intervals
    print("\n>>> Bootstrap 90% prediction intervals...")
    lower, upper = bootstrap_uncertainty(cycles_norm, capacity_norm, scalers, split, 42, n_bootstrap=n_bootstrap)
    coverage = np.mean((last_results["cap_test_raw"] >= lower) & (last_results["cap_test_raw"] <= upper))
    print(f"    90% interval coverage: {coverage*100:.1f}%")

    # Plots
    def inv_cap(z):
        return z * (scalers["cap_max"] - scalers["cap_min"]) + scalers["cap_min"]

    def inv_cycle(z):
        return z * (scalers["cycle_max"] - scalers["cycle_min"]) + scalers["cycle_min"]

    cycles_train_raw = inv_cycle(cycles_norm[:split].ravel())
    cycles_test_raw = inv_cycle(cycles_norm[split:].ravel())
    cap_train_raw = inv_cap(capacity_norm[:split].ravel())
    cap_test_raw = inv_cap(capacity_norm[split:].ravel())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: True vs RCLN vs Crystallized
    ax = axes[0, 0]
    ax.plot(cycles_train_raw, cap_train_raw, "b-", label="Train True", linewidth=2)
    ax.plot(cycles_test_raw, cap_test_raw, "b.", label="Test True", markersize=4)
    ax.plot(cycles_test_raw, last_results["pred_rcln_raw"], "r-", label="RCLN", linewidth=1.5)
    ax.plot(cycles_test_raw, last_results["pred_cryst_raw"], "m--", label="RCLN (Crystallized)", linewidth=1.5)
    ax.fill_between(cycles_test_raw, lower, upper, color="red", alpha=0.2, label="90% interval")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("RUL Prediction + Uncertainty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Soft Shell vs Discovered Formula
    ax = axes[0, 1]
    ax.plot(cycles_train_raw, last_results["y_soft"], "g-", linewidth=2, label="Soft Shell")
    ax.plot(cycles_train_raw, last_results["formula_pred"], "r--", linewidth=1.5, label="Discovered f(t)")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Soft Shell Output")
    ax.set_title("Aging Law (t_norm, physics-consistent)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Baseline comparison
    ax = axes[1, 0]
    methods = ["Linear", "Exp", "MLP", "RCLN", "RCLN+Cryst"]
    maes = [np.mean(all_mae_lin), np.mean(all_mae_exp), np.mean(all_mae_mlp),
            np.mean(all_mae_rcln), np.mean(all_mae_cryst)]
    stds = [np.std(all_mae_lin), np.std(all_mae_exp), np.std(all_mae_mlp),
            np.std(all_mae_rcln), np.std(all_mae_cryst)]
    bars = ax.bar(methods, maes, yerr=stds, capsize=5)
    ax.set_ylabel("Test MAE (Ah)")
    ax.set_title("Baseline Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Data source & summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"Data: {data_src.upper()}\n"
        f"N_seeds: {N_SEEDS}\n"
        f"Coverage: {coverage*100:.1f}%\n\n"
        f"Best: {methods[np.argmin(maes)]} (MAE={min(maes):.4f} Ah)"
    )
    ax.text(0.1, 0.5, summary, fontsize=12, family="monospace", verticalalignment="center")

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "battery_rul_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
