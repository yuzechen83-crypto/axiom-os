"""
RAR 训练收敛性诊断：不同 epochs 下的 Log R² / Linear R²，收敛曲线与建议。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def check_rar_convergence(
    n_galaxies: int = 20,
    max_epochs: int = 600,
    step: int = 50,
) -> dict:
    """
    对 epochs = [50, 100, ..., max_epochs] 运行 RAR，记录 r2_linear / r2_log，
    打印表格（含 Δ R²、收敛状态），绘图并保存为 convergence_analysis.png。
    返回: {epochs_list, r2_log_list, r2_linear_list, is_converged, recommended_epochs}
    """
    from axiom_os.experiments.discovery_rar import run_rar_discovery

    # 固定为 [50, 100, 150, 200, 300, 400, 500, 600] 或按 step 生成到 max_epochs
    default_epochs = [50, 100, 150, 200, 300, 400, 500, 600]
    epochs_list = [e for e in default_epochs if e <= max_epochs]
    if not epochs_list or epochs_list[-1] != max_epochs:
        if max_epochs not in epochs_list:
            epochs_list.append(max_epochs)
        epochs_list = sorted(set(epochs_list))

    r2_log_list = []
    r2_linear_list = []
    for ep in epochs_list:
        res = run_rar_discovery(
            n_galaxies=n_galaxies,
            epochs=ep,
            apply_mass_calibration=True,
        )
        if res.get("error"):
            r2_log_list.append(None)
            r2_linear_list.append(None)
            continue
        r2_log_list.append(res.get("r2_log"))
        r2_linear_list.append(res.get("r2"))

    # 表格：Epochs | Log R² | Δ R² | 收敛状态
    print("\n" + "=" * 56)
    print("RAR convergence diagnostic")
    print("=" * 56)
    print(f"{'Epochs':>8} | {'Log R2':>7} | {'dR2':>7} | Status")
    print("-" * 56)
    prev_log = None
    deltas = []
    for i, ep in enumerate(epochs_list):
        r2_log = r2_log_list[i] if i < len(r2_log_list) else None
        delta_str = "  -"
        if r2_log is not None and prev_log is not None:
            delta = r2_log - prev_log
            deltas.append(delta)
            delta_str = f"{delta:+.3f}"
        else:
            if i > 0:
                deltas.append(None)
        status = "initial" if i == 0 else ""
        if i > 0 and r2_log is not None and prev_log is not None:
            delta_val = r2_log - prev_log
            if abs(delta_val) < 0.01:
                status = "OK converged (d<1%)"
            elif delta_val > 0.03:
                status = "need more epochs"
            else:
                status = "flattening"
        r2_str = f"{r2_log:.3f}" if r2_log is not None else "  -"
        print(f"{ep:>8} | {r2_str:>7} | {delta_str:>7} | {status}")
        if r2_log is not None:
            prev_log = r2_log

    # 判断是否收敛、建议 epochs
    is_converged = False
    recommended_epochs = max_epochs
    valid_deltas = [d for d in deltas if d is not None]
    if len(valid_deltas) >= 2:
        last_two = valid_deltas[-2:]
        if all(abs(d) < 0.01 for d in last_two):
            is_converged = True
            recommended_epochs = epochs_list[-1] if epochs_list else max_epochs
        elif any(d and d > 0.03 for d in last_two):
            recommended_epochs = max_epochs + 100
    print("-" * 56)
    if is_converged:
        print("Conclusion: converged. STANDARD 400 epochs is reasonable.")
    else:
        print(f"Conclusion: recommend epochs >= {recommended_epochs}.")

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt and r2_log_list and any(x is not None for x in r2_log_list):
        fig, ax = plt.subplots(figsize=(8, 5))
        x, y_log, y_lin = [], [], []
        for i, ep in enumerate(epochs_list):
            if i < len(r2_log_list) and r2_log_list[i] is not None:
                x.append(ep)
                y_log.append(r2_log_list[i])
                y_lin.append(r2_linear_list[i] if i < len(r2_linear_list) and r2_linear_list[i] is not None else None)
        ax.plot(x, y_log, "o-", color="red", label="Log R²")
        y_lin_plot = [v for v in y_lin if v is not None]
        if len(y_lin_plot) == len(x):
            ax.plot(x, y_lin_plot, "s-", color="blue", label="Linear R²")
        ax.axhline(0.92, color="gray", linestyle="--", label="theory limit 0.92")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("R²")
        ax.set_title("RAR convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = Path(__file__).resolve().parent / "convergence_analysis.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"Chart saved: {out_path}")
    else:
        out_path = None

    return {
        "epochs_list": epochs_list,
        "r2_log_list": r2_log_list,
        "r2_linear_list": r2_linear_list,
        "is_converged": is_converged,
        "recommended_epochs": recommended_epochs,
    }


if __name__ == "__main__":
    check_rar_convergence(n_galaxies=20, max_epochs=600, step=50)
