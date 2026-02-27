"""
暗物质发现统一管线
串联 RAR Discovery、Theory Validator、Inverse Projection，输出综合报告。

运行: python -m axiom_os.experiments.dark_matter_discovery
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _to_json_serializable(obj: Any) -> Any:
    """Convert numpy/torch to JSON-serializable types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):  # numpy scalar
        return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj


def run_rar_pipeline(n_galaxies: int = 30, epochs: int = 400, use_pysr: bool = True) -> Dict[str, Any]:
    """RAR 空间：g_obs vs g_bar，发现 nu(g_bar)"""
    from axiom_os.experiments.discovery_rar import run_rar_discovery

    res = run_rar_discovery(
        n_galaxies=n_galaxies,
        epochs=epochs,
        apply_mass_calibration=True,
        use_pysr=use_pysr,
    )
    if res.get("error"):
        return {"ok": False, "error": res["error"], "source": "rar"}
    cryst = res.get("crystallized") or {}
    return {
        "ok": True,
        "source": "rar",
        "r2_log": float(res.get("r2_log", 0)) if res.get("r2_log") is not None else None,
        "r2_linear": float(res.get("r2", 0)) if res.get("r2") is not None else None,
        "n_samples": int(res.get("n_samples", 0)),
        "n_galaxies": int(res.get("n_galaxies", 0)),
        "formula_nu": res.get("formula_nu"),
        "form_mond": res.get("form_mond"),
        "r2_mond": float(res.get("r2_mond", 0)) if res.get("r2_mond") is not None else None,
        "g_dagger": float(cryst.get("g_dagger")) if cryst.get("g_dagger") is not None else None,
    }


def run_theory_validator_pipeline(
    n_galaxies: int = 15,
    epochs: int = 400,
    data_source: str = "real",
) -> Dict[str, Any]:
    """理论验证：(r, V_bary²) -> V_def²，与 K(z) 积分形式对比"""
    from axiom_os.experiments.theory_validator import run_theory_validator

    res = run_theory_validator(
        n_galaxies=n_galaxies,
        epochs_per_galaxy=epochs,
        data_source=data_source,
    )
    tm = res.get("theory_match", {})
    return {
        "ok": True,
        "source": "theory_validator",
        "formula": res.get("formula"),
        "formula_direct": res.get("formula_direct"),
        "formula_theory": res.get("formula_theory"),
        "r2_direct": res.get("r2_direct"),
        "r2_theory": res.get("r2_theory"),
        "theory_match_score": tm.get("score", 0),
        "theory_match_details": tm.get("details", ""),
        "theory_match": tm,
        "n_samples": res.get("n_samples"),
        "data_source": res.get("data_source"),
    }


def run_inverse_projection_pipeline(
    n_galaxies: int = 30,
    epochs: int = 300,
) -> Dict[str, Any]:
    """逆投影：(Sigma_baryon, r) -> Sigma_halo"""
    from axiom_os.experiments.discovery_inverse_projection import run_discovery_mode

    res = run_discovery_mode(
        n_galaxies=n_galaxies,
        epochs=epochs,
    )
    return {
        "ok": True,
        "source": "inverse_projection",
        "formula": res.get("formula"),
        "formula_theory": res.get("formula_theory"),
        "r2_fit": res.get("r2_fit"),
        "r2_theory": res.get("r2_theory"),
        "n_samples": res.get("n_samples"),
        "n_galaxies": res.get("n_galaxies"),
    }


def generate_report(results: list, out_dir: Path) -> str:
    """生成 Markdown 报告"""
    lines = [
        "# 暗物质发现统一管线报告",
        "",
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. RAR 空间 (g_obs vs g_bar)",
        "",
    ]
    rar = next((r for r in results if r.get("source") == "rar"), None)
    if rar and rar.get("ok"):
        lines.extend([
            f"- **Log R²**: {rar.get('r2_log', 0):.4f}",
            f"- **样本数**: {rar.get('n_samples', 0)}",
            f"- **星系数**: {rar.get('n_galaxies', 0)}",
            f"- **ν(g_bar) 公式**: {rar.get('formula_nu', '—')}",
            f"- **McGaugh 形式**: {rar.get('form_mond', '—')} (R²={rar.get('r2_mond', 0):.4f})",
            f"- **g†**: {rar.get('g_dagger', '—')}",
            "",
        ])
    else:
        lines.append(f"- 状态: {rar.get('error', '未运行')}\n")

    lines.extend(["## 2. 理论验证 (Theory Validator)", ""])
    tv = next((r for r in results if r.get("source") == "theory_validator"), None)
    if tv and tv.get("ok"):
        lines.extend([
            f"- **发现公式**: {tv.get('formula_direct', '—')}",
            f"- **理论形式**: {tv.get('formula_theory', '—')}",
            f"- **理论匹配分数**: {tv.get('theory_match_score', 0):.2f}",
            f"- **样本数**: {tv.get('n_samples', 0)}",
            "",
        ])
    else:
        lines.append("- 未运行\n")

    lines.extend(["## 3. 逆投影 (Sigma_baryon, r) -> Sigma_halo", ""])
    inv = next((r for r in results if r.get("source") == "inverse_projection"), None)
    if inv and inv.get("ok"):
        lines.extend([
            f"- **发现公式**: {inv.get('formula', '—')}",
            f"- **理论形式**: {inv.get('formula_theory', '—')}",
            f"- **R²**: {inv.get('r2_fit', 0):.4f}",
            f"- **样本数**: {inv.get('n_samples', 0)}",
            "",
        ])
    else:
        lines.append("- 未运行\n")

    lines.append("---")
    lines.append("*Axiom-OS 暗物质发现管线*")

    report = "\n".join(lines)
    report_path = out_dir / "dark_matter_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report


def main(
    n_galaxies: int = 25,
    rar_epochs: int = 400,
    use_pysr: bool = True,
    run_rar: bool = True,
    run_theory: bool = True,
    run_inverse: bool = True,
) -> Dict[str, Any]:
    from axiom_os.benchmarks.seed_utils import set_global_seed

    set_global_seed(42)
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("暗物质发现统一管线")
    print("=" * 60)

    results = []

    if run_rar:
        print("\n[1/3] RAR Discovery (g_obs vs g_bar)...")
        try:
            r = run_rar_pipeline(n_galaxies=n_galaxies, epochs=rar_epochs, use_pysr=use_pysr)
            results.append(r)
            if r.get("ok"):
                print(f"    OK  R2_log={r.get('r2_log', 0):.4f}  n={r.get('n_samples', 0)}")
            else:
                print(f"    FAIL: {r.get('error', 'unknown')}")
        except Exception as e:
            results.append({"ok": False, "source": "rar", "error": str(e)})
            print(f"    FAIL: {e}")

    if run_theory:
        print("\n[2/3] Theory Validator (r, V_bary^2) -> V_def^2...")
        try:
            r = run_theory_validator_pipeline(n_galaxies=min(15, n_galaxies), epochs=rar_epochs)
            results.append(r)
            if r.get("ok"):
                print(f"    OK  match={r.get('theory_match_score', 0):.2f}  formula={str(r.get('formula_direct', ''))[:50]}...")
            else:
                print(f"    FAIL: {r.get('error', 'unknown')}")
        except Exception as e:
            results.append({"ok": False, "source": "theory_validator", "error": str(e)})
            print(f"    FAIL: {e}")

    if run_inverse:
        print("\n[3/3] Inverse Projection (Sigma_baryon, r) -> Sigma_halo...")
        try:
            r = run_inverse_projection_pipeline(n_galaxies=min(30, n_galaxies), epochs=300)
            results.append(r)
            if r.get("ok"):
                print(f"    OK  R2={r.get('r2_fit', 0):.4f}  n={r.get('n_samples', 0)}")
            else:
                print(f"    FAIL: {r.get('error', 'unknown')}")
        except Exception as e:
            results.append({"ok": False, "source": "inverse_projection", "error": str(e)})
            print(f"    FAIL: {e}")

    # 保存 JSON（确保可序列化）
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_galaxies": n_galaxies, "use_pysr": use_pysr},
        "results": _to_json_serializable(results),
    }
    json_path = out_dir / "dark_matter_discovery.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成报告
    report = generate_report(results, out_dir)
    print(f"\n已保存: {json_path}")
    print(f"已保存: {out_dir / 'dark_matter_report.md'}")
    print("=" * 60)

    return {"results": results, "report": report}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="暗物质发现统一管线")
    parser.add_argument("--n-galaxies", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--no-pysr", action="store_true", help="禁用 PySR")
    parser.add_argument("--rar-only", action="store_true", help="仅 RAR")
    parser.add_argument("--theory-only", action="store_true", help="仅 Theory Validator")
    parser.add_argument("--inverse-only", action="store_true", help="仅 Inverse Projection")
    args = parser.parse_args()

    run_rar = args.rar_only or not (args.theory_only or args.inverse_only)
    run_theory = args.theory_only or not (args.rar_only or args.inverse_only)
    run_inverse = args.inverse_only or not (args.rar_only or args.theory_only)

    main(
        n_galaxies=args.n_galaxies,
        rar_epochs=args.epochs,
        use_pysr=not args.no_pysr,
        run_rar=run_rar,
        run_theory=run_theory,
        run_inverse=run_inverse,
    )
