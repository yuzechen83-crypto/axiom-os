"""
Discovery 发现能力基准
在混沌复杂的现实数据下，评估发现的公式与已知物理定律的相似度。
证明系统具有从复杂现实中复现规律的能力。

运行: python -m axiom_os.benchmarks.discovery_capability_benchmark
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============ 已知定律 (Ground Truth) ============
GROUND_TRUTH = {
    "rar": {
        "form": "g_obs = g_bar * nu(g_bar/g†), nu 为 g_bar 的普适函数",
        "structure_hints": ["g_bar", "log_g_bar"],
        "param_g_dagger_range": (300.0, 5000.0),  # (km/s)^2/kpc
    },
    "theory_validator": {
        "form": "V_def^2 = C * V_bary_sq / r^alpha (rho 形式)",
        "structure_hints": ["v_bary", "r", "/", "r^"],
        "param_alpha_range": (1.0, 3.0),
    },
    "inverse_projection": {
        "form": "Sigma_halo = C * Sigma_baryon / r^beta",
        "structure_hints": ["sigma_baryon", "r", "/"],
        "param_beta_range": (0.0, 3.0),
    },
}


def _similarity_rar(
    formula_nu: Optional[str],
    formula_mond: Optional[str],
    g_dagger: Optional[float],
) -> Dict[str, Any]:
    """
    RAR 相似度：nu(g_bar) 形式 + g† 参数合理性
    """
    result = {
        "formula_discovered": formula_nu or formula_mond or "",
        "formula_ground_truth": GROUND_TRUTH["rar"]["form"],
        "structure_score": 0.0,
        "param_score": 0.0,
        "overall_similarity": 0.0,
        "details": {},
    }
    f = (formula_nu or formula_mond or "").lower()
    if not f:
        return result

    # 结构：是否包含 g_bar 或 log_g_bar（普适 nu 形式）
    has_g_bar = "g_bar" in f
    has_log_g_bar = "log_g_bar" in f
    structure_score = 0.0
    if has_g_bar or has_log_g_bar:
        structure_score = 0.5
        if has_log_g_bar:
            structure_score = 0.7  # log 形式更接近 RAR 经验
        if has_g_bar and has_log_g_bar:
            structure_score = 1.0
    result["details"]["has_g_bar"] = has_g_bar
    result["details"]["has_log_g_bar"] = has_log_g_bar
    result["structure_score"] = structure_score

    # 参数：g† 是否在合理范围
    lo, hi = GROUND_TRUTH["rar"]["param_g_dagger_range"]
    param_score = 0.0
    if g_dagger is not None and lo <= g_dagger <= hi:
        param_score = 1.0
    elif g_dagger is not None:
        # 量级正确（同数量级）也算部分分
        if 100 <= g_dagger <= 10000:
            param_score = 0.5
        elif 50 <= g_dagger <= 20000:
            param_score = 0.3
    result["details"]["g_dagger"] = g_dagger
    result["details"]["g_dagger_valid"] = lo <= (g_dagger or 0) <= hi
    result["param_score"] = param_score

    # 综合：结构 60% + 参数 40%
    result["overall_similarity"] = 0.6 * structure_score + 0.4 * param_score
    return result


def _similarity_theory(
    formula: Optional[str],
    theory_match_score: float,
    has_rho_form: bool,
    rho_form_r2: float,
) -> Dict[str, Any]:
    """
    Theory Validator 相似度：rho 形式 V_def^2 ~ V_bary_sq/r^alpha
    """
    result = {
        "formula_discovered": formula or "",
        "formula_ground_truth": GROUND_TRUTH["theory_validator"]["form"],
        "structure_score": 0.0,
        "theory_match_score": theory_match_score,
        "overall_similarity": 0.0,
        "details": {},
    }
    f = (formula or "").lower()
    if not f:
        result["overall_similarity"] = theory_match_score
        return result

    # 结构：是否有 V_bary_sq 与 r，是否有 /r 形式
    has_v_bary = "v_bary" in f or "x1" in f
    has_r = "r" in f or "x0" in f
    has_rho = ("/" in formula or "**-" in formula) and (has_r or "x0" in f)
    structure_score = 0.0
    if has_v_bary:
        structure_score += 0.33
    if has_r:
        structure_score += 0.33
    if has_rho:
        structure_score += 0.34
    result["details"]["has_v_bary_sq"] = has_v_bary
    result["details"]["has_r"] = has_r
    result["details"]["has_rho_form"] = has_rho
    result["structure_score"] = structure_score

    # 综合：结构 40% + theory_match 60%
    result["overall_similarity"] = 0.4 * structure_score + 0.6 * theory_match_score
    return result


def _similarity_inverse(
    formula: Optional[str],
    formula_theory: Optional[str],
    r2_theory: float,
) -> Dict[str, Any]:
    """
    Inverse Projection 相似度：Sigma_halo ~ Sigma_baryon/r^beta
    """
    result = {
        "formula_discovered": formula or "",
        "formula_ground_truth": GROUND_TRUTH["inverse_projection"]["form"],
        "formula_theory_fit": formula_theory or "",
        "structure_score": 0.0,
        "theory_fit_score": max(0.0, min(1.0, r2_theory)) if r2_theory is not None else 0.0,
        "overall_similarity": 0.0,
        "details": {},
    }
    f = (formula or "").lower()
    if not f:
        result["overall_similarity"] = result["theory_fit_score"]
        return result

    has_sigma = "sigma" in f or "sigma_baryon" in f
    has_r = "r" in f
    has_ratio = "/" in (formula or "") and has_r
    structure_score = 0.0
    if has_sigma:
        structure_score += 0.4
    if has_r:
        structure_score += 0.3
    if has_ratio:
        structure_score += 0.3
    result["details"]["has_sigma_baryon"] = has_sigma
    result["details"]["has_r"] = has_r
    result["details"]["has_ratio_form"] = has_ratio
    result["structure_score"] = structure_score

    result["overall_similarity"] = 0.5 * structure_score + 0.5 * result["theory_fit_score"]
    return result


def run_benchmark(
    n_galaxies: int = 25,
    rar_epochs: int = 400,
    use_pysr: bool = True,
) -> Dict[str, Any]:
    """运行完整管线并计算发现能力相似度"""
    from axiom_os.benchmarks.seed_utils import set_global_seed
    from axiom_os.experiments.dark_matter_discovery import (
        run_rar_pipeline,
        run_theory_validator_pipeline,
        run_inverse_projection_pipeline,
    )

    set_global_seed(42)
    results = {}
    similarities = {}

    # 1. RAR
    print("\n[1/3] RAR Discovery...")
    try:
        rar = run_rar_pipeline(n_galaxies=n_galaxies, epochs=rar_epochs, use_pysr=use_pysr)
        results["rar"] = rar
        sim = _similarity_rar(
            rar.get("formula_nu"),
            rar.get("form_mond"),
            rar.get("g_dagger"),
        )
        similarities["rar"] = sim
        print(f"    formula_nu: {str(rar.get('formula_nu', ''))[:60]}...")
        print(f"    g_dagger: {rar.get('g_dagger')}")
        print(f"    similarity: {sim['overall_similarity']:.3f}")
    except Exception as e:
        results["rar"] = {"ok": False, "error": str(e)}
        similarities["rar"] = {"overall_similarity": 0.0, "error": str(e)}
        print(f"    FAIL: {e}")

    # 2. Theory Validator
    print("\n[2/3] Theory Validator...")
    try:
        tv = run_theory_validator_pipeline(n_galaxies=min(15, n_galaxies), epochs=rar_epochs)
        results["theory_validator"] = tv
        tm = tv.get("theory_match") or {}
        tm_score = tm.get("score", tv.get("theory_match_score", 0))
        has_rho = tm.get("has_rho_form", False)
        rho_r2 = tm.get("rho_form_r2", 0.0)
        if not tm and tv.get("theory_match_details"):
            has_rho = "rho_form=True" in (tv.get("theory_match_details") or "")
            m = re.search(r"rho_R2=([\d.]+)", tv.get("theory_match_details", ""))
            if m:
                rho_r2 = float(m.group(1))
        sim = _similarity_theory(
            tv.get("formula_direct"),
            tm_score,
            has_rho,
            rho_r2,
        )
        similarities["theory_validator"] = sim
        print(f"    formula: {str(tv.get('formula_direct', ''))[:50]}...")
        print(f"    theory_match: {tm_score:.2f}, similarity: {sim['overall_similarity']:.3f}")
    except Exception as e:
        results["theory_validator"] = {"ok": False, "error": str(e)}
        similarities["theory_validator"] = {"overall_similarity": 0.0, "error": str(e)}
        print(f"    FAIL: {e}")

    # 3. Inverse Projection
    print("\n[3/3] Inverse Projection...")
    try:
        inv = run_inverse_projection_pipeline(n_galaxies=min(30, n_galaxies), epochs=300)
        results["inverse_projection"] = inv
        sim = _similarity_inverse(
            inv.get("formula"),
            inv.get("formula_theory"),
            inv.get("r2_theory"),
        )
        similarities["inverse_projection"] = sim
        print(f"    formula: {str(inv.get('formula', ''))[:50]}...")
        print(f"    r2_theory: {inv.get('r2_theory', 0):.3f}, similarity: {sim['overall_similarity']:.3f}")
    except Exception as e:
        results["inverse_projection"] = {"ok": False, "error": str(e)}
        similarities["inverse_projection"] = {"overall_similarity": 0.0, "error": str(e)}
        print(f"    FAIL: {e}")

    # 综合
    avg_similarity = sum(s.get("overall_similarity", 0) for s in similarities.values()) / max(len(similarities), 1)
    return {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_galaxies": n_galaxies, "use_pysr": use_pysr},
        "results": results,
        "similarities": similarities,
        "avg_similarity": float(avg_similarity),
        "ground_truth": GROUND_TRUTH,
    }


def _to_json_serializable(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj


def generate_report(data: Dict[str, Any], out_dir: Path) -> str:
    """生成发现能力基准报告"""
    sims = data.get("similarities", {})
    lines = [
        "# Discovery 发现能力基准报告",
        "",
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 评估目标",
        "",
        "在混沌复杂的现实数据（SPARC）下，评估发现的公式与**已知物理定律**的相似度。",
        "高相似度证明系统具有从复杂现实中复现规律的能力。",
        "",
        "## 已知定律 (Ground Truth)",
        "",
    ]
    for task, gt in data.get("ground_truth", GROUND_TRUTH).items():
        lines.append(f"- **{task}**: {gt['form']}")
    lines.extend(["", "## 相似度结果", ""])

    for task, sim in sims.items():
        if "error" in sim:
            lines.append(f"### {task}: 失败 - {sim['error']}")
        else:
            s = sim.get("overall_similarity", 0)
            pct = f"{s*100:.1f}%"
            fd = sim.get("formula_discovered", "") or ""
            fd_short = (fd[:80] + "...") if len(fd) > 80 else fd
            lines.extend([
                f"### {task}",
                f"- **相似度**: {pct}",
                f"- 发现公式: `{fd_short}`",
                f"- 已知定律: {sim.get('formula_ground_truth', '')}",
                "",
            ])
            if sim.get("details"):
                lines.append("  细节: " + str(sim["details"]))
                lines.append("")

    avg = data.get("avg_similarity", 0)
    lines.extend([
        "",
        f"## 综合",
        "",
        f"**平均相似度**: {avg*100:.1f}%",
        "",
        "---",
        "*Axiom-OS Discovery 发现能力基准*",
    ])
    report = "\n".join(lines)
    path = out_dir / "discovery_capability_report.md"
    path.write_text(report, encoding="utf-8")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Discovery 发现能力基准")
    parser.add_argument("--n-galaxies", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--no-pysr", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Discovery 发现能力基准")
    print("评估: 发现公式 vs 已知物理定律 相似度")
    print("=" * 60)

    data = run_benchmark(
        n_galaxies=args.n_galaxies,
        rar_epochs=args.epochs,
        use_pysr=not args.no_pysr,
    )

    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "discovery_capability_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(data), f, indent=2, ensure_ascii=False)

    generate_report(data, out_dir)
    print(f"\n平均相似度: {data['avg_similarity']*100:.1f}%")
    print(f"已保存: {json_path}")
    print(f"已保存: {out_dir / 'discovery_capability_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
