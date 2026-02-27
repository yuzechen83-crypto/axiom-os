"""
公式结晶 - Formula Crystallization

运行 RAR Discovery，提取符号公式并保存到 JSON。
可选：结晶到 Hippocampus。

用法:
  python -m axiom_os.experiments.crystallize_formulas
  python -m axiom_os.experiments.crystallize_formulas --rar-only
  python -m axiom_os.experiments.crystallize_formulas --to-hippocampus
"""

import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def crystallize_rar() -> dict:
    """RAR 公式结晶：run_rar_discovery 内含 crystallize_rar_law。"""
    from axiom_os.experiments.discovery_rar import run_rar_discovery

    res = run_rar_discovery(n_galaxies=80, epochs=400)
    if "error" in res:
        return {"ok": False, "error": res["error"], "crystallized": {}}

    cry = res.get("crystallized", {})
    return {
        "ok": True,
        "domain": "rar",
        "n_galaxies": res.get("n_galaxies", 0),
        "n_samples": res.get("n_samples", 0),
        "r2": res.get("r2"),
        "r2_log": res.get("r2_log"),
        "crystallized": {
            "g_dagger": cry.get("g_dagger"),
            "a0_or_gdagger": cry.get("a0_or_gdagger"),
            "g_dagger_prior": cry.get("g_dagger_prior"),
            "a0_si": cry.get("a0_si"),
            "r2_log_rar": cry.get("r2_log_rar"),
            "formulas": cry.get("formulas", []),
            "r2_scores": cry.get("r2_scores", []),
            "best3": [
                {"name": r[0], "formula": r[1], "r2": r[2]}
                for r in (cry.get("best3") or [])
            ],
        },
    }


def save_crystallized(out: dict, out_dir: Path) -> Path:
    """保存结晶结果到 JSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "crystallized_formulas.json"
    # 序列化：numpy 转 float，best3 中 pred 数组不保存
    def _serialize(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(x) for x in obj]
        return obj

    payload = {
        "timestamp": datetime.now().isoformat(),
        "results": _serialize(out),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def crystallize_to_hippocampus(cry: dict) -> bool:
    """结晶到 Hippocampus。RAR 公式存为知识条目。"""
    try:
        from axiom_os.core import Hippocampus
        from axiom_os.layers.rcln import RCLNLayer
        hippo = Hippocampus(dim=64, capacity=5000)
        rcln = RCLNLayer(input_dim=1, hidden_dim=8, output_dim=1, hard_core_func=None, lambda_res=1.0)
        g0 = cry.get("g_dagger") or cry.get("a0_or_gdagger") or 3700.0
        formula = f"x0/(1-np.exp(-np.sqrt(np.maximum(x0/{g0:.2f}, 1e-10))))"  # RAR McGaugh
        fid = hippo.crystallize(formula, rcln, formula_id="rar_law")
        return fid is not None
    except Exception:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="公式结晶")
    parser.add_argument("--rar-only", action="store_true", help="仅 RAR 结晶")
    parser.add_argument("--to-hippocampus", action="store_true", help="结晶到 Hippocampus")
    parser.add_argument("--n-galaxies", type=int, default=80, help="RAR 星系数")
    args = parser.parse_args()

    print("=" * 60)
    print("公式结晶 - Formula Crystallization")
    print("=" * 60)

    results = {}

    # RAR 结晶
    print("\n[1] RAR 公式结晶...")
    rar_res = crystallize_rar()
    if rar_res.get("ok"):
        cry = rar_res.get("crystallized", {})
        print(f"    n_galaxies={rar_res.get('n_galaxies')}, R2={rar_res.get('r2', 0):.4f}")
        best3 = cry.get("best3", [])
        for i, b in enumerate(best3[:3], 1):
            print(f"    #{i} {b.get('name', '')} R2={b.get('r2', 0):.4f}: {b.get('formula', '')[:60]}...")
        g0 = cry.get("g_dagger") or cry.get("a0_or_gdagger")
        if g0 is not None:
            print(f"    g0 = {g0:.2f} [(km/s)^2/kpc]")
        results["rar"] = rar_res

        if args.to_hippocampus and crystallize_to_hippocampus(cry):
            print("    [Hippocampus] 结晶成功")
        elif args.to_hippocampus:
            print("    [Hippocampus] 跳过")
    else:
        print(f"    FAIL: {rar_res.get('error', 'unknown')}")
        results["rar"] = rar_res

    # 保存
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_path = save_crystallized(results, out_dir)
    print(f"\n[2] 已保存: {out_path}")
    print("=" * 60)
    return 0 if results.get("rar", {}).get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
