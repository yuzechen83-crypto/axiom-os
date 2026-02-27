#!/usr/bin/env python3
"""
创建 axiom_core_proprietary 私有包
运行此脚本后，将核心实现提取到 axiom_core_proprietary/，供推送到私有仓库。

用法: python scripts/create_proprietary_package.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AXIOM = ROOT / "axiom_os"
PROP = ROOT / "axiom_core_proprietary"


def main():
    PROP.mkdir(parents=True, exist_ok=True)

    # 1. 复制核心模块（保持目录结构）
    modules = [
        ("axiom_os/layers/rcln.py", "axiom_core_proprietary/rcln.py"),
        ("axiom_os/engine/discovery.py", "axiom_core_proprietary/discovery.py"),
        ("axiom_os/core/hippocampus.py", "axiom_core_proprietary/hippocampus.py"),
        ("axiom_os/core/wind_hard_core.py", "axiom_core_proprietary/wind_hard_core.py"),
        ("axiom_os/coach/spnn_evo_coach.py", "axiom_core_proprietary/coach.py"),
        ("axiom_os/core/adaptive_hard_core.py", "axiom_core_proprietary/adaptive_hard_core.py"),
        ("axiom_os/core/partition.py", "axiom_core_proprietary/partition.py"),
    ]
    for src_rel, dst_rel in modules:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            content = src.read_text(encoding="utf-8")
            # 调整导入路径
            content = content.replace("from .fno import", "from axiom_os.layers.fno import")
            content = content.replace("from .clifford_nn import", "from axiom_os.layers.clifford_nn import")
            content = content.replace("from .clifford_transformer import", "from axiom_os.layers.clifford_transformer import")
            content = content.replace("from .tensor_net import", "from axiom_os.layers.tensor_net import")
            content = content.replace("from .perturbation_gate import", "from axiom_os.layers.perturbation_gate import")
            content = content.replace("from .forms import", "from axiom_os.engine.forms import")
            content = content.replace("from .gflownet import", "from axiom_os.engine.gflownet import")
            # hippocampus 与 bundle_field 同处私有包，保留 .bundle_field
            # content = content.replace("from .bundle_field import", ...)  # 不替换
            content = content.replace("from . import wrap_adaptive_hard_core", "from .adaptive_hard_core import wrap_adaptive_hard_core")
            dst.write_text(content, encoding="utf-8")
            print(f"  Copied: {src_rel} -> {dst_rel}")

    # 2. 复制 bundle_field 目录（仅当 axiom_os 含完整实现时）
    # 注意：若 axiom_os/core/bundle_field 仅剩 __init__.py stub，则跳过复制，
    # 避免覆盖 axiom_core_proprietary 中已有的完整实现
    bf_src = AXIOM / "core" / "bundle_field"
    bf_dst = PROP / "bundle_field"
    bf_impl_files = ["axes.py", "base.py", "fiber.py", "connection.py"]
    has_impl = bf_src.exists() and all((bf_src / f).exists() for f in bf_impl_files)
    if has_impl:
        if bf_dst.exists():
            shutil.rmtree(bf_dst)
        shutil.copytree(bf_src, bf_dst)
        section_py = bf_dst / "section.py"
        if section_py.exists():
            txt = section_py.read_text(encoding="utf-8")
            txt = txt.replace("from ..perturbation import", "from axiom_os.core.perturbation import")
            section_py.write_text(txt, encoding="utf-8")
        print(f"  Copied: axiom_os/core/bundle_field/ -> axiom_core_proprietary/bundle_field/")
    elif bf_dst.exists():
        print(f"  Skipped bundle_field copy (axiom_os has stub only); preserving axiom_core_proprietary/bundle_field/")

    # 3. 创建 __init__.py
    init_content = '''"""
axiom_core_proprietary - 核心算法私有包
PROPRIETARY - Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.
需授权获取。请联系 yuzechen83-crypto。
"""

from .rcln import RCLNLayer, DiscoveryHotspot, ActivityMonitor, SpectralConv1d, SpectralSoftShell, HAS_CLIFFORD
from .discovery import DiscoveryEngine
from .hippocampus import Hippocampus
from .coach import coach_score, coach_loss_torch, coach_score_batch

__all__ = [
    "RCLNLayer", "DiscoveryHotspot", "ActivityMonitor",
    "SpectralConv1d", "SpectralSoftShell", "HAS_CLIFFORD",
    "DiscoveryEngine", "Hippocampus",
    "coach_score", "coach_loss_torch", "coach_score_batch",
]
'''
    (PROP / "__init__.py").write_text(init_content, encoding="utf-8")

    # 4. 创建 pyproject.toml（flat layout：包即当前目录）
    (PROP / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "axiom-core-proprietary"
version = "1.0.0"
description = "Axiom-OS 核心算法 (需授权)"
requires-python = ">=3.9"
dependencies = ["numpy", "torch", "scipy"]

[project.optional-dependencies]
full = ["scikit-learn", "pysr", "pysindy", "sentence-transformers"]

[tool.setuptools]
package-dir = {"axiom_core_proprietary" = "."}
packages = ["axiom_core_proprietary", "axiom_core_proprietary.bundle_field"]
''', encoding="utf-8")

    print(f"\nDone. axiom_core_proprietary created at {PROP}")
    print("Next: cd axiom_core_proprietary && git init && git add . && git commit -m 'Proprietary core'")
    print("      git remote add origin <your-private-repo-url> && git push -u origin main")


if __name__ == "__main__":
    main()
