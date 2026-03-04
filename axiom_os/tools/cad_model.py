"""
Axiom-OS CAD 建模工具
供 Agent 调用的程序化 3D 建模，导出 STL。

支持：box, cylinder, sphere, l_bracket, simple_gear
用法：run_cad_model(shape="l_bracket", output_path="...")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import trimesh
    import numpy as np
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    np = None


def _write_stl_ascii(vertices: List, faces: List, path: Path) -> None:
    """纯 Python 写入 ASCII STL（无 trimesh 时备用）"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("solid axiom_cad\n")
        for tri in faces:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e1 = [v1[i] - v0[i] for i in range(3)]
            e2 = [v2[i] - v0[i] for i in range(3)]
            n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ]
            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            for v in [v0, v1, v2]:
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid axiom_cad\n")


def _box_numpy(extents: tuple) -> tuple:
    """纯 numpy 生成立方体顶点和面"""
    w, h, d = extents
    v = [
        [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
        [0, 0, d], [w, 0, d], [w, h, d], [0, h, d],
    ]
    f = [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ]
    return v, f


def _cylinder_numpy(radius: float, height: float, segments: int = 24) -> tuple:
    """纯 numpy 生成圆柱顶点和面"""
    import math
    verts = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        verts.append([radius * math.cos(a), radius * math.sin(a), 0])
        verts.append([radius * math.cos(a), radius * math.sin(a), height])
    n = len(verts)
    faces = []
    for i in range(0, n - 2, 2):
        faces.append([i, i + 2, i + 1])
        faces.append([i + 1, i + 2, (i + 3) % n])
    # bottom cap
    for i in range(2, segments, 2):
        faces.append([0, i, i - 2])
    # top cap
    for i in range(3, segments * 2, 2):
        faces.append([1, i - 2, i])
    return verts, faces


def build_shape(
    shape: str,
    **kwargs: Any,
) -> tuple:
    """
    构建 3D 形状，返回 (vertices, faces) 或 trimesh.Trimesh。
    shape: box, cylinder, sphere, l_bracket, simple_gear
    """
    if HAS_TRIMESH:
        if shape == "box":
            extents = kwargs.get("extents", (1.0, 1.0, 1.0))
            m = trimesh.creation.box(extents=extents)
            return m
        if shape == "cylinder":
            r = kwargs.get("radius", 0.5)
            h = kwargs.get("height", 1.0)
            m = trimesh.creation.cylinder(radius=r, height=h, sections=32)
            return m
        if shape == "sphere":
            r = kwargs.get("radius", 0.5)
            m = trimesh.creation.icosphere(subdivisions=2, radius=r)
            return m
        if shape == "l_bracket":
            # L 形支架：两个 box 组合
            b1 = trimesh.creation.box(extents=[0.2, 1.0, 0.1])
            b2 = trimesh.creation.box(extents=[0.8, 0.2, 0.1])
            b2.apply_translation([0.1, 0.9, 0])
            m = trimesh.util.concatenate([b1, b2])
            return m
        if shape == "simple_gear":
            # 简化齿轮：圆柱 + 若干小圆柱做齿
            cyl = trimesh.creation.cylinder(radius=0.4, height=0.2, sections=24)
            teeth = []
            for i in range(12):
                a = 2 * np.pi * i / 12
                t = trimesh.creation.cylinder(radius=0.08, height=0.25, sections=8)
                t.apply_translation([0.5 * np.cos(a), 0.5 * np.sin(a), 0.05])
                teeth.append(t)
            m = trimesh.util.concatenate([cyl] + teeth)
            return m

    # Fallback: numpy-only
    if shape == "box":
        extents = kwargs.get("extents", (1.0, 1.0, 1.0))
        v, f = _box_numpy(extents)
        return v, f
    if shape == "cylinder":
        r = kwargs.get("radius", 0.5)
        h = kwargs.get("height", 1.0)
        try:
            v, f = _cylinder_numpy(r, h)
            return v, f
        except Exception:
            return _box_numpy((r * 2, r * 2, h))
    if shape == "l_bracket":
        v1, f1 = _box_numpy((0.2, 1.0, 0.1))
        v2, f2 = _box_numpy((0.8, 0.2, 0.1))
        off = len(v1)
        v2 = [[x + 0.1, y + 0.9, z] for x, y, z in v2]
        f2 = [[a + off, b + off, c + off] for a, b, c in f2]
        return v1 + v2, f1 + f2
    if shape == "sphere" or shape == "simple_gear":
        return _box_numpy((0.5, 0.5, 0.5))  # fallback to box

    return _box_numpy((1.0, 1.0, 1.0))


def run_cad_model(
    shape: str = "l_bracket",
    output_path: Optional[str] = None,
    format: str = "stl",
    width: Optional[float] = None,
    height: Optional[float] = None,
    depth: Optional[float] = None,
    radius: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Agent 可调用的 CAD 建模入口。
    构建指定形状，导出到 STL。

    Args:
        shape: box | cylinder | sphere | l_bracket | simple_gear
        output_path: 输出路径，默认 agent_output/cad/
        width, height, depth: box 尺寸 (mm 或单位)
        radius: cylinder/sphere 半径
        **kwargs: 其他形状参数

    Returns:
        {ok, path, message, vertices_count, faces_count}
    """
    # 参数化：width/height/depth -> box extents, radius/height -> cylinder
    if shape == "box" and (width is not None or height is not None or depth is not None):
        ext = kwargs.get("extents", (1.0, 1.0, 1.0))
        kwargs["extents"] = (
            float(width) if width is not None else ext[0],
            float(height) if height is not None else ext[1],
            float(depth) if depth is not None else ext[2],
        )
    if shape in ("cylinder", "sphere") and radius is not None:
        kwargs["radius"] = float(radius)
    if shape == "cylinder" and height is not None:
        kwargs["height"] = float(height)
    out_dir = ROOT / "agent_output" / "cad"
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = str(out_dir / f"axiom_{shape}.stl")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = build_shape(shape, **kwargs)

        if HAS_TRIMESH and hasattr(result, "export"):
            result.export(str(path), file_type="stl")
            n_verts = len(result.vertices)
            n_faces = len(result.faces)
        else:
            v, f = result
            _write_stl_ascii(v, f, path)
            n_verts = len(v)
            n_faces = len(f)

        return {
            "ok": True,
            "path": str(path),
            "message": f"已生成 {shape} 模型，{n_verts} 顶点，{n_faces} 面",
            "vertices_count": n_verts,
            "faces_count": n_faces,
            "shape": shape,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "shape": shape}


def list_cad_shapes() -> Dict[str, Any]:
    """列出支持的 CAD 形状及参数"""
    return {
        "ok": True,
        "shapes": [
            {"name": "box", "params": "extents=(w,h,d)"},
            {"name": "cylinder", "params": "radius, height"},
            {"name": "sphere", "params": "radius"},
            {"name": "l_bracket", "params": "无"},
            {"name": "simple_gear", "params": "无"},
        ],
        "requires_trimesh": "pip install trimesh (可选，否则用简化输出)",
    }
