"""Unit tests for CAD modeling tool."""

import tempfile
from pathlib import Path

import pytest


def test_run_cad_model_l_bracket():
    """Test run_cad_model with l_bracket shape."""
    from axiom_os.tools.cad_model import run_cad_model

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.stl"
        r = run_cad_model(shape="l_bracket", output_path=str(out))
        assert r["ok"] is True
        assert r["path"] == str(out)
        assert r["vertices_count"] > 0
        assert r["faces_count"] > 0
        assert out.exists()


def test_run_cad_model_box_parametrized():
    """Test run_cad_model with parametrized box."""
    from axiom_os.tools.cad_model import run_cad_model

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "box.stl"
        r = run_cad_model(shape="box", width=2.0, height=1.0, depth=0.5, output_path=str(out))
        assert r["ok"] is True
        assert out.exists()


def test_run_cad_model_cylinder_parametrized():
    """Test run_cad_model with parametrized cylinder."""
    from axiom_os.tools.cad_model import run_cad_model

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cyl.stl"
        r = run_cad_model(shape="cylinder", radius=0.3, height=2.0, output_path=str(out))
        assert r["ok"] is True
        assert out.exists()


def test_list_cad_shapes():
    """Test list_cad_shapes."""
    from axiom_os.tools.cad_model import list_cad_shapes

    r = list_cad_shapes()
    assert r["ok"] is True
    assert "shapes" in r
    names = [s["name"] for s in r["shapes"]]
    assert "box" in names
    assert "l_bracket" in names
