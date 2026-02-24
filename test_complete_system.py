"""Test SPNN Complete System integration"""
import torch
from spnn import SPNNCompleteSystem
from spnn.semantics import PhysicalLabelingSystem, StructureDistillationEncoder, AdaptiveScaleEncoder
from spnn.memory import StructuredHippocampus
from spnn.ionization import IonizationDataModule


def test_physical_labels():
    pl = PhysicalLabelingSystem(32)
    emb = pl.get_semantic_embedding(["gravity", "momentum"])
    assert emb.shape == (32,)
    print("Physical labels OK")


def test_structure_encoder():
    enc = StructureDistillationEncoder(4, 64)
    x = torch.randn(8, 4)
    out = enc(x, {"physical_labels": ["velocity"]})
    assert "vector" in out and "coupling_feature" in out
    print("Structure encoder OK")


def test_structured_hippocampus():
    h = StructuredHippocampus(capacity=100)
    p = torch.randn(4, 8)
    h.store_physical_pattern(p, ["gravity", "momentum"], 0.9)
    ret = h.retrieve_by_physical_label(["gravity", "momentum"])
    assert ret is not None
    print("Structured hippocampus OK")


def test_scale_encoder():
    se = AdaptiveScaleEncoder(num_scales=8)
    x = torch.randn(8, 4)
    w = se.encode(x)
    assert w.shape[-1] == 8
    print("Scale encoder OK")


def test_ionization():
    ion = IonizationDataModule()
    x = torch.randn(8, 4)
    out = ion.preprocess_ionization_data(x, "plasma")
    assert "ion_density" in out
    print("Ionization OK")


def test_complete_system():
    sys = SPNNCompleteSystem(in_dim=4, hidden_dim=32, out_dim=1, memory_capacity=100)
    x = torch.randn(8, 4)
    ctx = {"physical_labels": ["velocity", "pressure"]}
    y = sys(x, context=ctx)
    assert y.shape == (8, 1)
    print("Complete system OK")


if __name__ == "__main__":
    test_physical_labels()
    test_structure_encoder()
    test_structured_hippocampus()
    test_scale_encoder()
    test_ionization()
    test_complete_system()
    print("All complete system tests passed.")
