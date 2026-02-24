"""Quick UPI + RCLN test"""
import torch
from spnn_evo.core.upi import UPIState, Units, VELOCITY, LENGTH

def test_upi():
    t = torch.randn(4, 3, dtype=torch.float64)
    u1 = UPIState(t, VELOCITY, spacetime=torch.tensor([0.0, 0, 0, 0]), semantics="v_fluid")
    u2 = UPIState(t * 2, VELOCITY, spacetime=torch.tensor([1e-9, 0, 0, 0]), semantics="v_fluid2")
    u3 = u1 + u2
    assert u3.tensor.shape == t.shape
    ok = u1.assert_causality(u2)
    print("UPI test OK:", ok)

def test_rcln():
    from spnn_evo.core.hippocampus import init_default_library
    from spnn_evo.layers.rcln import RCLNModule
    lib = init_default_library()
    rcln = RCLNModule(16, library=lib, soft_threshold=0.01, monitor_window=5)
    x = torch.randn(8, 16) * 2
    for _ in range(10):
        y, h = rcln(x, return_hotspot=True)
    print("RCLN test OK, hotspot:", h is not None)

if __name__ == "__main__":
    test_upi()
    test_rcln()
    print("All tests passed.")
