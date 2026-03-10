"""Quick unit tests for gym_adapter.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def test_imports():
    """Test that all imports work"""
    try:
        from benchmarks.gym_adapter import (
            AcrobotPhysics,
            GymUPIWrapper,
            AxiomAgent,
            SimToRealAdapter,
            acrobot_hard_core,
            run_benchmark,
        )
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_dataclass():
    """Test AcrobotPhysics dataclass"""
    from benchmarks.gym_adapter import AcrobotPhysics
    
    phy = AcrobotPhysics()
    assert phy.L1 == 1.0
    assert phy.g == 9.8
    assert phy.g_over_L == 9.8
    print("[OK] AcrobotPhysics dataclass works")
    return True

def test_upi_wrapper():
    """Test UPIState wrapper"""
    import numpy as np
    from benchmarks.gym_adapter import GymUPIWrapper, AcrobotPhysics
    
    wrapper = GymUPIWrapper("Acrobot-v1", AcrobotPhysics())
    
    # Create mock Acrobot observation
    # [cos(q1), sin(q1), cos(q2), sin(q2), dq1, dq2]
    obs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    upi_state = wrapper.wrap(obs, timestamp=0.0)
    
    assert upi_state.values.shape[0] == 4  # [q1, q2, p1, p2]
    assert upi_state.semantics == "AcrobotState[q1,q2,p1,p2]"
    print("[OK] UPIState wrapper works")
    return True

def test_adapter_creation():
    """Test SimToRealAdapter creation"""
    from benchmarks.gym_adapter import SimToRealAdapter
    
    adapter = SimToRealAdapter(state_dim=4, hidden_dim=32)
    assert adapter is not None
    print("[OK] SimToRealAdapter creation works")
    return True

def test_agent_creation():
    """Test AxiomAgent creation"""
    from benchmarks.gym_adapter import AxiomAgent
    
    agent = AxiomAgent(
        env_name="Acrobot-v1",
        horizon_steps=30,
        n_samples=100,
        use_adaptation=True,
    )
    assert agent is not None
    assert agent.mpc is not None
    print("[OK] AxiomAgent creation works")
    return True

if __name__ == "__main__":
    tests = [
        test_imports,
        test_physics_dataclass,
        test_upi_wrapper,
        test_adapter_creation,
        test_agent_creation,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    sys.exit(0 if all(results) else 1)
