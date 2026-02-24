"""
Test 01: Physical Constitution (UPI Check)
验证 UPI 是否真的会拦截错误。
如果 Mass + Time 不报错，说明地基是豆腐渣。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.core.upi import UPIState, Units, DimensionError


def test_mass_plus_time_raises_dimension_error():
    """
    物理宪法：Mass + Time 必须崩溃。
    - Mass = 10 kg  -> units [1,0,0,0,0]
    - Time = 5 s    -> units [0,0,1,0,0]
    - 相加必须抛出 DimensionError
    """
    mass = UPIState(values=10.0, units=Units.MASS, semantics="Mass")
    time_val = UPIState(values=5.0, units=Units.TIME, semantics="Time")

    try:
        _ = mass + time_val
        raise AssertionError(
            "UPI 地基是豆腐渣！Mass + Time 应该抛出 DimensionError，但没有报错。"
        )
    except DimensionError as e:
        assert "Unit mismatch" in str(e) or "different" in str(e).lower()
        print(f"OK: UPI 正确拦截 - {e}")
    except ValueError as e:
        # DimensionError 继承自 ValueError，所以 ValueError 也算通过
        if "DimensionError" in type(e).__name__:
            print(f"OK: 抛出 {type(e).__name__} - {e}")
        else:
            raise AssertionError(f"应抛出 DimensionError，实际抛出: {type(e).__name__}: {e}")


def test_mass_minus_time_raises():
    """Mass - Time 同样必须报错。"""
    mass = UPIState(values=10.0, units=Units.MASS, semantics="Mass")
    time_val = UPIState(values=5.0, units=Units.TIME, semantics="Time")

    try:
        _ = mass - time_val
        raise AssertionError("Mass - Time 应该抛出 DimensionError")
    except (DimensionError, ValueError) as e:
        print(f"OK: 减法也被拦截 - {type(e).__name__}")


def test_same_units_add_ok():
    """同量纲相加应成功。"""
    m1 = UPIState(values=10.0, units=Units.MASS, semantics="Mass")
    m2 = UPIState(values=5.0, units=Units.MASS, semantics="Mass")
    result = m1 + m2
    assert float(result.values) == 15.0
    print("OK: 同量纲相加正确")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 01: Physical Constitution (UPI Check)")
    print("=" * 60)
    test_mass_plus_time_raises_dimension_error()
    test_mass_minus_time_raises()
    test_same_units_add_ok()
    print("\n物理宪法验证通过。")
    print("=" * 60)
