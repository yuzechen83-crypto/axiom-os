"""SPNN Core Components: Physical Scale, UPI, Constants, Calibration"""

from .constants import *
from .physical_scale import PhysicalScaleSystem, UniversalConstants
from .upi_interface import UPIInterface
from .upi_contract import UPIContract, PhysicsViolationError
from .calibration import (
    renormalization_scale,
    safe_add,
    safe_subtract,
    erf_gate,
    apply_das,
    detect_oscillation,
    OscillationPattern,
    physical_aware_clip_grad,
)
