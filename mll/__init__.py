"""
MLL - Multi-Layer Learning
多领域学习、协议耦合

MLL = 多领域协议 + 协议耦合 + 可生成协议
"""

from .domain_protocols import (
    DomainProtocol,
    ProtocolResult,
    TurbulenceProtocol,
    RARProtocol,
    BatteryProtocol,
    AcrobotProtocol,
    PROTOCOL_REGISTRY,
    generate_protocol_template,
)
from .orchestrator import MLLOrchestrator

__all__ = [
    "DomainProtocol",
    "ProtocolResult",
    "TurbulenceProtocol",
    "RARProtocol",
    "BatteryProtocol",
    "AcrobotProtocol",
    "PROTOCOL_REGISTRY",
    "generate_protocol_template",
    "MLLOrchestrator",
]
