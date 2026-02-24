"""
MLL Orchestrator - 多领域学习编排器

协议耦合：按依赖顺序执行，Hippocampus 跨域共享。
自迭代：根据 metrics 决定下一轮调度（优先改进差的 domain）。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .domain_protocols import DomainProtocol, ProtocolResult, PROTOCOL_REGISTRY


@dataclass
class CouplingConfig:
    """协议耦合配置"""
    order: List[str] = field(default_factory=lambda: ["rar", "battery", "turbulence", "acrobot"])
    hippocampus_shared: bool = True


class MLLOrchestrator:
    """
    多领域学习编排器
    - 按 order 或依赖顺序执行协议
    - Hippocampus 跨域共享（若 hippocampus_shared）
    - 协议耦合：前序 domain 的 Discovery 可影响后续
    """

    def __init__(
        self,
        protocols: Optional[Dict[str, DomainProtocol]] = None,
        hippocampus: Optional[Any] = None,
        config: Optional[CouplingConfig] = None,
    ):
        self.protocols = protocols or PROTOCOL_REGISTRY
        self.hippocampus = hippocampus
        self.config = config or CouplingConfig()
        self._results: Dict[str, List[ProtocolResult]] = {k: [] for k in self.protocols}
        self._metrics_history: Dict[str, List[Dict]] = {k: [] for k in self.protocols}

    def run_domain(
        self,
        domain_id: str,
        epochs: int = 500,
        do_discover: bool = True,
        do_crystallize: bool = False,
    ) -> ProtocolResult:
        """执行单个领域协议"""
        if domain_id not in self.protocols:
            return ProtocolResult(domain=domain_id, ok=False, error=f"Unknown domain {domain_id}")
        proto = self.protocols[domain_id]
        res = proto.run(
            hippocampus=self.hippocampus,
            epochs=epochs,
            do_discover=do_discover,
            do_crystallize=do_crystallize,
        )
        self._results[domain_id].append(res)
        if res.ok and res.metrics:
            self._metrics_history[domain_id].append(res.metrics)
        return res

    def run_all(
        self,
        epochs_per_domain: Optional[Dict[str, int]] = None,
        order: Optional[List[str]] = None,
        do_discover: bool = True,
        do_crystallize: bool = False,
    ) -> Dict[str, ProtocolResult]:
        """按顺序执行所有领域"""
        order = order or self.config.order
        order = [d for d in order if d in self.protocols]
        epochs_map = epochs_per_domain or {}
        results = {}
        for domain_id in order:
            epochs = epochs_map.get(domain_id, 500)
            if domain_id == "acrobot":
                epochs = 100  # acrobot 用 T_max 步数
            res = self.run_domain(domain_id, epochs=epochs, do_discover=do_discover, do_crystallize=do_crystallize)
            results[domain_id] = res
            if not res.ok and res.error:
                print(f"  [{domain_id}] FAIL: {res.error}")
            else:
                m = res.metrics
                mstr = ", ".join(f"{k}={v:.4f}" for k, v in (m or {}).items())
                print(f"  [{domain_id}] OK  {mstr}")
        return results
