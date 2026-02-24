"""
NeuronRegistry - Ecological neuron registry by physical domain.
Register, retrieve, list by domain. Dimensional chain validation.
"""

from typing import Dict, List, Optional, Type, Any
import logging

_log = logging.getLogger(__name__)

# Physical domains
DOMAINS = ["mechanics", "fluids", "electromagnetism", "thermodynamics", "control", "generic"]


def _units_match(upstream: List[List[int]], downstream: List[List[int]]) -> bool:
    """Check if upstream output_units are compatible with downstream input_units."""
    if not upstream or not downstream:
        return True
    if len(upstream) != len(downstream):
        return False
    for u, d in zip(upstream, downstream):
        if u != d and u != [0, 0, 0, 0, 0] and d != [0, 0, 0, 0, 0]:
            return False
    return True


class NeuronRegistry:
    """
    Ecological neuron registry.
    Organize by physical domain. Validate dimensional chain on connect.
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        neuron_id: str,
        neuron_class: Type,
        domain: str = "generic",
        author: str = "",
        description: str = "",
        version: str = "0.1.0",
    ) -> None:
        """
        Register a neuron class.
        domain: mechanics | fluids | electromagnetism | thermodynamics | control | generic
        """
        if domain not in DOMAINS:
            domain = "generic"
        self._registry[neuron_id] = {
            "class": neuron_class,
            "domain": domain,
            "author": author,
            "description": description,
            "version": version,
        }
        _log.info("Registered neuron %s (domain=%s)", neuron_id, domain)

    def get(
        self,
        neuron_id: str,
        version: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Instantiate neuron by id. kwargs passed to constructor."""
        if neuron_id not in self._registry:
            raise KeyError(f"Neuron {neuron_id} not registered")
        meta = self._registry[neuron_id]
        if version and meta.get("version") != version:
            _log.warning("Version mismatch: requested %s, have %s", version, meta.get("version"))
        return meta["class"](**kwargs)

    def list_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """List neurons in a physical domain."""
        return [
            {"id": k, **{kk: vv for kk, vv in v.items() if kk != "class"}}
            for k, v in self._registry.items()
            if v.get("domain") == domain
        ]

    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered neurons."""
        return [
            {"id": k, **{kk: vv for kk, vv in v.items() if kk != "class"}}
            for k, v in self._registry.items()
        ]

    def validate_chain(
        self,
        upstream_output_units: List[List[int]],
        downstream_neuron_id: str,
    ) -> bool:
        """Check if upstream can feed into downstream neuron."""
        if downstream_neuron_id not in self._registry:
            return False
        meta = self._registry[downstream_neuron_id]
        cls = meta["class"]
        if hasattr(cls, "INPUT_UNITS"):
            return _units_match(upstream_output_units, cls.INPUT_UNITS)
        return True

    def __len__(self) -> int:
        return len(self._registry)


# Global default registry
_default_registry: Optional[NeuronRegistry] = None


def get_registry() -> NeuronRegistry:
    """Get or create default NeuronRegistry with built-in neurons."""
    global _default_registry
    if _default_registry is None:
        _default_registry = NeuronRegistry()
        _register_builtins(_default_registry)
    return _default_registry


def _register_builtins(reg: NeuronRegistry) -> None:
    """Register built-in domain neurons."""
    try:
        from .mechanics import AcrobotResidualNeuron
        reg.register("acrobot_residual", AcrobotResidualNeuron, domain="mechanics", description="Double pendulum residual")
    except ImportError:
        pass
    try:
        from .fluids import TurbulenceResidualNeuron
        reg.register("turbulence_residual", TurbulenceResidualNeuron, domain="fluids", description="Turbulence u,v,T residual")
    except ImportError:
        pass
    try:
        from .control import ControlResidualNeuron
        reg.register("control_residual", ControlResidualNeuron, domain="control", description="Control residual torque")
    except ImportError:
        pass
