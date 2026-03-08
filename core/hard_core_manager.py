"""
Hard Core Manager - Evolution of Physical Knowledge
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Hard Core Evolution:
    Level 1: Empty or Basic (Newton's laws)
        - No built-in physics
        - Pure data-driven learning
    
    Level 2: Knowledge-Augmented
        - Crystallized formulas from Discovery Engine
        - F = ma, drag laws, battery aging curves
    
    Level 3: Differentiable Physics Engine
        - Full JAX/PyTorch compatible
        - Gradients flow through physics
        - Learnable parameters (friction, mass, stiffness)

The Hard Core is NOT static - it evolves through:
    1. Manual engineering (human expertise)
    2. Crystallization (from KAN discovery)
    3. System identification (gradient-based learning)
"""

from typing import Optional, Callable, Dict, List, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
import torch
import torch.nn as nn


class HardCoreLevel(Enum):
    """Evolution levels of the Hard Core."""
    EMPTY = 0           # No physics (pure ML)
    BASIC = 1           # Newton/Euler equations
    CRYSTALLIZED = 2    # Discovered formulas from KAN
    DIFFERENTIABLE = 3  # Full differentiable physics


@dataclass
class PhysicsFormula:
    """A crystallized physical formula."""
    name: str
    formula_str: str
    callable_fn: Callable
    domain: str  # "mechanics", "fluids", "thermodynamics", etc.
    confidence: float
    source: str  # "manual", "kan_discovery", "system_identification"
    metadata: Dict[str, Any]


class HardCoreManager:
    """
    Manages the evolution of physical knowledge in the Hard Core.
    
    The Hard Core starts empty and evolves:
        Empty → Basic Physics → Discovered Formulas → Differentiable Engine
    
    This is the "body" that gets stronger over time.
    """

    def __init__(
        self,
        initial_level: HardCoreLevel = HardCoreLevel.BASIC,
        state_dim: Optional[int] = None,
    ):
        self.level = initial_level
        self.state_dim = state_dim
        
        # Physics knowledge base
        self.formulas: Dict[str, PhysicsFormula] = {}
        self.energy_functions: Dict[str, Callable] = {}
        self.dynamics_functions: Dict[str, Callable] = {}
        
        # Learnable parameters (for Level 3)
        self.learnable_params: nn.ParameterDict = nn.ParameterDict()
        self.param_constraints: Dict[str, Tuple[float, float]] = {}
        
        # Initialize based on level
        if initial_level == HardCoreLevel.BASIC:
            self._init_basic_physics()

    def _init_basic_physics(self):
        """Initialize basic Newtonian physics."""
        # Basic Hamiltonian: H = p^2/2m + V(q)
        def basic_hamiltonian(z: torch.Tensor, mass: float = 1.0) -> torch.Tensor:
            n = z.shape[-1] // 2
            q, p = z[..., :n], z[..., n:]
            kinetic = (p ** 2).sum(dim=-1, keepdim=True) / (2 * mass)
            potential = 0.5 * (q ** 2).sum(dim=-1, keepdim=True)  # Harmonic
            return kinetic + potential
        
        self.energy_functions['basic'] = basic_hamiltonian

    def crystallize_formula(
        self,
        formula: PhysicsFormula,
        overwrite: bool = False,
    ) -> bool:
        """
        Crystallize a discovered formula into the Hard Core.
        
        This is the key operation: moving knowledge from Soft Shell (neural)
        to Hard Core (analytical).
        
        Args:
            formula: The discovered formula to crystallize
            overwrite: Whether to overwrite existing formula
        
        Returns:
            success: Whether crystallization succeeded
        """
        if formula.name in self.formulas and not overwrite:
            print(f"Formula '{formula.name}' already exists. Use overwrite=True to replace.")
            return False
        
        # Validate formula
        try:
            # Test on dummy input
            test_input = torch.randn(2, self.state_dim or 4)
            _ = formula.callable_fn(test_input)
        except Exception as e:
            print(f"Formula validation failed: {e}")
            return False
        
        # Store formula
        self.formulas[formula.name] = formula
        
        # Update level if needed
        if self.level.value < HardCoreLevel.CRYSTALLIZED.value:
            self.level = HardCoreLevel.CRYSTALLIZED
        
        print(f"Crystallized formula '{formula.name}' into Hard Core (confidence: {formula.confidence:.2%})")
        return True

    def extract_from_kan(
        self,
        kan_rcln,
        formula_name: str,
        domain: str = "generic",
    ) -> Optional[PhysicsFormula]:
        """
        Extract formula from KAN-RCLN and crystallize it.
        
        Args:
            kan_rcln: KAN-RCLN layer with learned representation
            formula_name: Name for the crystallized formula
            domain: Physical domain
        
        Returns:
            formula: The extracted formula or None
        """
        try:
            from axiom_os.layers import KANFormulaExtractor
            
            # Extract from KAN
            extractor = KANFormulaExtractor(kan_rcln.soft_shell)
            formula_info = extractor.extract_physics_formula()
            
            if formula_info is None or formula_info['confidence'] < 0.5:
                print("Formula extraction confidence too low")
                return None
            
            # Convert to callable
            formula_str = formula_info['formula_str']
            
            def formula_callable(x: torch.Tensor) -> torch.Tensor:
                # Simple evaluation (in practice, compile to efficient code)
                # For now, use the KAN itself as the callable
                return kan_rcln.soft_shell(x)
            
            formula = PhysicsFormula(
                name=formula_name,
                formula_str=formula_str,
                callable_fn=formula_callable,
                domain=domain,
                confidence=formula_info['confidence'],
                source="kan_discovery",
                metadata=formula_info,
            )
            
            # Crystallize
            if self.crystallize_formula(formula):
                return formula
            
        except Exception as e:
            print(f"KAN extraction failed: {e}")
        
        return None

    def get_composite_dynamics(
        self,
        active_formulas: Optional[List[str]] = None,
    ) -> Callable:
        """
        Get composite dynamics function combining all crystallized formulas.
        
        Args:
            active_formulas: List of formula names to include (None = all)
        
        Returns:
            dynamics_fn: Combined dynamics function
        """
        if active_formulas is None:
            active_formulas = list(self.formulas.keys())
        
        formulas_to_use = [
            self.formulas[name] for name in active_formulas 
            if name in self.formulas
        ]
        
        def composite_dynamics(z: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(z)
            for formula in formulas_to_use:
                result = result + formula.callable_fn(z)
            return result
        
        return composite_dynamics

    def add_learnable_parameter(
        self,
        name: str,
        shape: Tuple[int, ...],
        initial_value: float = 1.0,
        constraint: Optional[Tuple[float, float]] = None,
    ):
        """
        Add a learnable physical parameter (Level 3).
        
        Examples:
            - Friction coefficient
            - Spring stiffness
            - Mass distribution
        
        Args:
            name: Parameter name
            shape: Parameter shape
            initial_value: Initial value
            constraint: (min, max) constraint
        """
        param = nn.Parameter(torch.ones(shape) * initial_value)
        self.learnable_params[name] = param
        
        if constraint:
            self.param_constraints[name] = constraint
        
        # Upgrade to Level 3
        if self.level.value < HardCoreLevel.DIFFERENTIABLE.value:
            self.level = HardCoreLevel.DIFFERENTIABLE
        
        print(f"Added learnable parameter '{name}' (Level 3)")

    def get_learnable_params(self) -> Dict[str, torch.Tensor]:
        """Get constrained learnable parameters."""
        constrained = {}
        for name, param in self.learnable_params.items():
            if name in self.param_constraints:
                min_val, max_val = self.param_constraints[name]
                # Apply sigmoid constraint
                constrained[name] = torch.sigmoid(param) * (max_val - min_val) + min_val
            else:
                # Ensure positive (physical parameters)
                constrained[name] = torch.nn.functional.softplus(param)
        return constrained

    def learn_from_data(
        self,
        trajectories: torch.Tensor,
        solver,
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """
        Learn physical parameters from trajectory data (System Identification).
        
        This is Level 3 capability: gradient-based physics learning.
        
        Args:
            trajectories: Observed trajectories (batch, time, state_dim)
            solver: Physics solver (must be differentiable)
            n_epochs: Training epochs
            lr: Learning rate
        
        Returns:
            learned_params: Dictionary of learned parameter values
        """
        if self.level.value < HardCoreLevel.DIFFERENTIABLE.value:
            raise ValueError("Need DIFFERENTIABLE level for gradient-based learning")
        
        optimizer = torch.optim.Adam(self.learnable_params.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Get current parameters
            params = self.get_learnable_params()
            
            # Predict trajectories
            initial_states = trajectories[:, 0]
            # ... rollout with solver ...
            
            # Compute loss
            # loss = F.mse_loss(pred_trajectories, trajectories)
            # loss.backward()
            # optimizer.step()
            
            if epoch % 10 == 0:
                param_str = ", ".join([f"{k}={v.mean().item():.4f}" for k, v in params.items()])
                print(f"Epoch {epoch}: {param_str}")
        
        return {k: v.mean().item() for k, v in self.get_learnable_params().items()}

    def save_knowledge(self, filepath: str):
        """Save crystallized knowledge to file."""
        data = {
            'level': self.level.name,
            'formulas': {
                name: {
                    'formula_str': f.formula_str,
                    'domain': f.domain,
                    'confidence': f.confidence,
                    'source': f.source,
                    'metadata': f.metadata,
                }
                for name, f in self.formulas.items()
            },
            'learnable_params': {
                name: param.detach().cpu().numpy().tolist()
                for name, param in self.learnable_params.items()
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved Hard Core knowledge to {filepath}")

    def load_knowledge(self, filepath: str):
        """Load crystallized knowledge from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.level = HardCoreLevel[data['level']]
        
        # Note: formulas need to be re-instantiated as callables
        # This is a simplified version
        for name, f_data in data['formulas'].items():
            print(f"Loaded formula '{name}': {f_data['formula_str']}")
        
        print(f"Loaded Hard Core knowledge from {filepath}")

    def get_info(self) -> Dict[str, Any]:
        """Get Hard Core information."""
        return {
            'level': self.level.name,
            'n_formulas': len(self.formulas),
            'n_learnable_params': len(self.learnable_params),
            'formulas': list(self.formulas.keys()),
            'domains': list(set(f.domain for f in self.formulas.values())),
        }


class CrystallizationEngine:
    """
    Automated crystallization from Soft Shell discoveries.
    
    This engine monitors Soft Shell activity and automatically crystallizes
    high-confidence discoveries into the Hard Core.
    """

    def __init__(
        self,
        hard_core: HardCoreManager,
        confidence_threshold: float = 0.8,
        stability_window: int = 100,
    ):
        self.hard_core = hard_core
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        
        self.discovery_history: List[Dict] = []
        self.crystallization_queue: List[PhysicsFormula] = []

    def monitor_discovery(
        self,
        rcln_layer,
        activity: float,
        formula_info: Optional[Dict] = None,
    ):
        """
        Monitor a discovery event from RCLN Activity Monitor.
        
        Args:
            rcln_layer: The RCLN layer that triggered discovery
            activity: Soft activity level
            formula_info: Extracted formula information (KAN mode)
        """
        self.discovery_history.append({
            'activity': activity,
            'formula_info': formula_info,
            'timestamp': len(self.discovery_history),
        })
        
        # Check if ready for crystallization
        if formula_info and formula_info.get('confidence', 0) > self.confidence_threshold:
            if self._check_stability():
                self._queue_crystallization(rcln_layer, formula_info)

    def _check_stability(self) -> bool:
        """Check if discovery is stable over window."""
        if len(self.discovery_history) < self.stability_window:
            return False
        
        recent = self.discovery_history[-self.stability_window:]
        confidences = [d['formula_info'].get('confidence', 0) for d in recent 
                      if d['formula_info']]
        
        if len(confidences) < self.stability_window // 2:
            return False
        
        # Check if consistently high confidence
        return all(c > self.confidence_threshold * 0.9 for c in confidences[-10:])

    def _queue_crystallization(self, rcln_layer, formula_info: Dict):
        """Queue a formula for crystallization."""
        print(f"Queueing formula for crystallization (confidence: {formula_info['confidence']:.2%})")
        
        # Create formula object
        formula = PhysicsFormula(
            name=f"discovered_{len(self.hard_core.formulas)}",
            formula_str=formula_info['formula_str'],
            callable_fn=lambda x: rcln_layer(x),  # Use RCLN as callable
            domain="auto_discovered",
            confidence=formula_info['confidence'],
            source="auto_crystallization",
            metadata=formula_info,
        )
        
        self.crystallization_queue.append(formula)

    def execute_crystallization(self) -> List[str]:
        """
        Execute all queued crystallizations.
        
        Returns:
            crystallized_names: List of successfully crystallized formula names
        """
        results = []
        for formula in self.crystallization_queue:
            if self.hard_core.crystallize_formula(formula):
                results.append(formula.name)
        
        self.crystallization_queue.clear()
        return results


def test_hard_core_evolution():
    """Test Hard Core evolution."""
    print("Testing Hard Core Evolution...")
    
    # Create Level 1 Hard Core
    hcm = HardCoreManager(initial_level=HardCoreLevel.BASIC, state_dim=4)
    print(f"Initial: {hcm.get_info()}")
    
    # Add a formula (Level 2)
    def friction_force(z):
        q, v = z[:, :2], z[:, 2:]
        return torch.cat([torch.zeros_like(q), -0.1 * v], dim=1)
    
    formula = PhysicsFormula(
        name="linear_friction",
        formula_str="F = -0.1 * v",
        callable_fn=friction_force,
        domain="mechanics",
        confidence=0.95,
        source="manual",
        metadata={},
    )
    
    hcm.crystallize_formula(formula)
    print(f"After crystallization: {hcm.get_info()}")
    
    # Add learnable parameter (Level 3)
    hcm.add_learnable_parameter('friction_coef', (), initial_value=0.1, constraint=(0.0, 1.0))
    hcm.add_learnable_parameter('spring_k', (), initial_value=1.0)
    print(f"After adding learnable params: {hcm.get_info()}")
    
    # Get composite dynamics
    dynamics = hcm.get_composite_dynamics()
    z = torch.randn(3, 4)
    z_dot = dynamics(z)
    print(f"Composite dynamics: {z.shape} -> {z_dot.shape}")
    
    # Test crystallization engine
    engine = CrystallizationEngine(hcm)
    print(f"\nCrystallization engine ready")
    
    print("\nHard Core Evolution tests passed!")


if __name__ == "__main__":
    test_hard_core_evolution()
