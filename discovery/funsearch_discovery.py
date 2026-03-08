# -*- coding: utf-8 -*-
"""
FunSearch-Style LLM-Guided Discovery for Axiom-OS v4.0

Replaces genetic algorithms (PySR) with LLM-driven formula generation.
Inspired by DeepMind's FunSearch (2023).

Key innovation: LLM writes Python code for candidate formulas,
which are then evaluated, scored, and fed back to LLM for improvement.

Workflow:
1. LLM generates candidate function code
2. Code is executed in sandbox to compute loss
3. Best candidates are added to "prompt" for LLM
4. LLM evolves/improves based on feedback
5. UPI validates physical correctness

Reference: "Mathematical discoveries from program search with large language models"
(Romera-Paredes et al., Nature 2023)
"""

import os
import re
import json
import ast
import traceback
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from enum import Enum

# OpenAI/DeepSeek API
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class DiscoveryStatus(Enum):
    """Status of a formula candidate"""
    PROPOSED = "proposed"
    COMPILED = "compiled"  # Code is syntactically valid
    EXECUTED = "executed"  # Code ran without error
    VALIDATED = "validated"  # Passes UPI checks
    SCORED = "scored"  # Has evaluation metric
    ACCEPTED = "accepted"  # Best performing
    REJECTED = "rejected"  # Failed validation or too poor


@dataclass
class FunSearchCandidate:
    """A formula candidate in FunSearch"""
    # Code
    function_code: str  # The actual Python code
    function_name: str  # Name of the generated function
    
    # Metadata
    description: str
    physical_basis: str  # Physical reasoning
    parent_ids: List[str] = field(default_factory=list)  # Parent candidates (for evolution tracking)
    generation: int = 0
    
    # Evaluation
    status: DiscoveryStatus = DiscoveryStatus.PROPOSED
    loss: Optional[float] = None
    mse: Optional[float] = None
    physics_score: Optional[float] = None  # UPI validation score
    execution_error: Optional[str] = None
    
    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(10000)}")


class LLMFormulaGenerator:
    """
    LLM-based formula generator using DeepSeek/OpenAI API.
    Generates Python code for physical formulas.
    """
    
    # System prompt for formula generation
    SYSTEM_PROMPT = """You are a brilliant physicist and mathematician specializing in turbulence modeling.
Your task is to write Python functions that model Sub-Grid Scale (SGS) stress in fluid dynamics.

The function signature MUST be:
```python
def sgs_stress(velocity_gradient: np.ndarray, delta: float, **params) -> np.ndarray:
    '''
    Compute SGS stress tensor from velocity gradient.
    
    Args:
        velocity_gradient: [3, 3] array du_i/dx_j
        delta: filter width
        **params: learnable parameters
    
    Returns:
        stress: [6] array [tau_xx, tau_yy, tau_zz, tau_xy, tau_xz, tau_yz]
    '''
    # Your formula here
    return stress
```

Requirements:
1. Use only numpy operations
2. The formula should be physically motivated (eddy viscosity, strain-rate, etc.)
3. Include learnable parameters (like cs, nu, etc.)
4. Return shape must be (6,) for the 6 independent stress components

Example good formula (Smagorinsky):
```python
def sgs_stress_smagorinsky(velocity_gradient, delta, cs=0.1):
    S = 0.5 * (velocity_gradient + velocity_gradient.T)
    S_norm = np.sqrt(2 * np.sum(S * S))
    tau = -2 * (cs * delta)**2 * S_norm * S
    return np.array([tau[0,0], tau[1,1], tau[2,2], tau[0,1], tau[0,2], tau[1,2]])
```

Be creative! Try:
- Different eddy viscosity models
- Strain-rotation interactions
- Nonlinear combinations
- Anisotropic corrections
"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "deepseek-chat"):
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
        
        # Try to get API key from environment if not provided
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("[WARNING] No API key provided. Set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            # Support DeepSeek and OpenAI
            base_url = base_url or "https://api.deepseek.com"
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        self.model = model
        self.generation_history: List[List[FunSearchCandidate]] = []
    
    def generate_candidates(
        self,
        num_candidates: int = 3,
        best_candidates: Optional[List[FunSearchCandidate]] = None,
        temperature: float = 0.7,
    ) -> List[FunSearchCandidate]:
        """
        Generate formula candidates using LLM.
        
        Args:
            num_candidates: Number of candidates to generate
            best_candidates: Previous best candidates to use as inspiration
            temperature: LLM temperature (creativity)
        
        Returns:
            List of FunSearchCandidate
        """
        if self.client is None:
            print("[ERROR] LLM client not initialized. Using fallback templates.")
            return self._fallback_candidates(num_candidates)
        
        # Build prompt with context from best candidates
        user_prompt = self._build_prompt(best_candidates, num_candidates)
        
        candidates = []
        for i in range(num_candidates):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature + i * 0.1,  # Vary temperature
                    max_tokens=1000,
                )
                
                code = response.choices[0].message.content
                candidate = self._parse_code_to_candidate(code, generation=len(self.generation_history))
                candidates.append(candidate)
                
            except Exception as e:
                print(f"[ERROR] LLM generation failed: {e}")
                # Add fallback candidate
                candidates.append(self._fallback_candidates(1)[0])
        
        self.generation_history.append(candidates)
        return candidates
    
    def _build_prompt(
        self,
        best_candidates: Optional[List[FunSearchCandidate]],
        num_new: int,
    ) -> str:
        """Build prompt for LLM with context"""
        
        prompt = f"Generate {num_new} NEW and DIFFERENT Python function(s) for SGS stress modeling.\n\n"
        
        if best_candidates:
            prompt += "Here are some previously successful formulas for inspiration:\n\n"
            for i, cand in enumerate(best_candidates[:3], 1):
                prompt += f"--- Candidate {i} (Loss: {cand.loss:.6f}) ---\n"
                prompt += cand.function_code + "\n\n"
            
            prompt += "IMPORTANT: Create VARIATIONS or IMPROVEMENTS on these ideas.\n"
            prompt += "Try different physics: different viscosity models, anisotropic terms, rotation effects, etc.\n\n"
        else:
            prompt += "This is the first generation. Be creative!\n\n"
        
        prompt += "Return ONLY the Python function code, enclosed in triple backticks."
        
        return prompt
    
    def _parse_code_to_candidate(self, code: str, generation: int) -> FunSearchCandidate:
        """Parse LLM response into FunSearchCandidate"""
        
        # Extract code from markdown
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        code = code.strip()
        
        # Extract function name
        match = re.search(r'def\s+(\w+)\s*\(', code)
        function_name = match.group(1) if match else "unknown_function"
        
        # Generate description from docstring or code
        docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if docstring_match:
            description = docstring_match.group(1).strip().split('\n')[0]
        else:
            description = f"Formula generated by LLM: {function_name}"
        
        return FunSearchCandidate(
            function_code=code,
            function_name=function_name,
            description=description,
            physical_basis="LLM-generated based on physical intuition",
            generation=generation,
        )
    
    def _fallback_candidates(self, num: int) -> List[FunSearchCandidate]:
        """Fallback candidates when LLM is unavailable"""
        
        templates = [
            {
                'name': 'smagorinsky_basic',
                'code': '''def sgs_stress_smagorinsky(velocity_gradient, delta, cs=0.1):
    """Basic Smagorinsky eddy viscosity model"""
    S = 0.5 * (velocity_gradient + velocity_gradient.T)
    S_norm = np.sqrt(2 * np.sum(S * S) + 1e-10)
    nu_t = (cs * delta)**2 * S_norm
    tau = -2 * nu_t * S
    return np.array([tau[0,0], tau[1,1], tau[2,2], tau[0,1], tau[0,2], tau[1,2]])''',
                'basis': 'Eddy viscosity hypothesis',
            },
            {
                'name': 'strain_rotation',
                'code': '''def sgs_stress_strain_rotation(velocity_gradient, delta, cs=0.1, cr=0.01):
    """Strain-rotation interaction model"""
    S = 0.5 * (velocity_gradient + velocity_gradient.T)
    Omega = 0.5 * (velocity_gradient - velocity_gradient.T)
    S_norm = np.sqrt(2 * np.sum(S * S) + 1e-10)
    # Eddy viscosity term
    tau_eddy = -2 * (cs * delta)**2 * S_norm * S
    # Rotation interaction term
    tau_rot = -2 * cr * delta**2 * (S @ Omega - Omega @ S)
    tau = tau_eddy + tau_rot
    return np.array([tau[0,0], tau[1,1], tau[2,2], tau[0,1], tau[0,2], tau[1,2]])''',
                'basis': 'Strain-rotation interaction (Rodi 1976)',
            },
            {
                'name': 'nonlinear_model',
                'code': '''def sgs_stress_nonlinear(velocity_gradient, delta, c1=0.1, c2=0.01):
    """Nonlinear eddy viscosity model"""
    S = 0.5 * (velocity_gradient + velocity_gradient.T)
    S_norm = np.sqrt(2 * np.sum(S * S) + 1e-10)
    # Quadratic term
    S2 = S @ S
    tau = -2 * (c1 * delta)**2 * S_norm * S - 2 * (c2 * delta)**4 * S_norm**2 * S2
    return np.array([tau[0,0], tau[1,1], tau[2,2], tau[0,1], tau[0,2], tau[1,2]])''',
                'basis': 'Nonlinear constitutive relation',
            },
        ]
        
        candidates = []
        for i in range(min(num, len(templates))):
            t = templates[i]
            candidates.append(FunSearchCandidate(
                function_code=t['code'],
                function_name=t['name'],
                description=f"Fallback: {t['name']}",
                physical_basis=t['basis'],
                generation=0,
            ))
        
        return candidates


class CodeEvaluator:
    """
    Safely evaluate generated code and compute loss.
    """
    
    def __init__(self):
        self.execution_namespace = {
            'np': np,
            'torch': torch,
            'math': __import__('math'),
        }
    
    def evaluate_candidate(
        self,
        candidate: FunSearchCandidate,
        velocity_gradients: np.ndarray,
        targets: np.ndarray,
        delta: float = 1.0,
    ) -> FunSearchCandidate:
        """
        Evaluate a candidate formula.
        
        Args:
            candidate: FunSearchCandidate with function_code
            velocity_gradients: [N, 3, 3] velocity gradient tensors
            targets: [N, 6] target SGS stress values
            delta: filter width
        
        Returns:
            Updated candidate with loss and status
        """
        # Step 1: Syntax check
        try:
            ast.parse(candidate.function_code)
            candidate.status = DiscoveryStatus.COMPILED
        except SyntaxError as e:
            candidate.status = DiscoveryStatus.REJECTED
            candidate.execution_error = f"Syntax error: {e}"
            return candidate
        
        # Step 2: Execute and get function
        try:
            # Create isolated namespace
            local_ns = self.execution_namespace.copy()
            exec(candidate.function_code, local_ns)
            
            # Get the function
            func = local_ns.get(candidate.function_name)
            if func is None:
                # Try to find any function defined
                for name, obj in local_ns.items():
                    if callable(obj) and name != '__builtins__':
                        func = obj
                        candidate.function_name = name
                        break
            
            if func is None:
                raise ValueError("No function found in generated code")
            
        except Exception as e:
            candidate.status = DiscoveryStatus.REJECTED
            candidate.execution_error = f"Execution error: {traceback.format_exc()}"
            return candidate
        
        candidate.status = DiscoveryStatus.EXECUTED
        
        # Step 3: Evaluate on data
        try:
            predictions = []
            for grad in velocity_gradients:
                # Call function with default parameters
                result = func(grad, delta)
                predictions.append(result)
            
            predictions = np.array(predictions)
            
            # Compute MSE
            mse = np.mean((predictions - targets) ** 2)
            candidate.mse = float(mse)
            candidate.loss = float(mse)
            candidate.status = DiscoveryStatus.SCORED
            
        except Exception as e:
            candidate.status = DiscoveryStatus.REJECTED
            candidate.execution_error = f"Evaluation error: {traceback.format_exc()}"
            return candidate
        
        return candidate


class UPIChecker:
    """Universal Physical Invariants Checker"""
    
    def validate(self, candidate: FunSearchCandidate) -> Tuple[bool, float]:
        """
        Validate physical correctness of formula.
        
        Returns:
            (is_valid, physics_score)
        """
        code = candidate.function_code.lower()
        score = 1.0
        
        # Check for physical terms
        physical_terms = {
            'strain': 'strain' in code or 's =' in code,
            'gradient': 'gradient' in code or 'velocity_gradient' in code,
            'delta': 'delta' in code,
            'tensor_operation': '@' in code or 'np.dot' in code or 'np.einsum' in code,
        }
        
        # Check for non-physical terms (red flags)
        red_flags = {
            'random': 'random' in code or 'np.random' in code,
            'magic_number': bool(re.search(r'\b\d{3,}\b', code)),  # Large magic numbers
        }
        
        # Compute score
        score = sum(physical_terms.values()) / len(physical_terms)
        score -= sum(red_flags.values()) * 0.5
        
        is_valid = score > 0.3 and not any(red_flags.values())
        
        return is_valid, max(0.0, score)


class FunSearchDiscovery:
    """
    Main FunSearch discovery engine.
    Iteratively evolves formulas using LLM feedback.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        population_size: int = 10,
        num_iterations: int = 5,
    ):
        self.generator = LLMFormulaGenerator(api_key=api_key)
        self.evaluator = CodeEvaluator()
        self.upi_checker = UPIChecker()
        
        self.population_size = population_size
        self.num_iterations = num_iterations
        
        # Archive of all candidates
        self.archive: List[FunSearchCandidate] = []
        self.best_candidate: Optional[FunSearchCandidate] = None
    
    def discover(
        self,
        velocity_gradients: np.ndarray,
        targets: np.ndarray,
        delta: float = 1.0,
    ) -> FunSearchCandidate:
        """
        Run FunSearch discovery loop.
        
        Args:
            velocity_gradients: [N, 3, 3] training data
            targets: [N, 6] target stresses
            delta: filter width
        
        Returns:
            Best FunSearchCandidate
        """
        print("=" * 70)
        print("FunSearch Discovery - Axiom-OS v4.0")
        print("=" * 70)
        print(f"Dataset size: {len(velocity_gradients)}")
        print(f"Population size: {self.population_size}")
        print(f"Iterations: {self.num_iterations}")
        print()
        
        for iteration in range(self.num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.num_iterations} ---")
            
            # Get best candidates for prompting
            current_best = self._get_best_candidates(3)
            
            # Generate new candidates
            print(f"Generating {self.population_size} candidates...")
            new_candidates = self.generator.generate_candidates(
                num_candidates=self.population_size,
                best_candidates=current_best,
            )
            
            # Evaluate each candidate
            print("Evaluating candidates...")
            for candidate in new_candidates:
                # Evaluate
                candidate = self.evaluator.evaluate_candidate(
                    candidate, velocity_gradients, targets, delta
                )
                
                # UPI validation
                if candidate.status == DiscoveryStatus.SCORED:
                    is_valid, physics_score = self.upi_checker.validate(candidate)
                    candidate.physics_score = physics_score
                    if is_valid:
                        candidate.status = DiscoveryStatus.VALIDATED
                    else:
                        candidate.status = DiscoveryStatus.REJECTED
                
                # Add to archive
                self.archive.append(candidate)
                
                # Print status
                status_str = candidate.status.value
                loss_str = f"{candidate.loss:.6f}" if candidate.loss else "N/A"
                print(f"  {candidate.function_name:30s} | {status_str:12s} | Loss: {loss_str}")
                if candidate.execution_error:
                    print(f"    Error: {candidate.execution_error[:100]}")
            
            # Update best
            validated = [c for c in self.archive if c.status == DiscoveryStatus.VALIDATED]
            if validated:
                self.best_candidate = min(validated, key=lambda c: c.loss)
                print(f"\nBest so far: {self.best_candidate.function_name} (Loss: {self.best_candidate.loss:.6f})")
        
        print("\n" + "=" * 70)
        if self.best_candidate:
            print("[SUCCESS] Discovery complete!")
            print(f"Best formula: {self.best_candidate.function_name}")
            print(f"Loss: {self.best_candidate.loss:.6f}")
            print(f"Physics score: {self.best_candidate.physics_score:.2f}")
            print("\nCode:")
            print(self.best_candidate.function_code)
        else:
            print("[WARNING] No valid formula found")
        
        return self.best_candidate
    
    def _get_best_candidates(self, n: int) -> List[FunSearchCandidate]:
        """Get top n validated candidates by loss"""
        validated = [c for c in self.archive if c.status == DiscoveryStatus.VALIDATED]
        validated.sort(key=lambda c: c.loss)
        return validated[:n]


def test_funsearch_discovery():
    """Test FunSearch discovery"""
    print("Testing FunSearch Discovery Module...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Generate random velocity gradients
    velocity_gradients = np.random.randn(n_samples, 3, 3) * 0.1
    
    # Generate synthetic targets using Smagorinsky model
    def true_model(grad, delta=1.0):
        S = 0.5 * (grad + grad.T)
        S_norm = np.sqrt(2 * np.sum(S * S) + 1e-10)
        cs = 0.16  # Theoretical value
        tau = -2 * (cs * delta)**2 * S_norm * S
        return np.array([tau[0,0], tau[1,1], tau[2,2], tau[0,1], tau[0,2], tau[1,2]])
    
    targets = np.array([true_model(g) for g in velocity_gradients])
    
    # Run discovery
    discovery = FunSearchDiscovery(
        population_size=3,  # Small for testing
        num_iterations=2,
    )
    
    best = discovery.discover(velocity_gradients, targets, delta=1.0)
    
    return best


if __name__ == "__main__":
    test_funsearch_discovery()
