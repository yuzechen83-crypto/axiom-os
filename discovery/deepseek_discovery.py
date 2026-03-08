# -*- coding: utf-8 -*-
"""
DeepSeek-Powered LLM Discovery for RCLN
Real API integration with DeepSeek-Coder for physics formula generation
"""

import requests
import json
import time
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class DeepSeekDiscovery:
    """
    DeepSeek API-powered formula discovery
    """
    
    API_URL = "https://api.deepseek.com/chat/completions"
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.discovery_history = []
    
    def _call_api(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Call DeepSeek API"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000,
        }
        
        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def generate_formula(self, context: Dict) -> Dict:
        """
        Generate physics formula using DeepSeek
        
        Args:
            context: Contains 'current_loss', 'physics_params', 'error_pattern'
        """
        current_loss = context.get('current_loss', 0.02)
        physics_params = context.get('physics_params', {})
        error_pattern = context.get('error_pattern', 'high_dissipation_error')
        
        system_prompt = """You are an expert in turbulence modeling and subgrid-scale (SGS) stress parameterization.
Your task is to propose a Python function that computes SGS stress tau_ij based on velocity gradients.

The function should:
1. Take velocity field u (shape [B, 3, D, H, W]) as input
2. Compute velocity gradients, strain rate S_ij, and optionally vorticity Omega_ij
3. Return tau tensor of shape [B, 6, D, H, W] representing [xx, yy, zz, xy, xz, yz]

Physical constraints:
- tau must have correct units: [M L^-1 T^-2] (stress)
- Use learnable parameters (nn.Parameter) for coefficients
- Follow tensor operation conventions (PyTorch)
- Be numerically stable (add small epsilon where needed)

Example structure:
```python
def sgs_model(u, params):
    # Compute gradients
    du_dx = gradient(u[:, 0], 'x')
    ...
    # Strain rate
    S = 0.5 * (du_i_dx_j + du_j_dx_i)
    # SGS stress
    tau = -2 * params['cs']**2 * delta**2 * |S| * S
    return tau
```
"""
        
        user_prompt = f"""Current model performance:
- Validation MSE: {current_loss:.6f}
- Current physics parameters: {json.dumps(physics_params, indent=2)}
- Main error pattern: {error_pattern}

Please propose an improved SGS stress model. Consider:
1. If error is in high-shear regions, add non-linear corrections
2. If dissipation is wrong, adjust eddy viscosity formulation  
3. If anisotropic errors exist, add rotation terms

Return ONLY the Python function code, no explanations."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_api(messages, temperature=0.8)
        
        if response:
            # Extract code block
            import re
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = response
            
            return {
                'code': code,
                'raw_response': response,
                'timestamp': time.time(),
            }
        
        return None
    
    def refine_formula(self, parent_formula: Dict, feedback: Dict) -> Dict:
        """
        Refine formula based on evaluation feedback
        """
        parent_code = parent_formula.get('code', '')
        loss = feedback.get('loss', 0.02)
        error_analysis = feedback.get('error_analysis', 'Model underpredicts in high-shear regions')
        
        system_prompt = """You are refining a turbulence SGS model based on evaluation feedback.
The parent model had certain errors. Propose an improved version.

Rules:
1. Keep the successful parts of the parent model
2. Fix the identified issues
3. Add appropriate corrections
4. Return only Python function code
"""
        
        user_prompt = f"""Parent model code:
```python
{parent_code}
```

Evaluation feedback:
- Loss: {loss:.6f}
- Error analysis: {error_analysis}

Please refine the model to address these issues. Return only the improved Python function."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_api(messages, temperature=0.7)
        
        if response:
            import re
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            code = code_match.group(1) if code_match else response
            
            return {
                'code': code,
                'parent': parent_code,
                'raw_response': response,
                'timestamp': time.time(),
            }
        
        return None
    
    def analyze_neural_patterns(self, neural_stats: Dict) -> str:
        """
        Ask DeepSeek to interpret neural network patterns
        """
        system_prompt = "You are analyzing neural network activation patterns to infer underlying physics."
        
        user_prompt = f"""Neural network analysis shows:
{json.dumps(neural_stats, indent=2)}

What physical mechanism might these patterns represent? 
Suggest a symbolic formula that could replace this neural computation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._call_api(messages, temperature=0.6)


class FormulaEvaluator:
    """
    Evaluate generated formulas on actual data
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.upi_checker = UPIChecker()
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Check if code is valid Python and has correct structure"""
        try:
            compile(code, '<string>', 'exec')
            
            # Check for required components
            required = ['def ', 'return', 'u']
            for req in required:
                if req not in code:
                    return False, f"Missing required component: {req}"
            
            return True, "Code is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def evaluate_formula(self, code: str, test_data: Dict) -> Dict:
        """
        Execute formula on test data and compute metrics
        """
        # Validate first
        is_valid, msg = self.validate_code(code)
        if not is_valid:
            return {'is_valid': False, 'error': msg, 'mse': float('inf')}
        
        try:
            # Check if code defines a class (DeepSeek often returns classes)
            if 'class ' in code:
                # Extract forward method or create wrapper
                return self._evaluate_class_formula(code, test_data)
            
            # Create safe execution environment
            namespace = {
                'torch': torch,
                'nn': nn,
                'F': torch.nn.functional,
                'np': np,
            }
            
            # Execute code to define function
            exec(code, namespace)
            
            # Find the function or class
            func = None
            formula_class = None
            for name, obj in namespace.items():
                if name.startswith('__'):
                    continue
                if callable(obj) and hasattr(obj, '__name__') and obj.__name__ != '<module>':
                    if isinstance(obj, type):  # It's a class
                        formula_class = obj
                    else:
                        func = obj
            
            # If we found a class, instantiate it
            if formula_class is not None:
                instance = formula_class(delta=1.0)
                # Call forward
                u = test_data['velocity'][:10]
                tau_pred = instance(u)
                tau_true = test_data['tau_sgs'][:10]
                mse = torch.mean((tau_pred - tau_true)**2).item()
                return {'is_valid': True, 'mse': mse, 'mae': 0, 'max_error': 0}
            
            if func is None:
                return {'is_valid': False, 'error': 'No function or class found', 'mse': float('inf')}
            
            # Test on data
            u = test_data['velocity'][:10]  # Small batch
            tau_true = test_data['tau_sgs'][:10]
            
            # Create dummy params
            params = {'cs': torch.tensor(0.1, device=self.device)}
            
            # Run function
            tau_pred = func(u, params)
            
            # Check output shape
            if tau_pred.shape != tau_true.shape:
                return {
                    'is_valid': False, 
                    'error': f'Shape mismatch: {tau_pred.shape} vs {tau_true.shape}',
                    'mse': float('inf')
                }
            
            # Compute metrics
            mse = torch.mean((tau_pred - tau_true)**2).item()
            mae = torch.mean(torch.abs(tau_pred - tau_true)).item()
            max_error = torch.max(torch.abs(tau_pred - tau_true)).item()
            
            return {
                'is_valid': True,
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
            }
            
        except Exception as e:
            return {'is_valid': False, 'error': str(e), 'mse': float('inf')}


class UPIChecker:
    """Universal Physical Invariants Checker"""
    
    def check_dimensional_consistency(self, code: str) -> Tuple[bool, str]:
        """Basic dimensional analysis"""
        # Check for common dimensional errors
        if 'velocity * velocity' in code and 'gradient' not in code:
            return False, "Velocity*Velocity without gradient may have wrong units"
        
        if 'cs' in code and '** 2' not in code:
            return True, "Warning: Cs might need squaring for Smagorinsky"
        
        return True, "Basic checks passed"


if __name__ == "__main__":
    # Test with mock
    print("DeepSeek Discovery Module")
    print("Use with real API key for production")
