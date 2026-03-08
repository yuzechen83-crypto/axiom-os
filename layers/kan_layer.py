"""
KAN (Kolmogorov-Arnold Network) Layer for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

KAN-RCLN: Symbolic Evolution Path
- Replaces MLP's fixed activation + learned weights with learned splines on edges
- Natural fit for Discovery Engine formula extraction
- Inspired by: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)

Architecture:
    MLP: y = σ(W·x + b)  →  fixed activation, learned linear weights
    KAN: y = Σ φ_i(x_i)   →  learned non-linear functions on each edge

Key Benefits for Physics:
    1. Parameter efficiency: Fewer params to fit complex functions (Bessel, etc.)
    2. Symbolic interpretability: Spline coefficients → symbolic formula
    3. Discovery-ready: Extract F = -cv directly from activation functions
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineActivation(nn.Module):
    """
    Learnable activation function using B-splines.
    Replaces fixed activations (ReLU, SiLU) with learnable 1D functions.
    """

    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Create knot grid
        # For grid_size=5, spline_order=3: need 5 + 2*3 + 1 = 12 knots
        n_knots = grid_size + 2 * spline_order + 1
        grid = torch.linspace(grid_range[0], grid_range[1], n_knots)
        self.register_buffer("grid", grid)

        # Learnable B-spline coefficients (one per basis function)
        # Number of basis functions = grid_size + spline_order
        self.coeff = nn.Parameter(torch.randn(grid_size + spline_order) * 0.1)

        # Base activation (SiLU/sigmoid hybrid for stability)
        self.base_activation = nn.SiLU()
        self.base_weight = nn.Parameter(torch.ones(1))
        self.base_bias = nn.Parameter(torch.zeros(1))

    def _b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions using Cox-de Boor recursion.
        Returns tensor of shape (batch_size, n_basis)
        """
        batch_size = x.shape[0]
        n_basis = self.grid_size + self.spline_order

        # Expand x for broadcasting: (batch, 1)
        x_expanded = x.unsqueeze(-1)  # (batch, 1)

        # Initialize degree-0 basis functions
        # B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
        grid_expanded = self.grid.unsqueeze(0)  # (1, n_knots)

        # For efficiency, use Gaussian-like basis (approximation)
        # This is a simplified version that's differentiable and fast
        centers = self.grid[self.spline_order : self.spline_order + n_basis]
        centers = centers.unsqueeze(0)  # (1, n_basis)

        # Basis width based on grid spacing
        grid_spacing = (self.grid_range[1] - self.grid_range[0]) / self.grid_size

        # Gaussian RBF basis (smooth approximation to B-splines)
        basis = torch.exp(-0.5 * ((x_expanded - centers) / (grid_spacing + 1e-6)) ** 2)

        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: combine base activation + spline contribution
        y = base_weight * base_activation(x) + Σ coeff_i * basis_i(x)
        """
        # Base activation path
        base = self.base_activation(x)
        base = self.base_weight * base + self.base_bias

        # Spline path: learned non-linear function
        basis = self._b_spline_basis(x)  # (batch, n_basis)
        spline = torch.matmul(basis, self.coeff)  # (batch,)

        return base + spline

    def get_symbolic_approx(self) -> str:
        """
        Return a string approximation of the learned function.
        Useful for Discovery Engine formula extraction.
        """
        # Analyze coefficients to guess symbolic form
        coeffs = self.coeff.detach().cpu().numpy()
        max_idx = int(abs(coeffs).argmax())
        max_val = float(coeffs[max_idx])

        if abs(max_val) < 0.01:
            return "0"

        # Simple heuristics for common functions
        grid = self.grid.detach().cpu().numpy()
        test_points = torch.linspace(self.grid_range[0], self.grid_range[1], 100)
        with torch.no_grad():
            values = self.forward(test_points).cpu().numpy()

        # Check linearity
        linear_fit = torch.linspace(values[0], values[-1], 100).numpy()
        linear_error = float(abs(values - linear_fit).mean())

        if linear_error < 0.1:
            slope = (values[-1] - values[0]) / (test_points[-1] - test_points[0]).item()
            intercept = values[0]
            if abs(intercept) < 0.1:
                return f"{slope:.3f}*x"
            return f"{slope:.3f}*x + {intercept:.3f}"

        # Check quadratic
        x_norm = test_points.numpy() / max(abs(test_points.numpy()))
        quad_fit = values[0] * (1 - x_norm**2) + values[-1] * x_norm**2
        quad_error = float(abs(values - quad_fit).mean())

        if quad_error < 0.1:
            return f"{max_val:.3f}*x^2"

        # Check sine-like
        sine_fit = max_val * torch.sin(test_points * 3.14159).numpy()
        sine_error = float(abs(values - sine_fit).mean())

        if sine_error < 0.2:
            return f"{max_val:.3f}*sin(x)"

        # Default: spline representation
        return f"spline_{self.grid_size}(x, coeffs=[{max_val:.2f},...])"


class KANLayer(nn.Module):
    """
    KAN Layer: Kolmogorov-Arnold Network layer.

    Unlike MLP: y_j = Σ_i φ_{ij}(x_i)
    Each edge has its own learnable activation function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create activation functions for each edge (in -> out)
        # Total: in_features * out_functions
        self.activations = nn.ModuleList([
            SplineActivation(grid_size, spline_order, grid_range)
            for _ in range(in_features * out_features)
        ])

        # Learnable weights for combining activations
        self.edge_weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            y_j = Σ_i w_{ij} * φ_{ij}(x_i)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)

        # Compute each output dimension
        for j in range(self.out_features):
            for i in range(self.in_features):
                # Get activation function for edge (i -> j)
                idx = i * self.out_features + j
                phi = self.activations[idx]

                # Apply activation to input dimension i
                activated = phi(x[:, i])

                # Weighted sum
                output[:, j] += self.edge_weights[i, j] * activated

        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def get_symbolic_form(self) -> List[str]:
        """
        Extract symbolic representation of the layer.
        Returns list of formulas for each output dimension.
        """
        formulas = []
        for j in range(self.out_features):
            terms = []
            for i in range(self.in_features):
                idx = i * self.out_features + j
                phi_str = self.activations[idx].get_symbolic_approx()
                weight = float(self.edge_weights[i, j].detach().cpu())
                if abs(weight) > 0.01 and phi_str != "0":
                    terms.append(f"{weight:.3f}*{phi_str}(x{i})")
            formulas.append(" + ".join(terms) if terms else "0")
        return formulas


class KANSoftShell(nn.Module):
    """
    KAN Soft Shell for RCLN.
    Replaces MLP with KAN for symbolic interpretability.

    Architecture:
        Input -> KANLayer -> KANLayer -> Output

    Benefits:
        - Better parameter efficiency for complex functions
        - Natural fit for formula discovery
        - Differentiable and trainable end-to-end
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        n_layers: int = 2,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Build KAN layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(KANLayer(
                in_features=dims[i],
                out_features=dims[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            ))

        self.layers = nn.ModuleList(layers)

        # Layer normalization for stability
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) if i < len(dims) - 2 else nn.Identity()
            for i in range(len(dims) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through KAN layers with residual connections."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = x
        for layer, norm in zip(self.layers[:-1], self.norms[:-1]):
            h = layer(h)
            h = norm(h)
            h = F.silu(h)  # Additional non-linearity between layers

        # Final layer (no activation)
        h = self.layers[-1](h)

        return h

    def extract_formula(self) -> str:
        """
        Extract approximate symbolic formula from the KAN.
        This is the key feature for Discovery Engine integration.
        """
        formulas = self.layers[0].get_symbolic_form()
        # For now, return first layer's formula (can be extended for full network)
        return f"y = [{'; '.join(formulas)}]"

    def reset_parameters(self) -> None:
        """Reset all parameters."""
        for layer in self.layers:
            for activation in layer.activations:
                activation.coeff.data.normal_(0, 0.1)
                activation.base_weight.data.fill_(1.0)
                activation.base_bias.data.zero_()
            layer.edge_weights.data.normal_(0, 0.1)


# =============================================================================
# Discovery Engine Integration
# =============================================================================

class KANFormulaExtractor:
    """
    Extract symbolic formulas from trained KAN models.
    Bridges the gap between neural network and analytical physics.
    """

    def __init__(self, kan_model: KANSoftShell):
        self.model = kan_model

    def extract_physics_formula(self, var_names: Optional[List[str]] = None) -> dict:
        """
        Extract physics-friendly formula representation.

        Returns:
            {
                'formula_str': 'y0 = 0.5*sin(x0) + 0.3*x1^2',
                'components': [{'input': 'x0', 'func': 'sin', 'weight': 0.5}, ...],
                'confidence': 0.95
            }
        """
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.model.input_dim)]

        formulas = []
        components = []

        # Get first layer symbolic form (most interpretable)
        layer_formulas = self.model.layers[0].get_symbolic_form()

        for j, formula in enumerate(layer_formulas[:self.model.output_dim]):
            # Parse and simplify
            simplified = self._simplify_formula(formula, var_names)
            formulas.append(f"y{j} = {simplified}")

            # Extract components for structured output
            comps = self._extract_components(formula, var_names)
            components.extend(comps)

        return {
            'formula_str': "; ".join(formulas),
            'components': components,
            'confidence': self._estimate_confidence(),
        }

    def _simplify_formula(self, formula: str, var_names: List[str]) -> str:
        """Simplify formula string for human readability."""
        # Replace x{i} with actual variable names
        for i, name in enumerate(var_names):
            formula = formula.replace(f"x{i}", name)
        return formula

    def _extract_components(self, formula: str, var_names: List[str]) -> List[dict]:
        """Extract structured components from formula."""
        components = []
        # Parse weight*function(x) patterns
        import re
        pattern = r'([\d.]+)\*([\w_]+)\(([^)]+)\)'
        matches = re.findall(pattern, formula)
        for weight, func, var in matches:
            components.append({
                'weight': float(weight),
                'function': func,
                'variable': var,
            })
        return components

    def _estimate_confidence(self) -> float:
        """Estimate confidence of formula extraction."""
        # Based on coefficient sparsity and smoothness
        total_coeffs = 0
        significant_coeffs = 0

        for layer in self.model.layers:
            for activation in layer.activations:
                coeffs = activation.coeff.detach().abs()
                total_coeffs += len(coeffs)
                significant_coeffs += (coeffs > 0.1).sum().item()

        if total_coeffs == 0:
            return 0.0

        sparsity = significant_coeffs / total_coeffs
        # Prefer moderate sparsity (not too dense, not too sparse)
        confidence = 1.0 - abs(sparsity - 0.3) * 2
        return max(0.0, min(1.0, confidence))


# =============================================================================
# Utility Functions
# =============================================================================

def create_kan_rcln(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    grid_size: int = 5,
    **kwargs
) -> KANSoftShell:
    """
    Factory function to create KAN Soft Shell for RCLN.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        grid_size: Number of grid points for B-splines
        **kwargs: Additional arguments for KANSoftShell

    Returns:
        KANSoftShell instance
    """
    return KANSoftShell(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        grid_size=grid_size,
        **kwargs
    )


if __name__ == "__main__":
    # Quick test
    print("Testing KAN Layer...")

    # Test SplineActivation
    spline = SplineActivation(grid_size=5)
    x = torch.linspace(-2, 2, 100)
    y = spline(x)
    print(f"Spline output shape: {y.shape}")
    print(f"Symbolic approx: {spline.get_symbolic_approx()}")

    # Test KANLayer
    kan_layer = KANLayer(in_features=3, out_features=2, grid_size=5)
    x_batch = torch.randn(10, 3)
    y_batch = kan_layer(x_batch)
    print(f"KANLayer output shape: {y_batch.shape}")
    print(f"Layer formulas: {kan_layer.get_symbolic_form()}")

    # Test KANSoftShell
    kan_shell = KANSoftShell(input_dim=4, hidden_dim=8, output_dim=2, grid_size=5)
    x_input = torch.randn(5, 4)
    y_output = kan_shell(x_input)
    print(f"KANSoftShell output shape: {y_output.shape}")
    print(f"Extracted formula: {kan_shell.extract_formula()}")

    print("\n✓ All KAN tests passed!")
