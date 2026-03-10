"""
Enhanced DiscoveryEngine - PySR-First with Polynomial Fallback
===============================================================

Replace linear Lasso with PySR priority + Polynomial regression fallback.
Better for discovering physical formulas like E=mc^2, F=G*m1*m2/r^2.
"""

import numpy as np
from typing import Optional, List

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge, LassoCV
    from sklearn.metrics import mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def bic_score(n_samples, mse, n_params):
    """Bayesian Information Criterion for model selection."""
    return n_samples * np.log(mse + 1e-12) + n_params * np.log(n_samples)


def distill_enhanced(
    X: np.ndarray,
    y: np.ndarray,
    use_pysr: bool = True,
    niterations: int = 40,
    var_names: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Enhanced symbolic regression: PySR first, polynomial fallback.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        use_pysr: Whether to try PySR first
        niterations: PySR iterations
        var_names: Variable names for formula display
        
    Returns:
        Discovered formula string
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = X.shape
    
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_features)]
    
    best_formula = None
    best_bic = float('inf')
    
    # ==========================================================================
    # 1. PySR (Genetic Algorithm) - Primary method
    # ==========================================================================
    if use_pysr and HAS_PYSR:
        try:
            model = PySRRegressor(
                niterations=niterations,
                binary_operators=["+", "*", "-", "/"],
                unary_operators=["square", "sqrt", "abs", "exp", "log", "sin", "cos"],
                maxsize=20,
                populations=20,
                population_size=100,
                ncycles_per_iteration=500,
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y)
            eq = model.sympy()
            
            if eq is not None:
                formula = str(eq)
                # Validate
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                # Estimate complexity
                complexity = formula.count('+') + formula.count('*') + formula.count('/') + \
                           formula.count('square') + formula.count('sqrt') + 1
                bic = bic_score(n_samples, mse, complexity)
                
                if bic < best_bic:
                    best_bic = bic
                    best_formula = formula
                    print(f"  [PySR] BIC={bic:.2f}, MSE={mse:.6f}")
        except Exception as e:
            print(f"  [PySR] Failed: {e}")
            pass
    
    # ==========================================================================
    # 2. Polynomial Regression - Fallback (better than linear Lasso)
    # ==========================================================================
    if HAS_SKLEARN:
        # Try polynomial degrees from 1 to 3
        for degree in [2, 3, 1]:  # Try quadratic first (most physical formulas)
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly = poly.fit_transform(X)
                
                # Use Ridge with cross-validation for stability
                from sklearn.linear_model import RidgeCV
                model = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 0.1, 1.0], cv=3)
                model.fit(X_poly, y)
                
                y_pred = model.predict(X_poly)
                mse = mean_squared_error(y, y_pred)
                
                # Count non-zero coefficients
                n_params = np.sum(np.abs(model.coef_) > 1e-6) + 1
                bic = bic_score(n_samples, mse, n_params)
                
                # Build formula string
                feature_names = poly.get_feature_names_out(var_names)
                terms = []
                for coef, name in zip(model.coef_, feature_names):
                    if abs(coef) > 1e-6:
                        # Clean name: "x0^2" stays, "x0 x1" becomes "x0*x1"
                        clean_name = name.replace(" ", "*")
                        terms.append(f"{coef:.4f}*{clean_name}")
                
                if model.intercept_ and abs(model.intercept_) > 1e-6:
                    terms.append(f"{model.intercept_:.4f}")
                
                formula = " + ".join(terms) if terms else "0"
                
                print(f"  [Poly deg={degree}] BIC={bic:.2f}, MSE={mse:.6f}, terms={n_params}")
                
                if bic < best_bic:
                    best_bic = bic
                    best_formula = formula
                    
            except Exception as e:
                print(f"  [Poly deg={degree}] Failed: {e}")
                continue
    
    # ==========================================================================
    # 3. Ultimate Fallback: Linear Regression
    # ==========================================================================
    if best_formula is None:
        try:
            ones = np.ones((n_samples, 1))
            X_aug = np.hstack([X, ones])
            coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            
            terms = []
            for i, c in enumerate(coef[:-1]):
                if abs(c) > 1e-6:
                    terms.append(f"{c:.4f}*{var_names[i]}")
            if abs(coef[-1]) > 1e-6:
                terms.append(f"{coef[-1]:.4f}")
            
            best_formula = " + ".join(terms) if terms else "0"
            print(f"  [Linear] Fallback to linear regression")
        except Exception as e:
            print(f"  [Linear] Failed: {e}")
            best_formula = "0"
    
    return best_formula


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced DiscoveryEngine Test")
    print("=" * 60)
    
    # Test 1: E = m * c^2 (nonlinear)
    print("\n[Test 1] E = m * c^2")
    np.random.seed(42)
    m = np.random.uniform(1, 10, 100)
    c = np.random.uniform(1e8, 3e8, 100)
    E = m * c**2 + np.random.normal(0, 1e16, 100)  # Add noise
    
    X = np.column_stack([m, c])
    formula = distill_enhanced(X, E, use_pysr=HAS_PYSR, var_names=["m", "c"])
    print(f"Discovered: {formula}")
    print(f"Expected: m*c^2")
    
    # Test 2: F = m * a (linear)
    print("\n[Test 2] F = m * a")
    m = np.random.uniform(1, 100, 100)
    a = np.random.uniform(1, 10, 100)
    F = m * a + np.random.normal(0, 5, 100)
    
    X = np.column_stack([m, a])
    formula = distill_enhanced(X, F, use_pysr=HAS_PYSR, var_names=["m", "a"])
    print(f"Discovered: {formula}")
    print(f"Expected: m*a")
    
    # Test 3: d = v * t (linear interaction)
    print("\n[Test 3] d = v * t")
    v = np.random.uniform(1, 50, 100)
    t = np.random.uniform(1, 10, 100)
    d = v * t + np.random.normal(0, 2, 100)
    
    X = np.column_stack([v, t])
    formula = distill_enhanced(X, d, use_pysr=HAS_PYSR, var_names=["v", "t"])
    print(f"Discovered: {formula}")
    print(f"Expected: v*t")
    
    print("\n" + "=" * 60)
