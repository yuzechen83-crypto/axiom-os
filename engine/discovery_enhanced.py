"""
Enhanced Discovery Engine - 增强版符号回归
============================================

修复问题：
1. 添加对除法/反比关系的显式支持 (x/y)
2. 添加对倒数变换的支持 (1/x)
3. 增强对复杂物理公式的发现能力
"""

from typing import List, Optional, Any, Tuple
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import RidgeCV, Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class EnhancedDiscoveryEngine:
    """
    增强版发现引擎，支持：
    - 多项式关系: x*y, x^2
    - 除法关系: x/y, (x-y)/z
    - 倒数关系: 1/x, 1/(x+y)
    """
    
    def __init__(self, validation_mse_threshold: float = 0.5):
        self.validation_mse_threshold = validation_mse_threshold
    
    def _create_division_features(self, X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, List[str]]:
        """
        创建包含除法特征的特征矩阵
        
        Features:
        - 原始特征: x0, x1, ...
        - 两两比值: x0/x1, x1/x0, x0/x2, ...
        - 倒数: 1/x0, 1/x1, ...
        """
        n_samples, n_features = X.shape
        features = [X]
        feature_names = [f"x{i}" for i in range(n_features)]
        
        # 两两比值 x_i / x_j
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    ratio = X[:, i] / (X[:, j] + eps)
                    features.append(ratio.reshape(-1, 1))
                    feature_names.append(f"x{i}/x{j}")
        
        # 倒数 1/x_i
        for i in range(n_features):
            inv = 1.0 / (np.abs(X[:, i]) + eps)
            features.append(inv.reshape(-1, 1))
            feature_names.append(f"1/x{i}")
        
        return np.hstack(features), feature_names
    
    def _create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, List[str]]:
        """创建多项式特征"""
        if not HAS_SKLEARN:
            return X, [f"x{i}" for i in range(X.shape[1])]
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out([f"x{i}" for i in range(X.shape[1])])
        return X_poly, list(feature_names)
    
    def _fit_linear(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[str, float]:
        """拟合线性模型并返回公式字符串"""
        if not HAS_SKLEARN:
            # numpy fallback
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            mse = np.mean((X @ coef - y) ** 2)
            terms = [f"{c:.4f}*{n}" for c, n in zip(coef, feature_names) if abs(c) > 1e-4]
            return " + ".join(terms) if terms else "0", mse
        
        # Use RidgeCV for stable regression
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
        model = RidgeCV(alphas=alphas, cv=min(3, max(2, len(y)//10)))
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        # Build formula
        terms = []
        for coef, name in zip(model.coef_, feature_names):
            if abs(coef) > 1e-4:
                clean_name = name.replace(" ", "*")
                terms.append(f"{coef:.4f}*{clean_name}")
        if abs(model.intercept_) > 1e-4:
            terms.append(f"{model.intercept_:.4f}")
        
        formula = " + ".join(terms) if terms else "0"
        return formula, mse
    
    def distill(self, data_buffer: List[Tuple[np.ndarray, np.ndarray]], 
                input_units: Optional[List[List[int]]] = None,
                niterations: int = 20) -> Optional[str]:
        """
        从数据中发现符号公式
        
        策略：
        1. 尝试多项式回归 (degree=2)
        2. 尝试除法特征 (x/y, 1/x)
        3. 尝试线性回归 (degree=1)
        4. 选择MSE最低的模型
        """
        if not data_buffer:
            return None
        
        # Unpack data
        x_arr = np.stack([np.asarray(p[0]).ravel() for p in data_buffer])
        y_arr = np.stack([np.asarray(p[1]).ravel() for p in data_buffer])
        if y_arr.ndim > 1:
            y_arr = y_arr.ravel()
        
        n_samples, n_in = x_arr.shape
        
        candidates = []
        
        # Strategy 1: Polynomial features (degree=2)
        try:
            X_poly, names_poly = self._create_polynomial_features(x_arr, degree=2)
            formula, mse = self._fit_linear(X_poly, y_arr, names_poly)
            candidates.append(("poly2", mse, formula))
        except Exception:
            pass
        
        # Strategy 2: Division features (key improvement!)
        try:
            X_div, names_div = self._create_division_features(x_arr)
            # Also add polynomial features of original
            X_poly_div, names_poly_div = self._create_polynomial_features(x_arr, degree=1)
            X_combined = np.hstack([X_div, X_poly_div])
            names_combined = names_div + names_poly_div
            
            formula, mse = self._fit_linear(X_combined, y_arr, names_combined)
            candidates.append(("division", mse, formula))
        except Exception:
            pass
        
        # Strategy 3: Linear (degree=1)
        try:
            X_linear, names_linear = self._create_polynomial_features(x_arr, degree=1)
            formula, mse = self._fit_linear(X_linear, y_arr, names_linear)
            candidates.append(("linear", mse, formula))
        except Exception:
            pass
        
        if not candidates:
            return None
        
        # Select best by MSE
        best = min(candidates, key=lambda x: x[1])
        
        # Post-process formula: simplify common patterns
        formula = best[2]
        formula = self._simplify_formula(formula)
        
        return formula
    
    def _simplify_formula(self, formula: str) -> str:
        """简化公式字符串"""
        import re
        
        # Remove very small coefficients
        formula = re.sub(r'\d+\.?\d*e-\d+\*', '0*', formula)
        
        # Simplify "1.0000*x" to "x"
        formula = re.sub(r'1\.0+\*', '', formula)
        
        # Simplify "0.5000" to "0.5"
        formula = re.sub(r'(\d)\.0{2,}', r'\1', formula)
        
        return formula
    
    def _evaluate_formula(self, formula: str, X: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        """评估公式在给定输入下的输出"""
        try:
            n_samples, n_features = X.shape
            safe_dict = {
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "sqrt": np.sqrt,
                "exp": np.exp,
                "log": np.log,
                "abs": np.abs,
                "pi": np.pi,
            }
            
            # Add variables
            for i in range(n_features):
                safe_dict[f"x{i}"] = X[:, i]
            
            # Parse and evaluate the formula safely
            # Convert formula to use safe_dict
            import re
            eval_formula = formula
            
            # Replace patterns like x0/x1 with safe access
            # Handle division
            eval_formula = re.sub(r'x(\d+)/x(\d+)', r'x\1/(x\2 + 1e-8)', eval_formula)
            eval_formula = re.sub(r'1/x(\d+)', r'1/(x\1 + 1e-8)', eval_formula)
            
            # Handle powers
            eval_formula = eval_formula.replace("^", "**")
            
            # Handle patterns like 041 (missing decimal)
            eval_formula = re.sub(r'\b0+(\d)', r'\1', eval_formula)
            
            # Split by + and evaluate each term
            terms = [t.strip() for t in eval_formula.split('+') if t.strip()]
            result = np.zeros(n_samples)
            
            for term in terms:
                try:
                    # Evaluate term
                    val = eval(term, {"__builtins__": {}}, safe_dict)
                    if np.isscalar(val):
                        result += val
                    else:
                        result += np.asarray(val).ravel()[:n_samples]
                except Exception as e:
                    # Skip problematic terms
                    continue
            
            return result
        except Exception as e:
            print(f"      Eval error: {e}")
            return None


# =============================================================================
# Hard Formula Dataset (Feynman Hard)
# =============================================================================

FEYNMAN_HARD_FORMULAS = [
    {
        "name": "I.9.18",
        "description": "Newton's law of gravitation",
        "formula_str": "F = G * m1 * m2 / r^2",
        "variables": ["G", "m1", "m2", "r"],
        "target": "F",
        "func": lambda X: X[:, 0] * X[:, 1] * X[:, 2] / (X[:, 3] ** 2),
    },
    {
        "name": "I.12.2",
        "description": "Coulomb's law",
        "formula_str": "F = q1 * q2 / (4*pi*eps0*r^2)",
        "variables": ["q1", "q2", "eps0", "r"],
        "target": "F",
        "func": lambda X: X[:, 0] * X[:, 1] / (4 * np.pi * X[:, 2] * X[:, 3]**2),
    },
    {
        "name": "I.13.4",
        "description": "Pendulum period",
        "formula_str": "T = 2*pi*sqrt(L/g)",
        "variables": ["L", "g"],
        "target": "T",
        "func": lambda X: 2 * np.pi * np.sqrt(X[:, 0] / X[:, 1]),
    },
    {
        "name": "I.15.3x",
        "description": "Projectile motion x",
        "formula_str": "x = (v0 + v*t) * cos(theta)",
        "variables": ["v0", "v", "t", "theta"],
        "target": "x",
        "func": lambda X: (X[:, 0] + X[:, 1] * X[:, 2]) * np.cos(X[:, 3]),
    },
    {
        "name": "I.16.6",
        "description": "Dynamic pressure",
        "formula_str": "P = 0.5 * rho * v^2",
        "variables": ["rho", "v"],
        "target": "P",
        "func": lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2,
    },
    {
        "name": "I.18.12",
        "description": "Torque",
        "formula_str": "tau = r * F * sin(theta)",
        "variables": ["r", "F", "theta"],
        "target": "tau",
        "func": lambda X: X[:, 0] * X[:, 1] * np.sin(X[:, 2]),
    },
    {
        "name": "I.24.6",
        "description": "Capacitor energy",
        "formula_str": "E = 0.5 * C * V^2",
        "variables": ["C", "V"],
        "target": "E",
        "func": lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2,
    },
    {
        "name": "I.29.4",
        "description": "Wave number",
        "formula_str": "k = omega / c",
        "variables": ["omega", "c"],
        "target": "k",
        "func": lambda X: X[:, 0] / X[:, 1],
    },
    {
        "name": "I.30.5",
        "description": "de Broglie wavelength",
        "formula_str": "lambda = h / p",
        "variables": ["h", "p"],
        "target": "lambda",
        "func": lambda X: X[:, 0] / X[:, 1],
    },
    {
        "name": "I.32.17",
        "description": "Power in circuit",
        "formula_str": "P = E^2 * R / (R + r)^2",
        "variables": ["E", "R", "r"],
        "target": "P",
        "func": lambda X: X[:, 0]**2 * X[:, 1] / (X[:, 1] + X[:, 2])**2,
    },
    {
        "name": "I.37.4",
        "description": "Total energy",
        "formula_str": "E = p^2/(2*m) + U",
        "variables": ["p", "m", "U"],
        "target": "E",
        "func": lambda X: X[:, 0]**2 / (2 * X[:, 1]) + X[:, 2],
    },
    {
        "name": "I.39.1",
        "description": "Thermal energy",
        "formula_str": "E = 1.5 * kB * T",
        "variables": ["kB", "T"],
        "target": "E",
        "func": lambda X: 1.5 * X[:, 0] * X[:, 1],
    },
]


def run_hard_benchmark():
    """Run benchmark on Feynman Hard formulas"""
    import time
    
    print("="*70)
    print("Feynman HARD Benchmark - Enhanced Discovery Engine")
    print("="*70)
    
    engine = EnhancedDiscoveryEngine()
    results = []
    
    for i, formula in enumerate(FEYNMAN_HARD_FORMULAS, 1):
        print(f"\n[{i}/{len(FEYNMAN_HARD_FORMULAS)}] {formula['name']} - {formula['description']}")
        print(f"    Ground truth: {formula['formula_str']}")
        
        # Generate data
        np.random.seed(42)
        n_vars = len(formula['variables'])
        X = np.random.uniform(0.5, 2.0, size=(1000, n_vars))
        y = formula['func'](X)
        
        # Add small noise
        noise = np.random.normal(0, 0.01 * np.std(y), size=y.shape)
        y_noisy = y + noise
        
        # Run discovery
        start = time.time()
        data_buffer = list(zip(X, y_noisy.reshape(-1, 1)))
        discovered = engine.distill(data_buffer)
        elapsed = time.time() - start
        
        print(f"    Discovered: {discovered}")
        
        # Evaluate
        if discovered:
            # Compute R2
            y_pred = engine._evaluate_formula(discovered, X)
            if y_pred is not None:
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                
                # Check if formula contains key operations
                has_division = "/" in discovered
                has_multiplication = "*" in discovered
                
                # Success: good fit + has structure
                success = (r2 > 0.95) and (has_division or has_multiplication)
                
                print(f"    R2: {r2:.4f}, Time: {elapsed:.2f}s, Success: {success}")
                
                results.append({
                    "name": formula['name'],
                    "truth": formula['formula_str'],
                    "discovered": discovered,
                    "r2": r2,
                    "success": success,
                    "time": elapsed,
                })
            else:
                print(f"    Failed to evaluate")
                results.append({
                    "name": formula['name'],
                    "success": False,
                    "r2": 0.0,
                })
    
    # Summary
    n_success = sum(1 for r in results if r.get('success', False))
    mean_r2 = np.mean([r['r2'] for r in results if 'r2' in r])
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Success Rate: {n_success}/{len(results)} ({n_success/len(results)*100:.1f}%)")
    print(f"Mean R2: {mean_r2:.4f}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_hard_benchmark()
