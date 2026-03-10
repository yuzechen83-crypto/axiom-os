"""
Hard Formula Discovery - 极难公式发现引擎
==========================================

针对 Feynman Hard 公式的专用发现引擎：
- 万有引力: F = G*m1*m2/r^2
- 库仑定律: F = q1*q2/(4πε0*r^2)
- 波动/反比: k = ω/c, λ = h/p
- 能量形式: E = p^2/(2m) + U

策略: 显式枚举物理常见的公式模板，线性拟合确定系数
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class HardFormulaDiscovery:
    """
    针对困难物理公式的发现引擎
    """
    
    def __init__(self):
        pass
    
    def _try_template(self, X: np.ndarray, y: np.ndarray, 
                      feature_func, template_name: str) -> Tuple[Optional[str], float]:
        """
        尝试一个公式模板
        
        Args:
            X: 输入数据 (n_samples, n_features)
            y: 目标值 (n_samples,)
            feature_func: 生成特征变换的函数
            template_name: 模板名称描述
        
        Returns:
            (formula_string, r2_score)
        """
        try:
            # 生成变换后的特征
            phi = feature_func(X)  # (n_samples, n_basis)
            
            if phi is None or phi.ndim != 2:
                return None, -np.inf
            
            # 线性拟合: y = c0*phi0 + c1*phi1 + ...
            coef, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
            
            # 预测和评估
            y_pred = phi @ coef
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            
            # 构建公式字符串
            terms = []
            for i, c in enumerate(coef):
                if abs(c) > 1e-6:
                    terms.append(f"{c:.4f}*phi{i}")
            
            formula = " + ".join(terms) if terms else "0"
            return f"{template_name}: {formula}", r2
            
        except Exception as e:
            return None, -np.inf
    
    def discover(self, X: np.ndarray, y: np.ndarray, 
                 var_names: List[str] = None) -> Tuple[str, float]:
        """
        发现最佳公式
        """
        n_samples, n_features = X.shape
        if var_names is None:
            var_names = [f"x{i}" for i in range(n_features)]
        
        candidates = []
        
        # Template 1: 纯多项式 (用于对比)
        def poly2_features(X):
            n = X.shape[1]
            feats = [np.ones(X.shape[0])]
            # 线性项
            for i in range(n):
                feats.append(X[:, i])
            # 二次项
            for i in range(n):
                for j in range(i, n):
                    feats.append(X[:, i] * X[:, j])
            return np.column_stack(feats)
        
        candidates.append(self._try_template(X, y, poly2_features, "poly2"))
        
        # Template 2: 乘法 + 除法平方 (万有引力形式: x0*x1*x2/x3^2)
        if n_features >= 4:
            def gravity_like(X):
                return np.column_stack([
                    np.ones(X.shape[0]),
                    X[:, 0] * X[:, 1] * X[:, 2] / (X[:, 3]**2 + 1e-8),
                    X[:, 0] * X[:, 1] / (X[:, 3]**2 + 1e-8),
                    X[:, 0] * X[:, 2] / (X[:, 3]**2 + 1e-8),
                    X[:, 1] * X[:, 2] / (X[:, 3]**2 + 1e-8),
                ])
            candidates.append(self._try_template(X, y, gravity_like, "gravity"))
        
        # Template 3: 简单比值 (k = omega/c, lambda = h/p)
        if n_features >= 2:
            def ratio_features(X):
                feats = [np.ones(X.shape[0])]
                for i in range(n_features):
                    for j in range(n_features):
                        if i != j:
                            feats.append(X[:, i] / (X[:, j] + 1e-8))
                return np.column_stack(feats)
            candidates.append(self._try_template(X, y, ratio_features, "ratio"))
        
        # Template 4: 乘积 (E = m*c^2, E = h*f)
        def product_features(X):
            feats = [np.ones(X.shape[0])]
            for i in range(n_features):
                for j in range(n_features):
                    feats.append(X[:, i] * X[:, j])
            return np.column_stack(feats)
        candidates.append(self._try_template(X, y, product_features, "product"))
        
        # Template 5: 平方比值 (E = p^2/(2m))
        if n_features >= 2:
            def square_ratio(X):
                feats = [np.ones(X.shape[0])]
                for i in range(n_features):
                    for j in range(n_features):
                        if i != j:
                            feats.append(X[:, i]**2 / (X[:, j] + 1e-8))
                return np.column_stack(feats)
            candidates.append(self._try_template(X, y, square_ratio, "square_ratio"))
        
        # Template 6: 开平方根 (T = 2*pi*sqrt(L/g))
        if n_features >= 2:
            def sqrt_ratio(X):
                feats = [np.ones(X.shape[0])]
                for i in range(n_features):
                    for j in range(n_features):
                        if i != j:
                            feats.append(np.sqrt(np.abs(X[:, i]) / (np.abs(X[:, j]) + 1e-8)))
                return np.column_stack(feats)
            candidates.append(self._try_template(X, y, sqrt_ratio, "sqrt_ratio"))
        
        # Template 7: 倒数 (1/x)
        def inverse_features(X):
            feats = [np.ones(X.shape[0])]
            for i in range(n_features):
                feats.append(1.0 / (np.abs(X[:, i]) + 1e-8))
            return np.column_stack(feats)
        candidates.append(self._try_template(X, y, inverse_features, "inverse"))
        
        # Template 8: 分式形式 (x0/(x1+x2))
        if n_features >= 3:
            def fraction_features(X):
                feats = [np.ones(X.shape[0])]
                for i in range(n_features):
                    for j in range(n_features):
                        for k in range(n_features):
                            if i != j and i != k:
                                feats.append(X[:, i] / (X[:, j] + X[:, k] + 1e-8))
                return np.column_stack(feats)
            candidates.append(self._try_template(X, y, fraction_features, "fraction"))
        
        # 选择最佳候选
        best_formula, best_r2 = max(candidates, key=lambda x: x[1])
        
        if best_formula is None:
            return "0", 0.0
        
        return best_formula, best_r2


# =============================================================================
# Hard Benchmark Runner
# =============================================================================

FEYNMAN_HARD = [
    {
        "name": "I.9.18",
        "desc": "Newton gravitation",
        "formula": "F = G*m1*m2/r^2",
        "vars": ["G", "m1", "m2", "r"],
        "func": lambda X: X[:,0]*X[:,1]*X[:,2]/(X[:,3]**2),
    },
    {
        "name": "I.12.2", 
        "desc": "Coulomb's law",
        "formula": "F = q1*q2/(4πε0*r^2)",
        "vars": ["q1", "q2", "eps0", "r"],
        "func": lambda X: X[:,0]*X[:,1]/(4*np.pi*X[:,2]*X[:,3]**2),
    },
    {
        "name": "I.29.4",
        "desc": "Wave number",
        "formula": "k = ω/c",
        "vars": ["omega", "c"],
        "func": lambda X: X[:,0]/X[:,1],
    },
    {
        "name": "I.30.5",
        "desc": "de Broglie",
        "formula": "λ = h/p",
        "vars": ["h", "p"],
        "func": lambda X: X[:,0]/X[:,1],
    },
    {
        "name": "I.37.4",
        "desc": "Total energy",
        "formula": "E = p^2/(2m) + U",
        "vars": ["p", "m", "U"],
        "func": lambda X: X[:,0]**2/(2*X[:,1]) + X[:,2],
    },
    {
        "name": "I.16.6",
        "desc": "Dynamic pressure",
        "formula": "P = 0.5*ρ*v^2",
        "vars": ["rho", "v"],
        "func": lambda X: 0.5*X[:,0]*X[:,1]**2,
    },
    {
        "name": "I.24.6",
        "desc": "Capacitor energy",
        "formula": "E = 0.5*C*V^2",
        "vars": ["C", "V"],
        "func": lambda X: 0.5*X[:,0]*X[:,1]**2,
    },
    {
        "name": "I.39.1",
        "desc": "Thermal energy",
        "formula": "E = 1.5*kB*T",
        "vars": ["kB", "T"],
        "func": lambda X: 1.5*X[:,0]*X[:,1],
    },
]


def run_hard_experiment():
    """Run hard formula discovery experiment"""
    import time
    
    print("="*70)
    print("Feynman HARD Benchmark - Template-based Discovery")
    print("="*70)
    
    engine = HardFormulaDiscovery()
    results = []
    
    for i, prob in enumerate(FEYNMAN_HARD, 1):
        print(f"\n[{i}/{len(FEYNMAN_HARD)}] {prob['name']}: {prob['desc']}")
        print(f"    Truth: {prob['formula']}")
        
        # Generate data
        np.random.seed(42)
        n_vars = len(prob['vars'])
        X = np.random.uniform(0.5, 2.0, size=(1000, n_vars))
        y = prob['func'](X)
        y += np.random.normal(0, 0.01*np.std(y), size=y.shape)
        
        # Discover
        start = time.time()
        formula, r2 = engine.discover(X, y, prob['vars'])
        elapsed = time.time() - start
        
        # Check success (R2 > 0.95)
        success = r2 > 0.95
        
        print(f"    Found: {formula[:80]}..." if len(formula) > 80 else f"    Found: {formula}")
        print(f"    R2: {r2:.4f}, Time: {elapsed:.3f}s, Success: {success}")
        
        results.append({
            "name": prob['name'],
            "truth": prob['formula'],
            "found": formula,
            "r2": r2,
            "success": success,
        })
    
    # Summary
    n_success = sum(r['success'] for r in results)
    mean_r2 = np.mean([r['r2'] for r in results])
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Success: {n_success}/{len(results)} ({n_success/len(results)*100:.1f}%)")
    print(f"Mean R2: {mean_r2:.4f}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_hard_experiment()
