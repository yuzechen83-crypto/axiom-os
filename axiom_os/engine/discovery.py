"""
Discovery Engine: 优先使用 axiom_core_proprietary，否则使用公开回退（多项式+Lasso）。
无专有包时基准与 Demo 仍可运行，结果与完整版可能略有差异。
"""
try:
    from axiom_core_proprietary.discovery import DiscoveryEngine
except ImportError:
    from axiom_os.engine.forms import MultivariatePolyForm

    class DiscoveryEngine:
        """公开回退：多项式特征 + Lasso，使 clone 后无专有包也可跑基准与 Demo。"""

        def __init__(self, use_pysr=False):
            self.use_pysr = use_pysr

        def discover_multivariate(
            self,
            X,
            y,
            var_names=None,
            selector="bic",
            sample_weight=None,
        ):
            form = MultivariatePolyForm(alpha=1e-3, degree=2, include_interactions=True)
            pred, coefs, formula = form.fit(X, y, var_names=var_names, sample_weight=sample_weight)
            return formula, pred, coefs
