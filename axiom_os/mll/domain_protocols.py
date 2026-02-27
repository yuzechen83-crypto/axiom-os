"""
Domain Protocols - 多领域学习协议

每个协议：load_data → train → evaluate → discover → crystallize
协议可耦合：Hippocampus 跨域共享，Discovery 公式可跨域复用。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ProtocolResult:
    """协议执行结果"""
    domain: str
    ok: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    discovered: Optional[str] = None
    crystallized: bool = False
    error: Optional[str] = None


class DomainProtocol(ABC):
    """
    领域协议抽象基类
    子类实现：load_data, train, evaluate, discover, crystallize
    """

    domain: str = "generic"
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他 domain，用于耦合

    @abstractmethod
    def load_data(self) -> Tuple[Any, Any, Dict]:
        """加载数据，返回 (X, y, meta)"""
        pass

    @abstractmethod
    def train(self, X: Any, y: Any, hippocampus: Optional[Any] = None, **kwargs) -> Dict:
        """训练，返回 {model, loss_history, ...}"""
        pass

    @abstractmethod
    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        """
        评估，返回 {mae, r2, ...}。
        model 可为 nn.Module（可调用）或 dict（如 RAR 的 run_rar_discovery 结果）。
        """
        pass

    def discover(self, model: Any, X: Any, y: Any) -> Optional[str]:
        """发现公式，可选覆盖"""
        return None

    def crystallize(self, formula: str, hippocampus: Optional[Any] = None) -> bool:
        """结晶到 Hippocampus，可选覆盖"""
        return False

    def run(
        self,
        hippocampus: Optional[Any] = None,
        epochs: int = 500,
        do_discover: bool = True,
        do_crystallize: bool = False,
    ) -> ProtocolResult:
        """完整协议执行"""
        try:
            X, y, meta = self.load_data()
            train_res = self.train(X, y, hippocampus=hippocampus, epochs=epochs)
            model = train_res.get("model")
            if model is None:
                return ProtocolResult(domain=self.domain, ok=False, error="No model from train")

            metrics = self.evaluate(model, X, y)
            discovered = None
            crystallized = False
            if do_discover:
                discovered = self.discover(model, X, y)
                if do_crystallize and discovered and hippocampus:
                    crystallized = self.crystallize(discovered, hippocampus)

            return ProtocolResult(
                domain=self.domain,
                ok=True,
                metrics=metrics,
                discovered=discovered,
                crystallized=crystallized,
            )
        except Exception as e:
            return ProtocolResult(domain=self.domain, ok=False, error=str(e))


class TurbulenceProtocol(DomainProtocol):
    """湍流领域协议"""

    domain = "fluids"

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
        coords, targets, meta = load_atmospheric_turbulence_3d(
            n_lat=3, n_lon=3, delta_deg=0.15, forecast_days=3, use_synthetic_if_fail=True
        )
        return coords, targets, meta

    def train(self, X: Any, y: Any, hippocampus: Optional[Any] = None, epochs: int = 500, **kwargs) -> Dict:
        import torch
        from axiom_os.core.turbulence_scale import TurbulencePhysicalScale
        from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive

        X, y = np.asarray(X), np.asarray(y)
        split = int(0.8 * len(X))
        X_train = torch.from_numpy(X[:split]).float()
        Y_train = torch.from_numpy(y[:split]).float()
        u_mean = float(Y_train[:, 0].mean())
        v_mean = float(Y_train[:, 1].mean())
        wind_mag = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
        thresh = max(float(np.percentile(wind_mag, 85)), 5.0)
        hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=thresh, use_enhanced=True)

        from axiom_os.layers.rcln import RCLNLayer
        rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, hard_core_func=hard_core, lambda_res=1.0)
        opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
        loss_fn = torch.nn.HuberLoss(delta=1.0)
        w_u, w_v = 1.5, 1.0
        history = []
        for epoch in range(epochs):
            rcln.set_lambda_decay(epoch, epochs, decay_min=0.6)
            opt.zero_grad()
            pred = rcln(X_train)
            loss = w_u * loss_fn(pred[:, 0:1], Y_train[:, 0:1]) + w_v * loss_fn(pred[:, 1:2], Y_train[:, 1:2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rcln.parameters(), 1.0)
            opt.step()
            history.append(loss.item())
        with torch.no_grad():
            _ = rcln(X_train)  # populate _last_y_soft for discovery
            pred_test = rcln(torch.from_numpy(X[split:]).float()).numpy()
        mae_u = np.mean(np.abs(pred_test[:, 0] - y[split:, 0]))
        mae_v = np.mean(np.abs(pred_test[:, 1] - y[split:, 1]))
        return {"model": rcln, "loss_history": history, "mae_u": mae_u, "mae_v": mae_v}

    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        import torch
        X, y = np.asarray(X), np.asarray(y)
        split = int(0.8 * len(X))
        with torch.no_grad():
            pred = model(torch.from_numpy(X[split:]).float()).numpy()
        mae_u = float(np.mean(np.abs(pred[:, 0] - y[split:, 0])))
        mae_v = float(np.mean(np.abs(pred[:, 1] - y[split:, 1])))
        return {"mae_u": mae_u, "mae_v": mae_v, "mae": (mae_u + mae_v) / 2}

    def discover(self, model: Any, X: Any, y: Any) -> Optional[str]:
        if not hasattr(model, "_last_y_soft") or model._last_y_soft is None:
            return None
        from axiom_os.engine.discovery import DiscoveryEngine
        X, y = np.asarray(X), np.asarray(y)
        split = int(0.8 * len(X))
        y_soft = model._last_y_soft.numpy()
        engine = DiscoveryEngine(use_pysr=False)
        formula_u, _, _ = engine.discover_multivariate(X[:split], y_soft[:split, 0], var_names=["t","x","y","z"], selector="bic")
        return formula_u[:80] + "..." if formula_u and len(formula_u) > 80 else formula_u


class RARProtocol(DomainProtocol):
    """RAR 领域协议"""

    domain = "mechanics"

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        from axiom_os.datasets.sparc import load_sparc_rar
        g_bar, g_obs, names = load_sparc_rar(n_galaxies=100, use_mock_if_fail=True, use_real=True)
        return np.asarray(g_bar).reshape(-1, 1), np.asarray(g_obs), {"n": len(g_bar)}


    def train(self, X: Any, y: Any, hippocampus: Optional[Any] = None, epochs: int = 200, **kwargs) -> Dict:
        from axiom_os.experiments.discovery_rar import run_rar_discovery
        res = run_rar_discovery(n_galaxies=min(50, len(X) // 10) if len(X) > 100 else 30, epochs=epochs)
        return {"model": res, "r2": res.get("r2", 0), "formula": res.get("formula")}

    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        r2 = model.get("r2", 0) if isinstance(model, dict) else 0
        return {"r2": float(r2), "mae": max(0, 1.0 - r2)}

    def discover(self, model: Any, X: Any, y: Any) -> Optional[str]:
        if isinstance(model, dict):
            return model.get("formula")
        return None


class BatteryProtocol(DomainProtocol):
    """电池 RUL 领域协议"""

    domain = "battery"

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        from axiom_os.datasets.nasa_battery import load_battery_data
        cycles_norm, capacity_norm, scalers = load_battery_data()
        X = np.asarray(cycles_norm, dtype=np.float32).reshape(-1, 1)
        y = np.asarray(capacity_norm, dtype=np.float32)
        return X, y, {"scalers": scalers}

    def train(self, X: Any, y: Any, hippocampus: Optional[Any] = None, epochs: int = 300, **kwargs) -> Dict:
        import torch
        from axiom_os.layers.rcln import RCLNLayer
        X, y = np.asarray(X), np.asarray(y)
        split = int(0.6 * len(X))
        X_train = torch.from_numpy(X[:split]).float()
        Y_train = torch.from_numpy(y[:split]).float()
        if Y_train.dim() == 1:
            Y_train = Y_train.unsqueeze(1)
        rcln = RCLNLayer(input_dim=1, hidden_dim=32, output_dim=1, hard_core_func=None, lambda_res=1.0)
        opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            pred = rcln(X_train)
            if pred.shape != Y_train.shape:
                pred = pred.squeeze(-1) if pred.dim() > Y_train.dim() else pred
            loss = torch.nn.functional.mse_loss(pred, Y_train)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred_test = rcln(torch.from_numpy(X[split:]).float()).numpy().ravel()
        mae = float(np.mean(np.abs(pred_test - y[split:])))
        return {"model": rcln, "mae": mae}

    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        import torch
        X, y = np.asarray(X), np.asarray(y)
        split = int(0.6 * len(X))
        with torch.no_grad():
            pred = model(torch.from_numpy(X[split:]).float()).numpy().ravel()
        mae = float(np.mean(np.abs(pred - y[split:])))
        return {"mae": mae, "r2": 0.0}


class TurbulencePINNLSTMProtocol(DomainProtocol):
    """
    湍流时空协议：PINN-LSTM 时序建模
    使用 PhysicsInformedLSTM (Hard Core + LSTM + PDE 残差 + Coach)
    """

    domain = "fluids"
    dependencies: List[str] = []

    def load_data(self) -> Tuple[Any, Any, Dict]:
        from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
        coords, targets, meta = load_atmospheric_turbulence_3d(
            n_lat=3, n_lon=3, delta_deg=0.15, forecast_days=3, use_synthetic_if_fail=True
        )
        self._pinn_meta = meta
        return (np.asarray(coords), np.asarray(targets), meta)

    def train(
        self,
        X: Any,
        y: Any,
        hippocampus: Optional[Any] = None,
        epochs: int = 500,
        seq_len: int = 8,
        **kwargs,
    ) -> Dict:
        import torch
        from axiom_os.experiments.run_turbulence_pinn_lstm import build_sequences
        from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
        from axiom_os.layers.pinn_lstm import PhysicsInformedLSTM, pde_residual_loss_temporal

        coords, targets = np.asarray(X), np.asarray(y)
        meta = getattr(self, "_pinn_meta", {})
        result = build_sequences(coords, targets, seq_len, meta)
        if result[0] is None:
            return {"model": None, "mae": float("nan"), "error": "Could not build sequences"}

        X_seq, Y, _ = result
        split = int(0.8 * len(X_seq))
        X_train = torch.from_numpy(X_seq[:split]).float()
        Y_train = torch.from_numpy(Y[:split]).float()

        u_mean = float(Y_train[:, 0].mean())
        v_mean = float(Y_train[:, 1].mean())
        wind_mag = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
        thresh = max(float(np.percentile(wind_mag, 85)), 5.0)
        hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=thresh, use_enhanced=True)

        model = PhysicsInformedLSTM(
            input_dim=4,
            hidden_dim=64,
            output_dim=2,
            seq_len=seq_len,
            hard_core_func=hard_core,
            lambda_res=1.0,
            num_layers=2,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.HuberLoss(delta=1.0)
        lambda_pde = 0.02
        try:
            from axiom_os.coach import coach_loss_torch
            _coach_loss_fn = coach_loss_torch
            lambda_coach = 0.15
        except ImportError:
            _coach_loss_fn = None
            lambda_coach = 0.0

        for epoch in range(epochs):
            model.set_lambda_decay(epoch, epochs, decay_min=0.6)
            opt.zero_grad()
            pred = model(X_train)
            loss_data = 1.5 * loss_fn(pred[:, 0:1], Y_train[:, 0:1]) + loss_fn(pred[:, 1:2], Y_train[:, 1:2])
            if lambda_pde > 0 and X_train.shape[1] >= 2:
                with torch.no_grad():
                    u_prev = hard_core(X_train[:, -2, :])
                    if not isinstance(u_prev, torch.Tensor):
                        u_prev = torch.as_tensor(u_prev, dtype=torch.float32, device=pred.device)
                l_pde = pde_residual_loss_temporal(pred, u_prev, dt=1.0 / seq_len)
            else:
                l_pde = torch.tensor(0.0, device=pred.device)
            l_coach = _coach_loss_fn(pred, domain="fluids") if _coach_loss_fn and lambda_coach > 0 else torch.tensor(0.0, device=pred.device)
            loss = loss_data + lambda_pde * l_pde + lambda_coach * l_coach
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        with torch.no_grad():
            X_test = torch.from_numpy(X_seq[split:]).float()
            pred_test = model(X_test).numpy()
        mae_u = float(np.mean(np.abs(pred_test[:, 0] - Y[split:, 0])))
        mae_v = float(np.mean(np.abs(pred_test[:, 1] - Y[split:, 1])))
        res = {
            "model": model,
            "mae_u": mae_u,
            "mae_v": mae_v,
            "mae": (mae_u + mae_v) / 2,
        }
        self._last_train_res = res
        self._pinn_X_seq = X_seq
        self._pinn_Y = Y
        self._pinn_split = split
        return res

    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        train_res = getattr(self, "_last_train_res", None)
        if train_res and "mae" in train_res:
            return {"mae": train_res["mae"], "mae_u": train_res.get("mae_u", 0), "mae_v": train_res.get("mae_v", 0)}
        if model is not None and hasattr(model, "eval"):
            import torch
            X_seq = getattr(self, "_pinn_X_seq", None)
            Y = getattr(self, "_pinn_Y", None)
            split = getattr(self, "_pinn_split", 0)
            if X_seq is not None and Y is not None and split > 0:
                with torch.no_grad():
                    pred = model(torch.from_numpy(X_seq[split:]).float()).numpy()
                mae_u = float(np.mean(np.abs(pred[:, 0] - Y[split:, 0])))
                mae_v = float(np.mean(np.abs(pred[:, 1] - Y[split:, 1])))
                return {"mae": (mae_u + mae_v) / 2, "mae_u": mae_u, "mae_v": mae_v}
        return {"mae": 0.0, "mae_u": 0.0, "mae_v": 0.0}


class AcrobotProtocol(DomainProtocol):
    """双摆控制领域协议（简化：仅主循环）"""

    domain = "mechanics"

    def load_data(self) -> Tuple[Any, Any, Dict]:
        return None, None, {"simulated": True}

    def train(self, X: Any, y: Any, hippocampus: Optional[Any] = None, epochs: int = 200, **kwargs) -> Dict:
        from axiom_os.main import main
        from axiom_os.main import AxiomConfig
        cfg = AxiomConfig(T_max=100, discover_interval=25)
        main(config=cfg, use_ai=True)
        return {"model": None, "ok": True}

    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        return {"loss": 0.0, "mae": 0.0}


# 协议注册表
PROTOCOL_REGISTRY: Dict[str, DomainProtocol] = {
    "turbulence": TurbulenceProtocol(),
    "turbulence_pinn_lstm": TurbulencePINNLSTMProtocol(),
    "rar": RARProtocol(),
    "battery": BatteryProtocol(),
    "acrobot": AcrobotProtocol(),
}


def generate_protocol_template(domain: str, template: str = "minimal") -> str:
    """
    生成新领域协议代码模板
    返回可执行的 Python 代码字符串。
    """
    class_name = "".join(w.capitalize() for w in domain.split("_")) + "Protocol"
    if template == "minimal":
        return f'''"""
{class_name} - 自动生成模板
"""

from axiom_os.mll.domain_protocols import DomainProtocol
from typing import Any, Dict, Tuple
import numpy as np


class {class_name}(DomainProtocol):
    domain = "{domain}"

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        # TODO: 加载数据
        X = np.zeros((100, 1))
        y = np.zeros(100)
        return X, y, {{}}

    def train(self, X: Any, y: Any, hippocampus=None, epochs=500, **kwargs) -> Dict:
        # TODO: 训练 RCLN 或调用 Discovery
        return {{"model": None, "mae": 0.0}}

    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        return {{"mae": 0.0, "r2": 0.0}}
'''
    return ""
