"""
Physical Scale System B = (B_universal, B_characteristic, B_numerical)
物理标尺系统：从普朗克尺度到工程尺度的统一管理
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field

from .constants import (
    c, G, hbar, k_B, epsilon_0,
    ℓ_P, t_P, m_P,
    EPSILON, TAU_SCALE, EPSILON_MACHINE,
)


@dataclass
class UniversalConstants:
    """B_universal: 宇宙常数基准"""
    c: float = c
    G: float = G
    hbar: float = hbar
    k_B: float = k_B
    epsilon_0: float = epsilon_0
    ℓ_P: float = field(default_factory=lambda: ℓ_P)
    t_P: float = field(default_factory=lambda: t_P)
    m_P: float = field(default_factory=lambda: m_P)


@dataclass
class CharacteristicScales:
    """B_characteristic: 问题相关特征尺度"""
    ℓ_c: float = 1.0  # Characteristic length
    t_c: float = 1.0  # Characteristic time
    m_c: float = 1.0  # Characteristic mass
    T_c: float = 1.0  # Characteristic temperature
    q_c: float = 1.0  # Characteristic charge


@dataclass
class NumericalScales:
    """B_numerical: 数值计算层（硬件适配）"""
    epsilon_machine: float = EPSILON_MACHINE
    float_type: str = "float64"
    range_min: float = 1e-308
    range_max: float = 1e308


class PhysicalScaleSystem:
    """
    Physical Scale System B
    三层锚定体系：宇宙常数层 + 特征尺度层 + 数值计算层
    """

    def __init__(
        self,
        universal: Optional[UniversalConstants] = None,
        characteristic: Optional[CharacteristicScales] = None,
        numerical: Optional[NumericalScales] = None,
        scale_tolerance: float = TAU_SCALE,
    ):
        self.universal = universal or UniversalConstants()
        self.characteristic = characteristic or CharacteristicScales()
        self.numerical = numerical or NumericalScales()
        self.scale_tolerance = scale_tolerance

        # Physical mapping matrix diagonal exponents [L, M, T, Θ, I]
        self._dims_cache: Dict[str, np.ndarray] = {}

    def auto_detect_characteristic(
        self,
        x: np.ndarray,
        v_c: Optional[float] = None,
    ) -> CharacteristicScales:
        """
        自动检测特征尺度
        ℓ_c = median(||x_i - x_j||), t_c = ℓ_c / v_c
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = min(x.shape[0], 100)  # Sample for efficiency
        idx = np.random.choice(x.shape[0], min(n, x.shape[0]), replace=False)
        x_sample = x[idx]

        diffs = []
        for i in range(len(x_sample)):
            for j in range(i + 1, len(x_sample)):
                diffs.append(np.linalg.norm(x_sample[i] - x_sample[j]))
        ℓ_c = float(np.median(diffs)) if diffs else 1.0
        ℓ_c = max(ℓ_c, 1e-10)

        v_c = v_c or 1.0
        t_c = ℓ_c / v_c

        self.characteristic = CharacteristicScales(ℓ_c=ℓ_c, t_c=t_c)
        return self.characteristic

    def normalize(
        self,
        x: np.ndarray,
        dim_exponents: Optional[np.ndarray] = None,
        x_p: Optional[float] = None,
    ) -> np.ndarray:
        """
        物理锚定（普朗克尺度归一化）
        x̃ = (x / x_p) · [c/c0, G/G0, ℏ/ℏ0]^d
        """
        if dim_exponents is None:
            dim_exponents = np.zeros(5)
        x_p = x_p or self.characteristic.ℓ_c
        scale_factors = np.array([
            self.universal.c / self.universal.c,
            self.universal.G / self.universal.G,
            self.universal.hbar / self.universal.hbar,
        ])
        x_norm = np.asarray(x, dtype=np.float64)
        x_norm = x_norm / (x_p + EPSILON)
        return x_norm

    def denormalize(
        self,
        x_tilde: np.ndarray,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        尺度感知反推：x_output = B · x̃_output
        """
        scale = scale or self.characteristic.ℓ_c
        return np.asarray(x_tilde) * scale

    def get_physical_mapping_matrix(
        self,
        dim_exponents: np.ndarray,
        current_scales: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        M_phys = diag(∏_q (ℓ_q / ℓ_{q,0})^{d_{qk}})
        物理映射算子
        """
        keys = ["ℓ", "t", "m", "T", "q"]
        ref = [self.characteristic.ℓ_c, self.characteristic.t_c,
               self.characteristic.m_c, self.characteristic.T_c, self.characteristic.q_c]
        curr = current_scales or {k: r for k, r in zip(keys, ref)}
        dims = dim_exponents if len(dim_exponents) >= 5 else np.zeros(5)

        diag = 1.0
        for q, (key, ref_val) in enumerate(zip(keys, ref)):
            ℓ_q = curr.get(key, ref_val)
            ℓ_q0 = ref_val
            if abs(ℓ_q0) > EPSILON:
                d_qk = dims[q] if q < len(dims) else 0
                diag *= (ℓ_q / ℓ_q0) ** d_qk
        return np.array([diag])

    def dim_check(
        self,
        d1: np.ndarray,
        d2: np.ndarray,
        s1: Optional[np.ndarray] = None,
        s2: Optional[np.ndarray] = None,
    ) -> bool:
        """
        量纲一致性验证
        DimCheck(T1, T2) = I(d1=d2) · I(||log10(s1/s2)||_∞ < τ_scale)
        """
        d1 = np.asarray(d1).flatten()
        d2 = np.asarray(d2).flatten()
        if d1.shape != d2.shape or not np.allclose(d1, d2):
            return False
        if s1 is not None and s2 is not None:
            s1, s2 = np.asarray(s1).flatten(), np.asarray(s2).flatten()
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ratio = np.log10(np.abs(s1) / (np.abs(s2) + EPSILON) + EPSILON)
            if np.any(np.abs(log_ratio) >= self.scale_tolerance):
                return False
        return True

    def adjust_scale(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        physics_rules: Optional[str] = "geometric",
    ) -> np.ndarray:
        """
        量纲不匹配时的自动修复
        s_fixed = B · AdjustScale(s1, s2, physics_rules)
        """
        s1, s2 = np.asarray(s1).flatten(), np.asarray(s2).flatten()
        if physics_rules == "geometric":
            return np.sqrt(np.abs(s1) * np.abs(s2) + EPSILON)
        return (s1 + s2) / 2

    def brain_mediate(
        self,
        B1: "PhysicalScaleSystem",
        B2: "PhysicalScaleSystem",
        coupling_type: str = "geometric",
        alpha: Optional[float] = None,
    ) -> CharacteristicScales:
        """
        跨系统标尺协调（主脑管理）
        ℓ_coupling = √(ℓ1·ℓ2) or α·ℓ1 + (1-α)·ℓ2
        """
        c1, c2 = B1.characteristic, B2.characteristic
        if coupling_type == "geometric":
            return CharacteristicScales(
                ℓ_c=np.sqrt(c1.ℓ_c * c2.ℓ_c),
                t_c=np.sqrt(c1.t_c * c2.t_c),
                m_c=np.sqrt(c1.m_c * c2.m_c),
                T_c=np.sqrt(c1.T_c * c2.T_c),
                q_c=np.sqrt(c1.q_c * c2.q_c),
            )
        alpha = alpha or 0.5
        return CharacteristicScales(
            ℓ_c=alpha * c1.ℓ_c + (1 - alpha) * c2.ℓ_c,
            t_c=alpha * c1.t_c + (1 - alpha) * c2.t_c,
            m_c=alpha * c1.m_c + (1 - alpha) * c2.m_c,
            T_c=alpha * c1.T_c + (1 - alpha) * c2.T_c,
            q_c=alpha * c1.q_c + (1 - alpha) * c2.q_c,
        )
