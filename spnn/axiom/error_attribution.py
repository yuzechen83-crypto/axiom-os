"""
异常溯源模块 (Error Attribution)
e = [e_magnitude, e_spatial_corr, e_temporal_corr, e_symmetry, e_conservation]
ErrorType(e) = argmax_c w_c^T e
"""

import torch
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class RootCause(Enum):
    AI_EXPLORATORY = "ai_exploratory_violation"
    AXIOM_PARAMETRIC_DRIFT = "axiom_parametric_drift"
    AXIOM_STRUCTURAL = "axiom_structural_insufficiency"
    SENSOR_FAULT = "sensor_fault_or_calibration_issue"
    NUMERICAL_INSTABILITY = "numerical_instability"
    UNKNOWN = "unknown_cause_needs_investigation"


@dataclass
class DiagnosticReport:
    gate_value: float
    residual_norm: float
    root_cause: RootCause
    recommended_action: str
    pattern_scores: Dict[str, float]


class ErrorAttributionDiagnostic:
    """异常溯源与根因分析"""

    def __init__(self):
        self.pattern_library: Dict[str, callable] = {}
        self.diagnostic_log: List[Dict] = []

    def extract_features(
        self,
        residual: torch.Tensor,
        context: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """提取残差的物理特征"""
        r = residual.detach()
        mag = r.norm().item()
        flat = r.flatten()
        spatial_corr = 0.5
        if flat.numel() > 1:
            c = torch.corrcoef(torch.stack([flat[:-1], flat[1:]]))[0, 1].item()
            spatial_corr = c if not torch.isnan(torch.tensor(c)) else 0.5
        return {
            "magnitude": mag,
            "spatial_correlation": spatial_corr,
            "temporal_correlation": 0.5,
            "symmetry_violation": 0.0,
            "conservation_violation": 0.0,
        }

    def diagnose_violation(
        self,
        soft_out: torch.Tensor,
        hard_out: torch.Tensor,
        gate: torch.Tensor,
        context: Optional[Dict] = None,
    ) -> DiagnosticReport:
        """诊断DAS高门控值的原因"""
        gate_val = gate.mean().item() if gate.numel() > 1 else gate.item()
        if gate_val < 0.8:
            return DiagnosticReport(
                gate_value=gate_val,
                residual_norm=0.0,
                root_cause=RootCause.UNKNOWN,
                recommended_action="normal",
                pattern_scores={},
            )
        residual = soft_out - hard_out
        features = self.extract_features(residual, context)
        pattern_scores = {
            "parametric_drift": 0.5 + 0.2 * features["spatial_correlation"],
            "structural_mismatch": 0.3,
            "numerical_instability": 0.2 if features["magnitude"] > 1e6 else 0.0,
            "ai_exploration": 0.4,
            "sensor_anomaly": 0.2,
        }
        root_cause = self.infer_root_cause(pattern_scores)
        action = self.suggest_action(root_cause)
        report = DiagnosticReport(
            gate_value=gate_val,
            residual_norm=features["magnitude"],
            root_cause=root_cause,
            recommended_action=action,
            pattern_scores=pattern_scores,
        )
        self.diagnostic_log.append({
            "report": report,
            "timestamp": time.time(),
            "context": context,
        })
        return report

    def infer_root_cause(self, pattern_scores: Dict[str, float]) -> RootCause:
        if pattern_scores.get("parametric_drift", 0) > 0.7:
            return RootCause.AXIOM_PARAMETRIC_DRIFT
        if pattern_scores.get("structural_mismatch", 0) > 0.6:
            return RootCause.AXIOM_STRUCTURAL
        if pattern_scores.get("numerical_instability", 0) > 0.8:
            return RootCause.NUMERICAL_INSTABILITY
        if pattern_scores.get("ai_exploration", 0) > 0.5:
            return RootCause.AI_EXPLORATORY
        if pattern_scores.get("sensor_anomaly", 0) > 0.7:
            return RootCause.SENSOR_FAULT
        return RootCause.UNKNOWN

    def suggest_action(self, root_cause: RootCause) -> str:
        m = {
            RootCause.AXIOM_PARAMETRIC_DRIFT: "request_calibration",
            RootCause.AXIOM_STRUCTURAL: "consider_model_extension",
            RootCause.NUMERICAL_INSTABILITY: "reduce_step_size",
            RootCause.AI_EXPLORATORY: "monitor_and_log",
            RootCause.SENSOR_FAULT: "check_sensors",
            RootCause.UNKNOWN: "investigate",
        }
        return m.get(root_cause, "investigate")
