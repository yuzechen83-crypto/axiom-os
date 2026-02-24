"""
离子化数据模块 (Ionization Data Module)
plasma, electrolyte, gas_ionized 专用预处理与约束
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List


class PlasmaFeatureExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "ion_density": x.abs().mean() + 1e-10,
            "electron_temperature": x.std() + 1e-10,
            "ionization_degree": torch.sigmoid(x.mean()),
        }


class ElectrolyteFeatureExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "ion_density": x.abs().mean() + 1e-10,
            "electron_temperature": torch.tensor(0.1, device=x.device),
            "ionization_degree": torch.sigmoid(x.mean()),
        }


class GasIonizationExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "ion_density": x.abs().mean() + 1e-10,
            "electron_temperature": x.std() + 1e-10,
            "ionization_degree": torch.sigmoid(x.mean()),
        }


class IonizationConstraints:
    """离子化物理约束"""

    def __init__(self, params: Dict):
        self.params = params

    def check(self, output: torch.Tensor, context: Optional[Dict] = None) -> Dict[str, Any]:
        return {"valid": True, "violations": []}

    def correct(self, output: torch.Tensor, context: Dict, violations: List) -> torch.Tensor:
        return output


class IonizationDataModule:
    """离子化数据专用模块"""

    def __init__(self, ionization_types: Optional[List[str]] = None):
        self.ionization_types = ionization_types or ["plasma", "electrolyte", "gas_ionized"]
        self.feature_extractors = nn.ModuleDict({
            "plasma": PlasmaFeatureExtractor(),
            "electrolyte": ElectrolyteFeatureExtractor(),
            "gas_ionized": GasIonizationExtractor(),
        })
        self.ionization_labels = {
            "ion_density": "ions/m³",
            "electron_temperature": "eV",
            "ionization_degree": "dimensionless",
        }

    def preprocess_ionization_data(
        self,
        raw_data: torch.Tensor,
        ionization_type: str,
    ) -> Dict[str, torch.Tensor]:
        extractor = self.feature_extractors[ionization_type] if ionization_type in self.feature_extractors else self.feature_extractors["plasma"]
        features = extractor(raw_data)
        normalized = {}
        for k, v in features.items():
            if k == "ion_density":
                normalized[k] = torch.log10(v + 1)
            elif k == "electron_temperature":
                normalized[k] = v / 100.0
            elif k == "ionization_degree":
                normalized[k] = torch.clamp(v, 0, 1)
            else:
                normalized[k] = v
        return normalized
