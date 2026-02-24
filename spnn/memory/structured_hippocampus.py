"""
感知海马体 (Structured Hippocampus)
基于物理标签的关联模式存储、检索、灾难性遗忘防护
"""

import torch
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    pattern: torch.Tensor
    labels: List[str]
    confidence: float
    last_accessed: float
    access_count: int
    retention_strength: float


class MemoryActivationManager:
    """记忆活性管理"""

    def __init__(self, decay_time: float = 3600.0):
        self.activation: Dict[str, float] = {}
        self.decay_time = decay_time

    def update_activation(self, key: str) -> None:
        self.activation[key] = time.time()

    def is_active(self, key: str, last_accessed: float) -> bool:
        return (time.time() - last_accessed) < self.decay_time

    def enhance_activation(self, key: str) -> None:
        self.activation[key] = time.time()

    def get_important_memories(self) -> List[str]:
        recent = time.time() - 300
        return [k for k, t in self.activation.items() if t > recent]


class StructuredHippocampus:
    """原始SPNN的感知海马体"""

    def __init__(self, capacity: int = 10000, retention_decay: float = 0.99):
        self.capacity = capacity
        self.retention_decay = retention_decay
        self.memory_banks: Dict[str, List[MemoryEntry]] = {
            "physical_patterns": {},
            "causal_sequences": {},
            "anomaly_patterns": {},
            "successful_couplings": {},
        }
        self.activation_manager = MemoryActivationManager()

    def _create_label_key(self, labels: List[str]) -> str:
        return "|".join(sorted(labels))

    def _parse_label_key(self, key: str) -> List[str]:
        return key.split("|")

    def _compute_label_similarity(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(set(a) & set(b))
        return inter / max(len(set(a)), len(set(b)), 1)

    def store_physical_pattern(
        self,
        pattern: torch.Tensor,
        physical_labels: List[str],
        confidence: float = 1.0,
    ) -> None:
        label_key = self._create_label_key(physical_labels)
        if label_key not in self.memory_banks["physical_patterns"]:
            self.memory_banks["physical_patterns"][label_key] = []
        self.memory_banks["physical_patterns"][label_key].append(
            MemoryEntry(
                pattern=pattern.detach().cpu(),
                labels=physical_labels,
                confidence=confidence,
                last_accessed=time.time(),
                access_count=0,
                retention_strength=1.0,
            )
        )
        total = sum(len(v) for v in self.memory_banks["physical_patterns"].values())
        if total > self.capacity:
            self._enforce_capacity("physical_patterns")
        self.activation_manager.update_activation(label_key)

    def _enforce_capacity(self, bank: str) -> None:
        all_entries = []
        for k, v in self.memory_banks[bank].items():
            for e in v:
                all_entries.append((k, e))
        if len(all_entries) <= self.capacity:
            return
        all_entries.sort(key=lambda x: x[1].confidence * x[1].retention_strength)
        to_remove = len(all_entries) - self.capacity
        for _ in range(to_remove):
            k, e = all_entries.pop(0)
            self.memory_banks[bank][k].remove(e)
            if not self.memory_banks[bank][k]:
                del self.memory_banks[bank][k]

    def retrieve_by_physical_label(
        self,
        query_labels: List[str],
        similarity_threshold: float = 0.7,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        best_match = None
        best_sim = 0.0
        for label_key, entries in self.memory_banks["physical_patterns"].items():
            sim = self._compute_label_similarity(query_labels, self._parse_label_key(label_key))
            if sim >= similarity_threshold and sim > best_sim:
                active = [e for e in entries if self.activation_manager.is_active(label_key, e.last_accessed)]
                if active:
                    best = max(active, key=lambda e: e.confidence * e.retention_strength)
                    best_match = best.pattern
                    best_sim = sim
                    best.access_count += 1
                    best.last_accessed = time.time()
                    self.activation_manager.enhance_activation(self._create_label_key(query_labels))
        if best_match is not None and device is not None:
            best_match = best_match.to(device)
        return best_match

    def cognitive_maintenance_step(self) -> None:
        """单步认知维护"""
        for bank in self.memory_banks.values():
            for key, entries in list(bank.items()):
                for e in entries:
                    e.retention_strength *= self.retention_decay
                    if e.retention_strength < 0.1:
                        e.confidence *= 0.9
