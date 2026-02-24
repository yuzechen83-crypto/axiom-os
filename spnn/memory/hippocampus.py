"""
Hippocampus - Structured Memory System M_H
海马体：结构化记忆 (K, V, C, T, R)
m = (k, v, c, t, l, a, r)
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import OrderedDict

from ..core.constants import EPSILON


@dataclass
class MemoryItem:
    """m = (k, v, c, t, l, a, r)"""
    key: np.ndarray      # k = T(l) ⊕ d ⊕ log10(s)
    value: np.ndarray    # v
    confidence: float   # c
    timestamp: float    # t
    label: Any          # l (semantic)
    access_count: int   # a
    reward: float      # r


class Hippocampus:
    """
    M_H = (K, V, C, T, R)
    结构化记忆：物理标尺关联、检索、巩固、防遗忘
    """

    def __init__(
        self,
        dim: int,
        capacity: int = 10000,
        alpha_s: float = 0.5,
        beta_s: float = 0.3,
        gamma_s: float = 0.1,
        delta_s: float = 0.1,
        lambda_r: float = 0.05,
        replay_freq: float = 0.5,
        priority_fn: Optional[Callable] = None,
    ):
        self.dim = dim
        self.capacity = capacity
        self.alpha_s = alpha_s
        self.beta_s = beta_s
        self.gamma_s = gamma_s
        self.delta_s = delta_s
        self.lambda_r = lambda_r
        self.replay_freq = replay_freq
        self.priority_fn = priority_fn or (lambda m: 1.0)

        self._memory: OrderedDict[int, MemoryItem] = OrderedDict()
        self._index: Dict[str, List[int]] = {}
        self._counter = 0
        self._current_time = 0.0

    def _make_key(self, label: Any, dim_vec: np.ndarray, scale_vec: np.ndarray) -> np.ndarray:
        """
        k = T(l) ⊕ d ⊕ log10(s)
        """
        l_enc = np.array(hash(str(label)) % 2**31, dtype=np.float64) if not isinstance(label, (np.ndarray, list)) else np.asarray(label).flatten()
        d = np.asarray(dim_vec).flatten() if dim_vec is not None else np.zeros(5)
        s = np.asarray(scale_vec).flatten() if scale_vec is not None else np.ones(5)
        s_log = np.log10(np.abs(s) + EPSILON)
        if isinstance(l_enc, (int, float)):
            l_enc = np.array([float(l_enc)])
        return np.concatenate([np.atleast_1d(l_enc), d[:5], s_log[:5]])[:self.dim]

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        label: Any = None,
        confidence: float = 1.0,
        reward: float = 0.0,
        dim_vec: Optional[np.ndarray] = None,
        scale_vec: Optional[np.ndarray] = None,
    ) -> int:
        """
        存储记忆项
        """
        k = np.asarray(key).flatten()[:self.dim]
        v = np.asarray(value).flatten()[:self.dim]
        mid = self._counter
        self._counter += 1
        item = MemoryItem(
            key=k,
            value=v,
            confidence=confidence,
            timestamp=self._current_time,
            label=label,
            access_count=0,
            reward=reward,
        )
        if len(self._memory) >= self.capacity:
            self._evict()
        self._memory[mid] = item
        lbl_str = str(label) if label is not None else ""
        if lbl_str not in self._index:
            self._index[lbl_str] = []
        self._index[lbl_str].append(mid)
        return mid

    def _evict(self) -> None:
        """Evict lowest priority item"""
        if not self._memory:
            return
        scores = []
        for mid, m in self._memory.items():
            recency = 1.0 / (1.0 + self._current_time - m.timestamp)
            score = m.confidence * self.priority_fn(m) * recency
            scores.append((mid, score))
        scores.sort(key=lambda x: x[1])
        mid_to_remove = scores[0][0]
        del self._memory[mid_to_remove]
        for k, v in list(self._index.items()):
            if mid_to_remove in v:
                v.remove(mid_to_remove)
                if not v:
                    del self._index[k]
                break

    def retrieve(
        self,
        query_label: Any,
        context: Optional[Dict] = None,
        top_k: int = 5,
        task_relevance_fn: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, List[MemoryItem]]:
        """
        Retrieve(l_query, M_H) = Σ w_m · v_m
        Score = α_s·Sim + β_s·c_m + γ_s·Recency + δ_s·Relevance_task
        """
        context = context or {}
        q = np.asarray(query_label).flatten() if isinstance(query_label, (np.ndarray, list)) else np.array([hash(str(query_label)) % 2**31], dtype=np.float64)
        candidates = list(self._memory.items())
        if not candidates:
            return np.zeros(self.dim), []

        scores = []
        for mid, m in candidates:
            sim = (1.0 - np.linalg.norm(m.key - q) / (np.linalg.norm(q) + EPSILON)) if q.size == m.key.size else 0.5
            sim = np.clip(sim, 0, 1)
            recency = 1.0 / (1.0 + max(0, self._current_time - m.timestamp))
            if task_relevance_fn is not None:
                rel = task_relevance_fn(m, context)
            else:
                rel = 1.0
            score = self.alpha_s * sim + self.beta_s * m.confidence + self.gamma_s * recency + self.delta_s * rel
            scores.append((mid, score, m))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        weights = np.array([s[1] for s in top])
        weights = weights / (weights.sum() + EPSILON)
        result = sum(w * m.value for (_, _, m), w in zip(top, weights))
        return result, [m for _, _, m in top]

    def retrieve_tensor(
        self,
        query_label: Any,
        device: torch.device,
        top_k: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve for PyTorch (returns tensor)"""
        val, _ = self.retrieve(query_label, top_k=top_k, **kwargs)
        return torch.as_tensor(val, dtype=torch.float32, device=device)

    def consolidate(self, base_freq: float = 0.5) -> None:
        """
        记忆巩固
        f_replay(m) = f_base · (c_m · Priority_task(m)) / Σ(...)
        """
        if not self._memory:
            return
        total = sum(m.confidence * self.priority_fn(m) for m in self._memory.values())
        if total <= 0:
            return
        for m in self._memory.values():
            m.confidence = min(1.0, m.confidence + base_freq * (m.confidence * self.priority_fn(m) / total) * self.lambda_r)

    def advance_time(self, dt: float = 1.0) -> None:
        self._current_time += dt

    def get_consistency(self, label: Any) -> float:
        """历史表现记录，用于主脑筛选"""
        _, items = self.retrieve(label, top_k=3)
        if not items:
            return 1.0
        return float(np.mean([m.confidence for m in items]))
