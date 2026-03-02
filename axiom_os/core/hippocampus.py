"""
Hippocampus - Knowledge Registry
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Stores crystallized physical laws found by the system.

RAG: retrieve_by_query supports keyword (default) and optional semantic embeddings.
"""

from typing import Optional, List, Tuple, Any, Callable, Union, Dict
import re

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional: sentence-transformers for semantic RAG (heavy dependency)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def _formula_to_callable(
    formula: str,
    output_dim: int,
) -> Callable:
    """
    Convert a symbolic formula string to a callable for hard_core.
    Formula uses 'x' for input (numpy array). E.g., "x[0]**2 + x[1]**2".
    Restricted namespace for safety.
    """
    safe_namespace = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "abs": np.abs,
        "pow": np.power,
    }

    def fn(x: Any) -> np.ndarray:
        if HAS_TORCH and isinstance(x, torch.Tensor):
            vals = x.detach().cpu().numpy()
        elif hasattr(x, "values") and not (HAS_TORCH and isinstance(x, torch.Tensor)):
            vals = x.values
            if HAS_TORCH and isinstance(vals, torch.Tensor):
                vals = vals.detach().cpu().numpy()
        else:
            vals = x
        x_arr = np.asarray(vals, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        safe_namespace["x"] = x_arr
        for i in range(min(10, x_arr.shape[1])):
            safe_namespace[f"x{i}"] = x_arr[:, i]
        try:
            result = eval(formula, {"__builtins__": {}}, safe_namespace)
        except Exception:
            result = np.zeros((x_arr.shape[0], output_dim))
        result = np.asarray(result, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        if result.shape[1] != output_dim:
            result = np.broadcast_to(result[:, :1], (result.shape[0], output_dim))
        return result

    return fn


def _reset_module_weights(module: Any) -> None:
    """Reset trainable parameters of a module to default initialization."""
    if not HAS_TORCH:
        return
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


class Hippocampus:
    """
    Knowledge Registry: Stores crystallized physical laws.
    Crystallize: Move discovered formulas into hard_core, reset soft_shell.
    RAG: use_semantic_rag=True + sentence-transformers → 语义检索；否则关键词检索。
    """

    def __init__(
        self,
        dim: int = 32,
        capacity: int = 5000,
        use_semantic_rag: bool = False,
        embedding_model: str = "paraphrase-MiniLM-L3-v2",
        use_bundle_field: bool = True,
    ):
        self.dim = dim
        self.capacity = capacity
        self.knowledge_base: dict = {}  # formula_id -> {formula, callable, metadata}
        self._memory: List[Tuple[np.ndarray, Any, float]] = []
        self.use_semantic_rag = use_semantic_rag and HAS_SENTENCE_TRANSFORMERS
        self._embedding_model_name = embedding_model
        self._embedder: Optional[Any] = None
        self._bundle_field: Optional[Any] = None
        if use_bundle_field:
            try:
                from .bundle_field import MetaAxisBundleField
                self._bundle_field = MetaAxisBundleField()
            except ImportError:
                self._bundle_field = None

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        label: Any = None,
        confidence: float = 1.0,
    ) -> None:
        """Store a pattern with optional label."""
        if len(self._memory) >= self.capacity:
            self._memory.pop(0)
        self._memory.append((np.asarray(key), value, confidence))

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[float], List[Any]]:
        """Retrieve top-k similar patterns."""
        if not self._memory:
            return [], []
        q = np.asarray(query).ravel()
        scores = [float(-np.linalg.norm(np.asarray(emb).ravel() - q)) for k, val, emb in self._memory if emb is not None]
        idx = np.argsort(scores)[::-1][:top_k]
        return [scores[i] for i in idx], [self._memory[i][1] for i in idx]

    def _get_embedder(self) -> Optional[Any]:
        """Lazy load sentence-transformer for semantic RAG."""
        if not self.use_semantic_rag:
            return None
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformer(self._embedding_model_name)
            except Exception:
                self.use_semantic_rag = False
        return self._embedder

    def retrieve_by_query(
        self,
        question: str,
        top_k: int = 5,
        partition_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """
        RAG: Retrieve laws by keyword or semantic relevance to question.
        支持 partition_id / domain 过滤（分区检索增强）。

        Args:
            question: 查询
            top_k: 返回条数
            partition_id: 仅检索该分区的定律
            domain: 仅检索该 domain 的定律
        """
        if not self.knowledge_base:
            return ""
        # 分区/domain 过滤
        candidates = {}
        for fid, meta in self.knowledge_base.items():
            if not isinstance(meta, dict):
                continue
            if partition_id is not None and meta.get("partition_id") != partition_id:
                continue
            if domain is not None and meta.get("domain") != domain:
                continue
            candidates[fid] = meta
        if not candidates:
            return ""
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                q_emb = embedder.encode(question)
                scored = []
                for fid, meta in candidates.items():
                    formula = meta.get("formula", str(meta))
                    f_emb = embedder.encode(formula)
                    sim = float(np.dot(q_emb, f_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(f_emb) + 1e-9))
                    scored.append((sim, fid, formula))
                scored.sort(key=lambda x: -x[0])
                items = [f"- {fid}: {formula}" for _, fid, formula in scored[:top_k]]
                return "Known laws (by semantic relevance):\n" + "\n".join(items) if items else ""
            except Exception:
                pass  # fallback to keyword
        q_lower = question.lower()
        q_words = set(w for w in re.split(r"\W+", q_lower) if len(w) > 1)
        scored = []
        for fid, meta in candidates.items():
            formula = meta.get("formula", str(meta))
            f_lower = formula.lower()
            f_words = set(w for w in re.split(r"\W+", f_lower) if len(w) > 1)
            overlap = len(q_words & f_words) if q_words else 0
            scored.append((overlap, fid, formula))
        scored.sort(key=lambda x: (-x[0], x[1]))
        items = [f"- {fid}: {formula}" for _, fid, formula in scored[:top_k]]
        return "Known laws (by relevance):\n" + "\n".join(items) if items else ""

    def retrieve_by_partition(self, partition_id: str, top_k: int = 10) -> str:
        """
        分区检索：返回该分区的所有定律，用于「何时用哪块知识」的直觉。
        """
        laws = self.list_by_partition(partition_id)
        if not laws:
            return ""
        items = [f"- {l['id']}: {l.get('formula', '?')}" for l in laws[:top_k]]
        return f"Laws in partition [{partition_id}]:\n" + "\n".join(items)

    def crystallize(
        self,
        formula: Union[str, Callable],
        target_rcln: Any,
        formula_id: Optional[str] = None,
        partition_id: Optional[str] = None,
        **metadata,
    ) -> str:
        """
        Crystallize a discovered formula into the target RCLN.
        1. Update hard_core with the new formula (as callable).
        2. Reset soft_shell weights (forget neural part, keep symbolic).

        Args:
            formula: Symbolic formula string (e.g., "x[0]**2 + x[1]**2") or callable.
            target_rcln: RCLNLayer instance to update.
            formula_id: Optional id for knowledge_base. Auto-generated if None.
            partition_id: Optional partition id (智能分区).
            **metadata: 额外元数据 (domain, applicability, ...).

        Returns:
            formula_id used for storage.
        """
        if formula_id is None:
            formula_id = f"law_{len(self.knowledge_base)}"

        # Convert formula to callable if string
        if callable(formula):
            hard_core_func = formula
            formula_str = str(formula)
        else:
            formula_str = str(formula)
            output_dim = getattr(target_rcln, "output_dim", 1)
            hard_core_func = _formula_to_callable(formula_str, output_dim)

        # 1. Update hard_core
        target_rcln.hard_core = hard_core_func

        # 2. Reset soft_shell weights (forget neural part, keep symbolic)
        if hasattr(target_rcln, "soft_shell"):
            _reset_module_weights(target_rcln.soft_shell)

        # Store in knowledge_base (含 partition 元数据)
        entry = {
            "formula": formula_str,
            "callable": hard_core_func,
            "target": formula_id,
        }
        if partition_id is not None:
            entry["partition_id"] = partition_id
        entry.update(metadata)

        self.knowledge_base[formula_id] = entry

        # 元轴丛场：结晶时同步到 bundle_field
        if self._bundle_field is not None:
            domain = metadata.get("domain", "mechanics")
            pid = partition_id or "default"
            self._bundle_field.crystallize(
                formula=formula_str,
                callable_fn=hard_core_func,
                output_dim=getattr(target_rcln, "output_dim", 1),
                domain=domain,
                partition_id=pid,
                regime="weak",
                formula_id=formula_id,
                residual_role="principal",
            )

        return formula_id

    def list_by_partition(self, partition_id: str) -> List[Dict[str, Any]]:
        """按 partition_id 检索结晶定律。"""
        return [
            {"id": fid, **{k: v for k, v in meta.items() if k != "callable"}}
            for fid, meta in self.knowledge_base.items()
            if isinstance(meta, dict) and meta.get("partition_id") == partition_id
        ]

    # -------------------------------------------------------------------------
    # 扰动项：分区学习成分作为联想/直觉，决策参考（不替代主预测）
    # -------------------------------------------------------------------------

    def register_perturbation(
        self,
        formula: Union[str, Callable],
        partition_id: str,
        domain: str = "mechanics",
        formula_id: Optional[str] = None,
        output_dim: int = 1,
        **metadata,
    ) -> str:
        """
        注册扰动项：分区学习得到的公式作为联想/直觉，存入海马体。
        不结晶到 hard_core，仅作为决策时的参考成分。

        Args:
            formula: 公式字符串或 callable
            partition_id: 分区 id (rar_low, turb_z_low, ...)
            domain: mechanics | fluids | battery
            formula_id: 可选 id
            output_dim: 输出维度
            **metadata: g0, r2, n_samples 等

        Returns:
            formula_id
        """
        if formula_id is None:
            formula_id = f"pert_{partition_id}_{len([k for k in self.knowledge_base if k.startswith('pert_')])}"
        if callable(formula):
            callable_fn = formula
            formula_str = str(formula)
        else:
            formula_str = str(formula)
            callable_fn = _formula_to_callable(formula_str, output_dim)
        entry = {
            "formula": formula_str,
            "callable": callable_fn,
            "partition_id": partition_id,
            "domain": domain,
            "output_dim": output_dim,
            "type": "perturbation",
        }
        entry.update(metadata)
        self.knowledge_base[formula_id] = entry

        # 元轴丛场：扰动项同步到 bundle_field
        if self._bundle_field is not None:
            self._bundle_field.crystallize(
                formula=formula_str,
                callable_fn=callable_fn,
                output_dim=output_dim,
                domain=domain,
                partition_id=partition_id,
                regime="weak",
                formula_id=formula_id,
                residual_role="perturbation",
            )

        return formula_id

    def eval_perturbation(
        self,
        x: Any,
        partition_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        计算扰动项：检索匹配分区的定律，求值并平均（联想直觉的聚合）。
        若启用 bundle_field，优先使用元轴丛场截面选择。

        Args:
            x: 输入 (n,d) 或 tensor
            partition_id: 指定分区，None 则需由调用方根据 domain 推断
            domain: 过滤 domain

        Returns:
            (n, output_dim) 扰动值，无匹配则 None
        """
        if self._bundle_field is not None:
            try:
                out = self._bundle_field.eval_perturbation(
                    x, partition_id=partition_id, domain=domain
                )
                if out is not None:
                    return out
            except Exception:
                pass

        candidates = []
        for fid, meta in self.knowledge_base.items():
            if not isinstance(meta, dict) or meta.get("type") != "perturbation":
                continue
            if partition_id is not None and meta.get("partition_id") != partition_id:
                continue
            if domain is not None and meta.get("domain") != domain:
                continue
            fn = meta.get("callable")
            if callable(fn):
                candidates.append((fn, meta.get("output_dim", 1)))
        if not candidates:
            return None
        # 聚合：多定律时取平均（模拟多源直觉融合）
        results = []
        for fn, out_dim in candidates:
            try:
                out = fn(x)
                arr = np.asarray(out, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                results.append(arr)
            except Exception:
                continue
        if not results:
            return None
        return np.mean(results, axis=0)
