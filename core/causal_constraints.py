"""
因果约束统一入口：辛结构 + 光锥
供 Discovery 与因果发现模块调用，保证候选公式/边符合物理因果。
"""

from typing import List, Tuple, Optional, Set, Any
import numpy as np

from .symplectic_causal import (
    get_symplectic_causal_edges,
    get_symplectic_allowed_inputs,
    filter_formula_by_symplectic,
    build_symplectic_causal_mask,
)
from .light_cone_filter import (
    check_light_cone,
    filter_causal_edges_by_light_cone,
)


def allowed_edges(
    state_dim: int,
    spacetime_list: Optional[List[Tuple[float, Any]]] = None,
    use_symplectic: bool = True,
    use_light_cone: bool = True,
    c: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    统一入口：返回允许的因果边 (source, target)，即 source 可因果影响 target。
    - use_symplectic: 仅当 state_dim 为偶数时生效，约束 Hamiltonian dq/dt, dp/dt。
    - use_light_cone: 当 spacetime_list 非空时，过滤掉违反光锥的边。
    """
    if use_symplectic and state_dim % 2 == 0:
        edges = get_symplectic_causal_edges(state_dim)
    else:
        # 无辛约束时：全连接（所有 j -> i）
        edges = [(j, i) for i in range(state_dim) for j in range(state_dim)]

    if use_light_cone and spacetime_list is not None and len(spacetime_list) >= state_dim:
        st = []
        for i in range(state_dim):
            t, r = spacetime_list[i]
            r = np.asarray(r).ravel()[:3] if hasattr(r, "__iter__") and not isinstance(r, str) else np.zeros(3)
            st.append((float(t), r))
        edges = filter_causal_edges_by_light_cone(st, edges, c=c)
    return edges


def allowed_inputs_for_output(
    output_index: int,
    state_dim: int,
    spacetime_list: Optional[List[Tuple[float, Any]]] = None,
    use_symplectic: bool = True,
    use_light_cone: bool = True,
) -> Set[int]:
    """
    给定输出变量索引（如 dx_i/dt 对应 output_index=i），返回允许作为输入的变量索引集合。
    Discovery 为第 output_index 个方程拟合时，应只使用这些列。
    """
    edges = allowed_edges(
        state_dim,
        spacetime_list=spacetime_list,
        use_symplectic=use_symplectic,
        use_light_cone=use_light_cone,
    )
    return {j for (j, i) in edges if i == output_index}


__all__ = [
    "allowed_edges",
    "allowed_inputs_for_output",
    "get_symplectic_causal_edges",
    "get_symplectic_allowed_inputs",
    "filter_formula_by_symplectic",
    "build_symplectic_causal_mask",
    "check_light_cone",
    "filter_causal_edges_by_light_cone",
]
