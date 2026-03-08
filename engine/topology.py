"""
Topological Early Warning (TDA) - Axiom-OS v4.0
Persistent Homology along the Meta-Axis for phase transition detection.
Betti number jumps at z > 0 can precede macroscopic disasters.
"""

from typing import Optional, Tuple, List, Dict
import numpy as np

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

try:
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _point_cloud_from_activation(activation: np.ndarray) -> np.ndarray:
    """
    Convert latent activation map to point cloud for TDA.
    activation: (H, W) or (B, H, W) or (D,) - flatten to (N, d) points.
    """
    a = np.asarray(activation, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2:
        h, w = a.shape
        y, x = np.mgrid[:h, :w]
        return np.column_stack([x.ravel(), y.ravel(), a.ravel()])
    if a.ndim == 3:
        b, h, w = a.shape
        pts = []
        for i in range(b):
            y, x = np.mgrid[:h, :w]
            pts.append(np.column_stack([x.ravel(), y.ravel(), a[i].ravel()]))
        return np.vstack(pts)
    return a.reshape(-1, a.shape[-1])


def compute_persistence(
    latent_state: np.ndarray,
    max_dim: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute Persistent Homology (Persistence Diagrams).
    Input: latent activation map at depth z (from RCLN/Holographic).
    Output: Betti-0 (connected components), Betti-1 (loops/vortices).

    Returns:
        dict with keys: "betti0" (birth, death) pairs, "betti1", "n_components", "n_loops"
    """
    result: Dict[str, np.ndarray] = {
        "betti0": np.array([]).reshape(0, 2),
        "betti1": np.array([]).reshape(0, 2),
        "n_components": 0,
        "n_loops": 0,
    }
    if not HAS_GUDHI or not HAS_SCIPY:
        return result

    pts = _point_cloud_from_activation(latent_state)
    if pts.shape[0] < 4:
        result["n_components"] = 1
        return result

    # Subsample if too large (GUDHI can be slow)
    max_pts = 500
    if pts.shape[0] > max_pts:
        idx = np.random.choice(pts.shape[0], max_pts, replace=False)
        pts = pts[idx]

    try:
        # Normalize pts for stable Rips
        pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) + 1e-8)
        max_edge = float(np.percentile(pdist(pts), 90)) + 0.1
        rips = gudhi.RipsComplex(points=pts.tolist(), max_edge_length=max_edge)
        st = rips.create_simplex_tree(max_dimension=max_dim)
        st.compute_persistence()

        for dim in range(max_dim + 1):
            pers = st.persistence_intervals_in_dimension(dim)
            if len(pers) > 0:
                pers_arr = np.array([[float(a), float(b)] for a, b in pers])
                if dim == 0:
                    result["betti0"] = pers_arr
                    result["n_components"] = len(pers)
                elif dim == 1:
                    result["betti1"] = pers_arr
                    result["n_loops"] = len(pers)
    except Exception:
        pass
    return result


def compute_betti_summary(persistence: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """Extract (n_components, n_loops) from persistence result."""
    return (
        int(persistence.get("n_components", 0)),
        int(persistence.get("n_loops", 0)),
    )


class TopologicalEarlyWarning:
    """
    Monitor ∂(Betti)/∂z along Meta-Axis. Alert on phase transition.
    Unstable topology at z > 0 precedes macroscopic disasters.
    """

    def __init__(
        self,
        threshold_derivative: float = 2.0,
        window_size: int = 3,
    ):
        self.threshold = threshold_derivative
        self.window = window_size
        self._betti_history: List[Tuple[int, int]] = []
        self._z_history: List[float] = []

    def update(self, z: float, latent_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Record Betti numbers at depth z."""
        pers = compute_persistence(latent_state)
        n_comp, n_loops = compute_betti_summary(pers)
        self._betti_history.append((n_comp, n_loops))
        self._z_history.append(z)
        if len(self._betti_history) > self.window * 2:
            self._betti_history = self._betti_history[-self.window * 2 :]
            self._z_history = self._z_history[-self.window * 2 :]
        return pers

    def check_alert(self) -> Tuple[bool, str]:
        """
        Check if ∂(Betti)/∂z exceeds threshold (phase transition warning).
        Returns (alert, message). When alert=True, MPC should increase caution
        (e.g. reduce action magnitude, tighten constraints).
        """
        if len(self._betti_history) < self.window or len(self._z_history) < self.window:
            return False, ""

        z = np.array(self._z_history[-self.window :])
        comp = np.array([b[0] for b in self._betti_history[-self.window :]])
        loops = np.array([b[1] for b in self._betti_history[-self.window :]])

        dz = np.diff(z)
        if np.any(np.abs(dz) < 1e-10):
            return False, ""
        d_comp = np.diff(comp) / (dz + 1e-10)
        d_loops = np.diff(loops) / (dz + 1e-10)

        if np.any(np.abs(d_comp) > self.threshold):
            return True, f"Phase transition warning: Betti-0 derivative = {float(np.max(np.abs(d_comp))):.2f}"
        if np.any(np.abs(d_loops) > self.threshold):
            return True, f"Phase transition warning: Betti-1 derivative = {float(np.max(np.abs(d_loops))):.2f}"
        return False, ""

    def reset(self) -> None:
        self._betti_history.clear()
        self._z_history.clear()
