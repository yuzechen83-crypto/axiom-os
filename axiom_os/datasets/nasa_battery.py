"""
NASA Li-ion Battery Aging Dataset - Data Pipeline
Battery B0005: Discharge capacity over cycles.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# URLs to try for B0005.mat (multiple mirrors)
NASA_B0005_URLS = [
    "https://github.com/anindodas/SOH-and-RUL-Prediction-in-Lithium-Ion-Batteries/raw/main/B0005.mat",
    "https://github.com/anindodas/Remaining-Useful-Life-Prediction-NASA-dataset-/raw/main/B0005.mat",
    "https://raw.githubusercontent.com/ignavinuales/Battery_RUL_Prediction/master/data/mat/B0005.mat",
    "https://github.com/bnarms/NASA-Battery-Dataset/raw/main/battery_data/B0005.mat",
]

# Fallback: generate synthetic degradation (Q = Q0 * exp(-k*t)) when download fails
def _generate_synthetic_b0005() -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic capacity fade mimicking NASA B0005 (~168 cycles, 2.0->1.4 Ah)."""
    np.random.seed(42)
    n_cycles = 168
    cycles = np.arange(1, n_cycles + 1, dtype=np.float64).reshape(-1, 1)
    # Q(t) = 2.0 * exp(-k*t), k s.t. Q(168)≈1.4 => k = -ln(1.4/2.0)/168 ≈ 0.0021
    t = cycles.ravel()
    k = -np.log(1.4 / 2.0) / 168
    Q = 2.0 * np.exp(-k * t) + np.random.randn(n_cycles) * 0.02
    Q = np.clip(Q, 1.2, 2.1).reshape(-1, 1)
    return cycles, Q


def _get_field(obj, name):
    """Get field from dict or mat_struct (struct_as_record=True)."""
    try:
        if hasattr(obj, "_fieldnames") and name in obj._fieldnames:
            return getattr(obj, name)
        return obj[name]
    except (TypeError, KeyError):
        return getattr(obj, name, None)


def _parse_nasa_mat(data: dict, battery: str = "B0005") -> Tuple[np.ndarray, np.ndarray]:
    """Parse NASA .mat structure: cycle type, data.Capacity. Handles both dict and mat_struct."""
    if battery not in data:
        raise ValueError(f"Battery {battery} not in data")
    root = data[battery]
    # Handle (0,0) indexing for some .mat layouts
    if hasattr(root, "shape") and root.size > 0:
        root = root.flat[0] if root.size == 1 else root[0, 0]
    cycle_arr = _get_field(root, "cycle")
    if cycle_arr is None:
        raise ValueError("No 'cycle' field in .mat file")
    cycle_arr = np.atleast_1d(cycle_arr)
    if cycle_arr.ndim > 1:
        cycle_arr = cycle_arr.flat
    cycles_list = []
    capacities_list = []
    for i in range(len(cycle_arr)):
        row = cycle_arr[i]
        op_type = ""
        try:
            t = _get_field(row, "type")
            if t is not None:
                op_type = str(np.squeeze(t)).lower()
        except Exception:
            pass
        if "discharge" not in op_type:
            continue
        data_block = _get_field(row, "data")
        if data_block is None:
            continue
        if hasattr(data_block, "shape") and data_block.size > 0:
            data_block = data_block.flat[0] if data_block.size == 1 else data_block[0, 0]
        cap_val = _get_field(data_block, "Capacity")
        if cap_val is None:
            continue
        try:
            cap = float(np.squeeze(cap_val))
        except (TypeError, ValueError):
            continue
        cycles_list.append(len(cycles_list) + 1)
        capacities_list.append(cap)
    if not cycles_list:
        raise ValueError("No discharge cycles found in .mat file")
    cycles = np.array(cycles_list, dtype=np.float64).reshape(-1, 1)
    capacities = np.array(capacities_list, dtype=np.float64).reshape(-1, 1)
    return cycles, capacities


def load_battery_data(
    battery: str = "B0005",
    local_path: Optional[Path] = None,
    use_synthetic_if_fail: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load NASA battery discharge capacity data.

    Returns:
        cycles: (n, 1) cycle indices
        capacity: (n, 1) discharge capacity (Ah)
        scalers: dict with cycle_min, cycle_max, cap_min, cap_max for inverse transform
    """
    cycles_raw = None
    capacity_raw = None
    data_source = "synthetic"

    # 1. Try local file
    if local_path is not None and local_path.exists():
        for struct_record in (False, True):
            try:
                data = loadmat(str(local_path), struct_as_record=struct_record)
                cycles_raw, capacity_raw = _parse_nasa_mat(data, battery)
                data_source = "real"
                break
            except Exception as e:
                if struct_record:
                    print(f"Local load failed: {e}")
                continue

    # 2. Try download
    data_source = "synthetic"
    if cycles_raw is None and HAS_REQUESTS:
        for url in NASA_B0005_URLS:
            try:
                print(f"Downloading from {url}...")
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    from io import BytesIO
                    content = BytesIO(resp.content)
                    loaded = False
                    for struct_record in (False, True):
                        try:
                            content.seek(0)
                            data = loadmat(content, struct_as_record=struct_record)
                            cycles_raw, capacity_raw = _parse_nasa_mat(data, battery)
                            data_source = "real"
                            print(f"Loaded {len(cycles_raw)} discharge cycles (real NASA data).")
                            loaded = True
                            break
                        except Exception:
                            continue
                    if loaded:
                        break
            except Exception as e:
                print(f"Download failed: {e}")
                continue

    # 3. Try datasets/ subdir
    if cycles_raw is None:
        for candidate in [
            Path(__file__).parent / "B0005.mat",
            Path(__file__).parent.parent / "data" / "B0005.mat",
        ]:
            if not candidate.exists():
                continue
            for struct_record in (False, True):
                try:
                    data = loadmat(str(candidate), struct_as_record=struct_record)
                    cycles_raw, capacity_raw = _parse_nasa_mat(data, battery)
                    data_source = "real"
                    break
                except Exception:
                    continue
            else:
                continue
            break

    # 4. Synthetic fallback
    if cycles_raw is None and use_synthetic_if_fail:
        print("Using synthetic B0005-like data (download unavailable).")
        cycles_raw, capacity_raw = _generate_synthetic_b0005()

    if cycles_raw is None:
        raise RuntimeError("Could not load NASA battery data. Place B0005.mat in axiom_os/datasets/ or axiom_os/data/")

    # Normalize to [0, 1] for stable training
    cycle_min, cycle_max = float(cycles_raw.min()), float(cycles_raw.max())
    cap_min, cap_max = float(capacity_raw.min()), float(capacity_raw.max())
    cycles_norm = (cycles_raw - cycle_min) / (cycle_max - cycle_min + 1e-12)
    capacity_norm = (capacity_raw - cap_min) / (cap_max - cap_min + 1e-12)

    scalers = {
        "cycle_min": cycle_min,
        "cycle_max": cycle_max,
        "cap_min": cap_min,
        "cap_max": cap_max,
        "data_source": data_source,
    }

    return cycles_norm, capacity_norm, scalers
