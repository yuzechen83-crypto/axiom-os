"""
SPARC - Spitzer Photometry and Accurate Rotation Curves
Galaxy rotation curve data for Meta-Axis theory validation.
Target: V_def² = V_obs² - (V_gas² + V_disk² + V_bulge²) (missing velocity squared)
Real data: http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import zipfile
import io
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

SPARC_ZIP_URL = "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
SPARC_ZENODO_URL = "https://zenodo.org/records/16284118/files/Rotmod_LTG.zip?download=1"
SPARC_RAR_URL = "https://astroweb.cwru.edu/SPARC/RAR.mrt"
# KM_S_SQ_PER_KPC_TO_MS2: (km/s)^2/kpc -> m/s^2
KM_S_SQ_PER_KPC_TO_MS2 = 1e6 / 3.08567758128e19
# Local path override: set SPARC_ZIP_PATH env, or use axiom_os/data/Rotmod_LTG.zip if exists
_SPARC_CACHE: Optional[bytes] = None
_DEFAULT_LOCAL = Path(__file__).resolve().parent.parent / "data" / "Rotmod_LTG.zip"

# SPARC galaxy names (match Rotmod filenames: NGC6503, NGC3198, etc.)
SPARC_GALAXIES = [
    "NGC_6503", "NGC_3198", "NGC_2403", "NGC_2841", "NGC_925",
    "NGC_2976", "NGC_7331", "NGC_5055", "NGC_6946", "NGC_628",
    "NGC_3621", "NGC_4736", "NGC_3521", "NGC_2903", "NGC_3031",
    "NGC_4321", "NGC_4535", "NGC_4536", "NGC_4548", "NGC_4579",
]


def _mock_rotation_curve(
    n_points: int = 40,
    r_max: float = 25.0,
    v_flat: float = 150.0,
    r_disk: float = 3.0,
    r_bulge: float = 0.5,
    v_bulge_peak: float = 80.0,
    gas_fraction: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mock rotation curve with realistic profiles.
    Returns R, V_obs, V_gas, V_disk, V_bulge (all in kpc, km/s).
    """
    rng = np.random.default_rng(seed)
    R = np.linspace(0.2, r_max, n_points)
    R = np.sort(R)

    # Exponential disk: simplified rotation curve (Freeman-like)
    x = R / (r_disk + 1e-6)
    V_disk = v_flat * 0.6 * np.sqrt(1 - np.exp(-x) * (1 + x))
    V_disk = np.nan_to_num(V_disk, nan=0.0, posinf=0.0, neginf=0.0)
    V_disk = V_disk / (np.max(V_disk) + 1e-8) * v_flat * 0.6

    # Bulge: Plummer-like
    V_bulge = v_bulge_peak * R / np.sqrt(R**2 + r_bulge**2)

    # Gas: extended, lower amplitude
    V_gas = v_flat * gas_fraction * (1 - np.exp(-R / (r_disk * 2)))
    V_gas = np.clip(V_gas, 0, v_flat * 0.4)

    # Baryonic total (velocity squared adds)
    V_bary_sq = V_gas**2 + V_disk**2 + V_bulge**2

    # "Dark" / missing: NFW-like rise then flat (simplified)
    r_s = r_disk * 2
    x_nfw = R / r_s
    f_x = np.log(1 + x_nfw) - x_nfw / (1 + x_nfw)
    V_def_sq = (v_flat**2 - np.max(V_bary_sq)) * f_x / (np.max(f_x) + 1e-8)
    V_def_sq = np.maximum(V_def_sq, 0)

    # Observed: V_obs² = V_bary² + V_def²
    V_obs_sq = V_bary_sq + V_def_sq
    V_obs = np.sqrt(np.maximum(V_obs_sq, 1.0))
    V_obs = V_obs + rng.normal(0, 3.0, size=V_obs.shape)

    return R, V_obs, V_gas, V_disk, V_bulge


def _mock_rotation_curve_meta(
    n_points: int = 40,
    r_max: float = 25.0,
    v_flat: float = 150.0,
    r_disk: float = 3.0,
    r_bulge: float = 0.5,
    v_bulge_peak: float = 80.0,
    gas_fraction: float = 0.15,
    meta_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mock rotation curve with Meta-Axis ground truth.
    V_def² = meta_scale * ρ_proxy, ρ_proxy = V_bary² / (r² + ε).
    Returns R, V_obs, V_gas, V_disk, V_bulge (all in kpc, km/s).
    """
    rng = np.random.default_rng(seed)
    R = np.linspace(0.2, r_max, n_points)
    R = np.sort(R)

    # Baryonic (same as NFW mock)
    x = R / (r_disk + 1e-6)
    V_disk = v_flat * 0.6 * np.sqrt(1 - np.exp(-x) * (1 + x))
    V_disk = np.nan_to_num(V_disk, nan=0.0, posinf=0.0, neginf=0.0)
    V_disk = V_disk / (np.max(V_disk) + 1e-8) * v_flat * 0.6
    V_bulge = v_bulge_peak * R / np.sqrt(R**2 + r_bulge**2)
    V_gas = v_flat * gas_fraction * (1 - np.exp(-R / (r_disk * 2)))
    V_gas = np.clip(V_gas, 0, v_flat * 0.4)

    V_bary_sq = V_gas**2 + V_disk**2 + V_bulge**2

    # Meta-Axis: V_def² ∝ ρ_proxy = V_bary² / r²
    eps = 1e-6
    r_safe = np.maximum(R, 0.1)
    rho_proxy = V_bary_sq / (r_safe**2 + eps)
    rho_norm = np.max(rho_proxy) + 1e-8
    V_def_sq = meta_scale * (v_flat**2 - np.max(V_bary_sq)) * rho_proxy / rho_norm
    V_def_sq = np.maximum(V_def_sq, 0)

    V_obs_sq = V_bary_sq + V_def_sq
    V_obs = np.sqrt(np.maximum(V_obs_sq, 1.0))
    V_obs = V_obs + rng.normal(0, 3.0, size=V_obs.shape)

    return R, V_obs, V_gas, V_disk, V_bulge


def _load_mock_galaxy(name: str, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load one mock galaxy. Seed from hash of name for reproducibility."""
    h = hash(name) % (2**31)
    r_max = 15.0 + (h % 20)
    v_flat = 100.0 + (h % 150)
    r_disk = 2.0 + (h % 5) * 0.5
    R, V_obs, V_gas, V_disk, V_bulge = _mock_rotation_curve(
        n_points=45, r_max=r_max, v_flat=float(v_flat),
        r_disk=r_disk, seed=seed or h,
    )
    return {
        "R": R,
        "V_obs": np.maximum(V_obs, 5.0),
        "V_gas": V_gas,
        "V_disk": V_disk,
        "V_bulge": V_bulge,
        "name": name,
    }


def _load_mock_galaxy_meta(name: str, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load one mock galaxy with Meta-Axis ground truth."""
    h = hash(name) % (2**31)
    r_max = 15.0 + (h % 20)
    v_flat = 100.0 + (h % 150)
    r_disk = 2.0 + (h % 5) * 0.5
    meta_scale = 0.8 + (h % 10) * 0.1
    R, V_obs, V_gas, V_disk, V_bulge = _mock_rotation_curve_meta(
        n_points=45, r_max=r_max, v_flat=float(v_flat),
        r_disk=r_disk, meta_scale=meta_scale, seed=seed or h,
    )
    return {
        "R": R,
        "V_obs": np.maximum(V_obs, 5.0),
        "V_gas": V_gas,
        "V_disk": V_disk,
        "V_bulge": V_bulge,
        "name": name,
    }


def load_mock_batch(
    n_galaxies: int,
    data_source: str = "nfw",
    seed: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Load batch of mock galaxies for fair comparison.
    data_source: "nfw" (NFW ground truth) or "meta" (Meta-Axis ground truth).
    """
    names = SPARC_GALAXIES[:n_galaxies] if n_galaxies <= len(SPARC_GALAXIES) else [
        f"NGC_{7000 + i}" for i in range(n_galaxies)
    ]
    loader = _load_mock_galaxy_meta if data_source == "meta" else _load_mock_galaxy
    return [loader(n, seed) for n in names]


def _name_to_filename(name: str) -> str:
    """NGC_6503 -> NGC6503, NGC3198 -> NGC3198."""
    return name.replace("_", "").upper()


def _fetch_sparc_zip(timeout: int = 25) -> Optional[bytes]:
    """Download Rotmod_LTG.zip from SPARC or Zenodo. Caches result."""
    global _SPARC_CACHE
    if _SPARC_CACHE is not None and len(_SPARC_CACHE) > 100:
        return _SPARC_CACHE
    import os
    local = os.environ.get("SPARC_ZIP_PATH") or str(_DEFAULT_LOCAL)
    if local and Path(local).exists():
        try:
            data = Path(local).read_bytes()
            if len(data) > 1000:
                _SPARC_CACHE = data
                return data
        except Exception:
            pass
    urls = [SPARC_ZIP_URL, SPARC_ZENODO_URL]
    try:
        import requests
        for url in urls:
            try:
                r = requests.get(url, timeout=timeout, headers={"User-Agent": "AxiomOS/1.0"})
                if r.status_code == 200 and len(r.content) > 1000:
                    _SPARC_CACHE = r.content
                    return r.content
            except Exception:
                continue
    except ImportError:
        pass
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "AxiomOS/1.0 SPARC-Loader"})
            with urlopen(req, timeout=timeout) as r:
                data = r.read()
                if len(data) > 1000:
                    _SPARC_CACHE = data
                    return data
        except (URLError, HTTPError, OSError, TimeoutError):
            continue
    return None


def set_sparc_zip_path(path: str) -> None:
    """Set local path to Rotmod_LTG.zip for offline use."""
    global _SPARC_CACHE
    _SPARC_CACHE = None
    import os
    os.environ["SPARC_ZIP_PATH"] = str(path)


def _parse_rotmod_file(content: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse Rotmod MRT/ASCII file. Columns: R Vobs [eVobs] Vgas Vdisk [Vbulge].
    Skip header, handle comments. Tolerant of 4-6 columns.
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    data = []
    for line in lines:
        if line.startswith("#") or line.startswith("!") or line.upper().startswith("R("):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                row = [float(p) for p in parts[:8]]
                if row[0] >= 0 and row[1] >= 0:
                    data.append(row[:8])
            except (ValueError, IndexError):
                continue
    if len(data) < 5:
        return None
    arr = np.array(data)
    ncol = arr.shape[1]
    R = arr[:, 0]
    V_obs = np.abs(arr[:, 1])
    # Column layout: 0=R, 1=Vobs, 2=eVobs(opt), 3=Vgas, 4=Vdisk, 5=Vbulge(opt)
    if ncol >= 5:
        V_gas = np.abs(arr[:, 3])
        V_disk = np.abs(arr[:, 4])
        V_bulge = np.abs(arr[:, 5]) if ncol > 5 else np.zeros_like(R)
    elif ncol >= 4:
        V_gas = np.abs(arr[:, 2])
        V_disk = np.abs(arr[:, 3])
        V_bulge = np.zeros_like(R)
    else:
        return None
    return {"R": R, "V_obs": V_obs, "V_gas": V_gas, "V_disk": V_disk, "V_bulge": V_bulge}


def _load_real_sparc(galaxy_name: str, cache_dir: Optional[Path] = None) -> Optional[Dict[str, np.ndarray]]:
    """Load from real SPARC Rotmod_LTG.zip."""
    fname = _name_to_filename(galaxy_name)
    fname_alt = galaxy_name.replace("_", "").replace("-", "").upper()
    zip_data = _fetch_sparc_zip()
    if zip_data is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf:
            names = zf.namelist()
            # Match: NGC6503, NGC6503.rotmod, Rotmod/NGC6503.dat, etc.
            def matches(n: str) -> bool:
                base = Path(n).stem.upper().replace("-", "").replace("_", "").replace(" ", "")
                return fname in base or fname_alt in base or base.startswith(fname)
            candidates = [n for n in names if matches(n) and not n.endswith("/")]
            # Prefer .rotmod, .dat, .txt, .mrt
            for ext in [".rotmod", ".dat", ".txt", ".mrt"]:
                for n in names:
                    if fname in Path(n).stem.upper().replace("_", "") and n.endswith(ext):
                        candidates.insert(0, n)
                        break
            for c in candidates:
                try:
                    raw = zf.read(c).decode("utf-8", errors="ignore")
                    d = _parse_rotmod_file(raw)
                    if d is not None and len(d["R"]) >= 5:
                        d["name"] = galaxy_name
                        return d
                except Exception:
                    continue
    except (zipfile.BadZipFile, OSError):
        pass
    return None


def get_available_sparc_galaxies() -> List[str]:
    """Return list of galaxy names available in real SPARC zip (if downloadable)."""
    import re
    zip_data = _fetch_sparc_zip()
    if zip_data is None:
        return []
    seen = set()
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf:
            for n in zf.namelist():
                if n.endswith("/"):
                    continue
                stem = Path(n).stem.upper().replace("-", "").replace("_", "").replace(" ", "")
                m = re.search(r"NGC\s*(\d+)", stem, re.I)
                if m:
                    key = f"NGC_{m.group(1)}"
                    if key not in seen:
                        seen.add(key)
                        out.append(key)
    except Exception:
        pass
    return sorted(out, key=lambda x: int(x.split("_")[1]) if "_" in x else 0)


def load_sparc_galaxy(
    galaxy_name: str,
    use_mock_if_fail: bool = True,
    seed: Optional[int] = None,
    use_real: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load one galaxy from SPARC (real or mock).
    Returns dict with R, V_obs, V_gas, V_disk, V_bulge, name.
    All velocities in km/s, R in kpc.
    use_real=True: try real SPARC download first.
    """
    if use_real:
        d = _load_real_sparc(galaxy_name)
        if d is not None:
            return d
    if use_mock_if_fail:
        return _load_mock_galaxy(galaxy_name, seed)
    raise RuntimeError(f"Could not load SPARC data for {galaxy_name}")


def load_sparc_multi(
    n_galaxies: int = 20,
    galaxy_list: Optional[List[str]] = None,
    use_mock_if_fail: bool = True,
    use_real: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load multiple galaxies. Returns stacked arrays with galaxy index.
    Returns: R_all, V_def_sq_all, V_bary_sq_all, names
    V_def² = V_obs² - (V_gas² + V_disk² + V_bulge²)
    """
    if galaxy_list is None:
        if use_real:
            avail = get_available_sparc_galaxies()
            names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]
        else:
            names = SPARC_GALAXIES[:n_galaxies]
    else:
        names = galaxy_list
    all_R = []
    all_V_def_sq = []
    all_V_bary_sq = []
    all_names = []
    for name in names:
        d = load_sparc_galaxy(name, use_mock_if_fail=use_mock_if_fail, use_real=use_real)
        V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
        V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1.0)
        V_def_sq = V_obs_sq - V_bary_sq
        V_def_sq = np.maximum(V_def_sq, 0.0)
        all_R.append(d["R"])
        all_V_def_sq.append(V_def_sq)
        all_V_bary_sq.append(V_bary_sq)
        all_names.append(d["name"])
    return (
        np.concatenate(all_R),
        np.concatenate(all_V_def_sq),
        np.concatenate(all_V_bary_sq),
        all_names,
    )


def compute_accelerations(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute g_bar and g_obs from galaxy dict.
    g_bar = V_bary^2 / r  (Newtonian from visible matter)
    g_obs = V_obs^2 / r   (observed)
    Units: (km/s)^2 / kpc. Filter invalid points.
    Returns: g_bar, g_obs, mask (valid indices)
    """
    R = np.asarray(d["R"], dtype=np.float64)
    V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
    V_obs_sq = np.maximum(d["V_obs"]**2, 1.0)
    r_safe = np.maximum(R, 0.1)
    g_bar = V_bary_sq / r_safe
    g_obs = V_obs_sq / r_safe
    mask = (
        (R > 0.1) & (d["V_obs"] > 0) &
        np.isfinite(g_bar) & np.isfinite(g_obs) &
        (g_bar > 1e-12) & (g_obs > 1e-12)
    )
    return g_bar[mask], g_obs[mask], mask


_RAR_MRT_LOCAL = Path(__file__).resolve().parent.parent / "data" / "RAR.mrt"


def load_sparc_rar_mrt() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load canonical RAR data from SPARC RAR.mrt (McGaugh+2016, Lelli+2017).
    Tries axiom_os/data/RAR.mrt first, then fetches from URL and caches.
    Format: log10(g_bar), e_gbar, log10(g_obs), e_gobs in m/s^2.
    Returns g_bar, g_obs in (km/s)^2/kpc for consistency with Rotmod pipeline.
    """
    raw = None
    if _RAR_MRT_LOCAL.exists():
        try:
            raw = _RAR_MRT_LOCAL.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if raw is None or len(raw) < 100:
        try:
            req = Request(SPARC_RAR_URL, headers={"User-Agent": "AxiomOS/1.0"})
            with urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="replace")
            if raw and len(raw) > 100:
                _RAR_MRT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
                _RAR_MRT_LOCAL.write_text(raw, encoding="utf-8")
        except Exception:
            return np.array([]), np.array([])
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    g_bar_list, g_obs_list = [], []
    for line in lines:
        if line.startswith("#") or "Bytes" in line or "Format" in line or "Label" in line or "----" in line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                log_gbar = float(parts[0])
                log_gobs = float(parts[2])
                if np.isfinite(log_gbar) and np.isfinite(log_gobs):
                    g_bar_si = 10.0 ** log_gbar
                    g_obs_si = 10.0 ** log_gobs
                    if g_bar_si > 1e-20 and g_obs_si > 1e-20:
                        g_bar_list.append(g_bar_si / KM_S_SQ_PER_KPC_TO_MS2)
                        g_obs_list.append(g_obs_si / KM_S_SQ_PER_KPC_TO_MS2)
            except (ValueError, IndexError):
                continue
    if not g_bar_list:
        return np.array([]), np.array([])
    return np.array(g_bar_list), np.array(g_obs_list)


def load_sparc_rar_real_only_per_galaxy(
    n_galaxies: int = 175,
    galaxy_list: Optional[List[str]] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str]]:
    """Load (g_bar, g_obs) per galaxy from Rotmod. For per-galaxy fit then median (McGaugh-style)."""
    if galaxy_list is None:
        avail = get_available_sparc_galaxies()
        names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]
    else:
        names = galaxy_list
    per_gal = []
    used_names = []
    for name in names:
        d = _load_real_sparc(name)
        if d is None:
            continue
        g_bar, g_obs, mask = compute_accelerations(d)
        if len(g_bar) >= 5:  # need enough points per galaxy
            per_gal.append((g_bar, g_obs))
            used_names.append(name)
    return per_gal, used_names


def load_sparc_rar_real_only(
    n_galaxies: int = 175,
    galaxy_list: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load (g_bar, g_obs) from REAL SPARC data only. No mock fallback.
    Returns empty arrays if real data unavailable or insufficient.
    """
    if galaxy_list is None:
        avail = get_available_sparc_galaxies()
        names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]
    else:
        names = galaxy_list
    all_g_bar, all_g_obs, used_names = [], [], []
    for name in names:
        d = _load_real_sparc(name)
        if d is None:
            continue
        g_bar, g_obs, mask = compute_accelerations(d)
        if len(g_bar) >= 3:
            all_g_bar.append(g_bar)
            all_g_obs.append(g_obs)
            used_names.append(name)
    if not all_g_bar:
        return np.array([]), np.array([]), []
    return (
        np.concatenate(all_g_bar),
        np.concatenate(all_g_obs),
        used_names,
    )


def load_sparc_rar(
    n_galaxies: int = 175,
    galaxy_list: Optional[List[str]] = None,
    use_mock_if_fail: bool = True,
    use_real: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load (g_bar, g_obs) pairs for RAR discovery.
    Returns: g_bar_all, g_obs_all, names (per-galaxy labels for optional use)
    """
    if galaxy_list is None:
        if use_real:
            avail = get_available_sparc_galaxies()
            names = avail[:n_galaxies] if avail else SPARC_GALAXIES[:n_galaxies]
        else:
            names = SPARC_GALAXIES[:n_galaxies]
    else:
        names = galaxy_list
    all_g_bar, all_g_obs, used_names = [], [], []
    for name in names:
        d = load_sparc_galaxy(name, use_mock_if_fail=use_mock_if_fail, use_real=use_real)
        g_bar, g_obs, mask = compute_accelerations(d)
        if len(g_bar) >= 3:
            all_g_bar.append(g_bar)
            all_g_obs.append(g_obs)
            used_names.append(name)
    if not all_g_bar:
        return np.array([]), np.array([]), []
    return (
        np.concatenate(all_g_bar),
        np.concatenate(all_g_obs),
        used_names,
    )


def get_sparc_galaxy_tensors(
    galaxy_name: str,
    use_mock_if_fail: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get (R, V_def²) for one galaxy as numpy arrays.
    R in kpc, V_def² in (km/s)².
    """
    d = load_sparc_galaxy(galaxy_name, use_mock_if_fail=use_mock_if_fail)
    V_bary_sq = d["V_gas"]**2 + d["V_disk"]**2 + d["V_bulge"]**2
    V_obs_sq = np.maximum(d["V_obs"]**2, V_bary_sq + 1.0)
    V_def_sq = np.maximum(V_obs_sq - V_bary_sq, 0.0)
    return d["R"].astype(np.float64), V_def_sq.astype(np.float64)
