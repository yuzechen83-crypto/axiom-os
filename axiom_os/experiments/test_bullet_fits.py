"""Test Bullet Cluster FITS loader with a minimal synthetic FITS."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def create_test_fits(out_path: Path, nx: int = 50, ny: int = 50) -> None:
    """Create minimal 2D FITS for testing."""
    from astropy.io import fits

    mass = np.exp(-((np.arange(ny)[:, None] - 25) ** 2 + (np.arange(nx) - 30) ** 2) / 200)
    gas = np.exp(-((np.arange(ny)[:, None] - 20) ** 2 + (np.arange(nx) - 25) ** 2) / 150)
    hdu_m = fits.PrimaryHDU(mass)
    hdu_g = fits.PrimaryHDU(gas)
    hdu_m.writeto(out_path / "bullet_mass_test.fits", overwrite=True)
    hdu_g.writeto(out_path / "bullet_gas_test.fits", overwrite=True)


def main():
    from axiom_os.datasets.bullet_cluster import (
        load_bullet_cluster_fits,
        load_bullet_cluster_mvp,
        to_discovery_format,
    )

    out_dir = ROOT / "axiom_os" / "datasets" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    create_test_fits(out_dir)

    x, y, rho, gx, gy, v, t = load_bullet_cluster_fits(
        out_dir / "bullet_mass_test.fits",
        out_dir / "bullet_gas_test.fits",
    )
    print(f"FITS load: n={len(x)}, rho range [{rho.min():.2e}, {rho.max():.2e}]")

    X, y_out, names, src, _ = load_bullet_cluster_mvp(
        use_fits=True,
        mass_path=out_dir / "bullet_mass_test.fits",
        gas_path=out_dir / "bullet_gas_test.fits",
    )
    print(f"MVP with FITS: n={len(y_out)}, source={src}")
    assert len(X) > 0, "FITS should produce data"
    print("OK: FITS loader works")


if __name__ == "__main__":
    main()
