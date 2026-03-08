"""Axiom-OS datasets."""

from .sparc import load_sparc_galaxy, load_sparc_multi, SPARC_GALAXIES

__all__ = ["load_sparc_galaxy", "load_sparc_multi", "SPARC_GALAXIES"]

try:
    from .elia_grid import load_elia_grid, build_grid_sequences
    __all__ = list(__all__) + ["load_elia_grid", "build_grid_sequences"]
except ImportError:
    pass

try:
    from .bullet_cluster import load_bullet_cluster_mvp, load_bullet_cluster_fits
    __all__ = list(__all__) + ["load_bullet_cluster_mvp", "load_bullet_cluster_fits"]
except ImportError:
    pass

try:
    from .merging_clusters import (
        load_merging_clusters_multi_system,
        MERGING_CLUSTER_CATALOG,
        save_merging_cluster_catalog_csv,
    )
    __all__ = list(__all__) + [
        "load_merging_clusters_multi_system",
        "MERGING_CLUSTER_CATALOG",
        "save_merging_cluster_catalog_csv",
    ]
except ImportError:
    pass
