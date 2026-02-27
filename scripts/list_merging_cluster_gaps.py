"""
List Merging Clusters Needing v, t, M, x_offset

Golovich 2019 has 29 clusters with Name, z. We have 9 with full params.
Run download first: python scripts/download_merging_clusters.py

Output: clusters that need paper lookup for v_collision, t_since_collision,
M_total, x_offset (or d_proj).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.golovich_2019 import load_golovich_table1, golovich_clusters_needing_params
from axiom_os.datasets.merging_clusters import MERGING_CLUSTER_CATALOG


def main():
    print("=" * 70)
    print("Merging Cluster Data Status")
    print("=" * 70)

    curated = MERGING_CLUSTER_CATALOG
    print(f"\nCurated catalog (full v,t,M,x_offset): {len(curated)} systems")
    for r in curated:
        print(f"  {r['name']:15} v={r['v_collision_km_s']:.0f} t={r['t_since_collision_Myr']:.0f} x={r['x_offset_kpc']:.0f}")

    golovich = load_golovich_table1()
    print(f"\nGolovich 2019 (Name, z only): {len(golovich)} clusters")

    missing = golovich_clusters_needing_params()
    print(f"\nGolovich clusters needing v,t,M,x_offset from papers: {len(missing)}")
    print("  (Run discovery with curated only until these are filled)")
    for i, name in enumerate(missing, 1):
        row = next((r for r in golovich if r["name"] == name), {})
        z = row.get("z", "?")
        print(f"  {i:2}. {name:30} z={z}")

    print("\n" + "=" * 70)
    print("To reach 30+ systems: need 21+ more with v,t,M,x_offset.")
    print("Sources: individual papers, MCMAC runs, Golovich 2019 ApJ 882 follow-up.")
    print("=" * 70)


if __name__ == "__main__":
    main()
