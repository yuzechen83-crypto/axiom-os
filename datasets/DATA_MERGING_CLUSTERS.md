# Merging Cluster Data for τ_response Discovery

## Goal

30+ systems with v_collision, t_since_collision, M_total, x_offset (or d_proj).

Target formula: `x_offset/R_cluster = f(v_collision, t_since_collision, M_total)`

## Current Status

| Source | Systems | v,t,M,x_offset | Notes |
|--------|---------|----------------|-------|
| Curated (literature) | 9 | ✓ Full | Bullet, MusketBall, MACS_J0025, Abell_520, Abell_56, Abell_1758N, MACS_J0018, ZwCl_2341, CIZA_J2242 |
| Golovich 2019 table1 | 29 | ✗ Name,z only | ApJS 240, 39. Need v,t,M,x from papers |
| MCC MCMAC FITS | 2 | ✓ Posterior | Bullet, Musket Ball only |

## Data Sources (Priority)

### 1. Golovich et al. 2019
- **VizieR**: J/ApJS/240/39
- **CDS FTP**: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/240/39/
- **table1**: 29 clusters, Name, z, RA, Dec, Band
- **table7**: 4431 spectroscopic galaxies
- **Gap**: table1 does NOT contain v_collision, t_since_collision, M_total, projected_separation

### 2. Merging Cluster Collaboration (MCC)
- **URL**: http://www.mergingclustercollaboration.org/merger-mc-samples.html
- **Data**: FITS with ~2M MC samples, 13 params (v_col, TSC0, TSC1, d_proj, M1, M2, ...)
- **Available**: Bullet, Musket Ball only
- **Use**: Extract median/percentiles for v, t, d_proj

### 3. Individual Papers
Each Golovich cluster needs lookup for:
- v_collision (km/s)
- t_since_collision (Myr)
- M_total (M_sun)
- x_offset or d_proj (kpc) — gas-mass separation or subcluster separation

## Scripts

```bash
# Download Golovich 2019 table1 (29 clusters)
python scripts/download_merging_clusters.py

# Optional: MCC FITS (~112 MB each)
python scripts/download_merging_clusters.py --mcc

# Extract MCC MCMAC medians (v_col, TSM_0, d_proj, M1, M2) for Bullet/Musket Ball
python scripts/extract_mcc_mcmac_medians.py

# List clusters needing params
python scripts/list_merging_cluster_gaps.py
```

## Golovich Clusters Needing Params (27)

Run `list_merging_cluster_gaps.py` for full list. Overlap with curated: CIZA J2242.8+5301, ZwCl 2341+0000.

## 13-System Extended Results (2025-02)

Extended catalog: curated 9 + 4 filled from 29-collection (Abell_115, Abell_2744, RXCJ1314, ZwCl_0008).

| Metric | 9 systems | 13 systems |
|--------|-----------|-----------|
| Physical form LOOCV MSE | 0.143 | **0.030** |
| Poly LOOCV MSE | 0.162 | 0.051 |
| Poly/Phys LOOCV | ~1.1x | **1.69x** |

Physical form generalizes better with more data. Run: `python -m axiom_os.experiments.validate_merging_clusters --extended`

## Sensitivity Analysis

Run `python -m axiom_os.experiments.sensitivity_merging_clusters` to test excluding outliers.
Best result: exclude MusketBall + CIZA_J2242 → LOOCV MSE 0.099 (vs 0.162 full).

Validate with exclusions:
```bash
python -m axiom_os.experiments.validate_merging_clusters --exclude MusketBall CIZA_J2242
python -m axiom_os.experiments.validate_merging_clusters --weighted  # inverse-variance
```

## Next Steps

1. **Paper lookup**: For each of 27 Golovich clusters, find v,t,M,x_offset from weak-lensing/dynamics papers
2. **MCMAC runs**: If MCC adds more systems, or run MCMAC on Golovich clusters with available mass/redshift/separation priors
3. **Golovich 2019 ApJ 882**: Check follow-up dynamics paper for compiled parameters
4. **Add sigma_x_offset_frac** to catalog entries for weighted regression (1/sigma^2)
