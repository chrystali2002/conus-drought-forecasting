# conus-drought-forecasting

# Sub-seasonal vegetation drought forecasting over CONUS croplands

Code for an irrigation-aware ConvLSTM that forecasts vegetation drought
(standardised MODIS NIRv anomalies) across water-limited US croplands at 8–40 day
lead times, and for a controlled test of whether correcting the irrigation signal
in satellite soil moisture improves those forecasts.

**Write-up:** [blog post](https://chrystali2002.github.io/olusegun.christiana.github.io/)

## Result in one paragraph

The framework reaches an anomaly correlation of 0.502 ± 0.007 across 15 random
seeds on a held-out 2023–2024 test period, 0.661 at 8-day lead. Correcting the
irrigation signal in SMAP L4 root-zone soil moisture makes no detectable
difference to skill at 9 km, either domain-averaged or restricted to the 554
cells where the correction acts. It does help where irrigation dominates the
cell: +0.013 ACC at 32-day lead in cells more than half irrigated. The correction
is small and sparse — a mean absolute adjustment of 0.036 σ, with 52 % of
irrigated cell-timesteps receiving none at all — which is consistent with the
signal being diluted at 9 km resolution.

---

## Order of execution

Scripts are numbered in the order they must run. Steps 01–09 build the dataset
and need only be run once; 10–17 are the model and experiments.

### Data preparation

| # | script | what it does | writes |
|---|---|---|---|
| 01 | `01_check_smap_times.py` | picks which 3-hourly SMAP granule has complete CONUS coverage | diagnostic only |
| 01 | `01_download_smap.py` | downloads one SMAP L4 granule per day (21:00–00:00 UTC window) | `data/raw/smap/*.h5` |
| 02 | `02_read_smap.py` | HDF5 → daily NetCDF, CONUS subset | `smap_daily_2230utc_conus.nc` |
| 03 | `03a_qa_diagnostic.py` | MODIS QA flag inspection | diagnostic only |
| 03 | `03b_gee_nirv_export.py` | exports MOD09A1 NIRv from Earth Engine | GEE Drive exports |
| 03 | `03c_nirv_assemble.py` | assembles exports, QA-screens, interpolates 8-day → daily | `nirv_daily_conus.zarr` |
| 04 | `04a_cdl_mask.py` | eight semi-arid crop classes from CDL 2020 → binary crop mask | `cdl_crop_mask.npy` |
| 04 | `04b_irrigation_fraction_V3.py` | LANID 30 m → irrigated fraction per cell | `irrigation_fraction_smap_grid.npy` |
| 04 | `04c_aridity_and_indices.py` | aridity index (Global AI_ET0 v3.1) and NOAA teleconnections | `aridity_index.npy`, `teleconnection_indices_daily.csv` |
| 05 | `05_harmonize_grids.py` | everything onto the common 321 × 656 / 0.09° grid | `master_dataset.zarr` |
| 06 | `06_compute_anomalies.py` | per-pixel, ±15-day day-of-year z-scores | `anomalies.zarr` |
| 07 | `07_irrigation_deconvolution.py` | LANID-weighted regional-excess correction of RZSM | `anomalies_corrected.zarr` |
| 08 | `08_aridity_stratification.py` | domain definition (AI 0.05–0.65 ∩ cropland) and reporting zones | `domain_masks.npz` |
| 09 | `09_build_dataset.py` | normalisation statistics over the training window | `dataset_statistics.json` |

### Model and experiments

| # | script | what it does |
|---|---|---|
| 10 | `10_model.py` | ConvLSTM architecture (imported, not run directly) |
| 11 | `11_train.py` + `.sh` | single-run training — superseded by 15, kept for reference |
| 12 | `12_evaluate.py` + `.sh` | single-run evaluation and diagnostic plots |
| 13 | `13_train_nirv_only.py` + `.sh` | NIRv-only ablation: isolates the contribution of soil moisture |
| 14 | `14_evaluate_nirv_only.py` + `.sh` | scores the NIRv-only arm |
| 15 | `15_train_multiseed.py` + `.sh` | **main experiment.** 15 seeds × 2 arms (corrected / uncorrected RZSM) |
| 16 | `16_evaluate_multiseed.py` + `.sh` | scores every seed on 8 stratified masks |
| 17 | `17_ablation_compare.py` | paired per-seed comparison of the two arms |

The three arms are selected by an environment variable and resolved through a
single shared module, so they cannot drift apart:

```bash
DROUGHT_ARM=corrected   sbatch 15_train_multiseed.sh
DROUGHT_ARM=uncorrected sbatch 15_train_multiseed.sh
```

### Configuration

| file | contents |
|---|---|
| `config.py` | paths, grid, lead times, hyperparameters, seed list, reporting zones |
| `arm_config.py` | resolves which experimental arm a run belongs to; aborts if the two arms load identical input |
| `show_arm_paths.py` | prints every path each arm reads and writes, before committing GPU time |
| `drought_forecast.yml` | conda environment |

---

## Verification and diagnostics

These are not part of the pipeline. They exist because each was written to check
a specific claim, and several found real problems.

| script | question it answers | what it found |
|---|---|---|
| `check_correction.py` | does the irrigation correction alter cells it should not touch? | no — zero cells below the 0.15 threshold differ |
| `ablation_scope.py` | how much of the evaluation domain does the correction reach? | 554 of 3,093 cells (17.9 %), so a domain average dilutes any effect ~5.6× |
| `hyperarid_test.py` | is the aridity index contaminated at water margins? | yes — hyper-arid cells are enriched 26.8× within one cell of water, a multiplicative signature consistent with nodata treated as zero during regridding |
| `smap_coverage_stats.py` | how complete is the soil moisture record? | 15 cells over open water have no data at any timestep and are excluded from evaluation |
| `verify_paper_numbers.py` | do the quoted pixel counts and correction magnitudes hold? | counts reconcile; mean correction 0.036 σ over the full record |
| `extract_methods_params.py` | pulls every parameter the methods section needs from the code and data | — |
| `preflight_check.py` | are the inputs aligned before a long run? | — |
| `rebuild_masks.py` | rebuilds reporting-zone masks on the current grid | the originals were on a stale 278-row grid |

## Figures

| script | produces |
|---|---|
| `make_post_figures.py` | all four blog-post figures from current data, printing every plotted value |
| `agu_poster.py` | AGU poster panels — **contains hardcoded literals from an early single run; do not reuse its numbers** |

---

## Known limitations

**Aridity contamination at water margins.** The aridity index used to define the
domain appears to have been aggregated from 30 arcsec to 0.09° with nodata filled
as zero, so cells adjacent to water read drier than they are, in proportion to
their water fraction. 46 domain cells (1.5 %) are affected. None exceeds an
irrigated fraction of 0.09, so the irrigation ablation is unaffected; removing
them raises domain-averaged skill by 0.003. See `hyperarid_test.py`.

**Regional concentration.** 88 % of the cells above 0.50 irrigated fraction are in
Nebraska and Kansas. The dose-response gradient reflects a few irrigation systems
more than a national pattern.

**Cropland definition.** The domain uses eight CDL crop classes (winter, spring
and durum wheat, corn, sorghum, cotton, alfalfa, barley), not all cropland. This
is why the Central Valley contributes only 55 cells.

**Five seeds were not enough.** An earlier five-seed version of this analysis gave
two conclusions that fifteen seeds overturned. Multi-seed results are reported as
mean ± SD throughout, and paired by seed where two arms are compared.

---

## Data sources

| dataset | product | use |
|---|---|---|
| SMAP L4 | SPL4SMGP v008, 9 km, 3-hourly | root-zone soil moisture |
| MODIS | MOD09A1 v061, 500 m, 8-day | NIRv |
| USDA CDL | 2020, 30 m | cropland mask |
| USGS LANID | 2020, 30 m | irrigated fraction |
| Global AI_ET0 | v3.1, 30 arcsec | aridity index |
| NOAA | Niño 3.4, AMO, PDO | teleconnection indices |

## Environment

```bash
conda env create -f drought_forecast.yml
conda activate drought_forecast
```

Trained on NVIDIA A100-SXM4-80GB. One seed takes roughly 3.5 hours; the full
15-seed two-arm experiment is about 37 hours with both arms running in parallel.

## Citation

If this is useful, please cite the blog post. Christiana F. Olusegun,
Michigan State University.
