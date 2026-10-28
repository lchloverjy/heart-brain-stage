# heart-brain-stage

Code for: **"A heart–brain MRI stage and risk of cardiovascular and cerebrovascular events beyond SCORE2: a prospective analysis of UK Biobank"** (submitted to *eClinicalMedicine*).

The repository reproduces every number, table, and figure in the manuscript and supplementary appendix from UK Biobank inputs and the fitted Subtype and Stage Inference (SuStaIn) models. UK Biobank data cannot be redistributed: approved researchers can obtain them from the UK Biobank Access Management System. **This repository contains no participant-level data.**

## Layout

| Path | Content |
|---|---|
| `analysis/` | Cohort construction, SuStaIn summaries, outcome analyses, independent-sample validation, sex-specific analysis, de novo refit, regional heart–brain atlas analysis (`regional_atlas.py`), subtype-by-stage regional anatomy with out-of-model measures (`regional_subtype.py`), independent-sample cumulative incidence (`independent_km.py`), tables 1–2 |
| `figs/` | Figures 1–5 and appendix figures S1–S4 (matplotlib with PyVista/VTK off-screen 3D rendering; vector PDF/SVG and PNG): `fig1v7.py` (graphical-abstract overview with measured subtype signatures), `fig2_sequence_v7.py` (model-implied order; measured regional abnormality by subtype and stage painted on the 3D heart and brain; also appendix figure S4), `fig3_regional_v7.py` (stage-matched subtype differences, regional anatomy, and heart–brain coupling), `fig4_events_v6.py` (events after imaging in both samples), `fig5_value_v6.py` (value beyond SCORE2), `figS_v5.py` (appendix). `render3d.py` renders the four-chamber heart template (16 AHA segments painted with data), the cortical surface, hippocampus, and white-matter tracts; `heart_assets.py` builds the heart surfaces from the published template; `anat3d.py` (bull's-eye plots, white-matter sections) and `regional_data.py` and `regional_subtype_data.py` (results mapped to atlas labels) are shared helpers. Earlier figure scripts (`fig*v4.py`, `fig*_v5.py`, and the v6 scripts) are kept because the v7 scripts import their panels and constants. `_coronal_cache.pkl` is a vector cache of the Harvard-Oxford atlas (no participant data). |
| `figs/assets/` | Display-only templates. Included: ENIGMA Toolbox fsaverage5 surfaces, subcortical meshes, and Desikan–Killiany labels (BSD-3, `ENIGMA_LICENSE.txt`); surfaces derived from the population-average four-chamber heart of Rodero et al. (Zenodo doi:10.5281/zenodo.4593739, CC BY 4.0; `heart/ATTRIBUTION.txt`), rebuildable with `fetch_heart_template.sh`. Not redistributed: JHU-ICBM white-matter atlases (FSL licence), downloaded by `fetch_atlases.sh` (called by `run_all.sh`) and verified against `ASSETS.md5`. No result depends on these files; they are not the anatomy of any participant. |
| `manuscript/` | `facts.py` computes every number quoted in the text; `build_appendix.py` and `build_v4.py` build the appendix and manuscript DOCX from `src_v4.md`, `src_back_v4.md`, and `refs_v4.json`; `serial_comma.py` is the copy-editing pass applied to the English sources |
| `data_aggregate/` | Model-selection CVIC values (aggregate, no participant data) |
| `run_all.sh` | End-to-end pipeline |
| `INPUT_MANIFEST.csv` | The 33 restricted input files (30 UK Biobank-derived inputs and 3 pipeline-generated de novo pickles), with size and MD5 checksum (files not included) |

## Inputs (not distributed)

Set three environment variables:

* `HB_RAW`: upstream outputs of the SuStaIn pipeline (default `$HB_ROOT/raw`):
  `up/module3_preparation/{03_inputs,04_thresholds}` (SuStaIn input z scores and z-score thresholds),
  `up2/sustain_final/*` (final model pickles and subtype/stage assignments),
  `up/score_reconstruction/e_cohort_visits.csv` (assessment-centre dates, derivation sample), and
  `ukb_area/data_all.csv` (baseline covariates); for the regional analysis also
  `brainraw/原始脑区数据/{data_587idp_v2.csv,brain_idps_587used_new.csv}` (brain imaging-derived phenotypes and their dictionary),
  `cmr/cmr82.csv` (cardiac MRI segmental and global measures),
  `ukb_area_up/*_weights*.csv` (domain-score loadings), and
  `up/module3_preparation/{01_reference,02_normative_models}` (normative reference IDs and domain z scores).
* `UKB_DATA`: UK Biobank field exports: `41270.csv` and `41280.csv` (HES ICD-10 diagnoses and dates), `ALL_death.csv` (fields 40000, 40001, and 40002), and a field-53 export (assessment-centre dates, instances 0–3; any CSV in `UKB_DATA` with a `53-2.0` column is detected).
* `HB_ROOT`: this repository (default: the folder containing `run_all.sh`).

`INPUT_MANIFEST.csv` lists the exact files and checksums used for the submitted results.

## Run

```bash
pip install -r requirements.txt
export HB_RAW=/path/to/upstream UKB_DATA=/path/to/ukb_exports
RUN_DENOVO=1 bash run_all.sh  # full pipeline, including the de novo SuStaIn refit in 6000 independent participants (several hours)
bash run_all.sh               # about 1 h on 6 cores; reuses an existing de novo refit
```

The de novo refit pickles (`results/replication_sustain/pickle_files/`) contain participant-level fields and are not distributed. On a first run, use `RUN_DENOVO=1`; afterwards `replication_compare.py` reads the saved pickles.

## Reproducibility check

On Oct 25, 2026, the repository was run from a clean directory with the restricted inputs listed in `INPUT_MANIFEST.csv`. All 45 result tables (derivation outcomes, SuStaIn summaries, figure-preparation tables, independent-sample analyses under both follow-up designs, de novo comparison, and tables 1–2), and the participant list defining the de novo subsample, were identical to the submitted results (same text, missing-value pattern, and maximum absolute difference 0, including all bootstrap confidence intervals). The file of quoted numbers (`manuscript/facts.json`), the appendix page map, the word counts, and the reference list were identical, and the manuscript and appendix text were identical apart from local file paths. The sex-specific analysis (`sex_stratified.py`; appendix table S21), added after this run, was then executed in the same directory and its two output tables were also identical. The de novo MCMC refit itself was not re-executed in this check (saved pickles were used); it is seeded (`seed=20260712`). The regional analysis and the v5 figures added later were reproduced in the same way: the four regional result tables (including 1000-replicate bootstrap CIs), all 285 numbers quoted in the text, the manuscript text, and all 30 figure PNGs were identical to the submitted versions (pixel-identical figures). The v6 additions (independent-sample cumulative incidence, `independent_km.py`; figures 1–5 redrawn with the 3D heart and brain templates) were reproduced in the same directory: both new result tables were identical, as were all 285 quoted numbers, the manuscript text, and all 30 figure PNGs (v6 figures 1–5 with their single panels, and appendix figures S1–S3; pixel-identical). The v7 additions (subtype-by-stage regional anatomy with 313 measures, 151 of them not used to fit the model; `regional_subtype.py`, seven tables with 1000 bootstrap replicates; figures 1–3 and appendix figure S4 redrawn) were reproduced in the same directory: all seven tables were identical, as were all 310 quoted numbers, the appendix page map, the manuscript and appendix text, and the figure PNGs (pixel-identical).

## Key design choices in the code

* **Time zero** is the imaging visit (field 53-2.0). Participants with the outcome on or before imaging are excluded from that outcome's incidence analysis. Follow-up ends at the first event, death, or Sept 30, 2021 (the last date in the HES extract), whichever comes first. Deaths with the outcome as the underlying cause (field 40001) and no earlier admission count as events; deaths from other causes are censored.
* **Stage** is expressed as a percentage of the maximum stage (30, 24, or 18 events) so that hazard ratios per 10 percentage points are comparable across models.
* **Subtype labels.** `common.py` maps the pySuStaIn internal subtype order to the published labels (ordered by mean posterior fraction and checked with synthetic individuals). Check this mapping whenever models are refitted.
* **Independent sample.** The MRI-only model is applied without refitting, with stages assigned from the posterior MCMC samples (`restage.py`), to participants with complete heart and brain MRI markers who were not used for model fitting. These participants come from the same cohort, so this is an independent-sample (not external) validation. `validation_independent.py imaging` is the primary design (follow-up from each participant's own imaging date); `validation_independent.py landmark` is a fixed-landmark sensitivity analysis (time zero Apr 1, 2020; appendix table S15). In this sample, model 2 and the SCORE2 analyses are complete-case.
* **SCORE2/SCORE2-OP** is implemented from the 2021 ESC publications with the low-risk-region (UK) recalibration; 10-year risk enters the Cox models as a probability on the complementary log–log scale. In the independent-sample bootstrap the linear predictors are held fixed.
* **Missing covariates (derivation sample)** are imputed once by chained equations.
* **Seeds.** Imputation uses `random_state=20260712`. In the derivation sample, each outcome-by-model bootstrap task *(i, j)* uses `numpy.random.default_rng(20260712 + 100*i + j)`, with 1000 replicates for the All-10 model and the primary outcome and 200 for the other combinations. In the independent sample, replicate *i* uses `default_rng(1000 + i)`. The de novo SuStaIn refit uses `seed=20260712`.
* **Regional analysis** (`regional_atlas.py`) rebuilds normative z scores for the regional measures underlying the MRI domain scores (131 brain and 60 cardiac measures) in the same disease-free reference group as the domain scores (age and sex; hippocampus also intracranial volume), oriented so that higher = more abnormal. Regressions use Benjamini–Hochberg FDR control; tissue-class medians use a participant-level bootstrap (1000 replicates, `numpy.random.default_rng(20261025)`). Atlases are used for display only.
* **Disclosure control.** Only aggregate results are written to the Source Data workbook; counts of 1–4 are shown as "<5".

* **Display templates.** The 3D heart in figures 1–3 is a population-average template, not participant anatomy. In figure 3 the colours on the template (16 AHA segments, aortic segments, cortical regions, white-matter tracts) are the regional results; in figure 2B they show the model-implied z level reached at each stage.
## Software

Python 3.12; numpy, pandas, scipy, lifelines, scikit-learn, statsmodels, pySuStaIn 0.1 (commit 708fa22), matplotlib, shapely, scikit-image, nilearn, nibabel, python-docx, and openpyxl (exact versions in `requirements.txt`). LibreOffice (24.2) and poppler (`pdftotext`) are used only to paginate the appendix.

## Licence

MIT (see `LICENSE`).
