#!/usr/bin/env bash
# Reproduce every result, figure and document of the eClinicalMedicine submission.
#   HB_ROOT  : this repository (default: directory of this script)
#   HB_RAW   : upstream inputs (module3_preparation, sustain_final pickles, score_reconstruction, ukb_area);
#              restricted UK Biobank-derived data, not distributed (default: $HB_ROOT/raw)
#   UKB_DATA : UK Biobank field exports (41270.csv, 41280.csv, ALL_death.csv, field-53 dates)
#   RUN_DENOVO=1 also re-runs the de novo SuStaIn refit in 6000 independent participants (several hours).
set -euo pipefail
export HB_ROOT="${HB_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export HB_RAW="${HB_RAW:-$HB_ROOT/raw}"
export UKB_DATA="${UKB_DATA:-$HB_ROOT/data_ukb}"
export MPLBACKEND=Agg
cd "$HB_ROOT/analysis"
echo "[1/6] derivation cohort, SuStaIn summaries"
python3 build_cohort.py
python3 sustain_deep.py
python3 fig2_prep.py
echo "[2/6] derivation outcomes (1000 bootstrap replicates)"
python3 outcomes.py 1000
echo "[3/6] independent sample: transfer of the MRI-only model, imaging-anchored follow-up"
# fixed common landmark (2020-04-01 to 2021-09-30): sensitivity analysis for appendix table S15 and the text
python3 validation_independent.py landmark
# primary design: follow-up from each participant's own imaging date (field 53-2.0); run last
python3 validation_independent.py imaging
python3 validation_extra.py imaging
python3 independent_km.py          # figure 4B: cumulative incidence by stage group, independent sample
python3 independent_score2.py 1000
python3 sex_stratified.py          # women vs men, primary outcome (appendix table S21)
echo "[4/6] de novo refit"
if [ "${RUN_DENOVO:-0}" = "1" ]; then
  python3 replication_sustain.py 10 20000 3 6000 rep6000
fi
python3 replication_compare.py 2
python3 regional_atlas.py 1000      # regional anatomy: figure 3, appendix table S22 and figure S3
python3 regional_subtype.py 1000    # subtype x stage regional anatomy (313 measures, 151 out-of-model): figures 1C, 2B, 3A, 3D; table S23; figure S4
python3 table1.py
python3 table2.py
echo "[5/6] figures (figure 1 is drawn after facts.py, below)"
cd "$HB_ROOT/figs"
[ -f assets/JHU-ICBM-labels-1mm.nii.gz ] || bash assets/fetch_atlases.sh   # display-only atlases (FSL licence)
[ -f assets/heart/heart_surface.vtp ] || bash assets/fetch_heart_template.sh   # display-only heart template (CC BY 4.0)
python3 fig2_sequence_v7.py all    # figure 2: model-implied order; B = measured regional abnormality by subtype x stage; appendix figure S4
python3 fig3_regional_v7.py all    # figure 3: stage-matched subtype difference map and heart-brain coupling
python3 fig4_events_v6.py all      # figure 4: events after imaging (derivation and independent)
python3 fig5_value_v6.py all       # figure 5: value beyond SCORE2
mkdir -p output_v7 && cp -r output_v6/Figure4 output_v6/Figure5 output_v7/   # figures 4-5 unchanged in v7
python3 figS_v5.py all             # appendix figures S1-S3 (unchanged since v5)
echo "[6/6] appendix and manuscript"
cd "$HB_ROOT/manuscript"
python3 build_appendix.py          # first pass: locates appendix pages
python3 build_appendix.py --final  # second pass: table of contents with page numbers
python3 facts.py                   # every number quoted in the text
(cd "$HB_ROOT/figs" && python3 fig1v7.py all)   # figure 1 (graphical-abstract overview) reads facts.json
python3 build_v4.py                # manuscript DOCX
echo "done"
