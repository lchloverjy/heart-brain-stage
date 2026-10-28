#!/usr/bin/env bash
# Fetch the display-only atlas assets used by figs/anat3d.py (Figures 1-3 and appendix figure S3).
#
# Included in this repository (BSD-3-Clause, see ENIGMA_LICENSE.txt):
#   fsa5_lh.gii, fsa5_rh.gii, sctx_lh.gii, sctx_rh.gii, aparc_fsa5.csv  (ENIGMA Toolbox, commit b08974b5)
#   dk68_order.txt  (Desikan-Killiany label order used to map aparc_fsa5.csv; written for this project)
# Downloaded by this script (FSL licence; academic use; not redistributed here):
#   JHU-ICBM-labels-1mm.nii.gz, JHU-ICBM-FA-1mm.nii.gz, JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz,
#   JHU-labels.xml, JHU-tracts.xml  (FSL conda package fsl-data_atlases 2103.0)
# The atlases are used only to draw the figures; no analysis result depends on them.
set -euo pipefail
cd "$(dirname "$0")"
ENIGMA=https://raw.githubusercontent.com/MICA-MNI/ENIGMA/b08974b55243060cbc1fad12c87048037446e8f7/enigmatoolbox/datasets
for f in fsa5_lh.gii fsa5_rh.gii sctx_lh.gii sctx_rh.gii; do
  [ -f "$f" ] || curl -sSL -o "$f" "$ENIGMA/surfaces/$f"
done
[ -f aparc_fsa5.csv ] || curl -sSL -o aparc_fsa5.csv "$ENIGMA/parcellations/aparc_fsa5.csv"
if [ ! -f JHU-ICBM-labels-1mm.nii.gz ]; then
  tmp=$(mktemp -d)
  curl -sSL -o "$tmp/atl.tar.bz2" \
    https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/noarch/fsl-data_atlases-2103.0-0.tar.bz2
  tar -xjf "$tmp/atl.tar.bz2" -C "$tmp" data/atlases/JHU data/atlases/JHU-labels.xml data/atlases/JHU-tracts.xml
  cp "$tmp"/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz "$tmp"/data/atlases/JHU/JHU-ICBM-FA-1mm.nii.gz \
     "$tmp"/data/atlases/JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz "$tmp"/data/atlases/JHU-labels.xml \
     "$tmp"/data/atlases/JHU-tracts.xml .
  rm -rf "$tmp"
fi
md5sum -c ASSETS.md5
