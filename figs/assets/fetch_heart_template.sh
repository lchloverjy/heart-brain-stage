#!/usr/bin/env bash
# Download the population-average four-chamber heart template used for display in figures 1-3 and rebuild the
# derived surfaces in assets/heart/ (heart_surface.vtp, vessels.vtp, stump_geometry.npy).
# Source (CC BY 4.0): Rodero C, Strocchi M, Marciniak M, et al. Virtual cohort of extreme and average four-chamber
# heart meshes from statistical shape model. Zenodo 2021. doi:10.5281/zenodo.4593739 (file average.tar.gz).
# The template is a statistical-shape-model mean of adult hearts; it is NOT a UK Biobank participant and carries no data.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="$HERE/heart_template"; mkdir -p "$DST"; cd "$DST"
URL="https://zenodo.org/records/4593739/files/average.tar.gz?download=1"
MD5="992f31e20c1aa73c10c5d9a6b6ac903a"
for i in 1 2 3 4 5 6; do
  curl -sSL -C - --max-time 600 -o average.tar.gz "$URL" && break || { echo "retry $i"; sleep 5; }
done
echo "$MD5  average.tar.gz" | md5sum -c -
tar xzf average.tar.gz average.vtk
cd "$HERE/.." && python3 heart_assets.py "$DST/average.vtk"
