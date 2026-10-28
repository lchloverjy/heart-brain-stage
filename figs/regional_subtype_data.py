"""Data layer for the v7 subtype-specific regional figures (results/regional_subtype/*.csv -> atlas keys).

cell_maps(sample)      -> {cell: maps} mean regional z by subtype x stage window (stage 0 = shared reference)
contrast_maps(sample)  -> maps of the stage-matched Subtype 1 minus Subtype 2 difference (+ q < 0.05 masks)
Map keys: cortex_lh/rh (DK region -> v), wm_fw / wm_fa / wm_md (JHU label -> v), subc_lh/rh (structure -> v,
incl. hippocampus), heart_wt / heart_ecc / heart_err (AHA segment -> v), aorta (asc, desc), wmh; each with *_mask.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import anat3d as A

RES = Path(__file__).resolve().parent.parent / "results" / "regional_subtype"
CELLS = ["Stage 0", "Subtype 1, stage 1-2", "Subtype 1, stage >=3", "Subtype 2, stage 1-2", "Subtype 2, stage >=3"]
WM_KEY = {"White-matter free water": "wm_fw", "White-matter FA": "wm_fa", "White-matter MD": "wm_md"}
HEART_KEY = {"Wall thickness": "heart_wt", "Circumferential strain": "heart_ecc", "Radial strain": "heart_err"}


@lru_cache(None)
def dictionary():
    return pd.read_csv(RES / "dictionary_ext.csv")


@lru_cache(None)
def cell_means():
    return pd.read_csv(RES / "cell_means.csv")


@lru_cache(None)
def cell_meta():
    return pd.read_csv(RES / "cell_meta.csv")


@lru_cache(None)
def contrast():
    return pd.read_csv(RES / "subtype_contrast.csv")


@lru_cache(None)
def replication():
    return pd.read_csv(RES / "contrast_replication.csv")


@lru_cache(None)
def tissue_summary():
    return pd.read_csv(RES / "tissue_summary_ext.csv")


def split(series, mask=None):
    d = dictionary().set_index("var")
    keys = ["cortex_lh", "cortex_rh", "wm_fw", "wm_fa", "wm_md", "subc_lh", "subc_rh", "heart_wt", "heart_ecc",
            "heart_err", "aorta"]
    out = {k: {} for k in keys}
    out.update({k + "_mask": {} for k in keys})
    out["wmh"], out["wmh_mask"] = np.nan, True
    for var, val in series.items():
        sig = True if mask is None else bool(mask.get(var, True))
        if var not in d.index:
            continue
        r = d.loc[var]
        t = r.tissue
        if t == "White-matter hyperintensity":
            out["wmh"], out["wmh_mask"] = val, sig
        elif t == "Cortical thickness":
            if r.region == "GlobalMeanThickness":
                continue
            k = "cortex_lh" if r.hemi == "left" else "cortex_rh"
            out[k][r.region], out[k + "_mask"][r.region] = val, sig
        elif t in WM_KEY and r.method == "TBSS":
            idx = A.jhu_label_for(r.region, r.hemi)
            if idx is not None:
                out[WM_KEY[t]][idx], out[WM_KEY[t] + "_mask"][idx] = val, sig
        elif t in ("Hippocampal volume", "Subcortical volume"):
            k = "subc_lh" if r.hemi == "left" else "subc_rh"
            out[k][r.region], out[k + "_mask"][r.region] = val, sig
        elif t in HEART_KEY:
            seg = int(str(r.region).replace("AHA", ""))
            out[HEART_KEY[t]][seg], out[HEART_KEY[t] + "_mask"][seg] = val, sig
        elif t == "Aortic distensibility":
            out["aorta"][r.region], out["aorta_mask"][r.region] = val, sig
    return out


def cell_maps(sample="derivation"):
    cm = cell_means()
    cm = cm[cm["sample"] == sample]
    meta = cell_meta()
    meta = meta[meta["sample"] == sample].set_index("cell")
    res = {}
    for c in CELLS:
        x = cm[cm.cell == c].set_index("var")
        res[c] = split(x.mean_z)
        res[c]["n"] = int(meta.loc[c, "n"])
        res[c]["psub"] = meta.loc[c, "psub_median"]
    return res


def contrast_maps(sample="derivation", analysis="primary", q=0.05):
    c = contrast()
    c = c[(c["sample"] == sample) & (c.analysis == analysis)].set_index("var")
    m = split(c.beta, mask=(c.q < q).to_dict())
    m["n_s1"], m["n_s2"] = int(c.n_s1.max()), int(c.n_s2.max())
    return m, c


if __name__ == "__main__":
    m = cell_maps("derivation")
    for c in CELLS:
        x = m[c]
        print(c, x["n"], "ctx", len(x["cortex_lh"]), "fw/fa/md", len(x["wm_fw"]), len(x["wm_fa"]), len(x["wm_md"]),
              "subc", len(x["subc_lh"]), "wt/ecc/err", len(x["heart_wt"]), len(x["heart_ecc"]), len(x["heart_err"]),
              "aorta", x["aorta"], "wmh", round(x["wmh"], 2))
    k, t = contrast_maps("independent")
    print("contrast indep: n", k["n_s1"], k["n_s2"], "sig WT", sum(k["heart_wt_mask"].values()),
          "subc", {s: round(v, 2) for s, v in k["subc_lh"].items()})
    print(replication().round(3).to_string())
