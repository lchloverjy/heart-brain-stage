"""Data layer for the v5 regional heart-brain figures (results/regional/*.csv -> atlas keys)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import anat3d as A

RES = Path(__file__).resolve().parent.parent / "results" / "regional"
STAGES = ["Stage 0", "Stage 1-2", "Stage >=3"]
GLOBAL_ORDER = ["LV myocardial mass", "LV mean wall thickness", "LV end-diastolic volume", "LV ejection fraction",
                "Global circumferential strain", "Global longitudinal strain", "Global radial strain",
                "LV cardiac output", "Ascending aortic distensibility", "Descending aortic distensibility"]
GLOBAL_LABEL = {  # oriented so that higher = more abnormal
    "LV myocardial mass": "LV mass \u2191", "LV mean wall thickness": "Mean wall thickness \u2191",
    "LV end-diastolic volume": "LV end-diastolic volume \u2191", "LV ejection fraction": "LV ejection fraction \u2193",
    "Global circumferential strain": "Global circumferential strain \u2193",
    "Global longitudinal strain": "Global longitudinal strain \u2193",
    "Global radial strain": "Global radial strain \u2193", "LV cardiac output": "Cardiac output \u2193",
    "Ascending aortic distensibility": "Ascending aortic distensibility \u2193",
    "Descending aortic distensibility": "Descending aortic distensibility \u2193"}


@lru_cache(None)
def dictionary():
    return pd.read_csv(RES / "regional_dictionary.csv")


@lru_cache(None)
def stage_means():
    return pd.read_csv(RES / "regional_stage_means.csv")


@lru_cache(None)
def coupling():
    return pd.read_csv(RES / "regional_coupling.csv")


@lru_cache(None)
def tissue_summary():
    return pd.read_csv(RES / "regional_tissue_summary.csv")


def split_maps(series, mask=None):
    """series: pd.Series indexed by var -> value. Returns dict with atlas-keyed maps:
    cortex_lh/rh (region -> v), wm (JHU idx -> v), hippo_lh/rh, wmh, heart_wt/ecc/err (seg -> v), heart_global."""
    d = dictionary().set_index("var")
    out = {k: {} for k in ["cortex_lh", "cortex_rh", "wm", "wm_mask", "hippo", "heart_wt", "heart_ecc", "heart_err",
                           "heart_global", "cortex_lh_mask", "cortex_rh_mask", "heart_wt_mask", "heart_ecc_mask",
                           "heart_err_mask"]}
    out["wmh"] = np.nan
    out["wmh_sig"] = True
    for var, val in series.items():
        sig = True if mask is None else bool(mask.get(var, True))
        if var == "WMH_burden_Z_raw":
            out["wmh"], out["wmh_sig"] = val, sig
            continue
        if var not in d.index:
            continue
        r = d.loc[var]
        if r.tissue == "Cortical thickness":
            if r.region == "GlobalMeanThickness":
                continue
            h = "lh" if r.hemi == "left" else "rh"
            out[f"cortex_{h}"][r.region] = val
            out[f"cortex_{h}_mask"][r.region] = sig
        elif r.tissue == "White-matter free water" and r.method == "TBSS":
            idx = A.jhu_label_for(r.region, r.hemi)
            if idx is not None:
                out["wm"][idx] = val
                out["wm_mask"][idx] = sig
        elif r.tissue == "Hippocampal volume":
            out["hippo"]["lh" if r.hemi == "left" else "rh"] = (val, sig)
        elif r.tissue in ("Wall thickness", "Circumferential strain", "Radial strain"):
            key = {"Wall thickness": "heart_wt", "Circumferential strain": "heart_ecc", "Radial strain": "heart_err"}[r.tissue]
            seg = int(str(r.region).replace("AHA", ""))
            out[key][seg] = val
            out[key + "_mask"][seg] = sig
        elif r.tissue == "Global CMR":
            out["heart_global"][r.measure] = (val, sig)
    return out


def stage_maps(sample="independent"):
    s = stage_means()
    s = s[s["sample"] == sample]
    res = {}
    for g in STAGES:
        x = s[s.stage_grp == g].set_index("var")
        res[g] = split_maps(x.mean_z)
        res[g]["n"] = int(x.n.max())
    return res


def coupling_maps(exposure="cardiac_comp", sample="independent", model="A", q=0.05):
    c = coupling()
    c = c[(c["sample"] == sample) & (c.model == model) & (c.exposure == exposure)].set_index("var")
    return split_maps(c.beta, mask=(c.q < q).to_dict()), c


if __name__ == "__main__":
    import anat3d
    m = stage_maps()["Stage >=3"]
    _, _, vreg, _ = anat3d.cortex("lh")
    mesh_regions = set(vreg) - {""}
    print("cortex lh mapped", len(m["cortex_lh"]), "missing on mesh:", sorted(set(m["cortex_lh"]) - mesh_regions),
          "mesh regions without data:", sorted(mesh_regions - set(m["cortex_lh"])))
    print("wm", len(m["wm"]), "hippo", m["hippo"], "wmh", round(m["wmh"], 3))
    print("heart wt/ecc/err", len(m["heart_wt"]), len(m["heart_ecc"]), len(m["heart_err"]), "global", len(m["heart_global"]))
