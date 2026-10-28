"""Regional (sub-domain) anatomy of the heart-brain stage.

Rebuilds region-level normative z scores for the imaging measures that make
up the seven MRI domain scores (68 Desikan-Killiany cortical-thickness
measures, 60 white-matter tract ISOVF measures, left/right hippocampal
volume, 16 AHA-segment wall thickness and circumferential and radial strain
values, global CMR measures, and aortic distensibility) and describes, in
the independent imaging sample that was not used for model fitting:

  1. regional mean z by heart-brain stage group (0, 1-2, >=3);
  2. regional heart-brain coupling: association of each brain region with
     a cardiac composite (mean of the three cardiac domain z scores), and of
     each myocardial segment with a brain white-matter composite (mean of
     WMH and free-water z) and a grey-matter composite (cortical thinning and
     hippocampal atrophy z), adjusted for age and sex (model A) and further
     for vascular risk factors (model B);
  3. tissue-class summaries with participant-level bootstrap CIs.

Normative models mirror the domain scores used for SuStaIn: ordinary least
squares on age and sex (plus intracranial volume for hippocampal volume) in
the baseline disease-free reference group, residuals divided by the
reference residual SD, oriented so that higher = more abnormal, truncated
at +/-5. Outputs are aggregate tables only (results/regional/).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from common import RAW, RES, UPLOAD, PREP

OUT = RES / "regional"
OUT.mkdir(exist_ok=True)
BRAINRAW = RAW / "brainraw" / "原始脑区数据"
B_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 500
rng = np.random.default_rng(20261025)
ZCAP = 5.0


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        q[ok] = stats.false_discovery_control(p[ok])
    return q


def normative_z(df, cols, ref, covars, sign):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in covars])
    xok = np.all(np.isfinite(X), 1)
    out = {}
    for c in cols:
        y = df[c].to_numpy(float)
        ok = np.isfinite(y) & xok
        r = ok & ref
        b, *_ = np.linalg.lstsq(X[r], y[r], rcond=None)
        res = y - X @ b
        sd = np.std(res[r], ddof=X.shape[1])
        z = np.where(ok, res / sd * sign[c], np.nan)
        out[c] = np.clip(z, -ZCAP, ZCAP)
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- dictionary
dic = pd.read_csv(BRAINRAW / "brain_idps_587used_new.csv")
dic["field"] = dic["UKB ID"].astype(int)
dic["short"] = dic["IDP short name"].astype(str)
rows = []
for _, r in dic.iterrows():
    s, cat = r["short"], r["Category name"]
    if cat == "cortical thickness":
        hemi = "left" if "_lh_" in s else "right"
        reg = s.split("_thickness_")[1]
        rows.append(dict(var=r["IDP"], field=r["field"], tissue="Cortical thickness",
                         measure="thickness", region=reg, hemi=hemi, method="FreeSurfer DK",
                         sign=-1, label=r["IDP_name"]))
    elif cat == "WM tract ISOVF":
        meth = "TBSS" if "_TBSS_" in s else "ProbtrackX"
        tract = s.split("_ISOVF_")[1]
        hemi = "left" if tract[-2:] in ("_L", "_l") else ("right" if tract[-2:] in ("_R", "_r") else "midline")
        tract = tract[:-2] if hemi != "midline" else tract
        rows.append(dict(var=r["IDP"], field=r["field"], tissue="White-matter free water",
                         measure="ISOVF", region=tract, hemi=hemi, method=meth,
                         sign=1, label=r["IDP_name"]))
    elif r["field"] in (25019, 25020):
        rows.append(dict(var=r["IDP"], field=r["field"], tissue="Hippocampal volume",
                         measure="volume", region="hippocampus",
                         hemi="left" if r["field"] == 25019 else "right", method="FIRST",
                         sign=-1, label=r["IDP_name"]))
bdict = pd.DataFrame(rows)

# CMR fields (instance 2): AHA-segment wall thickness, circumferential and radial strain
SEG = [f"AHA{i}" for i in range(1, 17)]
crow = []
for i, s in enumerate(SEG):
    crow.append(dict(var=f"{24124+i}-2.0", field=24124 + i, tissue="Wall thickness", measure="wall thickness",
                     region=s, sign=1))
    crow.append(dict(var=f"{24141+i}-2.0", field=24141 + i, tissue="Circumferential strain", measure="Ecc",
                     region=s, sign=1))
    crow.append(dict(var=f"{24158+i}-2.0", field=24158 + i, tissue="Radial strain", measure="Err",
                     region=s, sign=-1))
GLOBAL_CMR = {24100: ("LV end-diastolic volume", 1), 24101: ("LV end-systolic volume", 1),
              24105: ("LV myocardial mass", 1), 24140: ("LV mean wall thickness", 1),
              24103: ("LV ejection fraction", -1), 24102: ("LV stroke volume", -1),
              24104: ("LV cardiac output", -1), 24157: ("Global circumferential strain", 1),
              24174: ("Global radial strain", -1), 24181: ("Global longitudinal strain", 1),
              24120: ("Ascending aortic distensibility", -1), 24123: ("Descending aortic distensibility", -1)}
for f, (lab, sg) in GLOBAL_CMR.items():
    crow.append(dict(var=f"{f}-2.0", field=f, tissue="Global CMR", measure=lab, region="global", sign=sg))
cdict = pd.DataFrame(crow)
cdict["hemi"] = ""
cdict["method"] = "CMR"
cdict["label"] = cdict["measure"] + " " + cdict["region"]

# domain loadings (composition of the SuStaIn domain scores)
bw = pd.read_csv(RAW / "ukb_area_up/brain_mri_domain_pca_weights.csv")
bw["field"] = bw.variable.str.extract(r"IDP(\d+)")[0].astype(float)
bw = bw.dropna(subset=["field"])
bw["field"] = bw.field.astype(int)
bdict = bdict.merge(bw[["field", "score", "loading", "weight"]], on="field", how="left")
cw = pd.read_csv(RAW / "ukb_area_up/cardio_mri_pca_weights_complete_case_instance2.csv")
cw["field"] = cw.variable.str.extract(r"X(\d+)_")[0].astype(float)
cw = cw.dropna(subset=["field"]).drop_duplicates("field")
cw["field"] = cw.field.astype(int)
cdict = cdict.merge(cw[["field", "score", "loading", "weight"]], on="field", how="left")
dictionary = pd.concat([bdict.assign(organ="brain"), cdict.assign(organ="heart")], ignore_index=True)
dictionary.to_csv(OUT / "regional_dictionary.csv", index=False)

# ---------------------------------------------------------------- normative z
ref_ids = set(pd.read_csv(PREP / "01_reference/reference_primary_eids.csv").eid.astype(int))
cov = pd.read_csv(RAW / "ukb_area/data_all.csv", usecols=lambda c: c in ("eid", "Eid", "Age", "Sex"))
cov = cov.rename(columns={"Eid": "eid"})

braw = pd.read_csv(BRAINRAW / "data_587idp_v2.csv", usecols=["Eid", "ICV"] + bdict["var"].tolist())
braw = braw.rename(columns={"Eid": "eid"}).merge(cov, on="eid", how="left")
ref_b = braw.eid.isin(ref_ids).to_numpy()
sign_b = dict(zip(bdict["var"], bdict["sign"]))
vol = bdict.loc[bdict.measure == "volume", "var"].tolist()
oth = [v for v in bdict["var"] if v not in vol]
zb = pd.concat([normative_z(braw, oth, ref_b, ["Age", "Sex"], sign_b),
                normative_z(braw, vol, ref_b, ["Age", "Sex", "ICV"], sign_b)], axis=1)
zb.insert(0, "eid", braw.eid.values)

craw = pd.read_csv(RAW / "cmr/cmr82.csv", usecols=["eid"] + cdict["var"].tolist())
craw = craw.dropna(subset=cdict["var"].tolist(), how="all")
strain = cdict.loc[cdict.measure.isin(["Ecc", "Err"]), "var"].tolist()
for v in strain:                       # 0 = failed tracking in the UKB strain pipeline
    craw.loc[craw[v] == 0, v] = np.nan
craw = craw.merge(cov, on="eid", how="left")
ref_c = craw.eid.isin(ref_ids).to_numpy()
zc = normative_z(craw, cdict["var"].tolist(), ref_c, ["Age", "Sex"], dict(zip(cdict["var"], cdict["sign"])))
zc.insert(0, "eid", craw.eid.values)
print("normative reference: brain", ref_b.sum(), "of", len(braw), "| CMR", ref_c.sum(), "of", len(craw))

# domain-level raw z (unfloored) for composites
dz = pd.read_csv(PREP / "02_normative_models/all_normative_zscores.csv",
                 usecols=["eid", "LV_remodeling_Z_raw", "LV_myocardial_dysfunction_Z_raw", "Aortic_stiffness_Z_raw",
                          "Hippocampal_atrophy_Z_raw", "Cortical_thinning_Z_raw", "WMH_burden_Z_raw",
                          "FreeWater_ISOVF_Z_raw"])


def std(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def analysis_frame(base, stage_col):
    d = base.merge(dz, on="eid", how="left")
    d["cardiac_comp"] = std(d[["LV_remodeling_Z_raw", "LV_myocardial_dysfunction_Z_raw", "Aortic_stiffness_Z_raw"]].mean(1))
    d["brain_wm_comp"] = std(d[["WMH_burden_Z_raw", "FreeWater_ISOVF_Z_raw"]].mean(1))
    d["brain_gm_comp"] = std(d[["Cortical_thinning_Z_raw", "Hippocampal_atrophy_Z_raw"]].mean(1))
    d["stage_grp"] = pd.cut(d[stage_col], [-1, 0, 2, 99], labels=["Stage 0", "Stage 1-2", "Stage >=3"])
    d = d.merge(zb, on="eid", how="left").merge(zc, on="eid", how="left")
    return d


ind = pd.read_pickle(RES / "validation/independent_cohort_full_imaging.pkl")
ind = ind[["eid", "stage", "age_imaging", "male", "SBP", "Diabetes", "Hypertension", "current_smoker", "BMI"]]
der = pd.read_pickle(RES / "cohort_derivation.pkl")
der = der[["eid", "all10_stage", "age_imaging", "male", "SBP", "Diabetes", "Hypertension", "current_smoker", "BMI"]]
D = {"independent": analysis_frame(ind, "stage"), "derivation": analysis_frame(der, "all10_stage")}
REG_B = bdict["var"].tolist() + ["WMH_burden_Z_raw"]
REG_C = cdict["var"].tolist()
LAB = dict(zip(dictionary["var"], dictionary["label"]))
LAB["WMH_burden_Z_raw"] = "White-matter hyperintensity volume"

# ---------------------------------------------------------------- 1. stage means
out = []
for samp, d in D.items():
    for v in REG_B + REG_C:
        for g, gd in d.groupby("stage_grp", observed=True):
            x = gd[v].dropna()
            if len(x) < 20:
                continue
            out.append(dict(sample=samp, var=v, stage_grp=g, n=len(x), mean_z=x.mean(),
                            lo=x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x)),
                            hi=x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x)),
                            pct_z1=100 * (x >= 1).mean()))
pd.DataFrame(out).to_csv(OUT / "regional_stage_means.csv", index=False)

# ---------------------------------------------------------------- 2. coupling
COV_A = ["age_imaging", "male"]
COV_B = COV_A + ["SBP", "Diabetes", "Hypertension", "current_smoker", "BMI"]


def ols_beta(y, x, C):
    X = np.column_stack([np.ones(len(y)), x, C])
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    X, y = X[ok], y[ok]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / (len(y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    p = 2 * stats.t.sf(abs(b[1] / se), len(y) - X.shape[1])
    return b[1], se, p, int(ok.sum())


rows = []
for samp, d in D.items():
    for model, C in (("A", COV_A), ("B", COV_B)):
        Cm = d[C].to_numpy(float)
        for expo, targets in (("cardiac_comp", REG_B), ("LV_remodeling_Z_raw", REG_B),
                              ("LV_myocardial_dysfunction_Z_raw", REG_B), ("Aortic_stiffness_Z_raw", REG_B),
                              ("brain_wm_comp", REG_C), ("brain_gm_comp", REG_C)):
            x = std(d[expo].to_numpy(float))
            for v in targets:
                b, se, p, n = ols_beta(d[v].to_numpy(float), x, Cm)
                rows.append(dict(sample=samp, model=model, exposure=expo, var=v, label=LAB.get(v, v),
                                 beta=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=p, n=n))
cp = pd.DataFrame(rows)
cp["q"] = np.nan
for k, g in cp.groupby(["sample", "model", "exposure"]):
    cp.loc[g.index, "q"] = bh(g.p)
cp = cp.merge(dictionary[["var", "organ", "tissue", "measure", "region", "hemi", "method"]], on="var", how="left")
cp.loc[cp["var"] == "WMH_burden_Z_raw", ["organ", "tissue", "measure", "region", "hemi", "method"]] = \
    ["brain", "White-matter hyperintensity", "WMH volume", "whole brain", "", "BIANCA"]
cp.to_csv(OUT / "regional_coupling.csv", index=False)

# ---------------------------------------------------------------- 3. tissue-class bootstrap
CLASSES = ["Cortical thickness", "Hippocampal volume", "White-matter free water"]


def class_medians(d, C, cols, tissue_of):
    X = np.column_stack([np.ones(len(d)), std(d["cardiac_comp"].to_numpy(float)), d[C].to_numpy(float)])
    Y = d[cols].to_numpy(float)
    ok = np.all(np.isfinite(X), 1) & np.all(np.isfinite(Y), 1)
    X, Y = X[ok], Y[ok]
    n = len(X)

    def med(Xs, Ys):
        B = np.linalg.solve(Xs.T @ Xs, Xs.T @ Ys)[1]
        return {c: np.median(B[[tissue_of[k] == c for k in cols]]) for c in CLASSES}

    est = med(X, Y)
    boot = []
    for _ in range(B_BOOT):
        i = rng.integers(0, n, n)
        boot.append(med(X[i], Y[i]))
    boot = pd.DataFrame(boot)
    return est, boot, n


tissue_of = dict(zip(bdict["var"], bdict["tissue"]))
cols = bdict["var"].tolist()
summ = []
for samp, d in D.items():
    for model, C in (("A", COV_A), ("B", COV_B)):
        est, boot, n = class_medians(d, C, cols, tissue_of)
        for c in CLASSES:
            summ.append(dict(sample=samp, model=model, contrast=c, median_beta=est[c],
                             lo=boot[c].quantile(.025), hi=boot[c].quantile(.975), n_complete=n))
        diff = boot["White-matter free water"] - boot["Cortical thickness"]
        summ.append(dict(sample=samp, model=model, contrast="WM free water minus cortical thickness",
                         median_beta=est["White-matter free water"] - est["Cortical thickness"],
                         lo=diff.quantile(.025), hi=diff.quantile(.975), n_complete=n,
                         p_boot=2 * min((diff <= 0).mean(), (diff >= 0).mean())))
pd.DataFrame(summ).to_csv(OUT / "regional_tissue_summary.csv", index=False)

# WMH is a single whole-brain measure: report its own beta alongside the class medians
print(pd.DataFrame(summ).round(4).to_string())
w = cp[(cp["var"] == "WMH_burden_Z_raw") & (cp.exposure == "cardiac_comp")]
print(w[["sample", "model", "beta", "lo", "hi", "n"]].round(4).to_string())
print("n regions brain", len(REG_B), "cardiac", len(REG_C))
