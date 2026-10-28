"""Subtype-specific regional anatomy of the heart-brain stage (v7).

Extends the regional analysis (regional_atlas.py, unchanged) in three ways:
  * adds imaging-derived phenotypes that are NOT inputs to any SuStaIn domain
    score ("out-of-model" measures): 16 AHA-segment wall thickness (already
    available), 12 FIRST subcortical grey-matter volumes, and white-matter
    fractional anisotropy (FA) and mean diffusivity (MD) in 60 tracts;
  * describes regional mean z by subtype x stage window (stage 0 is a shared
    reference because a subtype label carries no information at stage 0);
  * estimates the regional difference between the two subtypes at matched
    stage (z ~ subtype + stage + age + sex, stage >= 1) in the derivation
    sample (All-10 model) and, as a replication, in the independent sample
    (MRI-only model transferred without refitting; subtypes matched on their
    first event, the small third MRI-only subtype excluded), and tests
    heart-brain coupling for the out-of-model brain measures.
Normative z scores use exactly the same specification as regional_atlas.py.
Outputs are aggregate tables only (results/regional_subtype/); cells with
n < 20 are not reported.
Usage: python3 regional_subtype.py [n_bootstrap]
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from common import RAW, RES, PREP

OUT = RES / "regional_subtype"
OUT.mkdir(exist_ok=True)
BRAINRAW = RAW / "brainraw" / "原始脑区数据"
B_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 500
rng = np.random.default_rng(20261026)
ZCAP = 5.0
MIN_N = 20


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


def std(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


# ---------------------------------------------------------------- dictionary
dic = pd.read_csv(BRAINRAW / "brain_idps_587used_new.csv")
dic["field"] = dic["UKB ID"].astype(int)
dic["short"] = dic["IDP short name"].astype(str)
FIRST = {25011: "thalamus", 25012: "thalamus", 25013: "caudate", 25014: "caudate", 25015: "putamen",
         25016: "putamen", 25017: "pallidum", 25018: "pallidum", 25019: "hippocampus", 25020: "hippocampus",
         25021: "amygdala", 25022: "amygdala", 25023: "accumbens", 25024: "accumbens"}
WM_MEAS = {"WM tract ISOVF": ("ISOVF", "White-matter free water", 1, True),
           "WM tract FA": ("FA", "White-matter FA", -1, False),
           "WM tract diffusivity": ("MD", "White-matter MD", 1, False)}
rows = []
for _, r in dic.iterrows():
    s, cat, f = r["short"], r["Category name"], r["field"]
    if cat == "cortical thickness":
        rows.append(dict(var=r["IDP"], field=f, tissue="Cortical thickness", measure="thickness",
                         region=s.split("_thickness_")[1], hemi="left" if "_lh_" in s else "right",
                         method="FreeSurfer DK", sign=-1, in_model=True))
    elif cat in WM_MEAS:
        key, tissue, sg, inm = WM_MEAS[cat]
        tract = s.split(f"_{key}_")[1]
        hemi = "left" if tract[-2:] in ("_L", "_l") else ("right" if tract[-2:] in ("_R", "_r") else "midline")
        tract = tract[:-2] if hemi != "midline" else tract
        rows.append(dict(var=r["IDP"], field=f, tissue=tissue, measure=key, region=tract, hemi=hemi,
                         method="TBSS" if "_TBSS_" in s else "ProbtrackX", sign=sg, in_model=inm))
    elif f in FIRST:
        hip = FIRST[f] == "hippocampus"
        rows.append(dict(var=r["IDP"], field=f, tissue="Hippocampal volume" if hip else "Subcortical volume",
                         measure="volume", region=FIRST[f], hemi="left" if f % 2 == 1 else "right",
                         method="FIRST", sign=-1, in_model=hip))
bdict = pd.DataFrame(rows)
# ISOVF: 57 of 60 tracts load on the free-water domain score; flag the rest as out-of-model
bw = pd.read_csv(RAW / "ukb_area_up/brain_mri_domain_pca_weights.csv")
bw_fields = set(bw.variable.str.extract(r"IDP(\d+)")[0].dropna().astype(int))
iso = bdict.tissue == "White-matter free water"
bdict.loc[iso, "in_model"] = bdict.loc[iso, "field"].isin(bw_fields)

SEG = [f"AHA{i}" for i in range(1, 17)]
crow = []
for i, s in enumerate(SEG):
    crow += [dict(var=f"{24124+i}-2.0", field=24124 + i, tissue="Wall thickness", measure="wall thickness",
                  region=s, sign=1, in_model=False),
             dict(var=f"{24141+i}-2.0", field=24141 + i, tissue="Circumferential strain", measure="Ecc",
                  region=s, sign=1, in_model=True),
             dict(var=f"{24158+i}-2.0", field=24158 + i, tissue="Radial strain", measure="Err",
                  region=s, sign=-1, in_model=True)]
for f, lab in ((24120, "ascending"), (24123, "descending")):
    crow.append(dict(var=f"{f}-2.0", field=f, tissue="Aortic distensibility", measure="distensibility",
                     region=lab, sign=-1, in_model=True))
cdict = pd.DataFrame(crow)
cdict["hemi"], cdict["method"] = "", "CMR"
dictionary = pd.concat([bdict.assign(organ="brain"), cdict.assign(organ="heart")], ignore_index=True)
dictionary = pd.concat([dictionary, pd.DataFrame([dict(var="WMH_burden_Z_raw", field=25781, organ="brain",
                        tissue="White-matter hyperintensity", measure="WMH volume", region="whole brain",
                        hemi="", method="BIANCA", sign=1, in_model=True)])], ignore_index=True)
dictionary.to_csv(OUT / "dictionary_ext.csv", index=False)

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
for v in cdict.loc[cdict.measure.isin(["Ecc", "Err"]), "var"]:
    craw.loc[craw[v] == 0, v] = np.nan        # 0 = failed tracking in the UKB strain pipeline
craw = craw.merge(cov, on="eid", how="left")
ref_c = craw.eid.isin(ref_ids).to_numpy()
zc = normative_z(craw, cdict["var"].tolist(), ref_c, ["Age", "Sex"], dict(zip(cdict["var"], cdict["sign"])))
zc.insert(0, "eid", craw.eid.values)
dz = pd.read_csv(PREP / "02_normative_models/all_normative_zscores.csv",
                 usecols=["eid", "LV_remodeling_Z_raw", "LV_myocardial_dysfunction_Z_raw", "Aortic_stiffness_Z_raw",
                          "WMH_burden_Z_raw"])

COV = ["age_imaging", "male", "SBP", "Diabetes", "Hypertension", "current_smoker", "BMI"]
der = pd.read_pickle(RES / "cohort_derivation.pkl")
der = der[["eid", "all10_subtype", "all10_stage", "all10_p_subtype"] + COV].rename(
    columns={"all10_subtype": "sub", "all10_stage": "stg", "all10_p_subtype": "psub"})
ind = pd.read_pickle(RES / "validation/independent_cohort_full_imaging.pkl")
ind = ind[["eid", "subtype", "stage", "p_subtype"] + COV].rename(
    columns={"subtype": "sub", "stage": "stg", "p_subtype": "psub"})
# MRI-only subtype 1 starts with LV dysfunction and subtype 2 with LV remodelling (as All-10 subtypes 1 and 2);
# the third MRI-only subtype (n=281) has no All-10 counterpart and is excluded from subtype-specific analyses.


def frame(base):
    d = base.merge(dz, on="eid", how="left").merge(zb, on="eid", how="left").merge(zc, on="eid", how="left")
    d["cardiac_comp"] = std(d[["LV_remodeling_Z_raw", "LV_myocardial_dysfunction_Z_raw",
                               "Aortic_stiffness_Z_raw"]].mean(1))
    d["cell"] = np.select([d.stg == 0, (d["sub"] == 1) & d.stg.between(1, 2), (d["sub"] == 1) & (d.stg >= 3),
                           (d["sub"] == 2) & d.stg.between(1, 2), (d["sub"] == 2) & (d.stg >= 3)],
                          ["Stage 0", "Subtype 1, stage 1-2", "Subtype 1, stage >=3",
                           "Subtype 2, stage 1-2", "Subtype 2, stage >=3"], default="other")
    return d


D = {"derivation": frame(der), "independent": frame(ind)}
VARS = dictionary["var"].tolist()
CELLS = ["Stage 0", "Subtype 1, stage 1-2", "Subtype 1, stage >=3", "Subtype 2, stage 1-2", "Subtype 2, stage >=3"]

# ---------------------------------------------------------------- 1. subtype x stage cell means
meta, out = [], []
for samp, d in D.items():
    for c in CELLS:
        g = d[d.cell == c]
        meta.append(dict(sample=samp, cell=c, n=len(g),
                         psub_median=np.nan if c == "Stage 0" else g.psub.median(),
                         psub_q1=np.nan if c == "Stage 0" else g.psub.quantile(.25),
                         psub_q3=np.nan if c == "Stage 0" else g.psub.quantile(.75),
                         stage_median=g.stg.median()))
        for v in VARS:
            x = g[v].dropna()
            if len(x) < MIN_N:
                continue
            se = x.std(ddof=1) / np.sqrt(len(x))
            out.append(dict(sample=samp, cell=c, var=v, n=len(x), mean_z=x.mean(), lo=x.mean() - 1.96 * se,
                            hi=x.mean() + 1.96 * se))
pd.DataFrame(meta).to_csv(OUT / "cell_meta.csv", index=False)
cm = pd.DataFrame(out).merge(dictionary[["var", "organ", "tissue", "measure", "region", "hemi", "method",
                                         "in_model"]], on="var", how="left")
cm.to_csv(OUT / "cell_means.csv", index=False)


# ---------------------------------------------------------------- 2. subtype contrast at matched stage
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
    base = d[(d.stg >= 1) & d["sub"].isin([1, 2])]
    for analysis, g in (("primary", base), ("p_subtype>=0.70", base[base.psub >= 0.70]),
                        ("p_subtype>=0.60", base[base.psub >= 0.60])):
        x = (g["sub"] == 1).astype(float).to_numpy()
        C = g[["stg", "age_imaging", "male"]].to_numpy(float)
        for v in VARS:
            b, se, p, n = ols_beta(g[v].to_numpy(float), x, C)
            rows.append(dict(sample=samp, analysis=analysis, var=v, beta=b, lo=b - 1.96 * se, hi=b + 1.96 * se,
                             p=p, n=n, n_s1=int((g["sub"] == 1).sum()), n_s2=int((g["sub"] == 2).sum())))
sc = pd.DataFrame(rows)
sc["q"] = np.nan
for k, g in sc.groupby(["sample", "analysis"]):
    sc.loc[g.index, "q"] = bh(g.p)
sc = sc.merge(dictionary[["var", "organ", "tissue", "measure", "region", "hemi", "method", "in_model"]],
              on="var", how="left")
sc.to_csv(OUT / "subtype_contrast.csv", index=False)

# ---------------------------------------------------------------- 3. cross-sample replication of the contrast map
pr = sc[sc.analysis == "primary"].pivot_table(index="var", columns="sample", values="beta")
pr = pr.join(dictionary.set_index("var")[["organ", "tissue", "in_model"]])
rep = []
for lab, g in [("all regions", pr), ("out-of-model", pr[~pr.in_model.astype(bool)]),
               ("in-model", pr[pr.in_model.astype(bool)])] + [(t, g) for t, g in pr.groupby("tissue")]:
    g = g.dropna(subset=["derivation", "independent"])
    if len(g) < 3:
        continue
    r, p = stats.spearmanr(g.derivation, g.independent)
    sign = float(np.mean(np.sign(g.derivation) == np.sign(g.independent)))
    rep.append(dict(set=lab, n_regions=len(g), spearman_rho=r, p=p, sign_concordance=sign))
pd.DataFrame(rep).to_csv(OUT / "contrast_replication.csv", index=False)

# ---------------------------------------------------------------- 4. coupling of out-of-model brain measures
COV_A = ["age_imaging", "male"]
COV_B = COV_A + ["SBP", "Diabetes", "Hypertension", "current_smoker", "BMI"]
BR = bdict["var"].tolist()
rows = []
for samp, d in D.items():
    x = std(d["cardiac_comp"].to_numpy(float))
    for model, C in (("A", COV_A), ("B", COV_B)):
        Cm = d[C].to_numpy(float)
        for v in BR:
            b, se, p, n = ols_beta(d[v].to_numpy(float), x, Cm)
            rows.append(dict(sample=samp, model=model, var=v, beta=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=p, n=n))
cp = pd.DataFrame(rows)
cp["q"] = np.nan
for k, g in cp.groupby(["sample", "model"]):
    cp.loc[g.index, "q"] = bh(g.p)
cp = cp.merge(dictionary[["var", "tissue", "measure", "region", "hemi", "method", "in_model"]], on="var", how="left")
cp.to_csv(OUT / "coupling_ext.csv", index=False)

CLASSES = ["Cortical thickness", "Hippocampal volume", "White-matter free water", "Subcortical volume",
           "White-matter FA", "White-matter MD"]
tissue_of = dict(zip(bdict["var"], bdict["tissue"]))
summ = []
for samp, d in D.items():
    for model, C in (("A", COV_A), ("B", COV_B)):
        X = np.column_stack([np.ones(len(d)), std(d["cardiac_comp"].to_numpy(float)), d[C].to_numpy(float)])
        Y = d[BR].to_numpy(float)
        ok = np.all(np.isfinite(X), 1) & np.all(np.isfinite(Y), 1)
        X, Y = X[ok], Y[ok]
        mask = {c: np.array([tissue_of[v] == c for v in BR]) for c in CLASSES}

        def med(Xs, Ys):
            B = np.linalg.solve(Xs.T @ Xs, Xs.T @ Ys)[1]
            return {c: np.median(B[mask[c]]) for c in CLASSES}
        est = med(X, Y)
        boot = pd.DataFrame([med(X[i], Y[i]) for i in (rng.integers(0, len(X), len(X)) for _ in range(B_BOOT))])
        for c in CLASSES:
            summ.append(dict(sample=samp, model=model, contrast=c, n_regions=int(mask[c].sum()),
                             median_beta=est[c], lo=boot[c].quantile(.025), hi=boot[c].quantile(.975),
                             n_complete=len(X)))
        for a in ("White-matter FA", "White-matter MD", "Subcortical volume"):
            diff = boot[a] - boot["Cortical thickness"]
            summ.append(dict(sample=samp, model=model, contrast=f"{a} minus cortical thickness",
                             median_beta=est[a] - est["Cortical thickness"], lo=diff.quantile(.025),
                             hi=diff.quantile(.975), n_complete=len(X),
                             p_boot=2 * min((diff <= 0).mean(), (diff >= 0).mean())))
ts = pd.DataFrame(summ)
ts.to_csv(OUT / "tissue_summary_ext.csv", index=False)

# ---------------------------------------------------------------- console summary
print(pd.DataFrame(meta).round(3).to_string())
s = sc[sc.analysis == "primary"].copy()
s["sig"] = s.q < 0.05
print(s.groupby(["sample", "tissue"]).agg(n=("beta", "size"), median_beta=("beta", "median"),
                                          n_sig=("sig", "sum"), n_pos_sig=("beta", lambda b: 0)).round(3).to_string())
for samp in D:
    t = s[s["sample"] == samp]
    print(samp, "S1>S2 sig:", int(((t.q < .05) & (t.beta > 0)).sum()), "S2>S1 sig:", int(((t.q < .05) & (t.beta < 0)).sum()))
print(pd.DataFrame(rep).round(3).to_string())
print(ts.round(4).to_string())
