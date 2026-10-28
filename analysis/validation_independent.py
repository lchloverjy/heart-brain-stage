import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Independent-sample validation of the MRI-only heart-brain stage (UK Biobank imaging participants NOT in the
derivation cohort; no proteomics). The derivation MRI-only model (fitted in n=3,770) is transferred without refitting.

Design (no individual imaging dates are available for this sample):
  (i)  construct validity  - stage vs baseline risk factors and SCORE2 (baseline age);
  (ii) pre-imaging disease - HES diagnoses dated before 2014-08-01 (start of UK Biobank imaging) are certainly
       prevalent at imaging -> logistic association with stage;
  (iii) common-landmark prospective analysis - time zero 2020-04-01 (after the March 2020 suspension of imaging
       visits; assumes all imaging in this release preceded the landmark), follow-up to 2021-09-30, participants with
       any diagnosis of the outcome on/before the landmark excluded.
Also quantifies heart-brain co-abnormality (cross-organ odds ratios) in the full imaging sample."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
import sys
import common as C
DESIGN = sys.argv[1] if len(sys.argv) > 1 else "landmark"     # landmark | imaging
from restage import restage

warnings.filterwarnings("ignore")
OUT = C.RES / "validation"; OUT.mkdir(exist_ok=True)
IMG_START = pd.Timestamp("2014-08-01")
LANDMARK = pd.Timestamp("2020-04-01")
M = "mri_only_matched_all10"

# ---------------------------------------------------------------- sample
full = pd.read_csv(C.PREP / "03_inputs" / "sustain_mri_only.csv")
deriv = set(C.load_inputs("all10").eid)
ind = full[~full.eid.isin(deriv)].reset_index(drop=True)
th = C.load_thresholds(M)
print("full imaging sample", len(full), "independent", len(ind))

# ---------------------------------------------------------------- transfer staging (chunked)
stg_path = OUT / "independent_restaged.pkl"
if stg_path.exists():
    stg = pd.read_pickle(stg_path)
else:
    X = ind[th.biomarker].to_numpy(float)
    parts = []
    for i in range(0, len(X), 2000):
        r = restage(M, X[i:i + 2000], n_samples=1000)
        exp_stage = (r["prob_stage"] * np.arange(r["prob_stage"].shape[1])).sum(1)
        parts.append(pd.DataFrame(dict(eid=ind.eid.values[i:i + 2000], stage=r["ml_stage"], subtype=r["ml_subtype"] + 1,
                                       p_subtype=r["prob_ml_subtype"], p_stage=r["prob_ml_stage"], exp_stage=exp_stage)))
        print("  staged", i + len(parts[-1]), flush=True)
    stg = pd.concat(parts, ignore_index=True)
    stg.to_pickle(stg_path)
NEV = C.MODELS[M]["n_events"]
stg["stage_norm"] = stg.stage / NEV
stg["p10"] = stg.stage_norm * 10
stg["grp"] = pd.cut(stg.stage, [-1, 0, 1, 2, 99], labels=["0", "1", "2", ">=3"])

# model-implied organ involvement in the transferred model
seq = pd.read_csv(C.FINAL / M / "selected_event_sequences.csv")
ev = C.event_table(M)

# ---------------------------------------------------------------- heart-brain co-abnormality (cross-organ ORs)
H = ["LV_remodeling_Z", "LV_myocardial_dysfunction_Z", "Aortic_stiffness_Z"]
B = ["Hippocampal_atrophy_Z", "Cortical_thinning_Z", "WMH_burden_Z", "FreeWater_ISOVF_Z"]
rows = []
for sample, df in [("full", full), ("independent", ind)]:
    for h in H:
        for b in B:
            for z in (1, 2):
                a_ = (df[h] >= z); b_ = (df[b] >= z)
                n11 = int((a_ & b_).sum()); n10 = int((a_ & ~b_).sum()); n01 = int((~a_ & b_).sum()); n00 = int((~a_ & ~b_).sum())
                orr = (n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5)); se = np.sqrt(1 / (n11 + .5) + 1 / (n10 + .5) + 1 / (n01 + .5) + 1 / (n00 + .5))
                rho = df[[h, b]].corr("spearman").iloc[0, 1]
                rows.append(dict(sample=sample, heart=h, brain=b, z=z, n=len(df), n11=n11, OR=orr, lo=np.exp(np.log(orr) - 1.96 * se),
                                 hi=np.exp(np.log(orr) + 1.96 * se), spearman=rho, p_brain_given_heart=n11 / max(n11 + n10, 1),
                                 p_brain_given_noheart=n01 / max(n01 + n00, 1)))
    hz = df[H].max(1); bz = df[B].max(1)
    for z in (1, 2):
        a_ = hz >= z; b_ = bz >= z
        n11 = int((a_ & b_).sum()); n10 = int((a_ & ~b_).sum()); n01 = int((~a_ & b_).sum()); n00 = int((~a_ & ~b_).sum())
        orr = n11 * n00 / (n10 * n01); se = np.sqrt(1 / n11 + 1 / n10 + 1 / n01 + 1 / n00)
        rows.append(dict(sample=sample, heart="any_heart", brain="any_brain", z=z, n=len(df), n11=n11, OR=orr,
                         lo=np.exp(np.log(orr) - 1.96 * se), hi=np.exp(np.log(orr) + 1.96 * se),
                         spearman=pd.concat([hz, bz], axis=1).corr("spearman").iloc[0, 1],
                         p_brain_given_heart=n11 / (n11 + n10), p_brain_given_noheart=n01 / (n01 + n00)))
    # dose-response: number of abnormal heart markers vs mean number of abnormal brain markers
    nh = (df[H] >= 1).sum(1); nb = (df[B] >= 1).sum(1)
    for k in range(4):
        sel = nh == k
        rows.append(dict(sample=sample, heart=f"n_heart_abnormal={k}", brain="n_brain_abnormal_mean", z=1, n=int(sel.sum()),
                         OR=np.nan, spearman=nb[sel].mean(), p_brain_given_heart=(nb[sel] >= 1).mean()))
cooc = pd.DataFrame(rows); cooc.to_csv(OUT / "heart_brain_coabnormality.csv", index=False)

# ---------------------------------------------------------------- covariates and HES
cov = pd.read_csv(C.RAW / "ukb_area/data_all.csv", usecols=["Eid", "Sex", "Age", "Smoking_status", "Diabetes", "Hypertension",
                                                             "CHOL", "HDLc", "SBP", "BMI", "Ethnicity"]).rename(columns={"Eid": "eid"})
d = stg.merge(cov, on="eid", how="left")
d["male"] = (d.Sex == 1).astype(int)
d["current_smoker"] = np.where(d.Smoking_status < 0, np.nan, (d.Smoking_status == 2).astype(float))
r_, _ = C.score2(d.Age, d.male == 1, d.current_smoker, d.SBP, d.Diabetes, d.CHOL, d.HDLc)
d["score2_baseline"] = r_

hes_path = C.ROOT / "analysis/hes_long_independent.pkl"
if hes_path.exists():
    long = pd.read_pickle(hes_path)
else:
    ids = set(d.eid)
    codes = [c for c in pd.read_csv(C.UPLOAD / "41270.csv", nrows=0).columns if c != "Eid"]
    parts = []
    for cc, dd in zip(pd.read_csv(C.UPLOAD / "41270.csv", chunksize=20000, dtype=str),
                      pd.read_csv(C.UPLOAD / "41280.csv", chunksize=20000, dtype=str)):
        cc = cc[cc.Eid.astype(int).isin(ids)]; dd = dd[dd.Eid.astype(int).isin(ids)]
        if len(cc) == 0:
            continue
        assert (cc.Eid.values == dd.Eid.values).all()
        a = cc.melt(id_vars="Eid", var_name="k", value_name="code"); a["k"] = a.k.str.split(".").str[1]
        b = dd.melt(id_vars="Eid", var_name="k", value_name="date"); b["k"] = b.k.str.split(".").str[1]
        m = a.merge(b, on=["Eid", "k"]).dropna(subset=["code", "date"])
        parts.append(m[["Eid", "code", "date"]])
    long = pd.concat(parts, ignore_index=True)
    long["Eid"] = long.Eid.astype(int); long["date"] = pd.to_datetime(long.date)
    long.to_pickle(hes_path)
long = long.rename(columns={"Eid": "eid"})
print("HES rows", len(long), "max date", long.date.max())


deaths = C.load_deaths()
d["death_date"] = d.eid.map(deaths.set_index("eid").death_date)
_by = deaths.set_index("eid")
d["birth"] = pd.to_datetime(dict(year=d.eid.map(_by.birth_year), month=d.eid.map(_by.birth_month), day=15), errors="coerce")


def first_date(keys):
    return C.first_event_date(long, deaths, keys)


OUTS = {"hbve": C.COMPOSITES["hbve"], "mace": C.COMPOSITES["mace"], "cerebrovascular": C.COMPOSITES["cerebrovascular"],
        "cardiac": C.COMPOSITES["cardiac"], "stroke": ["stroke"], "ihd": ["ihd"], "af": ["af"], "hf": ["hf"],
        "depression": ["depression"], "parkinson": ["parkinson"], "ckd": ["ckd"]}
for o, keys in OUTS.items():
    fd = first_date(keys)
    d[f"first_{o}"] = d.eid.map(fd)
d["first_cv_death"] = d.eid.map(C.death_event_date(deaths, C.CV_DEATH_PREFIX))
d["first_mace_cvd"] = d[["first_mace", "first_cv_death"]].min(axis=1)
OUTS["cv_death"] = []; OUTS["mace_cvd"] = []
T0 = LANDMARK
TAG = ""
if DESIGN == "imaging":
    img = C.load_imaging_dates()
    assert img is not None, "no field-53 export found in UKB_DATA"
    d["date_imaging"] = d.eid.map(img.set_index("eid").date_imaging)
    d["age_imaging"] = (d.date_imaging - d.birth).dt.days / C.YEAR
    print("imaging dates matched", int(d.date_imaging.notna().sum()), "of", len(d), "| range", d.date_imaging.min(), d.date_imaging.max())
    T0 = d.date_imaging
    TAG = "_imaging"
    # same conventions as the derivation sample: age and SCORE2 at the imaging visit (recruitment risk factors)
    d["Age_recruit"] = d.Age
    d["Age"] = d.age_imaging
    r_, _ = C.score2(d.age_imaging, d.male == 1, d.current_smoker, d.SBP, d.Diabetes, d.CHOL, d.HDLc)
    d["score2_baseline"] = r_
    d["score2_cat"] = C.score2_category(d.age_imaging, r_)

for c in ["Age", "SBP", "CHOL", "HDLc", "BMI"]:
    d[c + "_s"] = (d[c] - d[c].mean()) / d[c].std()
ADJ1 = ["Age_s", "male"]
ADJ2 = ADJ1 + ["SBP_s", "current_smoker", "Diabetes", "CHOL_s", "HDLc_s", "BMI_s", "Hypertension"]

# (i) construct validity: stage vs risk factors (linear, standardized stage)
cv = []
z = (d.stage_norm - d.stage_norm.mean()) / d.stage_norm.std()
for v in ["Age", "male", "SBP", "current_smoker", "Diabetes", "Hypertension", "BMI", "CHOL", "HDLc", "score2_baseline"]:
    f = pd.DataFrame({"y": z, "x": d[v], "age": d.Age, "sex": d.male}).dropna()
    xv = f.x if f.x.nunique() <= 2 else (f.x - f.x.mean()) / f.x.std()
    X = pd.DataFrame({"x": xv}) if v in ("Age", "male", "score2_baseline") else pd.DataFrame({"x": xv, "age": f.age, "male": f.sex})
    res = sm.OLS(f.y, sm.add_constant(X)).fit()
    cv.append(dict(variable=v, beta=res.params["x"], lo=res.conf_int().loc["x", 0], hi=res.conf_int().loc["x", 1], p=res.pvalues["x"], n=len(f)))
pd.DataFrame(cv).to_csv(OUT / f"independent_stage_correlates{TAG}.csv", index=False)

# (ii) pre-imaging disease
pre = []
for o in [k for k in OUTS if k not in ("cv_death", "mace_cvd")]:
    d[f"pre_{o}"] = ((d[f"first_{o}"] <= d.date_imaging) if DESIGN == "imaging" else (d[f"first_{o}"] < IMG_START)).astype(int)
    f = d[[f"pre_{o}", "p10"] + ADJ1].dropna()
    res = sm.Logit(f[f"pre_{o}"], sm.add_constant(f[["p10"] + ADJ1])).fit(disp=0)
    ci = res.conf_int().loc["p10"]
    pre.append(dict(outcome=o, n=len(f), cases=int(f[f"pre_{o}"].sum()), OR=np.exp(res.params["p10"]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=res.pvalues["p10"]))
pd.DataFrame(pre).to_csv(OUT / f"independent_preimaging_disease{TAG}.csv", index=False)

# (iii) landmark prospective analysis
lm, inc = [], []
T_END = C.ADMIN_END
for o in OUTS:
    fd = d[f"first_{o}"]
    f = C.prospective_frame(d, fd, T0, d.death_date, T_END)
    if DESIGN == "imaging":       # age at the imaging visit replaces age at recruitment
        f["Age_s"] = (f.age_imaging - d.age_imaging.mean()) / d.age_imaging.std()
    for g, gg in f.groupby("grp", observed=True):
        inc.append(dict(outcome=o, grp=g, n=len(gg), events=int(gg.E.sum()), py=gg["T"].sum(), rate=1000 * gg.E.sum() / gg["T"].sum()))
    for adj, cols in [("M1", ADJ1), ("M2", ADJ2)]:
        ff = f[["T", "E", "p10"] + cols].dropna()
        cph = CoxPHFitter(penalizer=1e-4).fit(ff, "T", "E")
        s = cph.summary.loc["p10"]
        lm.append(dict(outcome=o, adj=adj, n=len(ff), events=int(ff.E.sum()), HR=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]),
                       hi=np.exp(s["coef upper 95%"]), p=s["p"]))
        # stage group >=3 vs 0
        g = ff.join(f["grp"]); g = g[g.grp.isin(["0", ">=3"])].copy(); g["hi_stage"] = (g.grp == ">=3").astype(float)
        cph = CoxPHFitter(penalizer=1e-4).fit(g[["T", "E", "hi_stage"] + cols], "T", "E")
        s = cph.summary.loc["hi_stage"]
        lm.append(dict(outcome=o, adj=adj + "_grp>=3vs0", n=len(g), events=int(g.E.sum()), HR=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]),
                       hi=np.exp(s["coef upper 95%"]), p=s["p"]))
pd.DataFrame(lm).to_csv(OUT / f"independent_landmark_cox{TAG}.csv", index=False)
pd.DataFrame(inc).to_csv(OUT / f"independent_landmark_incidence{TAG}.csv", index=False)
d.to_pickle(OUT / f"independent_cohort_full{TAG}.pkl")
if DESIGN == "landmark":
    d.drop(columns=[c for c in d.columns if c.startswith("first_")]).to_pickle(OUT / "independent_cohort.pkl")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
print(stg.stage.value_counts().sort_index().head(12).to_string())
print(cooc[cooc.heart.isin(["any_heart"]) | cooc.heart.str.startswith("n_heart")].round(3).to_string())
print(pd.DataFrame(cv).round(3).to_string())
print(pd.DataFrame(pre).round(3).to_string())
print(pd.DataFrame(inc).round(2).to_string())
print(pd.DataFrame(lm).round(3).to_string())
