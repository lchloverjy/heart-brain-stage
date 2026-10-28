"""Sex-specific analysis of the heart-brain stage (eClinicalMedicine sex/gender reporting policy; SAGER).

For the primary outcome (stroke, TIA, ischaemic heart disease, heart failure, or atrial fibrillation) in both samples:
  * stage distribution by sex (share at stage 0 and at stage >=3, mean normalised stage);
  * events, person-years, and rate per 1000 person-years by sex;
  * sex-specific HR per 10 percentage points of stage (model 2 without the sex term);
  * P for interaction from model 2 plus a stage x sex product term in the full sample.
Same conventions as outcomes.py (derivation: imputed covariates) and validation_independent.py imaging design
(independent: complete-case model 2, follow-up from each participant's imaging date, Sept 30, 2021).
Sex is UK Biobank field 31 (recorded at recruitment from NHS records, updated by self-report); gender was not assessed.
"""
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import common as C

warnings.filterwarnings("ignore")
OUT = C.RES / "sex_stratified.csv"
OUT_DIST = C.RES / "sex_stage_distribution.csv"
SEXLAB = {0: "Female", 1: "Male"}


def fit(f, expo, cols):
    cph = CoxPHFitter(penalizer=1e-4).fit(f[["T", "E", expo] + cols], "T", "E")
    s = cph.summary.loc[expo]
    return np.exp(s["coef"]), np.exp(s["coef lower 95%"]), np.exp(s["coef upper 95%"]), s["p"]


def analyse(sample, f, expo, cols, grp):
    rows = []
    cols_nosex = [c for c in cols if c != "male"]
    for sx in (0, 1):
        g = f[f.male == sx]
        hr, lo, hi, p = fit(g, expo, cols_nosex)
        rows.append(dict(sample=sample, sex=SEXLAB[sx], n=len(g), events=int(g.E.sum()), py=g["T"].sum(),
                         rate=1000 * g.E.sum() / g["T"].sum(), HR=hr, lo=lo, hi=hi, p=p, p_interaction=np.nan))
    ff = f.copy(); ff["stage_x_male"] = ff[expo] * ff.male
    cph = CoxPHFitter(penalizer=1e-4).fit(ff[["T", "E", expo, "stage_x_male"] + cols], "T", "E")
    s = cph.summary.loc["stage_x_male"]
    rows.append(dict(sample=sample, sex="Interaction (male vs female)", n=len(ff), events=int(ff.E.sum()), py=ff["T"].sum(),
                     rate=1000 * ff.E.sum() / ff["T"].sum(), HR=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]),
                     hi=np.exp(s["coef upper 95%"]), p=s["p"], p_interaction=s["p"]))
    dist = []
    for sx in (0, 1):
        g = grp[grp.male == sx]
        dist.append(dict(sample=sample, sex=SEXLAB[sx], n=len(g), stage0_pct=100 * (g.grp == "0").mean(),
                         stage3plus_pct=100 * (g.grp == ">=3").mean(), mean_stage_pct=100 * g.stage_norm.mean()))
    return rows, dist


rows, dist = [], []
# ---------------------------------------------------------------- derivation sample (All-10 model)
d = pd.read_pickle(C.RES / "cohort_derivation_imputed.pkl")
M2 = ["age_imaging_s", "male", "SBP_i_s", "current_smoker_i", "Diabetes_i", "CHOL_i_s", "HDLc_i_s", "BMI_i_s", "Hypertension_i"]
f = d.loc[d["prev_hbve"] == 0, ["t_hbve", "ev_hbve", "all10_p10"] + M2].dropna().copy()
f.columns = ["T", "E", "all10_p10"] + M2
f["T"] = f["T"].clip(lower=1 / C.YEAR)
grp = pd.DataFrame({"male": d.male, "grp": d.stage_grp.astype(str), "stage_norm": d.all10_stage_norm})
r, s = analyse("derivation", f, "all10_p10", M2, grp); rows += r; dist += s
# ---------------------------------------------------------------- independent sample (MRI-only model, imaging design)
v = pd.read_pickle(C.RES / "validation/independent_cohort_full_imaging.pkl")
ADJ2 = ["Age_s", "male", "SBP_s", "current_smoker", "Diabetes", "CHOL_s", "HDLc_s", "BMI_s", "Hypertension"]
fv = C.prospective_frame(v, v["first_hbve"], v.date_imaging, v.death_date, C.ADMIN_END)
fv["Age_s"] = (fv.age_imaging - v.age_imaging.mean()) / v.age_imaging.std()
fv = fv[["T", "E", "p10"] + ADJ2].dropna()
grpv = pd.DataFrame({"male": v.male, "grp": v.grp.astype(str), "stage_norm": v.stage_norm})
r, s = analyse("independent", fv, "p10", ADJ2, grpv); rows += r; dist += s

res = pd.DataFrame(rows); res.to_csv(OUT, index=False)
dd = pd.DataFrame(dist); dd.to_csv(OUT_DIST, index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string()); print(dd.round(2).to_string())
