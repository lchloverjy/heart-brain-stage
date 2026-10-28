"""Independent sample: organ-concordant risk and stage vs simple marker counts (landmark design, see validation_independent.py)."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import sys
import common as C
DESIGN = sys.argv[1] if len(sys.argv) > 1 else "landmark"
TAG = "_imaging" if DESIGN == "imaging" else ""

warnings.filterwarnings("ignore")
V = C.RES / "validation"
LANDMARK = pd.Timestamp("2020-04-01"); IMG_START = pd.Timestamp("2014-08-01"); T_END = C.ADMIN_END
d = pd.read_pickle(V / f"independent_cohort_full{TAG}.pkl")
deaths = C.load_deaths()
long = pd.read_pickle(C.ROOT / "analysis/hes_long_independent.pkl").rename(columns={"Eid": "eid"})
full = pd.read_csv(C.PREP / "03_inputs" / "sustain_mri_only.csv")
H = ["LV_remodeling_Z", "LV_myocardial_dysfunction_Z", "Aortic_stiffness_Z"]
B = ["Hippocampal_atrophy_Z", "Cortical_thinning_Z", "WMH_burden_Z", "FreeWater_ISOVF_Z"]
d = d.merge(full[["eid"] + H + B], on="eid", how="left")
d["heart_abn"] = (d[H] >= 1).any(axis=1); d["brain_abn"] = (d[B] >= 1).any(axis=1)
d["obs_cat"] = np.select([~d.heart_abn & ~d.brain_abn, d.heart_abn & ~d.brain_abn, ~d.heart_abn & d.brain_abn],
                         ["None", "Heart only", "Brain only"], "Heart and brain")
d["n_abn"] = (d[H + B] >= 1).sum(axis=1)
d["mean_z"] = d[H + B].mean(axis=1)
inv = pd.read_csv(C.RES / "sustain_organ_involvement_by_stage.csv")
inv = inv[inv.model == "mri_only_matched_all10"]
key = {(int(r.subtype), int(r.stage)): ("None" if r.stage == 0 else ("Heart and brain" if r.brain else ("Heart only" if r.heart else "Brain only")))
       for r in inv.itertuples()}
d["mod_cat"] = [key.get((int(s), int(g)), "None") if g > 0 else "None" for s, g in zip(d.subtype, d.stage)]

OUTS = {"hbve": C.COMPOSITES["hbve"], "cardiac": C.COMPOSITES["cardiac"], "cerebrovascular": C.COMPOSITES["cerebrovascular"],
        "stroke": ["stroke"], "af": ["af"], "hf": ["hf"], "ihd": ["ihd"]}
ADJ = ["Age_s", "male", "SBP_s", "current_smoker", "Diabetes", "CHOL_s", "HDLc_s", "BMI_s", "Hypertension"]


def first_date(keys):
    return C.first_event_date(long, deaths, keys)


T0 = d.date_imaging if DESIGN == "imaging" else LANDMARK


rows, pre, disc = [], [], []
for o, keys in OUTS.items():
    fd = d.eid.map(first_date(keys))
    # pre-imaging (prevalent before Aug 2014) by category, age+sex logistic
    y = ((fd <= d.date_imaging) if DESIGN == "imaging" else (fd < IMG_START)).astype(int)
    for scheme in ["obs_cat", "mod_cat"]:
        X = pd.get_dummies(d[scheme]).drop(columns="None").astype(float)
        X = pd.concat([X, d[["Age_s", "male"]]], axis=1)
        res = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
        for lev in [c for c in X.columns if c not in ("Age_s", "male")]:
            ci = res.conf_int().loc[lev]
            pre.append(dict(outcome=o, scheme=scheme, level=lev, cases=int(y[d[scheme] == lev].sum()), n=int((d[scheme] == lev).sum()),
                            OR=np.exp(res.params[lev]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=res.pvalues[lev]))
    # landmark prospective
    f = C.prospective_frame(d, fd, T0, d.death_date, T_END)
    if DESIGN == "imaging":
        f["Age_s"] = (f.age_imaging - d.age_imaging.mean()) / d.age_imaging.std()
    for scheme in ["obs_cat", "mod_cat"]:
        X = pd.get_dummies(f[scheme]).drop(columns="None").astype(float)
        g = pd.concat([f[["T", "E"] + ADJ], X], axis=1).dropna()
        cph = CoxPHFitter(penalizer=1e-4).fit(g, "T", "E")
        for lev in X.columns:
            s_ = cph.summary.loc[lev]
            sel = f[scheme] == lev
            rows.append(dict(outcome=o, scheme=scheme, level=lev, n=int(sel.sum()), events=int(f.E[sel].sum()),
                             py=float(f["T"][sel].sum()), HR=np.exp(s_["coef"]), lo=np.exp(s_["coef lower 95%"]),
                             hi=np.exp(s_["coef upper 95%"]), p=s_["p"]))
        sel = f[scheme] == "None"
        rows.append(dict(outcome=o, scheme=scheme, level="None", n=int(sel.sum()), events=int(f.E[sel].sum()),
                         py=float(f["T"][sel].sum()), HR=1.0, lo=np.nan, hi=np.nan, p=np.nan))
    # discrimination: risk factors vs + stage vs + n_abn vs + mean_z (apparent C; landmark)
    g = f[["T", "E", "p10", "n_abn", "mean_z"] + ADJ].dropna()
    base = CoxPHFitter(penalizer=1e-4).fit(g[["T", "E"] + ADJ], "T", "E")
    c0 = concordance_index(g["T"], -base.predict_log_partial_hazard(g), g["E"])
    rec = dict(outcome=o, n=len(g), events=int(g.E.sum()), C_rf=c0)
    for v in ["p10", "n_abn", "mean_z"]:
        m = CoxPHFitter(penalizer=1e-4).fit(g[["T", "E", v] + ADJ], "T", "E")
        rec[f"C_rf_{v}"] = concordance_index(g["T"], -m.predict_log_partial_hazard(g), g["E"])
        rec[f"LR_{v}"] = 2 * (m.log_likelihood_ - base.log_likelihood_)
    disc.append(rec)

pd.DataFrame(rows).to_csv(V / f"independent_involvement_landmark{TAG}.csv", index=False)
pd.DataFrame(pre).to_csv(V / f"independent_involvement_preimaging{TAG}.csv", index=False)
pd.DataFrame(disc).to_csv(V / f"independent_discrimination_vs_counts{TAG}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(d.obs_cat.value_counts().to_string()); print(d.mod_cat.value_counts().to_string())
print(pd.DataFrame(rows).round(3).to_string())
print(pd.DataFrame(pre).round(3).to_string())
print(pd.DataFrame(disc).round(4).to_string())
