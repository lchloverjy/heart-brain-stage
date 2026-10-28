"""Imaging-anchored outcome analyses for the heart-brain SuStaIn stage (derivation, n=3,770).

Time zero = imaging visit; outcomes = first HES inpatient diagnosis after imaging; administrative censoring
2021-09-30; prevalent cases (diagnosis on/before imaging) excluded outcome-by-outcome.
Covariates with missing values are singly imputed by chained equations (sensitivity: complete case)."""
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import common as C

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260712)
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
d = pd.read_pickle(C.RES / "cohort_derivation.pkl").copy()

# ------------------------------------------------------------------ imputation of risk-factor covariates
imp_cols = ["age_imaging", "male", "SBP", "current_smoker", "Diabetes", "CHOL", "HDLc", "BMI", "Hypertension",
            "HbA1c", "LDLc", "CysC", "townsend_index"]
d["cc"] = d[imp_cols].notna().all(1).astype(int)
imp = IterativeImputer(max_iter=20, random_state=20260712, sample_posterior=False)
X = pd.DataFrame(imp.fit_transform(d[imp_cols]), columns=imp_cols, index=d.index)
for c in ["current_smoker", "Diabetes", "Hypertension"]:
    X[c] = X[c].clip(0, 1).round()
for c in imp_cols:
    d[c + "_i"] = X[c]
risk, _ = C.score2(d.age_imaging, d.male == 1, d.current_smoker_i, d.SBP_i, d.Diabetes_i, d.CHOL_i, d.HDLc_i)
d["score2_i"] = risk
d["score2_cat_i"] = C.score2_category(d.age_imaging, risk)
d["score2_cll"] = np.log(-np.log(1 - risk / 100))           # complementary log-log of predicted risk
for c in ["SBP_i", "CHOL_i", "HDLc_i", "BMI_i", "HbA1c_i", "age_imaging"]:
    d[c + "_s"] = (d[c] - d[c].mean()) / d[c].std()

# exposures
MODEL_STAGE = {m: f"{m}_stage_norm" for m in C.MODELS}
for m in C.MODELS:
    d[f"{m}_p10"] = d[f"{m}_stage_norm"] * 10                # per 10 percentage points of the event sequence
    d[f"{m}_sd"] = (d[f"{m}_stage_norm"] - d[f"{m}_stage_norm"].mean()) / d[f"{m}_stage_norm"].std()
d["stage_grp"] = pd.cut(d.all10_stage, [-1, 0, 1, 2, 99], labels=["0", "1", "2", ">=3"])
d["mri_grp"] = pd.cut(d.mri_only_matched_all10_stage, [-1, 0, 1, 2, 99], labels=["0", "1", "2", ">=3"])
for c in ["n_abnormal_z1", "mean_abn_z"]:
    d[c + "_sd"] = (d[c] - d[c].mean()) / d[c].std()
d.to_pickle(C.RES / "cohort_derivation_imputed.pkl")

ADJ = {
    "M1": ["age_imaging_s", "male"],
    "M2": ["age_imaging_s", "male", "SBP_i_s", "current_smoker_i", "Diabetes_i", "CHOL_i_s", "HDLc_i_s", "BMI_i_s", "Hypertension_i"],
    "SCORE2": ["score2_cll"],
}
PRIMARY = ["hbve", "mace", "cerebrovascular", "cardiac"]
SECOND = ["stroke", "tia", "mi", "ihd", "af", "hf", "cv_death"]
EXPLOR = ["depression", "anxiety", "parkinson", "ckd", "death", "mace_cvd"]
ALL_OUT = PRIMARY + SECOND + EXPLOR


def frame(outcome, cols, data=None):
    data = d if data is None else data
    f = data.loc[data[f"prev_{outcome}"] == 0, [f"t_{outcome}", f"ev_{outcome}"] + cols].dropna().copy()
    f.columns = ["T", "E"] + cols
    f["T"] = f["T"].clip(lower=1 / C.YEAR)
    return f


def cox(f, cols, strata=None):
    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(f[["T", "E"] + cols], "T", "E")
    return cph


def hr_row(cph, var):
    s = cph.summary.loc[var]
    return dict(HR=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]), hi=np.exp(s["coef upper 95%"]), p=s["p"])


# ------------------------------------------------------------------ 1. incidence by stage group
rows = []
for o in ALL_OUT:
    f = d[d[f"prev_{o}"] == 0]
    for g, gg in f.groupby("stage_grp", observed=True):
        ev = int(gg[f"ev_{o}"].sum()); py = gg[f"t_{o}"].sum()
        lo = stats.chi2.ppf(0.025, 2 * ev) / 2 if ev else 0.0; hi = stats.chi2.ppf(0.975, 2 * ev + 2) / 2
        rows.append(dict(outcome=o, stage_grp=g, n=len(gg), events=ev, person_years=py, rate=1000 * ev / py,
                         rate_lo=1000 * lo / py, rate_hi=1000 * hi / py))
inc = pd.DataFrame(rows); inc.to_csv(C.RES / "out_incidence_by_stage.csv", index=False)

# ------------------------------------------------------------------ 2. Cox: stage per 10 pp and stage groups
rows = []
for o in ALL_OUT:
    for m in C.MODELS:
        for adj in ["M1", "M2", "SCORE2"]:
            cols = [f"{m}_p10"] + ADJ[adj]
            f = frame(o, cols)
            try:
                r = hr_row(cox(f, cols), f"{m}_p10")
            except Exception as e:
                r = dict(HR=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
            rows.append(dict(outcome=o, model=m, adj=adj, exposure="per10pp", n=len(f), events=int(f.E.sum()), **r))
    # stage groups (All-10)
    for adj in ["M1", "M2"]:
        f = frame(o, ["stage_grp"] + ADJ[adj])
        dm = pd.get_dummies(f.stage_grp, prefix="g", drop_first=True).astype(float)
        f2 = pd.concat([f.drop(columns="stage_grp"), dm], axis=1)
        cols = list(dm.columns) + ADJ[adj]
        try:
            cph = cox(f2, cols)
            for c in dm.columns:
                rows.append(dict(outcome=o, model="all10", adj=adj, exposure="grp" + c[2:], n=len(f), events=int(f.E.sum()), **hr_row(cph, c)))
            # trend
            f["grp_num"] = f.stage_grp.cat.codes.astype(float)
            r = hr_row(cox(f, ["grp_num"] + ADJ[adj]), "grp_num")
            rows.append(dict(outcome=o, model="all10", adj=adj, exposure="trend", n=len(f), events=int(f.E.sum()), **r))
        except Exception:
            pass
coxr = pd.DataFrame(rows); coxr.to_csv(C.RES / "out_cox_main.csv", index=False)

# ------------------------------------------------------------------ 3. SuStaIn stage vs simple abnormality summaries (HBVE)
rows = []
cands = {"all10_sd": "Heart-brain stage (All-10)", "mri_plus_m13_sd": "Heart-brain stage (MRI + M13)",
         "mri_only_matched_all10_sd": "Heart-brain stage (MRI only)", "n_abnormal_z1_sd": "Number of abnormal markers (z>=1)",
         "mean_abn_z_sd": "Mean abnormality z"}
for o in PRIMARY:
    base = frame(o, ADJ["SCORE2"] + list(cands))
    cb = cox(base, ADJ["SCORE2"]); llb = cb.log_likelihood_; cbase = concordance_index(base["T"], -cb.predict_partial_hazard(base), base["E"])
    for v, lbl in cands.items():
        c1 = cox(base, ADJ["SCORE2"] + [v])
        lr = 2 * (c1.log_likelihood_ - llb)
        rows.append(dict(outcome=o, predictor=v, label=lbl, **hr_row(c1, v), LR_chi2=lr, LR_p=stats.chi2.sf(lr, 1),
                         C_base=cbase, C_new=concordance_index(base["T"], -c1.predict_partial_hazard(base), base["E"]),
                         AIC_delta=-2 * c1.log_likelihood_ + 2 * 2 - (-2 * llb + 2)))
    # nested: does stage add beyond the count?
    c2 = cox(base, ADJ["SCORE2"] + ["n_abnormal_z1_sd"]); c3 = cox(base, ADJ["SCORE2"] + ["n_abnormal_z1_sd", "all10_sd"])
    lr = 2 * (c3.log_likelihood_ - c2.log_likelihood_)
    rows.append(dict(outcome=o, predictor="all10_sd|count", label="Stage beyond count", **hr_row(c3, "all10_sd"), LR_chi2=lr, LR_p=stats.chi2.sf(lr, 1)))
    c4 = cox(base, ADJ["SCORE2"] + ["mean_abn_z_sd"]); c5 = cox(base, ADJ["SCORE2"] + ["mean_abn_z_sd", "all10_sd"])
    lr = 2 * (c5.log_likelihood_ - c4.log_likelihood_)
    rows.append(dict(outcome=o, predictor="all10_sd|meanz", label="Stage beyond mean z", **hr_row(c5, "all10_sd"), LR_chi2=lr, LR_p=stats.chi2.sf(lr, 1)))
    lr = 2 * (c5.log_likelihood_ - c5.log_likelihood_)
    c6 = cox(base, ADJ["SCORE2"] + ["all10_sd", "mean_abn_z_sd"]); lr = 2 * (c6.log_likelihood_ - cox(base, ADJ["SCORE2"] + ["all10_sd"]).log_likelihood_)
    rows.append(dict(outcome=o, predictor="meanz|all10_sd", label="Mean z beyond stage", **hr_row(c6, "mean_abn_z_sd"), LR_chi2=lr, LR_p=stats.chi2.sf(lr, 1)))
cmp_ = pd.DataFrame(rows); cmp_.to_csv(C.RES / "out_compare_predictors.csv", index=False)

# ------------------------------------------------------------------ 4. heart-brain involvement categories
rows = []
d["obs_inv"] = np.select([(d.heart_any_z1 == 0) & (d.brain_any_z1 == 0), (d.heart_any_z1 == 1) & (d.brain_any_z1 == 0),
                          (d.heart_any_z1 == 0) & (d.brain_any_z1 == 1)], ["None", "Heart only", "Brain only"], "Heart and brain")
for scheme, col, levels in [("model_all10", "all10_inv", ["None", "Heart only", "Heart and brain"]),
                            ("model_mri", "mri_only_matched_all10_inv", ["None", "Heart only", "Heart and brain"]),
                            ("observed", "obs_inv", ["None", "Heart only", "Brain only", "Heart and brain"])]:
    for o in PRIMARY + ["stroke", "af", "hf", "ihd"]:
        f = frame(o, [col] + ADJ["M2"])
        dm = pd.get_dummies(pd.Categorical(f[col], categories=levels), prefix="i", drop_first=True).astype(float)
        dm.index = f.index; dm.columns = [c.replace(" ", "_") for c in dm.columns]
        f2 = pd.concat([f.drop(columns=col), dm], axis=1)
        try:
            cph = cox(f2, list(dm.columns) + ADJ["M2"])
        except Exception:
            continue
        for lv in levels:
            g = f[f[col] == lv]
            rec = dict(scheme=scheme, outcome=o, level=lv, n=len(g), events=int(g.E.sum()), py=g["T"].sum(),
                       rate=1000 * g.E.sum() / g["T"].sum())
            cname = "i_" + lv.replace(" ", "_")
            rec.update(hr_row(cph, cname) if cname in cph.summary.index else dict(HR=1.0, lo=np.nan, hi=np.nan, p=np.nan))
            rows.append(rec)
invr = pd.DataFrame(rows); invr.to_csv(C.RES / "out_involvement.csv", index=False)
obs_tab = pd.crosstab(d.heart_any_z1, d.brain_any_z1, margins=True)
obs_tab.to_csv(C.RES / "out_observed_heart_brain_crosstab.csv")

# ------------------------------------------------------------------ 5. subtype contrasts among stage-positive
rows = []
for m in ["all10", "mri_plus_m13"]:
    pos = d[d[f"{m}_stage"] > 0].copy(); pos["S2"] = (pos[f"{m}_subtype"] == 2).astype(float)
    for o in PRIMARY + ["stroke", "af", "hf", "ihd"]:
        f = frame(o, ["S2", f"{m}_p10"] + ADJ["M1"], data=pos)
        try:
            r = hr_row(cox(f, ["S2", f"{m}_p10"] + ADJ["M1"]), "S2")
            f["inter"] = f.S2 * f[f"{m}_p10"]
            ri = hr_row(cox(f, ["S2", f"{m}_p10", "inter"] + ADJ["M1"]), "inter")
        except Exception:
            continue
        rows.append(dict(model=m, outcome=o, n=len(f), events=int(f.E.sum()), ev_S1=int(f.E[f.S2 == 0].sum()),
                         ev_S2=int(f.E[f.S2 == 1].sum()), **{k + "_S2vsS1": v for k, v in r.items()}, p_interaction=ri["p"]))
subr = pd.DataFrame(rows); subr.to_csv(C.RES / "out_subtype.csv", index=False)

# ------------------------------------------------------------------ 6. incremental value over SCORE2
HOR = 3.0


def surv_at(cph, f, t):
    return cph.predict_survival_function(f, times=[t]).T.iloc[:, 0].to_numpy()


def nri_idi(f, p0, p1, t):
    """Continuous NRI(>0) and IDI at horizon t using KM-weighted event/non-event sets (Pencina/Steyerberg)."""
    up = p1 > p0; dn = p1 < p0
    km = KaplanMeierFitter().fit(f["T"], f["E"]); S = km.predict(t)
    def km_frac(mask):
        if mask.sum() == 0: return 0.0, 1.0
        k = KaplanMeierFitter().fit(f["T"][mask], f["E"][mask]); return mask.mean(), k.predict(t)
    pu, su = km_frac(up); pd_, sd_ = km_frac(dn)
    p_ev = 1 - S
    nri_e = (pu * (1 - su) - pd_ * (1 - sd_)) / p_ev
    nri_ne = (pd_ * sd_ - pu * su) / S
    # IDI with event status at t (exclude censored before t)
    known = (f["T"] >= t) | (f["E"] == 1)
    ev = (f["E"] == 1) & (f["T"] <= t)
    idi = (p1[known & ev].mean() - p0[known & ev].mean()) - (p1[known & ~ev].mean() - p0[known & ~ev].mean())
    return nri_e, nri_ne, nri_e + nri_ne, idi


def net_benefit(f, p, t, thr):
    out = []
    km = KaplanMeierFitter()
    n = len(f)
    for pt in thr:
        sel = p >= pt
        if sel.sum() == 0:
            out.append(0.0); continue
        km.fit(f["T"][sel], f["E"][sel]); risk_sel = 1 - km.predict(t)
        tp = risk_sel * sel.sum() / n; fp = (1 - risk_sel) * sel.sum() / n
        out.append(tp - fp * pt / (1 - pt))
    return np.array(out)


THR_DCA = np.round(np.arange(0.005, 0.1001, 0.0025), 4)


def inc_task(args):
    o, m, nb, seed = args
    rng = np.random.default_rng(seed)
    v = f"{m}_p10"
    f = frame(o, ["score2_cll", v, "score2_cat_i"]).reset_index(drop=True)
    c0 = cox(f, ["score2_cll"]); c1 = cox(f, ["score2_cll", v])
    lp0 = c0.predict_log_partial_hazard(f).to_numpy(); lp1 = c1.predict_log_partial_hazard(f).to_numpy()
    C0 = concordance_index(f["T"], -lp0, f["E"]); C1 = concordance_index(f["T"], -lp1, f["E"])
    dcs, opt = [], []
    Tn, En = f["T"].to_numpy(), f["E"].to_numpy()
    for b in range(nb):
        ix = rng.integers(0, len(f), len(f)); fb = f.iloc[ix]
        try:
            b0 = cox(fb, ["score2_cll"]); b1 = cox(fb, ["score2_cll", v])
        except Exception:
            continue
        l0b = b0.predict_log_partial_hazard(fb).to_numpy(); l1b = b1.predict_log_partial_hazard(fb).to_numpy()
        dcb = concordance_index(fb["T"], -l1b, fb["E"]) - concordance_index(fb["T"], -l0b, fb["E"])
        l0o = b0.predict_log_partial_hazard(f).to_numpy(); l1o = b1.predict_log_partial_hazard(f).to_numpy()
        dco = concordance_index(Tn, -l1o, En) - concordance_index(Tn, -l0o, En)
        dcs.append(dcb); opt.append(dcb - dco)
    dcs = np.array(dcs); opt = np.array(opt)
    p0 = 1 - surv_at(c0, f, HOR); p1 = 1 - surv_at(c1, f, HOR)
    nri_e, nri_ne, nri, idi = nri_idi(f, p0, p1, HOR)
    lr = 2 * (c1.log_likelihood_ - c0.log_likelihood_)
    h = hr_row(c1, v)
    urow = dict(outcome=o, model=m, n=len(f), events=int(f.E.sum()), n_boot=len(dcs), C_score2=C0, C_score2_stage=C1, dC=C1 - C0,
                dC_lo=np.percentile(dcs, 2.5), dC_hi=np.percentile(dcs, 97.5), dC_optimism=opt.mean(),
                dC_corrected=C1 - C0 - opt.mean(), p_dC_le0=(dcs <= 0).mean(), LR_chi2=lr, LR_p=stats.chi2.sf(lr, 1),
                NRI_events=nri_e, NRI_nonevents=nri_ne, NRI=nri, IDI=idi, HR_stage=h["HR"], HR_lo=h["lo"], HR_hi=h["hi"])
    drows = []
    if m == "all10" and o in ("hbve", "mace", "cerebrovascular", "cardiac"):
        thr = THR_DCA
        nb0 = net_benefit(f, p0, HOR, thr); nb1 = net_benefit(f, p1, HOR, thr)
        km = KaplanMeierFitter().fit(f["T"], f["E"]); prev = 1 - km.predict(HOR)
        nball = prev - (1 - prev) * thr / (1 - thr)
        for t_, a_, b_, c_ in zip(thr, nb0, nb1, nball):
            drows.append(dict(outcome=o, threshold=t_, nb_score2=a_, nb_score2_stage=b_, nb_all=c_, nb_none=0.0))
    return urow, drows


from concurrent.futures import ProcessPoolExecutor
tasks = []
for i, o in enumerate(PRIMARY + ["stroke", "af", "hf"]):
    for j, m in enumerate(C.MODELS):
        tasks.append((o, m, N_BOOT if (m == "all10" or o == "hbve") else min(N_BOOT, 200), 20260712 + 100 * i + j))
util, dca, strata = [], [], []
with ProcessPoolExecutor(max_workers=5) as ex:
    for urow, drows in ex.map(inc_task, tasks):
        util.append(urow); dca.extend(drows)
for o in PRIMARY + ["stroke", "af", "hf"]:
    # stage within SCORE2 categories (All-10)
    f = frame(o, ["score2_cat_i", "all10_stage", "stage_grp"])
    f["s2grp"] = np.where(f.score2_cat_i == 0, "Low-to-moderate", "High or very high")
    f["stgrp"] = np.where(f.all10_stage == 0, "Stage 0", np.where(f.all10_stage <= 2, "Stage 1-2", "Stage >=3"))
    for (a_, b_), g in f.groupby(["s2grp", "stgrp"]):
        ev = int(g.E.sum()); py = g["T"].sum()
        lo = stats.chi2.ppf(0.025, 2 * ev) / 2 if ev else 0.0; hi = stats.chi2.ppf(0.975, 2 * ev + 2) / 2
        km = KaplanMeierFitter().fit(g["T"], g["E"])
        strata.append(dict(outcome=o, score2_group=a_, stage_group=b_, n=len(g), events=ev, py=py, rate=1000 * ev / py,
                           rate_lo=1000 * lo / py, rate_hi=1000 * hi / py, cuminc_3y=1 - km.predict(3.0), cuminc_5y=1 - km.predict(5.0)))
pd.DataFrame(util).to_csv(C.RES / "out_score2_increment.csv", index=False)
pd.DataFrame(dca).to_csv(C.RES / "out_dca.csv", index=False)
pd.DataFrame(strata).to_csv(C.RES / "out_score2_strata.csv", index=False)

# ------------------------------------------------------------------ 7. cumulative incidence curves (HBVE, cerebrovascular) by stage group
rows = []
for o in ["hbve", "mace", "cerebrovascular", "cardiac"]:
    for scheme, col in [("all10", "stage_grp"), ("mri", "mri_grp")]:
        f = d[d[f"prev_{o}"] == 0]
        for g, gg in f.groupby(col, observed=True):
            km = KaplanMeierFitter().fit(gg[f"t_{o}"], gg[f"ev_{o}"])
            ci = km.confidence_interval_cumulative_density_
            sf = km.cumulative_density_
            for t in sf.index:
                rows.append(dict(outcome=o, scheme=scheme, group=g, t=t, cuminc=sf.loc[t].iloc[0],
                                 lo=ci.loc[t].iloc[0], hi=ci.loc[t].iloc[1]))
            for t in [0, 1, 2, 3, 4, 5, 6]:
                rows.append(dict(outcome=o, scheme=scheme, group=g, t=float(t), at_risk=int((gg[f"t_{o}"] >= t).sum()), kind="risk_table"))
        # log-rank trend
pd.DataFrame(rows).to_csv(C.RES / "out_km.csv", index=False)

# ------------------------------------------------------------------ 8. sensitivity analyses (All-10, per 10 pp, M2)
rows = []
def sens(label, data=None, o="hbve", exposure="all10_p10", adj="M2", min_t=None):
    data = d if data is None else data
    f = frame(o, [exposure] + ADJ[adj], data=data)
    if min_t is not None:
        f = f[f["T"] > min_t].copy(); f["T"] = f["T"] - min_t
    cols = [c for c in [exposure] + ADJ[adj] if f[c].nunique() > 1]
    r = hr_row(cox(f, cols), exposure)
    rows.append(dict(analysis=label, outcome=o, n=len(f), events=int(f.E.sum()), **r))
d["all10_exp_p10"] = d.all10_stage_exp / C.MODELS["all10"]["n_events"] * 10
for o in PRIMARY:
    sens("Main (imputed covariates, M2)", o=o)
    sens("Complete-case covariates", data=d[d.cc == 1], o=o)
    sens("Excluding first year of follow-up", o=o, min_t=1.0)
    sens("Posterior expected stage", o=o, exposure="all10_exp_p10")
    sens("Excluding prevalent CVD or PAD", data=d[d.prev_any_cvd == 0], o=o)
    sens("Excluding diabetes", data=d[d.Diabetes_i == 0], o=o)
    sens("High subtype certainty (P>=0.60) or stage 0", data=d[(d.all10_p_subtype >= 0.6) | (d.all10_stage == 0)], o=o)
    sens("Adjusted for SCORE2 only", o=o, adj="SCORE2")
    sens("White ethnicity only", data=d[d.white == 1], o=o)
    d["M2x"] = 0
    try:
        f = frame(o, ["all10_p10"] + ADJ["M2"] + ["HbA1c_i_s", "CysC_i", "townsend_index_i"])
        f["CysC_i"] = (f.CysC_i - f.CysC_i.mean()) / f.CysC_i.std(); f["townsend_index_i"] = (f.townsend_index_i - f.townsend_index_i.mean()) / f.townsend_index_i.std()
        cols = ["all10_p10"] + ADJ["M2"] + ["HbA1c_i_s", "CysC_i", "townsend_index_i"]
        rows.append(dict(analysis="Further adjusted (HbA1c, cystatin C, Townsend)", outcome=o, n=len(f), events=int(f.E.sum()), **hr_row(cox(f, cols), "all10_p10")))
    except Exception:
        pass
# Schoenfeld-based PH test for main model
from lifelines.statistics import proportional_hazard_test
ph = []
for o in PRIMARY:
    f = frame(o, ["all10_p10"] + ADJ["M2"]); cph = cox(f, ["all10_p10"] + ADJ["M2"])
    t = proportional_hazard_test(cph, f, time_transform="rank")
    ph.append(dict(outcome=o, p_stage=float(t.summary.loc["all10_p10", "p"]), p_global_min=float(t.summary.p.min())))
sensr = pd.DataFrame(rows); sensr.to_csv(C.RES / "out_sensitivity.csv", index=False)
pd.DataFrame(ph).to_csv(C.RES / "out_ph_test.csv", index=False)

# ------------------------------------------------------------------ print summary
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(inc[inc.outcome.isin(PRIMARY + ["stroke", "af", "hf"])].round(2).to_string())
print(coxr[(coxr.exposure == "per10pp")].pivot_table(index=["outcome", "adj"], columns="model", values="HR").round(2).to_string())
print(coxr[(coxr.model == "all10") & (coxr.exposure == "per10pp")][["outcome", "adj", "n", "events", "HR", "lo", "hi", "p"]].round(3).to_string())
print(coxr[coxr.exposure.str.startswith("grp") | (coxr.exposure == "trend")].round(3).to_string())
print(cmp_.round(3).to_string())
print(invr.round(3).to_string())
print(obs_tab)
print(subr.round(3).to_string())
print(pd.DataFrame(util).round(4).to_string())
print(pd.DataFrame(strata).round(3).to_string())
print(sensr.round(3).to_string())
print(pd.DataFrame(ph).round(3).to_string())
