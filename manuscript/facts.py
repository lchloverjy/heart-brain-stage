import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Compute every number quoted in the manuscript from the result tables and write facts.json.
Lancet house style: mid-line decimal point, thin-space-free thousands below 10 000, space above."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, _os.path.join(HB_ROOT, "analysis"))
import common as C

R = C.RES; V = R / "validation"
F = {}


def d(x, k=2):
    s = f"{x:.{k}f}"
    return s.replace("-", "\u2212").replace(".", "\u00b7")


def n(x):
    x = int(round(x))
    return f"{x:,}".replace(",", " ") if x >= 10000 else str(x)


def pct(x, k=1):
    return d(100 * x, k) + "%"


def ci(lo, hi, k=2):
    sep = " to " if (lo < 0 or hi < 0) else "\u2013"
    return f"{d(lo, k)}{sep}{d(hi, k)}"


def full(est, lo, hi, lab="HR", k=2):
    return f"{lab} {d(est, k)}, 95% CI {ci(lo, hi, k)}"


def bare(est, lo, hi, k=2):
    return f"{d(est, k)}, 95% CI {ci(lo, hi, k)}"


def paren(est, lo, hi, k=2):
    return f"{d(est, k)} (95% CI {ci(lo, hi, k)})"


def short(est, lo, hi, k=2):
    return f"{d(est, k)} ({ci(lo, hi, k)})"


WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def nw(x):
    x = int(round(x)); return WORDS.get(x, n(x))


def sig2(p):
    from math import floor, log10
    k = max(2, 1 - int(floor(log10(p))) )
    return d(p, k)


def pval(p):
    if p < 0.0001:
        return "<0\u00b70001"
    return "=" + sig2(p)


# ------------------------------------------------------------------ cohorts
der = pd.read_pickle(R / "cohort_derivation.pkl")
ind = pd.read_pickle(V / "independent_cohort_full_imaging.pkl")
F["age_mean"], F["age_sd"] = d(der.age_imaging.mean(), 1), d(der.age_imaging.std(), 1)
F["male_pct"] = f"{n(der.male.sum())} ({pct(der.male.mean())})"
F["ind_age_mean"], F["ind_age_sd"] = d(ind.age_imaging.mean(), 1), d(ind.age_imaging.std(), 1)
F["ind_male_pct"] = f"{n(ind.male.sum())} ({pct(ind.male.mean())})"
F["gap_med"] = d(der.gap_years.median(), 1)
for nm, x, pre in [("der", der, ""), ("ind", ind, "ind_")]:
    end = x.death_date.where(x.death_date.notna() & (x.death_date < C.ADMIN_END), C.ADMIN_END)
    fu = (end - x.date_imaging).dt.days / C.YEAR
    F[pre + "fu_med"] = d(fu.median(), 1); F[pre + "fu_iqr"] = ci(fu.quantile(.25), fu.quantile(.75), 1)
    F[pre + "n_death_all"] = n(((x.death_date > x.date_imaging) & (x.death_date <= C.ADMIN_END)).sum())
F["n_death"] = F["n_death_all"]
F["n_full"] = n(27744); F["n_ind"] = n(23974)

# ------------------------------------------------------------------ SuStaIn
hf = pd.read_csv(R / "sustain_heart_first.csv")
a10 = hf[hf.model == "all10"].set_index("subtype")
F["fracS1"], F["fracS2"] = pct(a10.loc[1, "fraction"]), pct(a10.loc[2, "fraction"])
g = der.all10_stage
F["n_stage0"], F["pct_stage0"] = n((g == 0).sum()), pct((g == 0).mean())
F["n_stage12"], F["pct_stage12"] = n(g.isin([1, 2]).sum()), pct(g.isin([1, 2]).mean())
F["n_stage3"], F["pct_stage3"] = n((g >= 3).sum()), pct((g >= 3).mean())
pos = der[der.all10_stage > 0]
F["n_pos"] = n(len(pos)); F["n_posS1"] = n((pos.all10_subtype == 1).sum()); F["n_posS2"] = n((pos.all10_subtype == 2).sum())
mo = der[der.mri_only_matched_all10_stage > 0]
F["n_mri_s3"] = n((mo.mri_only_matched_all10_subtype == 3).sum())
F["p_mri_s3"] = d(hf[(hf.model == "mri_only_matched_all10") & (hf.subtype == 3)].p_heart_before_brain.iloc[0], 2)
inv = pd.read_csv(R / "sustain_organ_involvement_by_stage.csv"); inv = inv[inv.model == "all10"]
m = der[["all10_subtype", "all10_stage"]].merge(inv, left_on=["all10_subtype", "all10_stage"], right_on=["subtype", "stage"], how="left")
F["n_inv_heart"] = n(((m.heart == 1) & (m.brain == 0)).sum()); F["n_inv_both"] = n(((m.heart == 1) & (m.brain == 1)).sum())
F["n_inv_brainonly_stages"] = int(((inv.heart == 0) & (inv.brain == 1)).sum())
ag = pd.read_csv(R / "sustain_stage_agreement.csv")
F["rho_range"] = ci(ag.spearman.min(), ag.spearman.max())
po = pd.read_csv(R / "sustain_prevalence_vs_order.csv")
F["prev_rho_range"] = ci(po.spearman_order_vs_prevalence.min(), po.spearman_order_vs_prevalence.max())
F["prev_nsig"] = nw((po.p < 0.05).sum()); F["prev_ntot"] = nw(len(po))
sc = pd.read_csv(R / "sustain_stage_correlates.csv"); sc = sc[sc.model == "all10"].set_index("factor")
for k, f_ in [("beta_diab", "Diabetes"), ("beta_htn", "Hypertension"), ("beta_smk", "current_smoker"), ("beta_score2", "score2")]:
    F[k] = d(sc.loc[f_, "beta"])
# co-abnormality
co = pd.read_csv(V / "heart_brain_coabnormality.csv"); co = co[co["sample"] == "independent"]
r1 = co[(co.heart == "any_heart") & (co.z == 1)].iloc[0]; r2 = co[(co.heart == "any_heart") & (co.z == 2)].iloc[0]
F["co_or_z1"], F["co_or_z2"] = bare(r1.OR, r1.lo, r1.hi), bare(r2.OR, r2.lo, r2.hi)
F["dose0"] = pct(co[co.heart == "n_heart_abnormal=0"].p_brain_given_heart.iloc[0])
F["dose3"] = pct(co[co.heart == "n_heart_abnormal=3"].p_brain_given_heart.iloc[0])
F["dose3_n"] = n(co[co.heart == "n_heart_abnormal=3"].n.iloc[0])
pair = co[(co.z == 1) & co.heart.str.endswith("_Z")]
q = pair[(pair.heart == "LV_remodeling_Z") & (pair.brain == "WMH_burden_Z")].iloc[0]; F["co_lvr_wmh"] = bare(q.OR, q.lo, q.hi)
q = pair[(pair.heart == "LV_remodeling_Z") & (pair.brain == "FreeWater_ISOVF_Z")].iloc[0]; F["co_lvr_fw"] = bare(q.OR, q.lo, q.hi)
grey = pair[pair.brain.isin(["Hippocampal_atrophy_Z", "Cortical_thinning_Z"])]
white = pair[pair.brain.isin(["WMH_burden_Z", "FreeWater_ISOVF_Z"])]
F["co_grey_range"] = ci(grey.OR.min(), grey.OR.max()); F["co_white_range"] = ci(white.OR.min(), white.OR.max())
p2 = co[(co.z == 2) & co.heart.isin(["LV_remodeling_Z", "LV_myocardial_dysfunction_Z"])]
w2 = p2[p2.brain.isin(["WMH_burden_Z", "FreeWater_ISOVF_Z"])]; g2 = p2[p2.brain.isin(["Hippocampal_atrophy_Z", "Cortical_thinning_Z"])]
F["co_white_z2"] = ci(w2.OR.min(), w2.OR.max()); F["co_grey_z2"] = ci(g2.OR.min(), g2.OR.max())

# ------------------------------------------------------------------ derivation outcomes
cx = pd.read_csv(R / "out_cox_main.csv")


def cox(o, model="all10", adj="M2", exp="per10pp"):
    return cx[(cx.outcome == o) & (cx.model == model) & (cx.adj == adj) & (cx.exposure == exp)].iloc[0]


r = cox("hbve"); F["hbve_ev"], F["hbve_n"] = n(r.events), n(r.n)
F["hbve_m2_full"] = paren(r.HR, r.lo, r.hi); F["hbve_m2"] = full(r.HR, r.lo, r.hi)
r = cox("hbve", adj="M1"); F["hbve_m1"] = bare(r.HR, r.lo, r.hi)
r = cox("hbve", exp="grp>=3"); F["hbve_g3"] = paren(r.HR, r.lo, r.hi); F["hbve_g3_short"] = short(r.HR, r.lo, r.hi)
F["hbve_ptrend"] = pval(cox("hbve", exp="trend").p).lstrip("=")
for o, k in [("mace", "mace_m2"), ("cerebrovascular", "cer_m2"), ("stroke", "stroke_m2"), ("hf", "hf_m2"), ("ihd", "ihd_m2"),
             ("af", "af_m2"), ("cv_death", "cvd_m2"), ("depression", "dep_m2"), ("anxiety", "anx_m2"), ("parkinson", "pd_m2"),
             ("ckd", "ckd_m2"), ("mi", "mi_m2"), ("tia", "tia_m2"), ("cardiac", "card_m2")]:
    r = cox(o); F[k] = full(r.HR, r.lo, r.hi); F[k.replace("_m2", "_ev")] = nw(r.events) if r.events <= 10 else n(r.events)
r = cox("hbve", model="mri_plus_m13"); F["hbve_m13"] = full(r.HR, r.lo, r.hi)
r = cox("hbve", model="mri_only_matched_all10"); F["hbve_mri"] = full(r.HR, r.lo, r.hi)
ib = pd.read_csv(R / "out_incidence_by_stage.csv")
ib = ib[(ib.outcome == "hbve") & (ib.model == "all10")] if "model" in ib.columns else ib[ib.outcome == "hbve"]
gcol = "grp" if "grp" in ib.columns else ib.columns[1]
F["rate0"] = d(ib[ib[gcol].astype(str) == "0"].rate.iloc[0], 1); F["rate3"] = d(ib[ib[gcol].astype(str) == ">=3"].rate.iloc[0], 1)
iv = pd.read_csv(R / "out_involvement.csv"); iv = iv[(iv.scheme == "model_all10") & (iv.outcome == "hbve")].set_index("level")
F["inv_heart"] = paren(iv.loc["Heart only", "HR"], iv.loc["Heart only", "lo"], iv.loc["Heart only", "hi"])
F["inv_both"] = paren(iv.loc["Heart and brain", "HR"], iv.loc["Heart and brain", "lo"], iv.loc["Heart and brain", "hi"])
sb = pd.read_csv(R / "out_subtype.csv"); sb = sb[(sb.model == "all10") & (sb.outcome == "hbve")].iloc[0]
F["sub_hr"] = full(sb.HR_S2vsS1, sb.lo_S2vsS1, sb.hi_S2vsS1); F["sub_pint"] = pval(sb.p_interaction).lstrip("=")
se = pd.read_csv(R / "out_sensitivity.csv"); se = se[(se.outcome == "hbve") & ~se.analysis.str.startswith("Main")]
F["sens_min"], F["sens_max"] = d(se.HR.min()), d(se.HR.max()); F["n_sens"] = nw(len(se))
ph = pd.read_csv(R / "out_ph_test.csv").set_index("outcome"); F["ph_mace"] = sig2(ph.loc["mace", "p_stage"])
F["ph_hbve"] = sig2(ph.loc["hbve", "p_stage"])

# ------------------------------------------------------------------ SCORE2 increment (derivation)
inc = pd.read_csv(R / "out_score2_increment.csv"); inc = inc[inc.model == "all10"].set_index("outcome")
h = inc.loc["hbve"]
F["c0_hbve"], F["c1_hbve"] = d(h.C_score2, 3), d(h.C_score2_stage, 3)
F["dc_hbve"] = f"{d(h.dC, 3)}, 95% CI {ci(h.dC_lo, h.dC_hi, 3)}"; F["dc_hbve_corr"] = d(h.dC_corrected, 3)
F["dc_hbve_pt"] = d(h.dC, 2)
F["lr_hbve"] = pval(h.LR_p)
F["nri_hbve"], F["nri_ne_hbve"], F["nri_e_hbve"] = d(h.NRI), d(h.NRI_nonevents), d(h.NRI_events)
F["idi_hbve"] = d(h.IDI, 4)
for o, k in [("mace", "dc_mace"), ("cerebrovascular", "dc_cer"), ("stroke", "dc_stroke"), ("cardiac", "dc_card")]:
    if o in inc.index:
        q = inc.loc[o]; F[k] = f"{d(q.dC, 3)} (95% CI {ci(q.dC_lo, q.dC_hi, 3)})"
st = pd.read_csv(R / "out_score2_strata.csv"); st = st[st.outcome == "hbve"]


def rate(tab, sg, gg):
    return d(tab[(tab.score2_group == sg) & (tab.stage_group == gg)].rate.iloc[0], 1)


for a, sg in [("s2hi", "High or very high"), ("s2lo", "Low-to-moderate")]:
    for b, gg in [("g0", "Stage 0"), ("g12", "Stage 1-2"), ("g3", "Stage >=3")]:
        F[f"{a}_{b}"] = rate(st, sg, gg)
dca = pd.read_csv(R / "out_dca.csv"); dca = dca[dca.outcome == "hbve"].copy(); dca["dnb"] = dca.nb_score2_stage - dca.nb_score2
pos_ = dca[dca.dnb >= 0.001]
F["dca_range"] = f"{d(100 * pos_.threshold.min(), 0)}\u2013{d(100 * pos_.threshold.max(), 0)}%"
F["dnb_max"] = d(1000 * dca.dnb.max(), 1)
cp = pd.read_csv(R / "out_compare_predictors.csv"); cp = cp[cp.outcome == "hbve"].set_index("predictor")
F["cmp_c_stage"] = d(cp.loc["all10_sd", "C_new"], 3); F["cmp_c_count"] = d(cp.loc["n_abnormal_z1_sd", "C_new"], 3)
F["cmp_c_meanz"] = d(cp.loc["mean_abn_z_sd", "C_new"], 3)
F["cmp_p_count"] = sig2(cp.loc["all10_sd|count", "LR_p"]); F["cmp_p_meanz"] = sig2(cp.loc["all10_sd|meanz", "LR_p"])

# ------------------------------------------------------------------ independent sample
gi = ind.grp.astype(str)
F["ind_n0"], F["ind_pct0"] = n((gi == "0").sum()), pct((gi == "0").mean())
F["ind_n3"], F["ind_pct3"] = n((gi == ">=3").sum()), pct((gi == ">=3").mean())
pr = pd.read_csv(V / "independent_preimaging_disease_imaging.csv").set_index("outcome")
F["pre_hbve"] = bare(pr.loc["hbve", "OR"], pr.loc["hbve", "lo"], pr.loc["hbve", "hi"])
F["pre_hbve_cases"] = n(pr.loc["hbve", "cases"])
for o in ["stroke", "hf", "af"]:
    F[f"pre_{o}"] = full(pr.loc[o, "OR"], pr.loc[o, "lo"], pr.loc[o, "hi"], "OR")
ic = pd.read_csv(V / "independent_landmark_cox_imaging.csv")


def icx(o, adj="M2"):
    return ic[(ic.outcome == o) & (ic.adj == adj)].iloc[0]


r = icx("hbve"); F["ind_hbve"] = full(r.HR, r.lo, r.hi); F["ind_hbve_short"] = full(r.HR, r.lo, r.hi); F["ind_hbve_num"] = f"{d(r.HR, 2)} ({ci(r.lo, r.hi, 2)})"
F["ind_hbve_ev"], F["ind_hbve_n"] = n(r.events), n(r.n)
r = icx("hbve", "M1"); F["ind_hbve_m1"] = full(r.HR, r.lo, r.hi); F["ind_hbve_m1_ev"] = n(r.events)
r = icx("hbve", "M2_grp>=3vs0"); F["ind_hbve_g3"] = full(r.HR, r.lo, r.hi)
for o, k in [("mace", "ind_mace"), ("stroke", "ind_stroke"), ("hf", "ind_hf"), ("af", "ind_af"), ("ihd", "ind_ihd"),
             ("cv_death", "ind_cvd"), ("cerebrovascular", "ind_cer"), ("cardiac", "ind_card"), ("depression", "ind_dep"),
             ("ckd", "ind_ckd"), ("parkinson", "ind_pd")]:
    r = icx(o); F[k] = full(r.HR, r.lo, r.hi); F[k + "_ev"] = n(r.events)
F["ind_cvd_ev"] = F["ind_cvd_ev"]
lmk = pd.read_csv(V / "independent_landmark_cox.csv"); r = lmk[(lmk.outcome == "hbve") & (lmk.adj == "M2")].iloc[0]
F["lmk_hbve"] = full(r.HR, r.lo, r.hi); F["lmk_hbve_ev"] = n(r.events)
pe = pd.read_csv(V / "independent_involvement_preimaging_imaging.csv"); pe = pe[(pe.scheme == "obs_cat") & (pe.outcome == "stroke")].set_index("level")
ie = pd.read_csv(V / "independent_involvement_landmark_imaging.csv"); ie = ie[(ie.scheme == "obs_cat") & (ie.outcome == "stroke")].set_index("level")
F["oc_pre_brain"] = bare(pe.loc["Brain only", "OR"], pe.loc["Brain only", "lo"], pe.loc["Brain only", "hi"])
F["oc_pre_both"] = full(pe.loc["Heart and brain", "OR"], pe.loc["Heart and brain", "lo"], pe.loc["Heart and brain", "hi"], "OR")
F["oc_pre_heart"] = full(pe.loc["Heart only", "OR"], pe.loc["Heart only", "lo"], pe.loc["Heart only", "hi"], "OR")
F["oc_inc_heart"] = bare(ie.loc["Heart only", "HR"], ie.loc["Heart only", "lo"], ie.loc["Heart only", "hi"])
F["oc_inc_both"] = full(ie.loc["Heart and brain", "HR"], ie.loc["Heart and brain", "lo"], ie.loc["Heart and brain", "hi"])
F["oc_inc_brain"] = full(ie.loc["Brain only", "HR"], ie.loc["Brain only", "lo"], ie.loc["Brain only", "hi"])
ie2 = pd.read_csv(V / "independent_involvement_landmark_imaging.csv"); ie2 = ie2[(ie2.scheme == "obs_cat") & (ie2.outcome == "hf")].set_index("level")
F["oc_hf_heart"] = full(ie2.loc["Heart only", "HR"], ie2.loc["Heart only", "lo"], ie2.loc["Heart only", "hi"])
F["oc_hf_brain"] = full(ie2.loc["Brain only", "HR"], ie2.loc["Brain only", "lo"], ie2.loc["Brain only", "hi"])
dv = pd.read_csv(V / "independent_discrimination_vs_counts_imaging.csv").set_index("outcome").loc["hbve"]
F["ind_c_rf"], F["ind_c_stage"] = d(dv.C_rf, 3), d(dv.C_rf_p10, 3)
F["ind_c_count"], F["ind_c_meanz"] = d(dv.C_rf_n_abn, 3), d(dv.C_rf_mean_z, 3)
try:
    i2 = pd.read_csv(V / "independent_score2_increment_imaging.csv").set_index("outcome")
    q = i2.loc["hbve"]
    F["ind_c0"], F["ind_c1"] = d(q.C_score2, 3), d(q.C_score2_stage, 3)
    F["ind_s2_n"], F["ind_s2_ev"] = n(q.n), n(q.events)
    F["ind_s2_hr"] = f"{d(q.HR_stage)} (95% CI {ci(q.HR_lo, q.HR_hi)})"
    F["ind_dc"] = f"{d(q.dC, 3)}, 95% CI {ci(q.dC_lo, q.dC_hi, 3)}"; F["ind_lr"] = pval(q.LR_p)
    for o, k in [("stroke", "ind_dc_stroke"), ("hf", "ind_dc_hf"), ("mace", "ind_dc_mace")]:
        q = i2.loc[o]; F[k] = f"{d(q.dC, 3)} (95% CI {ci(q.dC_lo, q.dC_hi, 3)})"
    s2 = pd.read_csv(V / "independent_score2_strata_imaging.csv"); s2 = s2[s2.outcome == "hbve"]
    for a, sg in [("is2hi", "High or very high"), ("is2lo", "Low-to-moderate")]:
        for b, gg in [("g0", "Stage 0"), ("g12", "Stage 1-2"), ("g3", "Stage >=3")]:
            F[f"{a}_{b}"] = rate(s2, sg, gg)
except FileNotFoundError:
    print("independent SCORE2 files not ready")

# de novo refit in independent subsample
rh = pd.read_csv(R / "replication_sustain/denovo_heart_first_K2.csv").set_index("subtype")
rs = pd.read_csv(R / "replication_sustain/denovo_similarity_K2.csv")
F["rep_n"] = n(6000); F["rep_f1"], F["rep_f2"] = pct(rh.loc[1, "fraction"]), pct(rh.loc[2, "fraction"])
F["rep_p1"], F["rep_p2"] = d(rh.loc[1, "p_heart_before_brain"]), d(rh.loc[2, "p_heart_before_brain"])
q = rs[(rs.derivation_model == "all10")]
F["rep_rho1"] = d(q[(q.denovo_subtype == 1) & (q.derivation_subtype == 1)].spearman.iloc[0])
F["rep_rho2"] = d(q[(q.denovo_subtype == 2) & (q.derivation_subtype == 2)].spearman.iloc[0])

iv = pd.read_csv(V / "independent_discrimination_vs_counts_imaging.csv")
_dd = pd.concat([iv[f"C_rf_{c}"] - iv.C_rf for c in ["p10", "n_abn", "mean_z"]])
F["ind_cmp_min"], F["ind_cmp_max"] = d(_dd.min(), 3), d(_dd.max(), 3)

# sex-specific analysis (sex_stratified.py)
_sx = pd.read_csv(R / "sex_stratified.csv")
for _smp, _pre in [("derivation", "sex_der"), ("independent", "sex_ind")]:
    _g = _sx[_sx["sample"] == _smp].set_index("sex")
    F[_pre + "_f"] = full(_g.loc["Female", "HR"], _g.loc["Female", "lo"], _g.loc["Female", "hi"])
    F[_pre + "_m"] = bare(_g.loc["Male", "HR"], _g.loc["Male", "lo"], _g.loc["Male", "hi"])
    F[_pre + "_pint"] = sig2(_g.loc["Interaction (male vs female)", "p"])
# regional anatomy (results/regional; independent sample, model A unless stated)
_RG = R / "regional"
_dic = pd.read_csv(_RG / "regional_dictionary.csv")
_cp = pd.read_csv(_RG / "regional_coupling.csv")
_ts = pd.read_csv(_RG / "regional_tissue_summary.csv")
_sm = pd.read_csv(_RG / "regional_stage_means.csv").merge(_dic[["var", "tissue"]], on="var", how="left")
_sm.loc[_sm["var"] == "WMH_burden_Z_raw", "tissue"] = "White-matter hyperintensity"
F["reg_n_brain"] = n((_dic.organ == "brain").sum() + 1)          # + whole-brain WMH
F["reg_n_cort"] = n((_dic.tissue == "Cortical thickness").sum())
F["reg_n_wm"] = n((_dic.tissue == "White-matter free water").sum())
F["reg_n_heart"] = n((_dic.organ == "heart").sum())
_si = _sm[_sm["sample"] == "independent"]
_med = _si.groupby(["tissue", "stage_grp"]).mean_z.median().unstack()
for _k, _t in (("seg", None), ("cort", "Cortical thickness"), ("wm", "White-matter free water"),
               ("wmh", "White-matter hyperintensity"), ("hip", "Hippocampal volume")):
    if _t is None:
        _m = _si[_si.tissue.isin(["Wall thickness", "Circumferential strain", "Radial strain"])].groupby("stage_grp").mean_z.median()
    else:
        _m = _med.loc[_t]
    F[f"reg_{_k}_s0"], F[f"reg_{_k}_s12"], F[f"reg_{_k}_s3"] = d(_m["Stage 0"]), d(_m["Stage 1-2"]), d(_m["Stage >=3"])
def _cpl(ex, smp="independent", model="A"):
    return _cp[(_cp["sample"] == smp) & (_cp.model == model) & (_cp.exposure == ex)]
_cc = _cpl("cardiac_comp")
for _k, _t in (("cort", "Cortical thickness"), ("wm", "White-matter free water"), ("hip", "Hippocampal volume")):
    _g = _cc[_cc.tissue == _t]
    F[f"reg_sig_{_k}"] = f"{(_g.q < 0.05).sum()} of {len(_g)}"
_w = _cc[_cc["var"] == "WMH_burden_Z_raw"].iloc[0]
F["reg_wmh_b"] = short(_w.beta, _w.lo, _w.hi, 3)
_top = _cc[(_cc.organ == "brain") & (_cc["var"] != "WMH_burden_Z_raw")].sort_values("beta", ascending=False)
F["reg_top_range"] = f"{d(_top.beta.iloc[3], 3)}\u2013{d(_top.beta.iloc[0], 3)}"   # top four tracts (ACR L/R, ATR R, ALIC R)
def _ts1(smp, model, ct):
    r = _ts[(_ts["sample"] == smp) & (_ts.model == model) & (_ts.contrast == ct)].iloc[0]
    return r
for _k, _ct in (("wm", "White-matter free water"), ("cort", "Cortical thickness"), ("hip", "Hippocampal volume"),
                ("diff", "WM free water minus cortical thickness")):
    _r = _ts1("independent", "A", _ct); F[f"reg_ts_{_k}"] = short(_r.median_beta, _r.lo, _r.hi, 3)
_r = _ts1("independent", "A", "WM free water minus cortical thickness"); F["reg_ts_diff_p"] = d(max(_r.p_boot, 0.001), 3)
_r = _ts1("independent", "B", "WM free water minus cortical thickness"); F["reg_ts_diff_B"] = short(_r.median_beta, _r.lo, _r.hi, 3)
_r = _ts1("derivation", "A", "WM free water minus cortical thickness"); F["reg_ts_diff_der"] = short(_r.median_beta, _r.lo, _r.hi, 3)
_wmc, _gmc = _cpl("brain_wm_comp"), _cpl("brain_gm_comp")
_wt = _wmc[_wmc.tissue == "Wall thickness"]
F["reg_wt_sig"] = f"{(_wt.q < 0.05).sum()} of {len(_wt)}"
F["reg_wt_med"] = d(_wt.beta.median(), 3)
F["reg_gm_seg_med"] = d(_gmc[_gmc.tissue.isin(["Wall thickness", "Circumferential strain", "Radial strain"])].beta.median(), 3)
def _glob(df, key):
    r = df[df.label.str.startswith(key)].iloc[0]; return short(r.beta, r.lo, r.hi, 3)
F["reg_wm_mass"] = _glob(_wmc, "LV myocardial mass")
F["reg_wm_ado"] = _glob(_wmc, "Descending aortic distensibility")
F["reg_wm_aao"] = _glob(_wmc, "Ascending aortic distensibility")
F["reg_gm_mass"] = _glob(_gmc, "LV myocardial mass")
for _k, _ex in (("aos", "Aortic_stiffness_Z_raw"), ("lvr", "LV_remodeling_Z_raw"), ("lvd", "LV_myocardial_dysfunction_Z_raw")):
    _g = _cpl(_ex)
    F[f"reg_dec_{_k}_wm"] = d(_g[_g.tissue == "White-matter free water"].beta.median(), 3)
    F[f"reg_dec_{_k}_wmh"] = d(_g[_g["var"] == "WMH_burden_Z_raw"].beta.iloc[0], 3)
# v7: subtype-specific regional anatomy (results/regional_subtype/)
_RS = R / "regional_subtype"
_rd = pd.read_csv(_RS / "dictionary_ext.csv").set_index("var")
_rc = pd.read_csv(_RS / "subtype_contrast.csv"); _rc["tissue"] = _rc["var"].map(_rd.tissue)
_rr = pd.read_csv(_RS / "contrast_replication.csv").set_index("set")
_rt = pd.read_csv(_RS / "tissue_summary_ext.csv")
_pri = _rc[_rc.analysis == "primary"]
_n_reg = _pri[_pri["sample"] == "independent"]["var"].nunique()
F["rs_n"] = n(_n_reg)
F["rs_n_brain"] = n((_rd.loc[_pri[_pri["sample"] == "independent"]["var"]].organ == "brain").sum())
F["rs_n_heart"] = n((_rd.loc[_pri[_pri["sample"] == "independent"]["var"]].organ == "heart").sum())
F["rs_n_oom"] = n(int(_rr.loc["out-of-model", "n_regions"]))
for _k, _t in (("wt", "Wall thickness"), ("ao", "Aortic distensibility"), ("hip", "Hippocampal volume"),
               ("ct", "Cortical thickness"), ("sc", "Subcortical volume"), ("fw", "White-matter free water")):
    for _smp, _s in (("derivation", "der"), ("independent", "ind")):
        F[f"rs_{_k}_{_s}"] = d(_pri[(_pri["sample"] == _smp) & (_pri.tissue == _t)].beta.median())
F["rs_rho"] = d(_rr.loc["all regions", "spearman_rho"])
F["rs_rho_oom"] = d(_rr.loc["out-of-model", "spearman_rho"])
F["rs_sign"] = pct(_rr.loc["all regions", "sign_concordance"], 0)
for _k, _ct in (("fa", "White-matter FA minus cortical thickness"), ("md", "White-matter MD minus cortical thickness"),
                ("sc", "Subcortical volume minus cortical thickness")):
    _r = _rt[(_rt["sample"] == "independent") & (_rt.model == "A") & (_rt.contrast == _ct)].iloc[0]
    F[f"rs_{_k}_diff"] = short(_r.median_beta, _r.lo, _r.hi, 3)
_ip = pd.read_pickle(R / "validation/independent_cohort_full_imaging.pkl")
_ip = _ip[(_ip.stage >= 1) & _ip.subtype.isin([1, 2])]
F["rs_psub_ind"] = d(_ip.p_subtype.median())
_dp = pd.read_pickle(R / "cohort_derivation.pkl")
_dp = _dp[(_dp.all10_stage >= 1) & _dp.all10_subtype.isin([1, 2])]
F["rs_psub_der"] = d(_dp.all10_p_subtype.median())
# de novo refit: number of shared MRI events in the rank comparison
_sim = pd.read_csv(R / "replication_sustain/denovo_similarity_K2.csv")
F["rep_n_shared"] = n(_sim.n_shared_events.iloc[0])
# appendix map
_AP = json.load(open(_os.path.join(HB_ROOT, "manuscript/output/appendix_pages.json")))
def _ap(*keys):
    pg = sorted({_AP[k] for k in keys})
    if len(pg) == 1:
        return f"p {pg[0]}"
    return f"pp {pg[0]}\u2013{pg[-1]}" if pg[-1] - pg[0] == len(pg) - 1 else "pp " + ", ".join(map(str, pg))
_MAP = dict(app_icd=("t1",), app_miss=("t2",), app_cvic=("t3",), app_seq=("t4",), app_hf=("t5",), app_denovo=("t6", "f1"),
            app_agree=("t7",), app_corr=("t8",), app_coab=("t9",), app_cox_all=("t10",), app_rates=("t11",), app_ind=("t12",),
            app_sens=("t13",), app_ph=("t14",), app_landmark=("t15",), app_sub=("t16",), app_s2=("t17",), app_strata=("t18",),
            app_cmp=("t19",), app_inv=("t20",), app_sex=("t21",), app_m2=("m2",), app_m3=("m3",), app_m4=("m4",), app_m5=("m5",), app_m6=("m6",), app_flow=("fS1",), app_regm=("m7", "m8"), app_reg=("t22", "fS3"), app_regs=("m8", "t23", "fS4"))
F.update({k: _ap(*v) for k, v in _MAP.items()})
json.dump(F, open(_os.path.join(HB_ROOT, "manuscript/facts.json"), "w"), ensure_ascii=False, indent=1)
for k in sorted(F):
    print(f"{k:22s} {F[k]}")
