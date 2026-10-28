"""Build the derivation analytic cohort (n=3770) with imaging-anchored outcomes.
Time zero = imaging visit (field 53-2.0). Outcomes = first HES inpatient ICD-10 diagnosis (fields 41270/41280)
or, for cardiovascular death, the death registry (fields 40000/40001). Participants with a diagnosis on or before
the imaging date are prevalent cases for that outcome. Follow-up ends at the first event, death or the
administrative date 2021-09-30 (last date in the HES extract), whichever came first."""
import numpy as np
import pandas as pd
import common as C

R = C.RAW
# ---------------------------------------------------------------- dates
v = pd.read_csv(R / "up/score_reconstruction/e_cohort_visits.csv", parse_dates=["53-0.0", "53-2.0"])
v = v.rename(columns={"eid": "eid", "53-0.0": "date_baseline", "53-2.0": "date_imaging"})[["eid", "date_baseline", "date_imaging"]]

# ---------------------------------------------------------------- covariates (baseline, instance 0)
cov = pd.read_csv(R / "ukb_area/data_all.csv", usecols=[
    "Eid", "Sex", "Age", "Smoking_status", "Diabetes", "Hypertension", "CHOL", "HDLc", "LDLc", "SBP", "DBP",
    "HbA1c", "BMI", "APOE4", "Ethnicity", "townsend_index", "CysC", "Creatinine"]).rename(columns={"Eid": "eid"})
d = v.merge(cov, on="eid", how="left")
d["male"] = (d.Sex == 1).astype(int)
d["gap_years"] = (d.date_imaging - d.date_baseline).dt.days / C.YEAR
d["age_imaging"] = d.Age + d.gap_years
d["current_smoker"] = (d.Smoking_status == 2).astype(float)
d.loc[d.Smoking_status < 0, "current_smoker"] = np.nan
d["white"] = (d.Ethnicity == "White").astype(float)
risk, lp = C.score2(d.age_imaging, d.male == 1, d.current_smoker, d.SBP, d.Diabetes, d.CHOL, d.HDLc)
d["score2"] = risk; d["score2_lp"] = lp
d["score2_cat"] = C.score2_category(d.age_imaging, risk)

# ---------------------------------------------------------------- SuStaIn final assignments
for m, meta in C.MODELS.items():
    a = pd.read_csv(C.FINAL / m / "selected_subject_subtype_stage.csv")
    seq = pd.read_csv(C.FINAL / m / "selected_event_sequences.csv")
    # map original (0-based) subtype -> display number used in the final sequence export
    o2d = seq.drop_duplicates("subtype_original_0based").set_index("subtype_original_0based").subtype_display.to_dict()
    pk = C.load_pickle(m)
    assert np.allclose(pk["ml_stage"].ravel(), a.ml_stage.values)
    assert np.array_equal(pk["ml_subtype"].ravel().astype(int), a.ml_subtype_0based.values.astype(int))
    ix = np.argsort(pk["samples_f"].mean(1))[::-1]
    assert [o2d[o] for o in ix] == list(range(1, len(ix) + 1)), "display order != f-rank order"
    a = a.assign(**{
        f"{m}_stage": a.ml_stage.astype(int),
        f"{m}_stage_norm": a.ml_stage / meta["n_events"],
        # pySuStaIn subtype_and_stage_individuals() re-orders subtypes by descending mean MCMC f, so the
        # individual index is a prevalence rank; display number = rank + 1 (verified against z-profiles).
        f"{m}_subtype": a.ml_subtype_0based.astype(int) + 1,
        f"{m}_p_subtype": a.prob_ml_subtype,
        f"{m}_p_stage": a.prob_ml_stage,
        # posterior expected stage (uses the full stage posterior rather than the ML stage)
        f"{m}_stage_exp": (pk["prob_stage"] * np.arange(pk["prob_stage"].shape[1])).sum(1) / pk["prob_stage"].sum(1),
    })
    d = d.merge(a[["eid"] + [c for c in a.columns if c.startswith(m + "_")]], on="eid", how="left")

# raw SuStaIn inputs (non-negative abnormality z) for comparator scores
z = C.load_inputs("all10")
z.columns = ["eid"] + ["z_" + c.replace("_Z", "") for c in z.columns[1:]]
d = d.merge(z, on="eid", how="left")
zc = [c for c in d.columns if c.startswith("z_")]
d["n_abnormal_z1"] = (d[zc] >= 1).sum(1)
d["mean_abn_z"] = d[zc].mean(1)
heart = ["z_LV_remodeling", "z_LV_myocardial_dysfunction", "z_Aortic_stiffness"]
brain = ["z_Hippocampal_atrophy", "z_Cortical_thinning", "z_WMH_burden", "z_FreeWater_ISOVF"]
d["heart_any_z1"] = (d[heart] >= 1).any(axis=1).astype(int)
d["brain_any_z1"] = (d[brain] >= 1).any(axis=1).astype(int)

# ---------------------------------------------------------------- outcomes from HES
long = C.hes_long(d.eid, C.ROOT / "analysis/hes_long_cohort.pkl").rename(columns={"Eid": "eid"})
long = long.dropna(subset=["date"])
img = d.set_index("eid").date_imaging
deaths = C.load_deaths()
dd = deaths.set_index("eid").death_date.reindex(d.eid)
dd.index = d.index
d["death_date"] = dd
# end of follow-up = earliest of death and administrative censoring (HES complete to 30 Sep 2021)
d["end_fu"] = d.death_date.where(d.death_date.notna() & (d.death_date < C.ADMIN_END), C.ADMIN_END)
first = {}
for k, pref in C.ICD.items():
    hes = long[long.code.str.startswith(tuple(pref))].groupby("eid").date.min()
    fatal = C.death_event_date(deaths, pref)            # deaths with this underlying cause and no admission
    first[k] = pd.concat([hes, fatal], axis=1).min(axis=1)
first["cv_death"] = C.death_event_date(deaths, C.CV_DEATH_PREFIX)
first["death"] = deaths.dropna(subset=["death_date"]).set_index("eid").death_date
for k, comps in C.COMPOSITES.items():
    first[k] = pd.concat([first[c] for c in comps], axis=1).min(axis=1)
for k, s in first.items():
    s = s.reindex(d.eid)
    s.index = d.index
    prev = s.notna() & (s.values <= d.date_imaging.values)
    inc = s.notna() & (s.values > d.date_imaging.values) & (s.values <= d.end_fu.values)
    end = np.where(inc, s, d.end_fu)
    d[f"prev_{k}"] = prev.astype(int)
    d[f"ev_{k}"] = inc.astype(int)
    d[f"t_{k}"] = (pd.to_datetime(end) - d.date_imaging).dt.days / C.YEAR
d["fu_years"] = (d.end_fu - d.date_imaging).dt.days / C.YEAR
print("deaths after imaging to admin end:", int(((d.death_date > d.date_imaging) & (d.death_date <= C.ADMIN_END)).sum()),
      "| deaths before imaging (should be 0):", int((d.death_date <= d.date_imaging).sum()))
d["prev_any_cvd"] = d[[f"prev_{k}" for k in ["stroke", "tia", "ihd", "hf", "af", "pad"]]].max(axis=1)

d.to_pickle(C.RES / "cohort_derivation.pkl")
print("n", len(d), "| median FU", d.fu_years.median().round(2), d.fu_years.quantile([.25, .75]).round(2).tolist())
print("SCORE2 median", d.score2.median().round(2), "missing", d.score2.isna().sum(), "cat", d.score2_cat.value_counts().to_dict())
for k in C.OUTCOME_LABEL:
    print(f"{k:18s} prevalent {d[f'prev_{k}'].sum():4d}  incident {d.loc[d[f'prev_{k}']==0, f'ev_{k}'].sum():4d}")
for m in C.MODELS:
    print(m, d.groupby(f"{m}_subtype")[f"{m}_stage"].agg(["size", lambda s: (s > 0).sum(), "mean"]).round(2).to_dict())
