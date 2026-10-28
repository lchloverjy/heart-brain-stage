import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Shared paths, constants and loaders for the v4 heart-brain SuStaIn re-analysis."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(HB_ROOT)
RAW = Path(_os.environ.get("HB_RAW", ROOT / "raw"))
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
UPLOAD = Path(UKB_DATA)

PREP = RAW / "up/module3_preparation"
FINAL = RAW / "up2/sustain_final"
MODELS = {
    "all10": dict(label="All-10", pickle="all10_final_subtype1.pickle", n_events=30),
    "mri_plus_m13": dict(label="MRI + M13", pickle="mri_plus_m13_final_subtype1.pickle", n_events=24),
    "mri_only_matched_all10": dict(label="MRI only", pickle="mri_only_matched_all10_final_subtype2.pickle", n_events=18),
}
ADMIN_END = pd.Timestamp("2021-09-30")   # last date in the HES extract (fields 41270/41280)
YEAR = 365.25

FEATURE_LABEL = {
    "Protein_M13_Z": "Protein M13", "Protein_M20_Z": "Protein M20", "Protein_M5_Z": "Protein M5",
    "LV_remodeling_Z": "LV remodelling", "LV_myocardial_dysfunction_Z": "LV dysfunction",
    "Aortic_stiffness_Z": "Aortic stiffness", "Hippocampal_atrophy_Z": "Hippocampal atrophy",
    "Cortical_thinning_Z": "Cortical thinning", "WMH_burden_Z": "WMH burden",
    "FreeWater_ISOVF_Z": "Free water",
}
FEATURE_ORGAN = {
    "Protein_M13_Z": "Protein", "Protein_M20_Z": "Protein", "Protein_M5_Z": "Protein",
    "LV_remodeling_Z": "Heart", "LV_myocardial_dysfunction_Z": "Heart", "Aortic_stiffness_Z": "Heart",
    "Hippocampal_atrophy_Z": "Brain", "Cortical_thinning_Z": "Brain", "WMH_burden_Z": "Brain",
    "FreeWater_ISOVF_Z": "Brain",
}

# ICD-10 definitions (prefix match on HES inpatient diagnoses, fields 41270/41280)
ICD = {
    "stroke": ["I60", "I61", "I63", "I64"],
    "ischaemic_stroke": ["I63"],
    "tia": ["G45"],
    "mi": ["I21", "I22", "I23"],
    "ihd": ["I20", "I21", "I22", "I23", "I24", "I25"],
    "af": ["I48"],
    "hf": ["I50", "I110", "I130", "I132"],
    "pad": ["I702", "I739"],
    "dementia": ["F00", "F01", "F02", "F03", "G30", "G311"],
    "depression": ["F32", "F33"],
    "anxiety": ["F40", "F41"],
    "parkinson": ["G20"],
    "ckd": ["N18"],
}
COMPOSITES = {
    # primary: any incident heart-brain vascular event
    "hbve": ["stroke", "tia", "ihd", "hf", "af"],
    # MACE-type hard events
    "mace": ["stroke", "mi", "hf"],
    "cerebrovascular": ["stroke", "tia"],
    "cardiac": ["ihd", "hf", "af"],
    # sensitivity: MACE including cardiovascular death
    "mace_cvd": ["stroke", "mi", "hf", "cv_death"],
}
OUTCOME_LABEL = {
    "hbve": "Heart-brain vascular events", "mace": "Stroke, MI or heart failure",
    "cerebrovascular": "Stroke or TIA", "cardiac": "Cardiac events (IHD, HF, AF)",
    "stroke": "Stroke", "ischaemic_stroke": "Ischaemic stroke", "tia": "TIA", "mi": "Myocardial infarction",
    "ihd": "Ischaemic heart disease", "af": "Atrial fibrillation", "hf": "Heart failure", "pad": "PAD",
    "dementia": "Dementia", "depression": "Depression", "anxiety": "Anxiety", "parkinson": "Parkinson's disease",
    "ckd": "Chronic kidney disease", "cv_death": "Cardiovascular death", "death": "All-cause death",
    "mace_cvd": "Stroke, MI, heart failure, or CV death",
}


def load_pickle(model):
    p = FINAL / model / "pickle_files" / MODELS[model]["pickle"]
    with open(p, "rb") as fh:
        return pickle.load(fh)


def load_thresholds(model):
    return pd.read_csv(PREP / "04_thresholds" / f"thresholds_{model}.csv")


def load_inputs(model):
    return pd.read_csv(PREP / "03_inputs" / f"sustain_{model}.csv")


def event_table(model):
    """Return the (biomarker, z) event list in pySuStaIn's internal stage_zscore order."""
    th = load_thresholds(model)
    zv = th[["z1", "z2", "z3"]].to_numpy(float)
    rows = []
    # pySuStaIn ZscoreSustain: stage_zscore = Z_vals.T.flatten() (non-zero only); stage_biomarker_index likewise
    nb = zv.shape[0]
    IX_vals = np.array([[x for x in range(nb)]] * 3).T
    stage_z = zv.T.flatten(); stage_b = IX_vals.T.flatten()
    keep = stage_z != 0
    for k, (b, z) in enumerate(zip(stage_b[keep], stage_z[keep])):
        f = th.biomarker.iloc[b]
        rows.append(dict(event_index=k, biomarker=f, label=FEATURE_LABEL[f], organ=FEATURE_ORGAN[f], z=int(z)))
    return pd.DataFrame(rows)


def score2(age, male, smoker, sbp, diabetes, tchol, hdl):
    """SCORE2 / SCORE2-OP (low-risk region, UK). Returns 10-year CVD risk (%) and the uncalibrated LP.
    Coefficients: SCORE2 working group, Eur Heart J 2021;42:2439-54; SCORE2-OP, Eur Heart J 2021;42:2455-67
    (cross-checked against CRAN RiskScorescvd::SCORE2)."""
    age = np.asarray(age, float); male = np.asarray(male, bool)
    smoker = np.asarray(smoker, float); sbp = np.asarray(sbp, float); dm = np.asarray(diabetes, float)
    tc = np.asarray(tchol, float); hdl = np.asarray(hdl, float)
    risk = np.full(age.shape, np.nan); lp = np.full(age.shape, np.nan)
    young = age < 70
    ca = (age - 60) / 5; cs = (sbp - 120) / 20; ct = tc - 6; ch = (hdl - 1.3) / 0.5
    lm = (0.3742 * ca + 0.6012 * smoker + 0.2777 * cs + 0.6457 * dm + 0.1458 * ct - 0.2698 * ch
          - 0.0755 * ca * smoker - 0.0255 * ca * cs - 0.0281 * ca * ct + 0.0426 * ca * ch - 0.0983 * ca * dm)
    lf = (0.4648 * ca + 0.7744 * smoker + 0.3131 * cs + 0.8096 * dm + 0.1002 * ct - 0.2606 * ch
          - 0.1088 * ca * smoker - 0.0277 * ca * cs - 0.0226 * ca * ct + 0.0613 * ca * ch - 0.1272 * ca * dm)
    r_m = 1 - 0.9605 ** np.exp(lm); r_f = 1 - 0.9776 ** np.exp(lf)
    cal = lambda r, s1, s2: 1 - np.exp(-np.exp(s1 + s2 * np.log(-np.log(1 - r))))
    sel = young & male; risk[sel] = cal(r_m[sel], -0.5699, 0.7476); lp[sel] = lm[sel]
    sel = young & ~male; risk[sel] = cal(r_f[sel], -0.7380, 0.7019); lp[sel] = lf[sel]
    a = age - 73; sb = sbp - 150; t2 = tc - 6; h2 = hdl - 1.4
    om = (0.0634 * a + 0.4245 * dm + 0.3524 * smoker + 0.0094 * sb + 0.0850 * t2 - 0.3564 * h2
          - 0.0174 * a * dm - 0.0247 * a * smoker - 0.0005 * a * sb + 0.0073 * a * t2 + 0.0091 * a * h2)
    of = (0.0789 * a + 0.6010 * dm + 0.4921 * smoker + 0.0102 * sb + 0.0605 * t2 - 0.3040 * h2
          - 0.0107 * a * dm - 0.0255 * a * smoker - 0.0004 * a * sb - 0.0009 * a * t2 + 0.0154 * a * h2)
    ro_m = 1 - 0.7576 ** np.exp(om - 0.0929); ro_f = 1 - 0.8082 ** np.exp(of - 0.229)
    sel = ~young & male; risk[sel] = cal(ro_m[sel], -0.34, 1.19); lp[sel] = om[sel]
    sel = ~young & ~male; risk[sel] = cal(ro_f[sel], -0.52, 1.01); lp[sel] = of[sel]
    return 100 * risk, lp


def score2_category(age, risk):
    """ESC 2021 age-specific categories: 0 low-to-moderate, 1 high, 2 very high."""
    age = np.asarray(age, float); risk = np.asarray(risk, float)
    lo = np.where(age < 50, 2.5, np.where(age < 70, 5.0, 7.5))
    hi = np.where(age < 50, 7.5, np.where(age < 70, 10.0, 15.0))
    cat = np.where(risk < lo, 0, np.where(risk < hi, 1, 2)).astype(float)
    cat[np.isnan(risk)] = np.nan
    return cat


# ---------------------------------------------------------------- death registry (fields 40000, 40001, 40002)
DEATH_CSV = UPLOAD / "ALL_death.csv"
CV_DEATH_PREFIX = tuple(f"I{i:02d}" for i in range(100))  # underlying cause I00-I99


def load_deaths():
    """eid, death_date, underlying cause (40001-0.0), birth year/month (34, 52)."""
    cache = RES / "deaths.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    d = pd.read_csv(DEATH_CSV, usecols=["Eid", "34-0.0", "52-0.0", "40000-0.0", "40001-0.0"], low_memory=False)
    d = d.rename(columns={"Eid": "eid", "34-0.0": "birth_year", "52-0.0": "birth_month",
                          "40000-0.0": "death_date", "40001-0.0": "death_cause"})
    d["death_date"] = pd.to_datetime(d.death_date, errors="coerce")
    d["death_cause"] = d.death_cause.astype(str).str.replace(".", "", regex=False).where(d.death_cause.notna())
    d.to_pickle(cache)
    return d


def death_event_date(deaths, prefixes):
    """Date of death whose underlying cause matches ICD-10 prefixes (fatal events without admission)."""
    m = deaths.death_cause.fillna("").str.startswith(tuple(prefixes)) & deaths.death_date.notna()
    return deaths.loc[m].set_index("eid").death_date


def first_event_date(long, deaths, keys):
    """First HES diagnosis or death with matching underlying cause (ICD-10 prefixes of C.ICD keys)."""
    pref = tuple(sum([ICD[k] for k in keys], []))
    hes = long[long.code.str.startswith(pref)].groupby("eid").date.min()
    fatal = death_event_date(deaths, pref)
    return pd.concat([hes, fatal], axis=1).min(axis=1)


def prospective_frame(d, fd, t0, death_date, t_end=None):
    """Exclude prevalent (event <= t0) and deceased (death <= t0); censor at min(death, t_end).
    t0: scalar Timestamp (common landmark) or Series aligned to d (individual imaging date)."""
    t_end = ADMIN_END if t_end is None else t_end
    t0 = pd.Series(t0, index=d.index) if not isinstance(t0, pd.Series) else t0
    keep = t0.notna() & (t0 < t_end) & ~(fd <= t0) & ~(death_date <= t0)
    f = d[keep].copy(); fdd = fd[keep]; t0k = t0[keep]; ddk = death_date[keep]
    end = ddk.where(ddk.notna() & (ddk < t_end), t_end)
    ev = fdd.notna() & (fdd > t0k) & (fdd <= end)
    f["E"] = ev.astype(int)
    f["T"] = (np.where(ev, (fdd - t0k).dt.days, (end - t0k).dt.days) / YEAR)
    f["T"] = f["T"].clip(lower=1 / YEAR)
    return f


IMAGING_DATES_CSV = None   # set when field 53 is supplied (see validation_independent.py --design imaging)


def load_imaging_dates():
    """Return DataFrame eid, date_imaging (53-2.0) if a field-53 export is present in UKB_DATA."""
    import glob
    for f in sorted(glob.glob(str(UPLOAD / "*.csv"))):
        try:
            cols = pd.read_csv(f, nrows=0).columns
        except Exception:
            continue
        if "53-2.0" in cols:
            idc = "Eid" if "Eid" in cols else ("eid" if "eid" in cols else cols[0])
            x = pd.read_csv(f, usecols=[idc, "53-2.0"]).rename(columns={idc: "eid", "53-2.0": "date_imaging"})
            x["date_imaging"] = pd.to_datetime(x.date_imaging, format="%d/%m/%Y", errors="coerce")
            return x.dropna()
    return None



def hes_long(ids, cache_path):
    """Long table (Eid, ICD-10 code, diagnosis date) of all HES inpatient diagnoses (fields 41270/41280)
    for the given participant ids; cached as a pickle. Used for both derivation and independent samples."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    ids = set(int(i) for i in ids)
    parts = []
    for cc, dd in zip(pd.read_csv(UPLOAD / "41270.csv", chunksize=20000, dtype=str),
                      pd.read_csv(UPLOAD / "41280.csv", chunksize=20000, dtype=str)):
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
    long.to_pickle(cache_path)
    return long
