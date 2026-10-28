import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Table 1 - characteristics of the derivation and independent imaging samples (Lancet style numbers)."""
import numpy as np
import pandas as pd
import common as C

R = _os.path.join(HB_ROOT, "results")
d = pd.read_pickle(f"{R}/cohort_derivation.pkl")
i = pd.read_pickle(f"{R}/validation/independent_cohort_full_imaging.pkl")
i["score2_img"] = C.score2(i.age_imaging, i.male == 1, i.current_smoker, i.SBP, i.Diabetes, i.CHOL, i.HDLc)[0]


def md(x, k=1):
    return f"{x:.{k}f}".replace(".", "\u00b7").replace("-", "\u2013") if pd.notna(x) else "\u2014"


def ms(s, k=1):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{md(s.mean(), k)} ({md(s.std(), k)})"


def miqr(s, k=1):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{md(s.median(), k)} ({md(s.quantile(.25), k)}\u2013{md(s.quantile(.75), k)})"


def npct(b):
    b = pd.Series(b).dropna().astype(float)
    return f"{int(b.sum()):,} ({md(100 * b.mean(), 1)}%)"


def miss(s):
    return int(pd.Series(s).isna().sum())


i["white"] = (i.Ethnicity == "White").astype(float).where(i.Ethnicity.notna() & (i.Ethnicity.astype(str) != "-1"))
rows = []
def add(lab, a, b, note=""):
    rows.append(dict(characteristic=lab, derivation=a, independent=b, note=note))

add("Participants, n", f"{len(d):,}", f"{len(i):,}")
add("Data used for staging", "Plasma proteomics, cardiac MRI, and brain MRI", "Cardiac MRI and brain MRI")
add("Age at recruitment, years", ms(d.Age), ms(i.Age_recruit))
add("Age at imaging visit, years", ms(d.age_imaging), ms(i.age_imaging))
add("Interval recruitment to imaging, years", miqr(d.gap_years), miqr(i.age_imaging - i.Age_recruit))
add("Male sex", npct(d.male), npct(i.male))
add("White ethnicity", npct(d.white), npct(i.white))
add("Body-mass index, kg/m\u00b2", ms(d.BMI), ms(i.BMI))
add("Systolic blood pressure, mm Hg", ms(d.SBP, 0), ms(i.SBP, 0))
add("Total cholesterol, mmol/L", ms(d.CHOL, 2), ms(i.CHOL, 2))
add("HDL cholesterol, mmol/L", ms(d.HDLc, 2), ms(i.HDLc, 2))
add("Current smoking", npct(d.current_smoker), npct(i.current_smoker))
add("Diabetes", npct(d.Diabetes), npct(i.Diabetes))
add("Hypertension", npct(d.Hypertension), npct(i.Hypertension))
add("SCORE2 or SCORE2-OP 10-year risk, %", miqr(d.score2), miqr(i.score2_img), "age at imaging visit; other risk factors at recruitment")
add("Heart\u2013brain vascular event before imaging\u2021", npct(d.prev_hbve), npct(i.pre_hbve), "on or before the imaging visit")
st_d = d.all10_stage
add("Heart\u2013brain stage (derivation All-10; independent MRI-only)", "", "")
add("    Stage 0", npct(st_d == 0), npct(i.stage == 0))
add("    Stage 1\u20132", npct(st_d.between(1, 2)), npct(i.stage.between(1, 2)))
add("    Stage \u22653", npct(st_d >= 3), npct(i.stage >= 3))
def fu(x):
    end = x.death_date.where(x.death_date.notna() & (x.death_date < C.ADMIN_END), C.ADMIN_END)
    return (end - x.date_imaging).dt.days / C.YEAR
add("Follow-up after imaging, years", miqr(fu(d)), miqr(fu(i)), "to death or 30 Sep 2021")
cx = pd.read_csv(f"{R}/out_cox_main.csv"); r = cx[(cx.outcome == "hbve") & (cx.model == "all10") & (cx.adj == "M1") & (cx.exposure == "per10pp")].iloc[0]
ic = pd.read_csv(f"{R}/validation/independent_landmark_cox_imaging.csv"); q = ic[(ic.outcome == "hbve") & (ic.adj == "M1")].iloc[0]
add("Incident heart\u2013brain vascular events (at risk)", f"{int(r.events)} ({int(r.n):,})", f"{int(q.events)} ({int(q.n):,})")
dd = d.death_date; di = i.death_date
add("Deaths after imaging", f"{int(((dd > d.date_imaging) & (dd <= C.ADMIN_END)).sum())}", f"{int(((di > i.date_imaging) & (di <= C.ADMIN_END)).sum())}")
t = pd.DataFrame(rows)
mis = dict(derivation={c: miss(d[c]) for c in ["BMI", "SBP", "CHOL", "HDLc", "current_smoker", "Diabetes", "Hypertension", "score2"]},
           independent={c: miss(i[c]) for c in ["BMI", "SBP", "CHOL", "HDLc", "current_smoker", "Diabetes", "Hypertension", "score2_img"]})
t.to_csv(f"{R}/table1.csv", index=False)
print(t.to_string())
print(mis)
