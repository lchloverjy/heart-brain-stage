"""Cumulative incidence by heart-brain stage group in the independent imaging sample (Figure 4B).

Same design as validation_independent.py --design imaging: time zero = individual imaging visit (field 53-2.0),
participants with the outcome on/before imaging excluded, follow-up to the hospital-data end date (2021-09-30),
deaths from other causes censored (common.prospective_frame). Stage groups come from the transferred MRI-only
model (no refitting). Output has the same layout as results/out_km.csv (curve rows + risk-table rows), and the
script checks that group sizes and event counts equal independent_landmark_incidence_imaging.csv.

Usage: python3 independent_km.py
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import common as C

V = C.RES / "validation"
d = pd.read_pickle(V / "independent_cohort_full_imaging.pkl")
inc = pd.read_csv(V / "independent_landmark_incidence_imaging.csv")
OUTS = [o for o in ["hbve", "mace", "cerebrovascular", "cardiac"] if f"first_{o}" in d.columns]
rows, chk, lr = [], [], []
for o in OUTS:
    f = C.prospective_frame(d, d[f"first_{o}"], d.date_imaging, d.death_date, C.ADMIN_END)
    for g, gg in f.groupby("grp", observed=True):
        km = KaplanMeierFitter().fit(gg["T"], gg["E"])
        ci = km.confidence_interval_cumulative_density_
        cd = km.cumulative_density_
        for t in cd.index:
            rows.append(dict(outcome=o, scheme="independent_mri", group=g, t=t, cuminc=cd.loc[t].iloc[0],
                             lo=ci.loc[t].iloc[0], hi=ci.loc[t].iloc[1]))
        for t in range(7):
            rows.append(dict(outcome=o, scheme="independent_mri", group=g, t=float(t),
                             at_risk=int((gg["T"] >= t).sum()), kind="risk_table"))
        ref = inc[(inc.outcome == o) & (inc.grp.astype(str) == str(g))]
        if len(ref):
            chk.append(dict(outcome=o, grp=g, n=len(gg), events=int(gg.E.sum()), n_ref=int(ref.n.iloc[0]),
                            events_ref=int(ref.events.iloc[0])))
    t = multivariate_logrank_test(f["T"], f["grp"].astype(str), f["E"])
    lr.append(dict(outcome=o, test="log-rank (4 groups)", chi2=t.test_statistic, df=3, p=t.p_value))
out = pd.DataFrame(rows)
out.to_csv(V / "independent_km_imaging.csv", index=False)
pd.DataFrame(lr).to_csv(V / "independent_km_logrank_imaging.csv", index=False)
chk = pd.DataFrame(chk)
print(chk.to_string(index=False))
bad = chk[(chk.n != chk.n_ref) | (chk.events != chk.events_ref)]
assert bad.empty, f"KM frame differs from incidence table:\n{bad}"
print("consistency check passed:", len(chk), "outcome x group cells")
print(pd.DataFrame(lr).to_string(index=False))
