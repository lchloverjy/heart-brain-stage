"""Bootstrap: 1000 resamples of the C-index difference with coefficients fixed at full-sample estimates.
Independent sample (imaging-anchored follow-up): incremental value of the MRI-only heart-brain stage beyond SCORE2
and event rates by stage within SCORE2 risk categories. Mirrors outcomes.py (derivation) so that both samples are
reported on the same scale. Usage: python3 independent_score2.py [n_boot]"""
import sys
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from multiprocessing import Pool
import common as C
warnings.filterwarnings("ignore")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
V = C.RES / "validation"
d = pd.read_pickle(V / "independent_cohort_full_imaging.pkl")
deaths = C.load_deaths()
OUTS = ["hbve", "mace", "cerebrovascular", "cardiac", "stroke", "af", "hf", "ihd", "cv_death"]
d["score2_cll"] = np.log(-np.log(1 - (d.score2_baseline / 100).clip(1e-4, 0.999)))   # SCORE2 stored in %


def frame(o):
    f = C.prospective_frame(d, d[f"first_{o}"], d.date_imaging, d.death_date, C.ADMIN_END)
    return f[["T", "E", "score2_cll", "p10", "score2_cat", "grp"]].dropna(subset=["T", "E", "score2_cll", "p10"])


def cidx(f, cols):
    m = CoxPHFitter(penalizer=1e-4).fit(f[["T", "E"] + cols], "T", "E")
    return concordance_index(f["T"], -m.predict_partial_hazard(f), f["E"]), m


def boot(args):
    o, seed = args
    f, lp0, lp1 = FR[o], LP0[o], LP1[o]; rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(f), len(f))
    T, E = f["T"].values[ix], f["E"].values[ix]
    try:
        return o, concordance_index(T, -lp1[ix], E) - concordance_index(T, -lp0[ix], E)
    except Exception:
        return o, np.nan


FR = {o: frame(o) for o in OUTS}
rows = []; LP0 = {}; LP1 = {}
for o, f in FR.items():
    c0, m0 = cidx(f, ["score2_cll"]); c1, m1 = cidx(f, ["score2_cll", "p10"])
    LP0[o] = m0.predict_log_partial_hazard(f).values; LP1[o] = m1.predict_log_partial_hazard(f).values
    lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    from scipy.stats import chi2
    s = m1.summary.loc["p10"]
    rows.append(dict(outcome=o, n=len(f), events=int(f.E.sum()), C_score2=c0, C_score2_stage=c1, dC=c1 - c0, LR_chi2=lr,
                     LR_p=chi2.sf(lr, 1), HR_stage=np.exp(s["coef"]), HR_lo=np.exp(s["coef lower 95%"]), HR_hi=np.exp(s["coef upper 95%"])))
    print(o, len(f), int(f.E.sum()), round(c0, 4), round(c1, 4), flush=True)
res = pd.DataFrame(rows)
jobs = [(o, 1000 + i) for o in OUTS for i in range(NB)]
with Pool(6) as p:
    out = p.map(boot, jobs, chunksize=20)
bd = pd.DataFrame(out, columns=["outcome", "dC"])
q = bd.groupby("outcome").dC.quantile([0.025, 0.975]).unstack()
res["dC_lo"] = res.outcome.map(q[0.025]); res["dC_hi"] = res.outcome.map(q[0.975]); res["n_boot"] = NB
res.to_csv(V / "independent_score2_increment_imaging.csv", index=False)
# event rates by stage within SCORE2 categories
st = []
for o in ["hbve", "mace", "stroke"]:
    f = FR[o].copy()
    f["score2_group"] = np.where(f.score2_cat >= 1, "High or very high", "Low-to-moderate")
    f["stage_group"] = np.select([f.grp == "0", f.grp.isin(["1", "2"])], ["Stage 0", "Stage 1-2"], "Stage >=3")
    for (a, b), g in f.groupby(["score2_group", "stage_group"]):
        ev, py = int(g.E.sum()), g["T"].sum()
        from scipy.stats import chi2 as _c
        lo = _c.ppf(0.025, 2 * ev) / 2 / py * 1000 if ev > 0 else 0.0
        hi = _c.ppf(0.975, 2 * (ev + 1)) / 2 / py * 1000
        st.append(dict(outcome=o, score2_group=a, stage_group=b, n=len(g), events=ev, py=py, rate=1000 * ev / py, rate_lo=lo, rate_hi=hi))
pd.DataFrame(st).to_csv(V / "independent_score2_strata_imaging.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string()); print(pd.DataFrame(st).round(2).to_string())
print("score2_cat values:", d.score2_cat.value_counts(dropna=False).to_dict())
