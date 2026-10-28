"""Deep characterisation of the final heart-brain SuStaIn models (derivation sample, n=3,770).

Outputs (results/sustain_*.csv):
  pvd_<model>.csv            posterior P(event e at position p) per subtype (positional variance diagram)
  event_position_summary.csv posterior median / 95% CrI of each event position, ML position, prevalence
  heart_first.csv            posterior probability that the first cardiac event precedes the first brain event
  prevalence_vs_order.csv    rank agreement between ML z=1 order and marginal prevalence of z>=1
  organ_involvement.csv      model-implied organ involvement by ML stage
  subtype_profiles.csv       mean abnormality z by subtype among stage-positive participants
  stage_correlates.csv       age/sex-adjusted associations of risk factors with stage
  stage_agreement.csv        cross-model stage concordance
"""
import numpy as np
import pandas as pd
from scipy import stats
import common as C

rng = np.random.default_rng(20260712)
d = pd.read_pickle(C.RES / "cohort_derivation.pkl")
THIN = 10   # use every 10th MCMC sample (10,000 of 100,000)

pos_rows, hf_rows, pv_rows, inv_rows = [], [], [], []
for m, meta in C.MODELS.items():
    ev = C.event_table(m)
    seq_exp = pd.read_csv(C.FINAL / m / "selected_event_sequences.csv")
    o2d = seq_exp.drop_duplicates("subtype_original_0based").set_index("subtype_original_0based").subtype_display.to_dict()
    pk = C.load_pickle(m)
    S = pk["samples_sequence"][:, :, ::THIN].astype(int)      # (K, N_events, n_samples): S[k, position, sample] = event index
    F = pk["samples_f"][:, ::THIN]
    K, NE, NS = S.shape
    # position of each event in each sample
    P = np.empty_like(S)
    for k in range(K):
        for s in range(NS):
            P[k, S[k, :, s], s] = np.arange(NE)
    ml = pk["ml_sequence_EM"].astype(int)
    pvd = []
    for k in range(K):
        disp = o2d[k]
        mlpos = np.empty(NE, int); mlpos[ml[k]] = np.arange(NE)
        # PVD matrix
        M = np.zeros((NE, NE))
        for e in range(NE):
            M[e] = np.bincount(P[k, e], minlength=NE) / NS
        for e in range(NE):
            r = ev.iloc[e]
            pos_rows.append(dict(model=m, subtype=disp, event_index=e, biomarker=r.biomarker, label=r.label, organ=r.organ,
                                 z=r.z, ml_position=mlpos[e] + 1, post_median=np.median(P[k, e]) + 1,
                                 post_lo=np.percentile(P[k, e], 2.5) + 1, post_hi=np.percentile(P[k, e], 97.5) + 1,
                                 post_mean=P[k, e].mean() + 1, fraction=float(np.median(F[k]))))
            pvd.append(dict(model=m, subtype=disp, event_index=e, label=r.label, organ=r.organ, z=r.z, ml_position=mlpos[e] + 1,
                            **{f"p{p + 1}": M[e, p] for p in range(NE)}))
        # heart-first: position of first heart event vs first brain event (any z level)
        hi = ev.index[ev.organ == "Heart"].to_numpy(); bi = ev.index[ev.organ == "Brain"].to_numpy()
        pi = ev.index[ev.organ == "Protein"].to_numpy()
        fh = P[k, hi].min(0); fb = P[k, bi].min(0)
        rec = dict(model=m, subtype=disp, fraction=float(np.median(F[k])), p_heart_before_brain=(fh < fb).mean(),
                   first_heart_pos_median=np.median(fh) + 1, first_brain_pos_median=np.median(fb) + 1)
        # z=1 events only
        h1 = ev.index[(ev.organ == "Heart") & (ev.z == 1)].to_numpy(); b1 = ev.index[(ev.organ == "Brain") & (ev.z == 1)].to_numpy()
        rec["p_heart_z1_before_brain_z1"] = (P[k, h1].min(0) < P[k, b1].min(0)).mean()
        # all z=1 heart events before all z=1 brain events
        rec["p_all_heart_z1_before_any_brain_z1"] = (P[k, h1].max(0) < P[k, b1].min(0)).mean()
        if len(pi):
            fp = P[k, pi].min(0)
            rec["p_protein_before_brain"] = (fp < fb).mean(); rec["p_heart_before_protein"] = (fh < fp).mean()
            rec["first_protein_pos_median"] = np.median(fp) + 1
        # aortic stiffness z1 before WMH z1 (pulsatility hypothesis)
        ao = ev.index[(ev.biomarker == "Aortic_stiffness_Z") & (ev.z == 1)][0]
        wmh = ev.index[(ev.biomarker == "WMH_burden_Z") & (ev.z == 1)][0]
        fw = ev.index[(ev.biomarker == "FreeWater_ISOVF_Z") & (ev.z == 1)][0]
        hip = ev.index[(ev.biomarker == "Hippocampal_atrophy_Z") & (ev.z == 1)][0]
        rec["p_aortic_before_wmh"] = (P[k, ao] < P[k, wmh]).mean()
        rec["p_aortic_before_freewater"] = (P[k, ao] < P[k, fw]).mean()
        rec["p_wmh_before_hippocampus"] = (P[k, wmh] < P[k, hip]).mean()
        hf_rows.append(rec)
        # prevalence vs ML order (z=1 events)
        z1 = ev[ev.z == 1]
        prev = np.array([(d["z_" + b.replace("_Z", "")] >= 1).mean() for b in z1.biomarker])
        order = mlpos[z1.index.to_numpy()]
        rho, p = stats.spearmanr(order, -prev)
        pv_rows.append(dict(model=m, subtype=disp, spearman_order_vs_prevalence=rho, p=p, n_events=len(z1),
                            order=" > ".join(z1.label.to_numpy()[np.argsort(order)]),
                            prevalence_order=" > ".join(z1.label.to_numpy()[np.argsort(-prev)])))
        # model-implied organ involvement by ML stage
        for st in range(NE + 1):
            reached = ml[k][:st]
            org = set(ev.organ.iloc[reached]) if st else set()
            inv_rows.append(dict(model=m, subtype=disp, stage=st, heart=int("Heart" in org), brain=int("Brain" in org),
                                 protein=int("Protein" in org), n_heart=int((ev.organ.iloc[reached] == "Heart").sum()) if st else 0,
                                 n_brain=int((ev.organ.iloc[reached] == "Brain").sum()) if st else 0))
    pd.DataFrame(pvd).to_csv(C.RES / f"sustain_pvd_{m}.csv", index=False)

pd.DataFrame(pos_rows).to_csv(C.RES / "sustain_event_position_summary.csv", index=False)
hf = pd.DataFrame(hf_rows); hf.to_csv(C.RES / "sustain_heart_first.csv", index=False)
pv = pd.DataFrame(pv_rows); pv.to_csv(C.RES / "sustain_prevalence_vs_order.csv", index=False)
inv = pd.DataFrame(inv_rows); inv.to_csv(C.RES / "sustain_organ_involvement_by_stage.csv", index=False)

# participant-level organ involvement category (model-implied)
lab = {(0, 0): "None", (1, 0): "Heart only", (0, 1): "Brain only", (1, 1): "Heart and brain"}
for m in C.MODELS:
    iv = inv[inv.model == m].set_index(["subtype", "stage"])
    key = list(zip(d[f"{m}_subtype"], d[f"{m}_stage"]))
    d[f"{m}_inv"] = [lab[(iv.loc[k].heart, iv.loc[k].brain)] for k in key]
    d[f"{m}_inv_protein"] = [int(iv.loc[k].protein) for k in key]
pd.to_pickle(d, C.RES / "cohort_derivation.pkl")

# subtype z-profiles among stage-positive participants
zc = [c for c in d.columns if c.startswith("z_")]
prof = []
for m in C.MODELS:
    pos = d[d[f"{m}_stage"] > 0]
    g = pos.groupby(f"{m}_subtype")[zc].mean()
    g["n"] = pos.groupby(f"{m}_subtype").size(); g["mean_stage"] = pos.groupby(f"{m}_subtype")[f"{m}_stage"].mean()
    g["model"] = m; prof.append(g.reset_index().rename(columns={f"{m}_subtype": "subtype"}))
pd.concat(prof).to_csv(C.RES / "sustain_subtype_profiles.csv", index=False)

# stage correlates (age/sex-adjusted standardised beta for stage_norm)
import statsmodels.api as sm
rows = []
facs = {"age_imaging": "Age at imaging (per SD)", "male": "Male sex", "SBP": "Systolic BP (per SD)", "Diabetes": "Diabetes",
        "current_smoker": "Current smoking", "BMI": "BMI (per SD)", "HbA1c": "HbA1c (per SD)", "LDLc": "LDL-C (per SD)",
        "HDLc": "HDL-C (per SD)", "CysC": "Cystatin C (per SD)", "Hypertension": "Hypertension", "APOE4": "APOE e4 alleles",
        "townsend_index": "Townsend index (per SD)", "score2": "SCORE2 risk (per SD)"}
for m in C.MODELS:
    y = d[f"{m}_stage_norm"]; ys = (y - y.mean()) / y.std()
    for f, lbl in facs.items():
        x = d[f].astype(float)
        if d[f].nunique() > 3: x = (x - x.mean()) / x.std()
        X = pd.DataFrame({"x": x}) if f in ("age_imaging", "male", "score2") else pd.DataFrame({"x": x, "age": d.age_imaging, "male": d.male})
        X = sm.add_constant(X); ok = X.notna().all(1) & ys.notna()
        r = sm.OLS(ys[ok], X[ok]).fit(cov_type="HC3")
        lo, hi = r.conf_int().loc["x"]
        rows.append(dict(model=m, factor=f, label=lbl, beta=r.params["x"], lo=lo, hi=hi, p=r.pvalues["x"], n=int(ok.sum())))
sc = pd.DataFrame(rows); sc.to_csv(C.RES / "sustain_stage_correlates.csv", index=False)

# cross-model agreement
ag = []
ms = list(C.MODELS)
for i in range(3):
    for j in range(i + 1, 3):
        a_, b_ = d[f"{ms[i]}_stage_norm"], d[f"{ms[j]}_stage_norm"]
        ag.append(dict(m1=ms[i], m2=ms[j], spearman=stats.spearmanr(a_, b_)[0],
                       both_positive=((d[f"{ms[i]}_stage"] > 0) == (d[f"{ms[j]}_stage"] > 0)).mean()))
pd.DataFrame(ag).to_csv(C.RES / "sustain_stage_agreement.csv", index=False)

pd.set_option("display.width", 220)
print(hf.round(3).to_string())
print(pv.round(3).drop(columns=["order", "prevalence_order"]).to_string())
print(pd.DataFrame(ag).round(3).to_string())
print(pd.concat(prof).round(2).to_string())
for m in C.MODELS:
    print(m, d[f"{m}_inv"].value_counts().to_dict())
print(sc[sc.model == "all10"].round(3).to_string())
