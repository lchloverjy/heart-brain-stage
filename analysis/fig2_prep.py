"""Data for Figure 2: organ-onset posterior intervals and stage snapshots (derived from final SuStaIn MCMC samples)."""
import numpy as np
import pandas as pd
import common as C

THIN = 10
d = pd.read_pickle(C.RES / "cohort_derivation.pkl")
onset, snap, reach = [], [], []
for m in C.MODELS:
    ev = C.event_table(m)
    seq_exp = pd.read_csv(C.FINAL / m / "selected_event_sequences.csv")
    o2d = seq_exp.drop_duplicates("subtype_original_0based").set_index("subtype_original_0based").subtype_display.to_dict()
    pk = C.load_pickle(m)
    S = pk["samples_sequence"][:, :, ::THIN].astype(int)
    K, NE, NS = S.shape
    P = np.empty_like(S)
    for k in range(K):
        for s in range(NS):
            P[k, S[k, :, s], s] = np.arange(NE)
    ml = pk["ml_sequence_EM"].astype(int)
    for k in range(K):
        disp = int(o2d[k])
        fr = {}
        for org in ["Heart", "Protein", "Brain"]:
            idx = ev.index[ev.organ == org].to_numpy()
            if len(idx) == 0:
                continue
            first = P[k, idx].min(0) + 1
            fr[org] = first
            onset.append(dict(model=m, subtype=disp, organ=org, n_events=NE, median=np.median(first), lo=np.percentile(first, 2.5),
                              hi=np.percentile(first, 97.5), median_frac=np.median(first) / NE, lo_frac=np.percentile(first, 2.5) / NE,
                              hi_frac=np.percentile(first, 97.5) / NE))
        onset.append(dict(model=m, subtype=disp, organ="P(heart<brain)", n_events=NE, median=(fr["Heart"] < fr["Brain"]).mean()))
        # ML-sequence snapshots: z level reached by each marker after k events
        seq = ml[k]
        for stage in range(0, NE + 1):
            reached = ev.iloc[seq[:stage]]
            lv = reached.groupby("biomarker").z.max() if len(reached) else pd.Series(dtype=float)
            for b in ev.biomarker.unique():
                snap.append(dict(model=m, subtype=disp, stage=stage, biomarker=b, organ=ev[ev.biomarker == b].organ.iloc[0],
                                 z_reached=float(lv.get(b, 0))))
        # share of participants assigned to this subtype who reached at least each stage
        sub = d[(d[f"{m}_subtype"] == disp) & (d[f"{m}_stage"] > 0)]
        for stage in range(1, NE + 1):
            reach.append(dict(model=m, subtype=disp, stage=stage, n_subtype=len(sub), n_reached=int((sub[f"{m}_stage"] >= stage).sum()),
                              frac_reached=(sub[f"{m}_stage"] >= stage).mean()))

pd.DataFrame(onset).to_csv(C.RES / "fig2_organ_onset.csv", index=False)
pd.DataFrame(snap).to_csv(C.RES / "fig2_stage_snapshots.csv", index=False)
pd.DataFrame(reach).to_csv(C.RES / "fig2_stage_reach.csv", index=False)
pd.set_option("display.width", 220)
print(pd.DataFrame(onset).round(3).to_string())
r = pd.DataFrame(reach); print(r[(r.model == "all10") & r.stage.isin([1, 3, 5, 8, 12, 18, 30])].to_string())
