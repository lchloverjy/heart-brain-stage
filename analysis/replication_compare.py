"""Compare the de novo SuStaIn fit in an independent random subsample (n = 6000; MRI-only events) with the derivation models.
Outputs (results/replication_sustain/):
  denovo_heart_first.csv   per de novo subtype: fraction, P(first cardiac event before first brain event), first-event positions, ML order
  denovo_similarity.csv    Spearman correlation of posterior mean event positions (shared (marker, z) events) with derivation subtypes
"""
import pickle, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import common as C

K_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = C.RES / "replication_sustain"
THIN = 5
ev = C.event_table("mri_only_matched_all10")
with open(OUT / "pickle_files" / f"rep6000_subtype{K_FIT - 1}.pickle", "rb") as fh:
    pk = pickle.load(fh)


def positions(pk, thin):
    S = pk["samples_sequence"][:, :, ::thin].astype(int)
    K, NE, NS = S.shape
    P = np.empty_like(S)
    for k in range(K):
        for s in range(NS):
            P[k, S[k, :, s], s] = np.arange(NE)
    return P


P = positions(pk, THIN)
F = pk["samples_f"][:, ::THIN]
K = P.shape[0]
frac = np.array([np.median(F[k]) for k in range(K)])
order = np.argsort(-frac)                      # display numbering by fraction
ml = pk["ml_sequence_EM"].astype(int)
hi = ev.index[ev.organ == "Heart"].to_numpy(); bi = ev.index[ev.organ == "Brain"].to_numpy()
rows, means = [], {}
for disp, k in enumerate(order, start=1):
    fh = P[k, hi].min(0); fb = P[k, bi].min(0)
    seq = [f"{ev.label.iloc[e]} z{ev.z.iloc[e]}" for e in ml[k]]
    rows.append(dict(K=K_FIT, subtype=disp, fraction=frac[k], p_heart_before_brain=(fh < fb).mean(),
                     first_heart_pos_median=np.median(fh) + 1, first_brain_pos_median=np.median(fb) + 1,
                     first_event=seq[0], first_five=" > ".join(seq[:5])))
    means[disp] = pd.Series(P[k].mean(1) + 1, index=[(ev.biomarker.iloc[e], ev.z.iloc[e]) for e in range(len(ev))])
hf = pd.DataFrame(rows); hf.to_csv(OUT / f"denovo_heart_first_K{K_FIT}.csv", index=False)
print(hf.to_string())

# derivation posterior mean positions
sim = []
for m in ["mri_only_matched_all10", "all10"]:
    evm = C.event_table(m)
    pkm = C.load_pickle(m)
    Pm = positions(pkm, 20)
    seq_exp = pd.read_csv(C.FINAL / m / "selected_event_sequences.csv")
    o2d = seq_exp.drop_duplicates("subtype_original_0based").set_index("subtype_original_0based").subtype_display.to_dict()
    for k in range(Pm.shape[0]):
        mm = pd.Series(Pm[k].mean(1) + 1, index=[(evm.biomarker.iloc[e], evm.z.iloc[e]) for e in range(len(evm))])
        for disp, s in means.items():
            common_ev = [i for i in s.index if i in mm.index]
            rho, p = spearmanr(s[common_ev].rank(), mm[common_ev].rank())
            sim.append(dict(K=K_FIT, denovo_subtype=disp, derivation_model=m, derivation_subtype=o2d[k], n_shared_events=len(common_ev),
                            spearman=rho, p=p))
sim = pd.DataFrame(sim); sim.to_csv(OUT / f"denovo_similarity_K{K_FIT}.csv", index=False)
print(sim.round(3).to_string())

# onset file for figure 2c (same layout as fig2_organ_onset.csv; model = "replication")
on = []
for disp, k in enumerate(order, start=1):
    fr = {}
    for org in ["Heart", "Brain"]:
        idx = ev.index[ev.organ == org].to_numpy(); fr[org] = P[k, idx].min(0) + 1
        on.append(dict(model="replication", subtype=disp, organ=org, n_events=len(ev), median=np.median(fr[org]),
                       lo=np.percentile(fr[org], 2.5), hi=np.percentile(fr[org], 97.5)))
    on.append(dict(model="replication", subtype=disp, organ="P(heart<brain)", n_events=len(ev), median=(fr["Heart"] < fr["Brain"]).mean()))
pd.DataFrame(on).to_csv(OUT / "rep6000_onset.csv", index=False)
print(pd.DataFrame(on))
