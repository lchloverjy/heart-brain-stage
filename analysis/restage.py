"""Re-derive subtype/stage of individuals from saved MCMC samples (pySuStaIn) to audit label alignment."""
import numpy as np, pandas as pd, common as C, sys
from pySuStaIn.ZscoreSustain import ZscoreSustain
def make(model, data):
    th = C.load_thresholds(model)
    zv = th[["z1","z2","z3"]].to_numpy(float); zm = th.zmax.to_numpy(float)
    return ZscoreSustain(data, zv, zm, th.label.tolist(), 1, 1, 1, "/tmp", "x", False, seed=1)
def restage(model, data, n_samples=1000, seed=20260712):
    pk = C.load_pickle(model); s = make(model, data)
    np.random.seed(seed)
    out = s.subtype_and_stage_individuals_newData(data, pk["samples_sequence"], pk["samples_f"], n_samples)
    ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage = out
    return dict(ml_subtype=ml_subtype.ravel().astype(int), prob_ml_subtype=prob_ml_subtype.ravel(), ml_stage=ml_stage.ravel(),
                prob_ml_stage=prob_ml_stage.ravel(), prob_subtype=prob_subtype, prob_stage=prob_stage, prob_subtype_stage=prob_subtype_stage)
if __name__ == "__main__":
    for m in C.MODELS:
        th = C.load_thresholds(m); a = C.load_inputs(m); X = a[th.biomarker].to_numpy(float)
        pk = C.load_pickle(m); r = restage(m, X)
        st_old = pk["ml_subtype"].ravel().astype(int); sg_old = pk["ml_stage"].ravel()
        pos = r["ml_stage"] > 0
        print(m, "stage agreement", (r["ml_stage"] == sg_old).mean().round(4),
              "| subtype agreement (stage>0)", (r["ml_subtype"][pos] == st_old[pos]).mean().round(4))
        print(pd.crosstab(pd.Series(st_old[pos], name="stored"), pd.Series(r["ml_subtype"][pos], name="recomputed")))
        for k in range(pk["samples_sequence"].shape[0]):
            sel = pos & (r["ml_subtype"] == k)
            print("   recomputed k=%d n=%d" % (k, sel.sum()), a.loc[sel, th.biomarker].mean().round(2).to_dict())
        np.savez_compressed(C.RES / f"restaged_{m}.npz", eid=a.eid.values, **r)
