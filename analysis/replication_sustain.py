import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Independent-sample replication of the MRI-only heart-brain SuStaIn model.

Fits ZscoreSustain on the 23,974 participants with complete cardiac + brain MRI features who were NOT part
of the 3,770-participant derivation sample, using the identical 18-event threshold configuration
(thresholds_mri_only_matched_all10.csv) and the same seed as the original final run."""
import sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pySuStaIn.ZscoreSustain import ZscoreSustain

PREP = Path(_os.path.join(_os.environ.get("HB_RAW", _os.path.join(HB_ROOT, "raw")), "up/module3_preparation"))
OUT = Path(_os.path.join(HB_ROOT, "results/replication_sustain")); OUT.mkdir(parents=True, exist_ok=True)
N_START = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_MCMC = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
N_SMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 3
N_SUB = int(sys.argv[4]) if len(sys.argv) > 4 else 0      # optional random subsample of the independent sample
TAG = sys.argv[5] if len(sys.argv) > 5 else "replication_mri_only"

a = pd.read_csv(PREP / "03_inputs/sustain_mri_only.csv")
dev = pd.read_csv(PREP / "03_inputs/sustain_all10.csv", usecols=["eid"])
rep = a[~a.eid.isin(dev.eid)].reset_index(drop=True)
if N_SUB:
    rep = rep.sample(n=N_SUB, random_state=20260712).reset_index(drop=True)
th = pd.read_csv(PREP / "04_thresholds/thresholds_mri_only_matched_all10.csv")
data = rep[th.biomarker].to_numpy(float)
z_vals = th[["z1", "z2", "z3"]].to_numpy(float); z_max = th.zmax.to_numpy(float)
rep[["eid"]].to_csv(OUT / f"{TAG}_eids.csv", index=False)
print("replication n =", data.shape, flush=True)
t0 = time.time()
s = ZscoreSustain(data, z_vals, z_max, th.label.tolist(), N_START, N_SMAX, N_MCMC, str(OUT),
                  TAG, False, seed=20260712)
res = s.run_sustain_algorithm(plot=False)
print("done in %.1f min" % ((time.time() - t0) / 60), flush=True)
with open(OUT / f"{TAG}_summary.pkl", "wb") as fh:
    pickle.dump({"n": data.shape[0], "n_startpoints": N_START, "n_mcmc": N_MCMC, "n_subtypes_max": N_SMAX,
                 "seconds": time.time() - t0}, fh)
