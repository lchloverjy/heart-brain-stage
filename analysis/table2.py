import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Table 2 - heart-brain stage and incident disease after imaging, derivation (All-10) and independent (MRI-only) samples."""
import pandas as pd

R = _os.path.join(HB_ROOT, "results")
c = pd.read_csv(f"{R}/out_cox_main.csv"); c = c[c.model == "all10"]
v = pd.read_csv(f"{R}/validation/independent_landmark_cox_imaging.csv")


def md(x, k=2):
    return f"{x:.{k}f}".replace(".", "\u00b7")


def hr(r):
    return f"{md(r.HR)} ({md(r.lo)}\u2013{md(r.hi)})"


def n(x):
    x = int(x); return f"{x:,}".replace(",", "\u2009") if x >= 10000 else str(x)


ROWS = [("Cardiovascular and cerebrovascular", None), ("hbve", "Heart\u2013brain vascular events (primary)"), ("mace", "Stroke, MI, or HF"),
        ("cardiac", "Cardiac composite"), ("mi", "Myocardial infarction"), ("ihd", "Ischaemic heart disease"), ("af", "Atrial fibrillation or flutter"),
        ("hf", "Heart failure"), ("cerebrovascular", "Cerebrovascular composite"), ("stroke", "Stroke"), ("tia", "Transient ischaemic attack"),
        ("cv_death", "Cardiovascular death"),
        ("Other outcomes", None), ("death", "All-cause death"), ("depression", "Depression"), ("anxiety", "Anxiety"), ("parkinson", "Parkinson's disease"),
        ("ckd", "Chronic kidney disease")]
out = []
for key, lab in ROWS:
    if lab is None:
        out.append(dict(outcome=key, header=True)); continue
    row = dict(outcome=lab, header=False)
    a = c[(c.outcome == key)]
    m1 = a[(a.adj == "M1") & (a.exposure == "per10pp")]; m2 = a[(a.adj == "M2") & (a.exposure == "per10pp")]; g3 = a[(a.adj == "M2") & (a.exposure == "grp>=3")]
    if len(m2):
        row["d_events"] = f"{int(m1.events.iloc[0])}/{n(m1.n.iloc[0])}"; row["d_hr"] = hr(m2.iloc[0]); row["d_g3"] = hr(g3.iloc[0]) if len(g3) else "NA"
    else:
        row["d_events"] = row["d_hr"] = row["d_g3"] = "NA"
    b = v[v.outcome == key]
    if len(b):
        b1 = b[b.adj == "M1"].iloc[0]; b2 = b[b.adj == "M2"].iloc[0]; b3 = b[b.adj == "M2_grp>=3vs0"].iloc[0]
        row["i_events"] = f"{int(b2.events)}/{n(b2.n)}"; row["i_hr"] = hr(b2); row["i_g3"] = hr(b3)
    else:
        row["i_events"] = row["i_hr"] = row["i_g3"] = "NA"
    out.append(row)
t = pd.DataFrame(out)
t.to_csv(f"{R}/table2.csv", index=False)
print(t.to_string())
