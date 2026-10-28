import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Appendix figure S1 - positional variance diagrams of the two sensitivity models (derivation) and of the MRI-only model
refitted de novo in a random independent subsample (n = 6000)."""
import sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
sys.path.insert(0, _os.path.join(HB_ROOT, "analysis"))
import style4 as S
import common as C

RES = _os.path.join(HB_ROOT, "results")
OUT = _os.path.join(HB_ROOT, "figs/output/Appendix")
import os; os.makedirs(OUT, exist_ok=True)


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2)); return tuple(a + (b - a) * t)


def pcol(org, p):
    p = float(np.clip(p, 0, 1)) ** 0.75
    return mix("#FFFFFF", mix(S.ORG[org], "#1A1A1A", 0.18), p)


def denovo_pvd():
    ev = C.event_table("mri_only_matched_all10")
    pk = pickle.load(open(f"{RES}/replication_sustain/pickle_files/rep6000_subtype1.pickle", "rb"))
    Sq = pk["samples_sequence"][:, :, ::5].astype(int); K, NE, NS = Sq.shape
    F = pk["samples_f"][:, ::5]; frac = np.array([np.median(F[k]) for k in range(K)]); order = np.argsort(-frac)
    ml = pk["ml_sequence_EM"].astype(int)
    rows = []
    for disp, k in enumerate(order, start=1):
        P = np.empty((NE, NS), int)
        for s in range(NS):
            P[Sq[k, :, s], s] = np.arange(NE)
        mlpos = np.empty(NE, int); mlpos[ml[k]] = np.arange(NE)
        for e in range(NE):
            r = dict(model="denovo", subtype=disp, event_index=e, label=ev.label.iloc[e], organ=ev.organ.iloc[e], z=ev.z.iloc[e],
                     ml_position=mlpos[e] + 1, fraction=frac[k])
            cnt = np.bincount(P[e], minlength=NE) / NS
            r.update({f"p{j + 1}": cnt[j] for j in range(NE)})
            rows.append(r)
    return pd.DataFrame(rows)


def draw(ax, g, ne, title):
    g = g.sort_values("ml_position")
    img = np.ones((len(g), ne, 3))
    for i, row in enumerate(g.itertuples()):
        for p in range(ne):
            img[i, p] = pcol(row.organ, getattr(row, f"p{p + 1}"))
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0.5, ne + 0.5, len(g) - 0.5, -0.5))
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{r.label} z{int(r.z)}" for r in g.itertuples()], fontsize=5.2)
    for t, r in zip(ax.get_yticklabels(), g.itertuples()):
        t.set_color(S.ORG[r.organ])
    tk = [1] + list(range(5, ne + 1, 5)); ax.set_xticks(tk); ax.set_xticklabels([str(x) for x in tk], fontsize=5.5)
    ax.tick_params(length=1.5, width=0.4, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#B7BCC4")
    ax.set_title(title, fontsize=6.4, loc="left", pad=3, color=S.INK, fontweight="bold")
    ax.set_xlabel("Position in sequence", fontsize=5.6, labelpad=1.5)


def main():
    S.set_style()
    dd = pd.read_pickle(f"{RES}/cohort_derivation.pkl"); sp = dd.mri_plus_m13_stage > 0
    m13_share = dd.mri_plus_m13_subtype[sp].value_counts(normalize=True).to_dict()
    hf = pd.read_csv(f"{RES}/sustain_heart_first.csv").set_index(["model", "subtype"])
    m13 = pd.read_csv(f"{RES}/sustain_pvd_mri_plus_m13.csv"); mo = pd.read_csv(f"{RES}/sustain_pvd_mri_only_matched_all10.csv")
    dn = denovo_pvd()
    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 215 * mm))
    gs = fig.add_gridspec(3, 6, height_ratios=[24, 18, 18], hspace=0.42, wspace=3.4, left=0.125, right=0.99, top=0.945, bottom=0.04)
    pos = {("m13", 1): gs[0, 0:3], ("m13", 2): gs[0, 3:6], ("mo", 1): gs[1, 0:2], ("mo", 2): gs[1, 2:4], ("mo", 3): gs[1, 4:6],
           ("dn", 1): gs[2, 0:3], ("dn", 2): gs[2, 3:6]}
    letters = iter("ABCDEFG")
    for (key, sub), spec in pos.items():
        ax = fig.add_subplot(spec)
        if key == "m13":
            g = m13[m13.subtype == sub]; ne = 24; fr = m13_share[sub]; nm = "MRI + M13"
        elif key == "mo":
            g = mo[mo.subtype == sub]; ne = 18; fr = hf.loc[("mri_only_matched_all10", sub), "fraction"]; nm = "MRI only"
        else:
            g = dn[dn.subtype == sub]; ne = 18; fr = g.fraction.iloc[0]; nm = "MRI only, de novo refit (n = 6000)"
        tag = "assigned " if key == "m13" else ""
        draw(ax, g, ne, f"{next(letters)}  {nm} \u00b7 S{sub} ({tag}{S.md(100 * fr, 1)}%)")
    fig.text(0.01, 0.99, "Posterior probability of each event position across MCMC samples; rows ordered by maximum-likelihood position.\n"
             "Label colour: heart (orange), plasma protein (blue), brain (purple). Percentages are model fractions, except for MRI + M13 "
             "(share of stage-positive participants assigned).", fontsize=5.6, color=S.GREY, va="top", linespacing=1.3)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT}/FigureS1_PVD_sensitivity_and_denovo.{ext}", dpi=400)
    dn.to_csv(f"{RES}/replication_sustain/denovo_pvd_K2.csv", index=False)
    print("saved")


if __name__ == "__main__":
    main()
