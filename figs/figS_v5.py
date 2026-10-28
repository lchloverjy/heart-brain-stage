"""Appendix figures (v5).

S1  participant flow, ten markers, and imaging-based time zero (v4 Figure 1, unchanged content)
S2  positional variance: sensitivity models and de novo refit (v4 Figure S1, unchanged content)
S3  brain coupling decomposed by cardiac phenotype, with tissue-class summary by model and sample (new)

Usage: python3 figS_v5.py [s1|s2|s3|all]
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import style4 as S
import base_style as BS
import anat3d as A
import regional_data as R
import fig3_regional_v5 as F2

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_v5" / "Appendix"
OUT.mkdir(parents=True, exist_ok=True)
BC, BN = F2.BC, F2.BN
W = 180


def _renumbered(module, func, old_stem, new_stem):
    M = __import__(module)
    with tempfile.TemporaryDirectory() as tmp:
        M.OUT = tmp
        S.set_style()
        getattr(M, func)()
        for f in sorted(Path(tmp).iterdir()):
            if f.name.startswith(old_stem):
                dst = OUT / f.name.replace(old_stem, new_stem)
                shutil.copy2(f, dst)
                print("saved", dst.name)


def s1():
    M = __import__("fig1v4")
    with tempfile.TemporaryDirectory() as tmp:
        M.OUT = tmp
        M.render("composite")
        for f in Path(tmp).glob("Figure1_composite.*"):
            dst = OUT / f.name.replace("Figure1_composite", "FigureS1_participant_flow_and_time_zero")
            shutil.copy2(f, dst)
            print("saved", dst.name)


def s2():
    _renumbered("figS1_pvd", "main", "FigureS1_PVD_sensitivity_and_denovo", "FigureS2_PVD_sensitivity_and_denovo")


# ----------------------------------------------------------------------------------------- S3
EXPOSURES = [("cardiac_comp", "Cardiac composite", "as in figure 3B"),
             ("LV_remodeling_Z_raw", "LV remodelling", "volumes, mass, wall thickness"),
             ("LV_myocardial_dysfunction_Z_raw", "LV myocardial dysfunction", "strain, ejection fraction"),
             ("Aortic_stiffness_Z_raw", "Aortic stiffness", "aortic distensibility")]
TISSUES = [("Cortical thickness", "cortex"), ("White-matter free water", "WM free water"),
           ("White-matter hyperintensity", "WMH"), ("Hippocampal volume", "hippocampus")]


def s3_panel_a(Fm, cv, x0=0, y0=0, rh=19.5):
    S.panel_letter(cv, x0 + 1, y0 + 1, "A", "Brain coupling decomposed by cardiac phenotype (independent sample)")
    cv.text(x0 + 5.0, y0 + 5.6, "\u03b2 per 1 SD of each cardiac exposure, age- and sex-adjusted (model A); "
            "faded regions, FDR q \u2265 0\u00b705; colour scale shared with figure 3B", fontsize=4.5, color=S.GREY,
            va="top")
    cx_lat, cx_med, cx_w1, cx_w2 = x0 + 44.0, x0 + 68.0, x0 + 91.0, x0 + 110.0
    hy = y0 + 12.4
    for cx, t in ((cx_lat, "left lateral"), (cx_med, "left medial"), (cx_w1, "WM z = +24 mm"),
                  (cx_w2, "WM z = +6 mm")):
        cv.text(cx, hy, t, fontsize=4.6, ha="center", va="center", color=S.INK)
    cv.text(x0 + 128.0, hy, "WMH, hippocampus", fontsize=4.6, ha="center", va="center", color=S.INK)
    cv.text(x0 + 160.0, hy, "median \u03b2 \u00b7 regions q < 0\u00b705", fontsize=4.6, ha="center", va="center",
            color=S.INK)
    for i, (ex, name, sub) in enumerate(EXPOSURES):
        yc = y0 + 24.0 + i * rh
        if i > 0:
            cv.plot([x0 + 2, W - 2], [yc - rh / 2, yc - rh / 2], color=S.RULE, lw=0.35, zorder=0)
        m, c = R.coupling_maps(ex)
        cv.text(x0 + 3.0, yc - 2.2, name, fontsize=5.4, fontweight="bold", color=S.ORG["Heart"], va="center")
        cv.text(x0 + 3.0, yc + 1.2, sub, fontsize=4.3, color=S.GREY, va="center")
        for cx, view in ((cx_lat, "lateral"), (cx_med, "medial")):
            A.draw_cortex(cv, "lh", view, m["cortex_lh"], cx, yc, 22.0, cmap=BC, norm=BN,
                          alpha_mask=m["cortex_lh_mask"], height=15.0)
        for z, cx in ((24, cx_w1), (6, cx_w2)):
            A.draw_wm_slice(cv, z, m["wm"], cx, yc, 13.0, cmap=BC, norm=BN, mask=m["wm_mask"])
        hip_l, _ = m["hippo"]["lh"]
        hip_r, _ = m["hippo"]["rh"]
        A.chip(cv, x0 + 121.0, yc - 4.6, 14.0, 3.2, m["wmh"], "WMH", cmap=BC, norm=BN, fontsize=4.0, nd=3)
        A.chip(cv, x0 + 121.0, yc - 0.6, 14.0, 3.2, hip_l, "Hipp L", cmap=BC, norm=BN, fontsize=4.0, nd=3)
        A.chip(cv, x0 + 121.0, yc + 3.4, 14.0, 3.2, hip_r, "Hipp R", cmap=BC, norm=BN, fontsize=4.0, nd=3)
        cb = c[c.organ == "brain"]
        for k, (tis, lab) in enumerate(TISSUES):
            g = cb[cb.tissue == tis]
            yy = yc - 5.4 + k * 3.6
            cv.text(x0 + 145.0, yy, lab, fontsize=4.3, color=S.GREY, va="center")
            cv.text(x0 + 167.0, yy, F2.fmt(g.beta.median(), 3), fontsize=4.4, color=S.INK, va="center", ha="right")
            cv.text(x0 + 168.5, yy, f"{int((g.q < 0.05).sum())}/{len(g)}", fontsize=4.2, color=S.GREY, va="center")
    A.colorbar_mm(Fm, x0 + 44.0, y0 + 24.0 + 3.5 * rh + 2.0, 50, 1.6, BC, BN, [-0.05, 0, 0.05, 0.10],
                  "\u03b2 (z per 1 SD exposure)", nd=2)
    return y0 + 24.0 + 3.5 * rh + 10.0


def s3_panel_b(Fm, cv, x0=0, y0=0, h=60):
    S.panel_letter(cv, x0 + 1, y0 + 1, "B", "Tissue-class summary by adjustment model and sample")
    t = pd.read_csv(R.RES / "regional_tissue_summary.csv")
    cv.text(x0 + 5.0, y0 + 5.6, "Median regional \u03b2 per 1 SD cardiac composite with bootstrap 95% CI (1000 "
            "resamples); model A, age and sex; model B, plus vascular risk factors", fontsize=4.5, color=S.GREY,
            va="top")
    contrasts = ["Cortical thickness", "Hippocampal volume", "White-matter free water",
                 "WM free water minus cortical thickness"]
    labels = ["Cortical thickness", "Hippocampal volume", "White-matter free water", "WM minus cortex"]
    combos = [("independent", "A", S.ORG["Brain"], "o", "independent, model A"),
              ("independent", "B", S.ORG["Brain"], "D", "independent, model B"),
              ("derivation", "A", "#6B7078", "o", "derivation, model A"),
              ("derivation", "B", "#6B7078", "D", "derivation, model B")]
    ax = Fm.add(x0 + 40.0, y0 + 10.0, 92.0, h - 19.0)
    for i, ct in enumerate(contrasts):
        for j, (smp, mdl, col, mk, _) in enumerate(combos):
            r = t[(t["sample"] == smp) & (t.model == mdl) & (t.contrast == ct)].iloc[0]
            yy = i + (j - 1.5) * 0.2
            ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=0.8, solid_capstyle="round", zorder=2)
            ax.plot(r.median_beta, yy, mk, ms=3.0 if mk == "o" else 2.6, mfc=col if mdl == "A" else "white",
                    mec=col, mew=0.6, zorder=3)
            ax.text(0.088, yy, f"{F2.fmt(r.median_beta, 3)} ({F2.fmt(r.lo, 3)} to {F2.fmt(r.hi, 3)})",
                    fontsize=3.9, va="center", color=col)
    ax.axvline(0, color=S.RULE, lw=0.6, zorder=0)
    ax.set_yticks(range(len(contrasts))); ax.set_yticklabels(labels, fontsize=4.8)
    ax.set_ylim(len(contrasts) - 0.45, -0.55)
    ax.set_xlim(-0.035, 0.085)
    ax.set_xticks([-0.03, 0, 0.03, 0.06])
    ax.set_xticklabels([F2.fmt(v, 2) for v in (-0.03, 0, 0.03, 0.06)])
    ax.tick_params(labelsize=4.5, length=1.4, pad=1)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("median regional \u03b2", fontsize=4.8, labelpad=1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    for j, (_, mdl, col, mk, lab) in enumerate(combos):
        xx = x0 + 6.0 + j * 33.0
        yy = y0 + h - 2.2
        cv.plot(xx, yy, mk, ms=3.0 if mk == "o" else 2.6, mfc=col if mdl == "A" else "white", mec=col, mew=0.6)
        cv.text(xx + 1.6, yy, lab, fontsize=4.4, va="center", color=S.INK)


def s3():
    S.set_style()
    ha = 24.0 + 3.5 * 19.5 + 10.0
    hb = 60.0
    Fm = BS.MMFig(W, ha + hb + 2); cv = Fm.canvas()
    s3_panel_a(Fm, cv, 0, 0)
    cv.plot([1, W - 1], [ha, ha], color=S.RULE, lw=0.4)
    s3_panel_b(Fm, cv, 0, ha + 2)
    Fm.save(str(OUT / "FigureS3_regional_coupling_by_cardiac_phenotype"))
    print("saved S3", W, round(ha + hb + 2, 1))


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("s1", s1), ("s2", s2), ("s3", s3)):
        if w in (name, "all"):
            fn()
