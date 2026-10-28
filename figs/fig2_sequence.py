"""Figure 2 (v7) - Heart-brain sequence and the measured regional anatomy of each subtype.
A  event sequence and positional variance (All-10), unchanged analysis (fig2v4.panel_a)
B  subtype x stage regional anatomy (MEASURED mean z; derivation sample, All-10 assignments): 3D heart with the
   16 AHA wall-thickness segments, strain bull's-eyes, 3D aorta, PyVista cortex, seven subcortical nuclei, white-
   matter FA and free-water slices, and WMH. Stage 0 is a shared reference row (no subtype information at stage 0).
   Measures marked with a dagger are not inputs to any SuStaIn domain score (out-of-model).
C  first cardiac versus first brain event across models (compact)
D  de novo refit in 6000 independent participants: rank agreement with the derivation order (compact)
Usage: python3 fig2_sequence_v7.py [a|b|c|d|composite|all] [sample]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
import style4 as S
import base_style as BS
import anat3d as A
import render3d as R3
import fig2v4 as G
import fig2_sequence_v6 as V6
import fig3_regional_v6 as F3
import regional_subtype_data as RS

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "output_v7" / "Figure2"
OUT.mkdir(parents=True, exist_ok=True)
W = 180
YB, HB = 84, 109
YC, HC = YB + HB, 44
H = YC + HC
SUBC = {1: "#1D7874", 2: "#B07A1E"}          # subtype identity (teal / ochre), distinct from organ hues
ZC, ZN = F3.ZC, F3.ZN                         # shared z scale (-0.5 to 1.4), as figure 3 v6
Z_SLICE = 6
XS = dict(wt=40.0, st=59.5, ao=75.0, ctx=101.5, sub=125.0, fa=145.0, fw=162.0, wmh=175.2)
DAG = "\u2020"


def fmt(v, nd=2):
    return S.md(v, nd)


# ----------------------------------------------------------------------------------------- shared cell renderers
def _fade(c, sig, fade=0.65):
    c = np.asarray(c[:3], float)
    return tuple(c if sig else c * (1 - fade) + R3._hex(R3.NEUTRAL) * fade)


def heart_cell(cv, cx, cy, vals, cmap, norm, mask=None, h=15.0):
    F3.heart_cell(cv, cx, cy, vals, cmap, norm, mask=mask, h=h, rb=2.45, px=900)


def strain_cell(cv, cx, cy, ecc, err, cmap, norm, me=None, mr=None, r=3.35):
    for dx, vals, m, lab in ((-4.2, ecc, me, "Ecc"), (4.2, err, mr, "Err")):
        A.draw_bullseye(cv, cx + dx, cy - 1.1, r, vals, cmap=cmap, norm=norm, mask=m, lw=0.2, zorder=4)
        cv.text(cx + dx, cy + r + 1.0, lab, fontsize=4.2, ha="center", va="center", color=S.GREY)


def aorta_cell(cv, cx, cy, ao, cmap, norm, mask=None, h=14.5, nd=2):
    asc, desc = ao.get("ascending", np.nan), ao.get("descending", np.nan)
    m = mask or {}
    ca = _fade(cmap(norm(asc)), m.get("ascending", True))
    cd = _fade(cmap(norm(desc)), m.get("descending", True))
    img = R3.render_aorta(px=900, asc_rgb=ca, desc_rgb=cd)
    R3.place(cv, img, cx - 1.8, cy, h=h, anchor="c", zorder=3)
    A.chip(cv, cx + 2.6, cy - 5.3, 8.4, 3.3, asc, "Asc", cmap=cmap, norm=norm, fontsize=4.0, nd=nd)
    A.chip(cv, cx + 2.6, cy + 1.6, 8.4, 3.3, desc, "Desc", cmap=cmap, norm=norm, fontsize=4.0, nd=nd)


def cortex_cell(cv, cx, cy, vals, cmap, norm, mask=None, view="lateral", w=23.0):
    rgb = R3.dk_vertex_rgb("lh", vals, cmap, norm, mask)
    img = R3.render_cortex("lh", view, px=1100, vertex_rgb=rgb)
    R3.place(cv, img, cx, cy, w=w, anchor="c", zorder=3)


def subc_cell(cv, cx, cy, vals, cmap, norm, mask=None, w=19.0):
    m = mask or {}
    rgb = {s: _fade(cmap(norm(v)), m.get(s, True)) for s, v in vals.items() if np.isfinite(v)}
    img = R3.render_subcortical("lh", "inferomedial", px=900, rgb=rgb)
    R3.place(cv, img, cx, cy, w=w, anchor="c", zorder=3)


def wm_cell(cv, cx, cy, vals, cmap, norm, mask=None, width=14.0):
    A.draw_wm_slice(cv, Z_SLICE, vals, cx, cy, width, cmap=cmap, norm=norm, mask=mask)


def wmh_cell(cv, cx, cy, v, cmap, norm, nd=2):
    A.chip(cv, cx - 4.4, cy - 1.9, 8.8, 3.8, v, "", cmap=cmap, norm=norm, fontsize=4.4, nd=nd)


def header(cv, x0, y0, ytitle):
    F3.organ_bar(cv, x0 + 31.0, x0 + 88.0, y0 + 8.2, "Heart (cardiac MRI)", S.ORG["Heart"])
    F3.organ_bar(cv, x0 + 89.5, x0 + 179.0, y0 + 8.2, "Brain (MRI; left hemisphere)", S.ORG["Brain"])
    heads = [("wt", f"Wall thickness{DAG}\n(16 AHA segments)"), ("st", "Myocardial strain\n(16 segments)"),
             ("ao", "Aortic\ndistensibility"), ("ctx", "Cortical thickness\n(33 DK regions)"),
             ("sub", f"Subcortical nuclei{DAG}\n(7 FIRST structures)"), ("fa", f"WM FA{DAG}\n(48 tracts)"),
             ("fw", "WM free water\n(48 tracts)"), ("wmh", "WMH\nvolume")]
    for k, t in heads:
        cv.text(x0 + XS[k], ytitle, t, fontsize=4.5, ha="center", va="top", color=S.GREY, linespacing=0.95)


def draw_row(cv, x0, y, m, cmap, norm, masks=False, nd=2):
    g = (lambda k: m.get(k + "_mask")) if masks else (lambda k: None)
    heart_cell(cv, x0 + XS["wt"], y, m["heart_wt"], cmap, norm, mask=g("heart_wt"))
    strain_cell(cv, x0 + XS["st"], y, m["heart_ecc"], m["heart_err"], cmap, norm, g("heart_ecc"), g("heart_err"))
    aorta_cell(cv, x0 + XS["ao"], y, m["aorta"], cmap, norm, mask=g("aorta"), nd=nd)
    cortex_cell(cv, x0 + XS["ctx"], y, m["cortex_lh"], cmap, norm, mask=g("cortex_lh"))
    subc_cell(cv, x0 + XS["sub"], y, m["subc_lh"], cmap, norm, mask=g("subc_lh"))
    wm_cell(cv, x0 + XS["fa"], y, m["wm_fa"], cmap, norm, mask=g("wm_fa"))
    wm_cell(cv, x0 + XS["fw"], y, m["wm_fw"], cmap, norm, mask=g("wm_fw"))
    wmh_cell(cv, x0 + XS["wmh"], y, m["wmh"], cmap, norm, nd=nd)


# ----------------------------------------------------------------------------------------- B
SIGNATURE = {1: "early: myocardial strain\nlate: hippocampus, cortex, and WM",
             2: "early: wall thickening\nlate: aorta and white matter"}


def panel_b(F, cv, x0, y0, sample="derivation", letter="B"):
    title = ("Measured regional heart and brain abnormality by subtype and stage"
             + (" (derivation sample)" if sample == "derivation" else " (independent sample)"))
    S.panel_letter(cv, x0 + 1, y0 + 1, letter, title, dx=3.8 if letter else 0.0)
    maps = RS.cell_maps(sample)
    header(cv, x0, y0, y0 + 9.3)
    row_y = [y0 + 24.0, y0 + 43.5, y0 + 59.5, y0 + 79.0, y0 + 95.0]
    rows = [("Stage 0", None, "Stage 0"), ("Subtype 1, stage 1-2", 1, "Stage 1\u20132"),
            ("Subtype 1, stage >=3", 1, "Stage \u22653"), ("Subtype 2, stage 1-2", 2, "Stage 1\u20132"),
            ("Subtype 2, stage >=3", 2, "Stage \u22653")]
    for i, (cell, s, lab) in enumerate(rows):
        y, m = row_y[i], maps[cell]
        if i in (1, 3):
            cv.plot([x0 + 1, x0 + W - 1], [y - 10.0, y - 10.0], color=S.RULE, lw=0.45)
        elif i in (2, 4):
            cv.plot([x0 + 4.0, x0 + W - 1], [y - 8.0, y - 8.0], color=S.RULE, lw=0.25, ls=(0, (1.5, 1.2)))
        draw_row(cv, x0, y, m, ZC, ZN)
        if s is None:
            cv.text(x0 + 1.5, y - 3.2, "Stage 0", fontsize=6.0, fontweight="bold", color=S.INK, va="center")
            cv.text(x0 + 1.5, y + 0.4, "shared reference", fontsize=4.5, color=S.GREY, va="center")
            cv.text(x0 + 1.5, y + 3.2, f"n = {S.md_int(m['n'])}", fontsize=4.5, color=S.GREY, va="center")
            continue
        cv.text(x0 + 4.2, y - 2.0, lab, fontsize=5.6, fontweight="bold", color=S.INK, va="center")
        cv.text(x0 + 4.2, y + 1.3, f"n = {S.md_int(m['n'])} \u00b7 p = {fmt(m['psub'])}", fontsize=4.3,
                color=S.GREY, va="center")
        if i in (1, 3):
            yt, yb = y - 8.6, row_y[i + 1] + 7.4
            cv.add_patch(Rectangle((x0 + 1.0, yt), 1.3, yb - yt, fc=SUBC[s], ec="none", zorder=3))
            cv.text(x0 + 4.2, y - 6.3, S.SUBTYPE_NAME[s], fontsize=5.0, fontweight="bold", color=SUBC[s],
                    va="center")
        else:
            cv.text(x0 + 4.2, y + 4.9, SIGNATURE[s], fontsize=4.1, color=SUBC[s], va="center", style="italic",
                    linespacing=1.05)
    for k in ("fa", "fw"):
        cv.text(x0 + XS[k] - 6.6, row_y[0] - 7.4, "L", fontsize=3.9, color=S.GREY, ha="left", va="center")
        cv.text(x0 + XS[k] + 6.6, row_y[0] - 7.4, "R", fontsize=3.9, color=S.GREY, ha="right", va="center")
    A.colorbar_mm(F, x0 + 2.0, y0 + 12.6, 25.0, 1.4, ZC, ZN, [-0.4, 0, 0.4, 0.8, 1.2], "", fontsize=4.0)
    cv.text(x0 + 2.0, y0 + 9.4, "Mean regional z score", fontsize=4.5, color=S.INK, va="bottom")
    cv.text(x0 + 2.0, y0 + 11.9, "(higher = more abnormal)", fontsize=4.0, color=S.GREY, va="bottom")
    model = "All-10 model" if sample == "derivation" else "transferred MRI-only model (subtype 3, n = 281, not shown)"
    cv.text(x0 + 1.5, y0 + 104.6,
            f"Subtype and stage from the {model}; p, median posterior probability of the assigned subtype. "
            f"{DAG} Not an input to any SuStaIn domain score (out-of-model). WM slices at z = +{Z_SLICE} mm (MNI).",
            fontsize=4.1, color=S.GREY, va="top")


# ----------------------------------------------------------------------------------------- C (compact)
def panel_c(cv, x0, y0, w=86):
    on = pd.read_csv(RES / "fig2_organ_onset.csv")
    S.panel_letter(cv, x0 + 1, y0 + 1, "C", "First cardiac event precedes first brain event")
    rows = [(m, s) for m in ["all10", "mri_plus_m13", "mri_only_matched_all10"]
            for s in sorted(on[on.model == m].subtype.unique())]
    rp = pd.read_csv(RES / "replication_sustain" / "rep6000_onset.csv")
    on = pd.concat([on, rp])
    rows += [("replication", s) for s in sorted(rp.subtype.unique())]
    L, R, xmax = x0 + 31.0, x0 + 70.0, 6

    def X(p):
        return L + (p - 0.5) / xmax * (R - L)
    top, dy = y0 + 10.6, 2.75
    bot = top + dy * (len(rows) - 0.4)
    for p in range(1, xmax + 1):
        cv.plot([X(p), X(p)], [top - 1.8, bot], color=S.RULE, lw=0.3, zorder=0)
        cv.text(X(p), bot + 0.7, str(p), fontsize=4.4, ha="center", va="top", color=S.GREY)
    cv.text((L + R) / 2, bot + 3.4, "Posterior median position of first event", fontsize=4.6, ha="center", va="top")
    cv.text(R + 6.5, top - 3.4, "P(heart\nfirst)", fontsize=4.2, ha="center", va="center", color=S.GREY,
            linespacing=0.95)
    nm = {"all10": "All-10", "mri_plus_m13": "MRI + M13", "mri_only_matched_all10": "MRI only",
          "replication": "De novo refit"}
    prev = None
    for i, (m, s) in enumerate(rows):
        yy = top + i * dy
        g = on[(on.model == m) & (on.subtype == s)].set_index("organ")
        if m != prev:
            cv.text(x0 + 2, yy, nm[m], fontsize=4.7, fontweight="bold", va="center",
                    color={"all10": "#0F4A38", "mri_plus_m13": "#2F7A60", "mri_only_matched_all10": "#4E7F6C",
                           "replication": "#3D5A80"}[m])
            if prev is not None:
                cv.plot([x0 + 2, R + 10], [yy - dy / 2, yy - dy / 2], color=S.RULE, lw=0.3)
            prev = m
        cv.text(L - 2.2, yy, f"S{s}", fontsize=4.5, va="center", ha="right", color=S.INK)
        hx, bx = X(g.loc["Heart", "median"]), X(g.loc["Brain", "median"])
        cv.plot([hx, bx], [yy, yy], color="#9AA0A8", lw=0.8, zorder=1, solid_capstyle="round")
        for org, mk, sz in [("Heart", "o", 3.0), ("Protein", "s", 2.5), ("Brain", "D", 2.4)]:
            if org in g.index:
                lo, hi = g.loc[org, "lo"], g.loc[org, "hi"]
                if hi > lo:
                    cv.plot([X(lo), X(hi)], [yy, yy], color=S.ORG[org], lw=0.5, alpha=0.7, zorder=2)
                cv.plot(X(g.loc[org, "median"]), yy, mk, ms=sz, mfc=S.ORG[org], mec="white", mew=0.35, zorder=3)
        cv.text(R + 6.5, yy, S.md(g.loc["P(heart<brain)", "median"], 2), fontsize=4.5, ha="center", va="center",
                color=S.INK)
    for i, (org, mk, lab) in enumerate([("Heart", "o", "first cardiac"), ("Protein", "s", "first protein"),
                                        ("Brain", "D", "first brain")]):
        cv.plot(x0 + 5.6 + i * 18.5, y0 + 6.6, mk, ms=2.7, mfc=S.ORG[org], mec="white", mew=0.35)
        cv.text(x0 + 7.0 + i * 18.5, y0 + 6.6, lab, fontsize=4.3, va="center", color=S.INK)


# ----------------------------------------------------------------------------------------- D (compact)
def panel_d(F, cv, x0, y0, w=92):
    S.panel_letter(cv, x0 + 1, y0 + 1, "D", "Order reproduced by a de novo refit (n = 6000)")
    d = V6._meanpos(RES / "sustain_pvd_all10.csv", "all10")
    r = V6._meanpos(RES / "replication_sustain" / "denovo_pvd_K2.csv")
    for i, s in enumerate((1, 2)):
        a = d[d.subtype == s].set_index(["label", "z"]).mpos
        b = r[r.subtype == s].set_index(["label", "z"]).mpos
        com = [k for k in b.index if k in a.index]
        ra, rb = a[com].rank(), b[com].rank()
        rho = spearmanr(ra, rb)[0]
        ax = F.add(x0 + 11.0 + i * 43.0, y0 + 10.0, 26.0, 26.0)
        ax.plot([0.5, 18.5], [0.5, 18.5], color=S.RULE, lw=0.5, zorder=0)
        for (lab, z) in com:
            org = V6.ORGAN[lab]
            ax.plot(ra[(lab, z)], rb[(lab, z)], V6.ZMARK[int(z)], ms=2.8 if z == 1 else 2.3, mfc=S.ORG[org],
                    mec="white", mew=0.3, zorder=3, alpha=1.0 if z == 1 else 0.75)
            if z == 1:
                ax.text(ra[(lab, z)] + 0.6, rb[(lab, z)] - 0.5, V6.LABEL_SHORT[lab], fontsize=3.6,
                        color=S.ORG[org], ha="left", va="bottom", fontweight="bold", zorder=4)
        ax.set_xlim(0.3, 18.7); ax.set_ylim(0.3, 18.7)
        ax.set_xticks([1, 6, 12, 18]); ax.set_yticks([1, 6, 12, 18])
        ax.tick_params(labelsize=4.1, length=1.2, pad=0.8)
        ax.set_xlabel("derivation rank", fontsize=4.4, labelpad=0.8)
        ax.set_ylabel("refit rank", fontsize=4.4, labelpad=0.8)
        ax.set_aspect("equal")
        ax.set_title(f"S{s} \u00b7 " + ("LV dysfunction-led" if s == 1 else "LV remodelling-led"), fontsize=4.6,
                     fontweight="bold", color=S.INK, pad=1.6)
        ax.text(1.2, 17.6, f"\u03c1 = {S.md(rho, 2)}", fontsize=4.6, color=S.INK, va="top", fontweight="bold")
    xs = x0 + 4.0
    for z in (1, 2, 3):
        cv.plot(xs, y0 + 6.6, V6.ZMARK[z], ms=2.5, mfc=S.GREY, mec="white", mew=0.3)
        cv.text(xs + 1.3, y0 + 6.6, f"z \u2265 {z}", fontsize=4.2, va="center", color=S.INK)
        xs += 10.5
    for o in ("Heart", "Brain"):
        cv.plot(xs, y0 + 6.6, "o", ms=2.5, mfc=S.ORG[o], mec="white", mew=0.3)
        cv.text(xs + 1.3, y0 + 6.6, o.lower(), fontsize=4.2, va="center", color=S.INK)
        xs += 10.5
    cv.text(xs + 1.0, y0 + 6.6, "18 shared MRI events; ranked mean positions", fontsize=4.0, va="center",
            color=S.GREY)


# ----------------------------------------------------------------------------------------- driver
def render(which, sample="derivation"):
    S.set_style()
    if which in ("a", "all"):
        Fm = BS.MMFig(W, 82); cv = Fm.canvas(); G.panel_a(cv, 0, 0); Fm.save(str(OUT / "Fig2a")); print("a")
    if which in ("b", "all"):
        Fm = BS.MMFig(W, HB); cv = Fm.canvas(); panel_b(Fm, cv, 0, 0, sample)
        Fm.save(str(OUT / ("Fig2b" if sample == "derivation" else "Fig2b_independent"))); print("b", sample)
    if which in ("s4", "all"):     # appendix figure S4: the same matrix in the independent sample (no panel letter)
        Fm = BS.MMFig(W, HB); cv = Fm.canvas(); panel_b(Fm, cv, 0, 0, "independent", letter="")
        AP = OUT.parent / "Appendix"; AP.mkdir(parents=True, exist_ok=True)
        Fm.save(str(AP / "FigureS4_regional_by_subtype_stage_independent")); print("s4")
    if which in ("c", "all"):
        Fm = BS.MMFig(88, HC); cv = Fm.canvas(); panel_c(cv, 0, 0); Fm.save(str(OUT / "Fig2c")); print("c")
    if which in ("d", "all"):
        Fm = BS.MMFig(92, HC); cv = Fm.canvas(); panel_d(Fm, cv, 0, 0); Fm.save(str(OUT / "Fig2d")); print("d")
    if which in ("composite", "all"):
        Fm = BS.MMFig(W, H); cv = Fm.canvas()
        G.panel_a(cv, 0, 0); panel_b(Fm, cv, 0, YB); panel_c(cv, 0, YC); panel_d(Fm, cv, 88, YC)
        cv.plot([1, W - 1], [YB - 1.0, YB - 1.0], color=S.RULE, lw=0.4)
        cv.plot([1, W - 1], [YC - 1.0, YC - 1.0], color=S.RULE, lw=0.4)
        cv.plot([87.0, 87.0], [YC + 1.0, H - 2.0], color=S.RULE, lw=0.4)
        Fm.save(str(OUT / "Figure2_composite")); print("composite", W, H)


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "all", sys.argv[2] if len(sys.argv) > 2 else "derivation")
