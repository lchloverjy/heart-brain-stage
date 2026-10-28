import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Figure 3 (v4) - heart-brain stage and subsequent cardiovascular and cerebrovascular events.
a-b, cumulative incidence by stage group (derivation, imaging-anchored); c, multi-outcome forest (derivation three models +
independent sample, imaging-anchored); d, organ-concordant risk before versus after imaging (independent sample); e, model-implied involvement."""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator
import style4 as S

RES = _os.path.join(HB_ROOT, "results"); V = f"{RES}/validation"
OUT = _os.path.join(HB_ROOT, "figs/output/Figure3")
W = 180
GRP = ["0", "1", "2", ">=3"]
GRP_LAB = {"0": "Stage 0", "1": "Stage 1", "2": "Stage 2", ">=3": "Stage \u22653"}
GRP_COL = {"0": "#A9B2BC", "1": "#86BFA9", "2": "#3C9A7C", ">=3": "#0F4A38"}
INDEP = "#3D5A80"
MODELS = ["all10", "mri_plus_m13", "mri_only_matched_all10"]
MCOL = {"all10": "#0F4A38", "mri_plus_m13": "#3E8E73", "mri_only_matched_all10": "#8FB9A8", "indep": INDEP}
MMK = {"all10": "o", "mri_plus_m13": "s", "mri_only_matched_all10": "D", "indep": "^"}
OUTLAB = {"hbve": "Heart\u2013brain vascular events", "mace": "Stroke, MI, or heart failure", "cardiac": "Cardiac composite",
          "ihd": "Ischaemic heart disease", "af": "Atrial fibrillation", "hf": "Heart failure",
          "cerebrovascular": "Cerebrovascular composite", "stroke": "Stroke", "depression": "Depression", "anxiety": "Anxiety",
          "parkinson": "Parkinson's disease", "ckd": "Chronic kidney disease", "mi": "Myocardial infarction", "tia": "TIA"}


def cox():
    return pd.read_csv(f"{RES}/out_cox_main.csv")


def curve_panel(F, cv, x0, y0, o, letter, title, w=86, h=63):
    km = pd.read_csv(f"{RES}/out_km.csv"); km = km[(km.outcome == o) & (km.scheme == "all10")]
    km["group"] = km.group.astype(str)
    S.panel_letter(cv, x0 + 1, y0 + 1, letter, title)
    ax = F.add(x0 + 12, y0 + 12.5, w - 15, h - 34.5)
    for g in GRP:
        c = km[(km.group == g) & km.kind.isna()].sort_values("t")
        c = c[c.t <= 6.0]
        if g in ("0", ">=3"):
            ax.fill_between(c.t, 100 * c.lo, 100 * c.hi, step="post", color=GRP_COL[g], alpha=0.12, lw=0, zorder=1)
        ax.step(c.t, 100 * c.cuminc, where="post", color=GRP_COL[g], lw=0.95 if g != ">=3" else 1.15, zorder=3)
    ax.set_xlim(0, 6); ax.set_xticks(range(0, 7)); ax.set_xticklabels([])
    ymax = {"hbve": 12, "mace": 6}.get(o, 8)
    step = {12: 3, 6: 2}.get(ymax, 2)
    ax.set_ylim(0, ymax); ax.yaxis.set_major_locator(FixedLocator(np.arange(0, ymax + 0.1, step)))
    ax.yaxis.set_major_formatter(S.md_formatter(0))
    ax.set_ylabel("Cumulative incidence (%)", fontsize=5.6, labelpad=2)
    ax.grid(axis="y", color=S.RULE, lw=0.35); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    # end labels
    ends = []
    for g in GRP:
        c = km[(km.group == g) & km.kind.isna()]; c = c[c.t <= 6.0]
        ends.append((g, 100 * c.cuminc.iloc[-1] if len(c) else 0))
    # HR annotation
    cx = cox(); cx = cx[(cx.outcome == o) & (cx.model == "all10") & (cx.adj == "M2")]
    r1 = cx[cx.exposure == "per10pp"].iloc[0]; r3 = cx[cx.exposure == "grp>=3"].iloc[0]
    ax.text(0.03, 0.97, f"HR per 10 percentage points {S.hr_ci(r1.HR, r1.lo, r1.hi)}", transform=ax.transAxes, fontsize=5.1,
            va="top", ha="left", color=S.INK)
    ax.text(0.03, 0.87, f"Stage \u22653 vs 0  {S.hr_ci(r3.HR, r3.lo, r3.hi)}", transform=ax.transAxes, fontsize=5.1, va="top",
            ha="left", color=S.INK)
    ax.text(0.03, 0.77, f"{int(r1.events)} events in {S.md_int(int(r1.n))}; adjusted for clinical risk factors", transform=ax.transAxes,
            fontsize=4.6, va="top", ha="left", color=S.GREY)
    # legend
    xx = x0 + 12.5; yy = y0 + 8.4
    for g in GRP:
        cv.plot([xx, xx + 3.0], [yy, yy], color=GRP_COL[g], lw=1.3, solid_capstyle="butt")
        cv.text(xx + 3.8, yy, GRP_LAB[g], fontsize=5.0, va="center", color=S.INK)
        xx += S.text_w_mm(GRP_LAB[g], 5.0) + 7.5
    cv.text(xx + 0.2, yy, "shaded: 95% CI", fontsize=4.5, va="center", color=S.GREY)
    # risk table
    rt = km[km.kind == "risk_table"]
    ty0 = y0 + h - 18.5
    axx0 = x0 + 12; axw = w - 15
    for t in range(0, 7):
        xx = axx0 + axw * t / 6
        cv.text(xx, ty0 - 2.6, str(t), fontsize=5.0, ha="center", va="top", color=S.GREY)
    cv.text(axx0 + axw / 2, ty0 + 0.6, "Years since imaging visit", fontsize=5.4, ha="center", va="top", color=S.INK)
    cv.text(x0 + 1.5, ty0 + 5.0, "Number at risk", fontsize=5.0, fontweight="bold", va="center", color=S.INK)
    for i, g in enumerate(GRP):
        yy = ty0 + 8.2 + i * 2.65
        cv.text(x0 + 1.5, yy, GRP_LAB[g], fontsize=4.8, va="center", color=GRP_COL[g] if g != "0" else "#7D8792", fontweight="bold")
        rg = rt[rt.group == g].set_index("t").at_risk
        for t in range(0, 7):
            v = rg.get(float(t), np.nan)
            if np.isnan(v):
                continue
            lab = S.md_int(int(v)) if v >= 5 or v == 0 else "<5"
            cv.text(axx0 + axw * t / 6, yy, lab, fontsize=4.7, ha="center", va="center", color=S.INK)


ROWS = [("Composite", ["hbve", "mace"]), ("Cardiac", ["cardiac", "ihd", "af", "hf"]),
        ("Cerebrovascular", ["cerebrovascular", "stroke"]), ("Mental health", ["depression", "anxiety"]),
        ("Other", ["parkinson", "ckd"])]


def panel_c(F, cv, x0, y0, w=98, h=96):
    cx = cox(); cx = cx[(cx.adj == "M2") & (cx.exposure == "per10pp")]
    ind = pd.read_csv(f"{V}/independent_landmark_cox_imaging.csv"); ind = ind[ind.adj == "M2"].set_index("outcome")
    S.panel_letter(cv, x0 + 1, y0 + 1, "c", "Stage and incident disease across outcomes")
    LBL = 30.0; AX = 27.0; ax0 = x0 + LBL; ax1 = ax0 + AX
    lo_, hi_ = 0.5, 4.0
    def X(v): return ax0 + (np.log(v) - np.log(lo_)) / (np.log(hi_) - np.log(lo_)) * AX
    top = y0 + 16.5; rh = 4.3; gh = 3.2
    # headers
    cv.text(ax1 + 2.4, top - 6.2, "Derivation", fontsize=5.0, fontweight="bold", color=MCOL["all10"], ha="left")
    cv.text(ax1 + 2.4, top - 3.7, "events  All-10 HR", fontsize=4.7, color=S.GREY, ha="left")
    cv.text(x0 + w - 1, top - 6.2, "Independent", fontsize=5.0, fontweight="bold", color=INDEP, ha="right")
    cv.text(x0 + w - 1, top - 3.7, "HR after imaging", fontsize=4.7, color=S.GREY, ha="right")
    yy = top; ys = []
    for gname, outs in ROWS:
        cv.text(x0 + 2, yy, gname, fontsize=5.2, fontweight="bold", color=S.INK, va="center")
        yy += gh
        for o in outs:
            ys.append((o, yy)); yy += rh
        yy += 0.6
    bottom = yy - 0.6
    for v in [0.5, 1, 2, 4]:
        cv.plot([X(v), X(v)], [top - 1.8, bottom], color="#8E949C" if v == 1 else S.RULE, lw=0.5 if v == 1 else 0.35, zorder=0,
                ls="-" if v == 1 else (0, (2, 1.4)))
        cv.text(X(v), bottom + 1.0, S.md(v, 1) if v < 1 else str(int(v)), fontsize=4.9, ha="center", va="top", color=S.GREY)
    cv.text((ax0 + ax1) / 2, bottom + 4.2, "HR per 10 percentage points of stage", fontsize=5.2, ha="center", va="top")
    for i, (o, y) in enumerate(ys):
        if i % 2 == 0:
            cv.add_patch(Rectangle((x0 + 1, y - rh / 2), w - 2, rh, fc=S.BAND, ec="none", zorder=-1))
        cv.text(x0 + 4, y, OUTLAB[o], fontsize=5.0, va="center", color=S.INK)
        offs = {"all10": -1.35, "mri_plus_m13": -0.45, "mri_only_matched_all10": 0.45, "indep": 1.35}
        for m in MODELS + ["indep"]:
            if m == "indep":
                if o not in ind.index:
                    continue
                r = ind.loc[o]
            else:
                r = cx[(cx.outcome == o) & (cx.model == m)].iloc[0]
            yo = y + offs[m] * 0.85
            lo, hi = max(r.lo, lo_ * 1.001), min(r.hi, hi_ * 0.999)
            cv.plot([X(lo), X(hi)], [yo, yo], color=MCOL[m], lw=0.75, zorder=2, solid_capstyle="butt")
            if r.hi > hi_:
                cv.annotate("", xy=(X(hi_) + 0.8, yo), xytext=(X(hi_) - 0.2, yo), arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=MCOL[m], lw=0.6))
            cv.plot(X(r.HR), yo, MMK[m], ms=2.5 if m != "indep" else 2.9, mfc=MCOL[m] if m != "indep" else "white",
                    mec=MCOL[m], mew=0.6, zorder=3)
        r = cx[(cx.outcome == o) & (cx.model == "all10")].iloc[0]
        cv.text(ax1 + 2.4, y, f"{int(r.events)}", fontsize=4.7, va="center", ha="left", color=S.GREY)
        cv.text(ax1 + 7.6, y, S.hr_ci(r.HR, r.lo, r.hi), fontsize=4.7, va="center", ha="left", color=S.INK)
        if o in ind.index:
            q = ind.loc[o]
            cv.text(x0 + w - 1, y, S.hr_ci(q.HR, q.lo, q.hi), fontsize=4.7, va="center", ha="right", color=INDEP)
        else:
            cv.text(x0 + w - 1, y, "\u2013", fontsize=4.9, va="center", ha="right", color=S.GREY)
    # legend
    ly = bottom + 9.0
    items = [("all10", "All-10"), ("mri_plus_m13", "MRI + M13"), ("mri_only_matched_all10", "MRI only"),
             ("indep", "Independent sample (MRI-only model, n = 23\u2009974)")]
    xx = x0 + 3
    for m, lab in items:
        cv.plot(xx, ly, MMK[m], ms=2.7, mfc=MCOL[m] if m != "indep" else "white", mec=MCOL[m], mew=0.6)
        cv.text(xx + 1.6, ly, lab, fontsize=4.8, va="center", color=S.INK)
        xx += S.text_w_mm(lab, 4.8) + 5.0


CAT = ["Heart only", "Brain only", "Heart and brain"]
CATCOL = {"Heart only": "#D9692F", "Brain only": "#A995D3", "Heart and brain": "#5B3C8C"}
CATMK = {"Heart only": "o", "Brain only": "s", "Heart and brain": "D"}


def panel_d(F, cv, x0, y0, w=80, h=60):
    pre = pd.read_csv(f"{V}/independent_involvement_preimaging_imaging.csv"); pre = pre[pre.scheme == "obs_cat"]
    lm = pd.read_csv(f"{V}/independent_involvement_landmark_imaging.csv"); lm = lm[lm.scheme == "obs_cat"]
    S.panel_letter(cv, x0 + 1, y0 + 1, "d", "Brain marks past stroke; heart marks future stroke")
    outs = [("stroke", "Stroke"), ("cerebrovascular", "Cerebrovascular"), ("hf", "Heart failure"), ("af", "Atrial fibrillation")]
    LBL = 18.5; AW = 26.0; G = 5.0
    a0 = x0 + LBL; b0 = a0 + AW + G
    lo_, hi_ = 0.5, 8.0
    def X(base, v): return base + (np.log(np.clip(v, lo_, hi_)) - np.log(lo_)) / (np.log(hi_) - np.log(lo_)) * AW
    top = y0 + 15.5; rh = 2.8; gh = 1.2
    cv.text(a0 + AW / 2, top - 6.4, "At imaging", fontsize=5.2, fontweight="bold", ha="center", color=S.INK)
    cv.text(a0 + AW / 2, top - 3.9, "prevalent disease, OR", fontsize=4.6, ha="center", color=S.GREY)
    cv.text(b0 + AW / 2, top - 6.4, "After imaging", fontsize=5.2, fontweight="bold", ha="center", color=S.INK)
    cv.text(b0 + AW / 2, top - 3.9, "incident events, HR", fontsize=4.6, ha="center", color=S.GREY)
    yy = top; ys = []
    for o, lab in outs:
        cv.text(x0 + 2, yy + rh, lab, fontsize=5.0, fontweight="bold", va="center", color=S.INK)
        for c in CAT:
            ys.append((o, c, yy)); yy += rh
        yy += gh
    bottom = yy - gh
    for base in [a0, b0]:
        for v in [0.5, 1, 2, 4, 8]:
            cv.plot([X(base, v), X(base, v)], [top - 1.7, bottom - 0.6], color="#8E949C" if v == 1 else S.RULE,
                    lw=0.5 if v == 1 else 0.35, zorder=0, ls="-" if v == 1 else (0, (2, 1.4)))
            cv.text(X(base, v), bottom + 0.3, S.md(v, 1) if v < 1 else str(int(v)), fontsize=4.7, ha="center", va="top",
                    color=S.GREY)
    for o, c, y in ys:
        for base, tab in [(a0, pre), (b0, lm)]:
            r = tab[(tab.outcome == o) & (tab.level == c)]
            if not len(r):
                continue
            r = r.iloc[0]
            est = r.OR if "OR" in r and not np.isnan(r.get("OR", np.nan)) else r.HR
            sig = r.lo > 1
            cv.plot([X(base, r.lo), X(base, r.hi)], [y, y], color=CATCOL[c], lw=0.75, zorder=2)
            cnt = int(r.cases) if base == a0 else int(r.events)
            cv.text(base + AW + 0.7, y, str(cnt), fontsize=4.2, va="center", ha="left", color=S.GREY)
            cv.plot(X(base, est), y, CATMK[c], ms=2.5, mfc=CATCOL[c] if sig else "white", mec=CATCOL[c], mew=0.6, zorder=3)
    # legend
    ly = bottom + 5.2
    xx = x0 + 3
    for c in CAT:
        cv.plot(xx, ly, CATMK[c], ms=2.6, mfc=CATCOL[c], mec=CATCOL[c], mew=0.6)
        cv.text(xx + 1.5, ly, c, fontsize=4.8, va="center", color=S.INK)
        xx += S.text_w_mm(c, 4.8) + 4.6
    cv.text(x0 + 3, ly + 3.0, "Independent imaging sample, MRI-only markers; reference: no marker at z \u2265 1. Open symbols:",
            fontsize=4.4, va="center", color=S.GREY)
    cv.text(x0 + 3, ly + 5.4, "95% CI includes 1. Grey numbers: prevalent cases (left) and incident events after imaging (right).",
            fontsize=4.4, va="center", color=S.GREY)


def panel_e(F, cv, x0, y0, w=80, h=34):
    inv = pd.read_csv(f"{RES}/out_involvement.csv"); inv = inv[inv.scheme == "model_all10"].copy()
    inv["level"] = inv.level.fillna("None")
    S.panel_letter(cv, x0 + 1, y0 + 1, "e", "Model-implied organ involvement (derivation)")
    outs = [("hbve", "Heart\u2013brain\nvascular"), ("cardiac", "Cardiac"), ("cerebrovascular", "Cerebro-\nvascular")]
    levels = ["None", "Heart only", "Heart and brain"]
    col = {"None": "#C9CED6", "Heart only": "#E9A27A", "Heart and brain": "#6B4C9A"}
    ax = F.add(x0 + 12, y0 + 10.5, w - 15, h - 19)
    bw = 0.26
    for i, (o, lab) in enumerate(outs):
        for j, lv in enumerate(levels):
            r = inv[(inv.outcome == o) & (inv.level == lv)].iloc[0]
            ev, py = r.events, r.py
            lo = stats.chi2.ppf(0.025, 2 * ev) / 2 / py * 1000 if ev else 0; hi = stats.chi2.ppf(0.975, 2 * ev + 2) / 2 / py * 1000
            xx = i + (j - 1) * (bw + 0.04)
            ax.bar(xx, r.rate, bw, color=col[lv], lw=0, zorder=2)
            ax.plot([xx, xx], [lo, hi], color="#3C4046", lw=0.5, zorder=3)
            ax.text(xx, hi + 0.6, f"{int(ev)}", fontsize=4.3, ha="center", va="bottom", color=S.GREY)
    ax.set_xticks(range(len(outs))); ax.set_xticklabels([l for _, l in outs], fontsize=4.9, linespacing=0.95)
    ax.set_ylabel("per 1000 person-years", fontsize=5.2, labelpad=2)
    ax.set_ylim(0, 30); ax.yaxis.set_major_locator(FixedLocator([0, 10, 20, 30]))
    ax.grid(axis="y", color=S.RULE, lw=0.35); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    xx = x0 + 13.0; yy = y0 + 7.6
    for lv in levels:
        lab = {"None": "Stage 0", "Heart only": "Heart only", "Heart and brain": "Heart + brain"}[lv]
        cv.add_patch(Rectangle((xx, yy - 0.9), 1.8, 1.8, fc=col[lv], ec="none"))
        cv.text(xx + 2.5, yy, lab, fontsize=4.8, va="center", color=S.INK)
        xx += S.text_w_mm(lab, 4.8) + 6.0
    cv.text(xx, yy, "numbers above bars = events", fontsize=4.4, color=S.GREY, va="center")


def render(which="all"):
    os.makedirs(OUT, exist_ok=True)
    S.set_style()
    if which in ("a", "all"):
        F = S.MMFig(88, 64); cv = F.canvas(zorder=5); curve_panel(F, cv, 0, 0, "hbve", "a", "Heart\u2013brain vascular events"); F.save(f"{OUT}/Fig3a")
    if which in ("b", "all"):
        F = S.MMFig(88, 64); cv = F.canvas(zorder=5); curve_panel(F, cv, 0, 0, "mace", "b", "Stroke, myocardial infarction, or heart failure"); F.save(f"{OUT}/Fig3b")
    if which in ("c", "all"):
        F = S.MMFig(98, 100); cv = F.canvas(zorder=5); panel_c(F, cv, 0, 0); F.save(f"{OUT}/Fig3c")
    if which in ("d", "all"):
        F = S.MMFig(80, 66); cv = F.canvas(zorder=5); panel_d(F, cv, 0, 0); F.save(f"{OUT}/Fig3d")
    if which in ("e", "all"):
        F = S.MMFig(80, 36); cv = F.canvas(zorder=5); panel_e(F, cv, 0, 0); F.save(f"{OUT}/Fig3e")
    if which in ("composite", "all"):
        F = S.MMFig(W, 166); cv = F.canvas(zorder=5)
        curve_panel(F, cv, 0, 0, "hbve", "a", "Heart\u2013brain vascular events")
        curve_panel(F, cv, 92, 0, "mace", "b", "Stroke, myocardial infarction, or heart failure")
        panel_c(F, cv, 0, 66); panel_d(F, cv, 100, 66); panel_e(F, cv, 100, 131)
        F.save(f"{OUT}/Figure3_composite")


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "all")
