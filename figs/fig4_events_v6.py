"""Figure 4 (v6) - Heart-brain stage and cardiovascular and cerebrovascular events after imaging.

A  derivation sample (All-10 model): cumulative incidence of the primary outcome by stage group + number at risk
B  independent imaging sample (MRI-only model transferred, not refitted): same outcome, axis, and palette (new)
C  all outcomes: derivation All-10 (solid circle) and independent sample (open diamond) as main estimates; the two
   derivation sensitivity models (MRI + M13, MRI only) as faint ticks; right-aligned numeric columns
D  organ pattern of abnormality and disease before (OR) and after (HR) imaging, independent sample; 3D organ icons
E  model-implied organ involvement and event rates (derivation); 3D organ icons

Time zero is the imaging visit for every panel; follow-up to 30 Sep 2021; deaths from other causes censored.
Usage: python3 fig4_events_v6.py [a|b|c|d|e|composite|all]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator

import style4 as S
import render3d as R3
from fig2v4 import mix
import fig3v4 as B4            # shared constants (outcome labels, palettes) of the v4 figure

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; V = RES / "validation"
OUT = Path(__file__).resolve().parent / "output_v6" / "Figure4"
OUT.mkdir(parents=True, exist_ok=True)
W = 180
GRP, GRP_LAB, GRP_COL = B4.GRP, B4.GRP_LAB, B4.GRP_COL
DER = "#0F4A38"; IND = B4.INDEP
OUTLAB = dict(B4.OUTLAB, hbve="Primary composite outcome")
YMAX = 12


# ----------------------------------------------------------------------------------------- icons
def organ_icon(cv, x, y, heart_on, brain_on, s=4.0, z=6):
    """Tiny realistic heart + cortex pair (x, y = left centre). Tinted = abnormal, neutral = not abnormal."""
    h = R3.render_heart("lao", px=360, style="data",
                        lv_rgb=S.ORG["Heart"] if heart_on else R3.NEUTRAL,
                        aorta_rgb=S.ORG_MID["Heart"] if heart_on else R3.NEUTRAL,
                        desc_rgb=S.ORG_MID["Heart"] if heart_on else R3.NEUTRAL, boundary=False)
    b = R3.render_cortex("lh", "lateral", px=420, base=S.ORG_MID["Brain"] if brain_on else "#E6E3DF", sulc_depth=0.25)
    _, _, wh, _ = R3.place(cv, h, x, y - s / 2, h=s, anchor="tl", zorder=z)
    bw = s * 0.72 * b.shape[1] / b.shape[0]
    R3.place(cv, b, x + wh + 0.4, y - s * 0.36, w=bw, anchor="tl", zorder=z)
    return x + wh + 0.4 + bw


# ----------------------------------------------------------------------------------------- A / B
def _cox_info(sample):
    if sample == "derivation":
        cx = pd.read_csv(RES / "out_cox_main.csv")
        cx = cx[(cx.outcome == "hbve") & (cx.model == "all10") & (cx.adj == "M2")]
        r1 = cx[cx.exposure == "per10pp"].iloc[0]; r3 = cx[cx.exposure == "grp>=3"].iloc[0]
        return r1, r3, int(r1.events), int(r1.n)
    ind = pd.read_csv(V / "independent_landmark_cox_imaging.csv"); ind = ind[ind.outcome == "hbve"]
    r1 = ind[ind.adj == "M2"].iloc[0]; r3 = ind[ind.adj == "M2_grp>=3vs0"].iloc[0]
    return r1, r3, int(r1.events), int(r1.n)


def km_panel(F, cv, x0, y0, sample, letter, w=88, h=66):
    if sample == "derivation":
        km = pd.read_csv(RES / "out_km.csv"); km = km[(km.outcome == "hbve") & (km.scheme == "all10")]
        title = "Derivation sample"; tag = "All-10 model fitted here"; col = DER
    else:
        km = pd.read_csv(V / "independent_km_imaging.csv"); km = km[km.outcome == "hbve"]
        title = "Independent sample"; tag = "MRI-only model transferred, not refitted"; col = IND
    km = km.copy(); km["group"] = km.group.astype(str)
    S.panel_letter(cv, x0 + 1, y0 + 1, letter, f"{title}: primary outcome after imaging")
    cv.add_patch(FancyBboxPatch((x0 + 5.0, y0 + 6.0), S.text_w_mm(tag, 4.4) + 3.0, 3.0,
                                boxstyle="round,pad=0,rounding_size=1.5", fc=mix(col, "#FFFFFF", 0.86), ec="none"))
    cv.text(x0 + 6.5, y0 + 7.5, tag, fontsize=4.4, color=col, va="center", fontweight="bold")
    # legend
    xx = x0 + 6.0 + S.text_w_mm(tag, 4.4) + 6.5; yy = y0 + 7.5
    for g in GRP:
        cv.plot([xx, xx + 2.6], [yy, yy], color=GRP_COL[g], lw=1.2, solid_capstyle="butt")
        cv.text(xx + 3.2, yy, GRP_LAB[g], fontsize=4.5, va="center", color=S.INK)
        xx += S.text_w_mm(GRP_LAB[g], 4.5) + 5.6
    axx0, axw, axy0, axh = x0 + 11.5, w - 15.0, y0 + 13.0, 29.0
    ax = F.add(axx0, axy0, axw, axh)
    for g in GRP:
        c = km[(km.group == g) & km.kind.isna()].sort_values("t"); c = c[c.t <= 6.0]
        if g in ("0", ">=3"):
            ax.fill_between(c.t, 100 * c.lo, 100 * c.hi, step="post", color=GRP_COL[g], alpha=0.13, lw=0, zorder=1)
        ax.step(c.t, 100 * c.cuminc, where="post", color=GRP_COL[g], lw=0.9 if g != ">=3" else 1.2, zorder=3)
        if len(c):
            ax.plot(c.t.iloc[-1], 100 * c.cuminc.iloc[-1], "o", ms=1.6, color=GRP_COL[g], zorder=4, clip_on=False)
    ax.set_xlim(0, 6); ax.set_xticks(range(0, 7)); ax.set_xticklabels([str(t) for t in range(7)], fontsize=4.6)
    ax.set_ylim(0, YMAX); ax.yaxis.set_major_locator(FixedLocator(np.arange(0, YMAX + 0.1, 3)))
    ax.yaxis.set_major_formatter(S.md_formatter(0)); ax.tick_params(axis="y", labelsize=4.6)
    ax.tick_params(length=1.6, pad=1.2)
    ax.set_ylabel("Cumulative incidence (%)", fontsize=5.0, labelpad=1.5)
    ax.set_xlabel("Years since imaging visit", fontsize=5.0, labelpad=1.2)
    ax.grid(axis="y", color=S.RULE, lw=0.3); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    r1, r3, ev, n = _cox_info(sample)
    bx, by = axx0 + 2.0, axy0 + 1.0
    cv.add_patch(FancyBboxPatch((bx, by), 39.5, 9.6, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec=S.RULE, lw=0.3, zorder=6))
    cv.text(bx + 1.4, by + 2.0, "HR per 10 percentage points", fontsize=4.3, color=S.GREY, va="center", zorder=7)
    cv.text(bx + 38.2, by + 2.0, S.hr_ci(r1.HR, r1.lo, r1.hi), fontsize=4.6, color=S.INK, va="center", ha="right",
            fontweight="bold", zorder=7)
    cv.text(bx + 1.4, by + 4.9, "Stage \u22653 vs stage 0", fontsize=4.3, color=S.GREY, va="center", zorder=7)
    cv.text(bx + 38.2, by + 4.9, S.hr_ci(r3.HR, r3.lo, r3.hi), fontsize=4.6, color=S.INK, va="center", ha="right",
            fontweight="bold", zorder=7)
    cv.text(bx + 1.4, by + 7.8, f"{S.md_int(ev)} events in {S.md_int(n)}; adjusted for clinical risk factors",
            fontsize=3.9, color=S.GREY, va="center", zorder=7)
    # risk table
    rt = km[km.kind == "risk_table"]
    ty0 = y0 + 50.6
    cv.text(x0 + 1.5, ty0, "Number at risk, by stage", fontsize=4.6, fontweight="bold", va="center", color=S.INK)
    for i, g in enumerate(GRP):
        yy = ty0 + 3.0 + i * 2.5
        if i % 2 == 0:
            cv.add_patch(Rectangle((x0 + 1.0, yy - 1.25), w - 2.0, 2.5, fc=S.BAND, ec="none", zorder=0))
        cv.plot([x0 + 1.6, x0 + 3.8], [yy, yy], color=GRP_COL[g], lw=1.2, solid_capstyle="butt")
        cv.text(x0 + 4.5, yy, {"0": "0", "1": "1", "2": "2", ">=3": "\u22653"}[g], fontsize=4.4, va="center",
                color=GRP_COL[g] if g != "0" else "#7D8792", fontweight="bold")
        rg = rt[rt.group == g].set_index("t").at_risk
        for t in range(0, 7):
            v = rg.get(float(t), np.nan)
            if np.isnan(v):
                continue
            lab = S.md_int(int(v)) if v >= 5 or v == 0 else "<5"
            cv.text(axx0 + axw * t / 6, yy, lab, fontsize=4.3, ha="center", va="center", color=S.INK)


# ----------------------------------------------------------------------------------------- C
ROWS = B4.ROWS


def panel_c(F, cv, x0, y0, w=98):
    cx = pd.read_csv(RES / "out_cox_main.csv"); cx = cx[(cx.adj == "M2") & (cx.exposure == "per10pp")]
    ind = pd.read_csv(V / "independent_landmark_cox_imaging.csv"); ind = ind[ind.adj == "M2"].set_index("outcome")
    S.panel_letter(cv, x0 + 1, y0 + 1, "C", "Stage and events after imaging across outcomes")
    LBL, AX = 29.0, 26.0
    ax0 = x0 + LBL; ax1 = ax0 + AX
    lo_, hi_ = 0.5, 4.0

    def X(v):
        return ax0 + (np.log(np.clip(v, lo_, hi_)) - np.log(lo_)) / (np.log(hi_) - np.log(lo_)) * AX
    top = y0 + 15.0; rh = 4.5; gh = 3.0
    cD, cI = ax1 + 1.8, ax1 + 21.8
    cv.text(cD, top - 6.4, "Derivation (All-10)", fontsize=4.7, fontweight="bold", color=DER, ha="left")
    cv.text(cD, top - 3.9, "events   HR (95% CI)", fontsize=4.2, color=S.GREY, ha="left")
    cv.text(x0 + w - 1, top - 6.4, "Independent (MRI only)", fontsize=4.7, fontweight="bold", color=IND, ha="right")
    cv.text(x0 + w - 1, top - 3.9, "events   HR (95% CI)", fontsize=4.2, color=S.GREY, ha="right")
    yy = top; ys = []
    for gname, outs in ROWS:
        cv.text(x0 + 2, yy, gname, fontsize=4.9, fontweight="bold", color=S.INK, va="center")
        yy += gh
        for o in outs:
            ys.append((o, yy)); yy += rh
        yy += 0.5
    bottom = yy - 0.5
    for v in [0.5, 1, 2, 4]:
        cv.plot([X(v), X(v)], [top - 1.8, bottom], color="#8E949C" if v == 1 else S.RULE,
                lw=0.5 if v == 1 else 0.3, zorder=0, ls="-" if v == 1 else (0, (2, 1.4)))
        cv.text(X(v), bottom + 0.9, S.md(v, 1) if v < 1 else str(int(v)), fontsize=4.5, ha="center", va="top",
                color=S.GREY)
    cv.text((ax0 + ax1) / 2, bottom + 3.8, "HR per 10 percentage points of stage", fontsize=4.8, ha="center", va="top")
    for i, (o, y) in enumerate(ys):
        if i % 2 == 0:
            cv.add_patch(Rectangle((x0 + 1, y - rh / 2), w - 2, rh, fc=S.BAND, ec="none", zorder=-1))
        cv.text(x0 + 4, y, OUTLAB[o], fontsize=4.7, va="center", color=S.INK)
        r = cx[(cx.outcome == o) & (cx.model == "all10")].iloc[0]
        yd, yi = y - 0.85, y + 0.85
        cv.plot([X(r.lo), X(r.hi)], [yd, yd], color=DER, lw=0.75, zorder=2, solid_capstyle="butt")
        if r.hi > hi_:
            cv.annotate("", xy=(X(hi_) + 0.8, yd), xytext=(X(hi_) - 0.2, yd),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=DER, lw=0.6))
        for m in ("mri_plus_m13", "mri_only_matched_all10"):
            q = cx[(cx.outcome == o) & (cx.model == m)]
            if len(q):
                cv.plot([X(q.HR.iloc[0])] * 2, [yd - 0.75, yd + 0.75], color="#8FB9A8", lw=0.55, zorder=2.5)
        cv.plot(X(r.HR), yd, "o", ms=2.5, mfc=DER, mec="white", mew=0.3, zorder=3)
        cv.text(cD + 3.0, y, f"{int(r.events)}", fontsize=4.4, va="center", ha="right", color=S.GREY)
        cv.text(cD + 5.0, y, S.hr_ci(r.HR, r.lo, r.hi), fontsize=4.4, va="center", ha="left", color=S.INK)
        if o in ind.index:
            q = ind.loc[o]
            cv.plot([X(q.lo), X(q.hi)], [yi, yi], color=IND, lw=0.75, zorder=2, solid_capstyle="butt")
            cv.plot(X(q.HR), yi, "D", ms=2.4, mfc="white", mec=IND, mew=0.6, zorder=3)
            cv.text(x0 + w - 15.6, y, f"{S.md_int(int(q.events))}", fontsize=4.4, va="center", ha="right", color=S.GREY)
            cv.text(x0 + w - 1, y, S.hr_ci(q.HR, q.lo, q.hi), fontsize=4.4, va="center", ha="right", color=IND)
        else:
            cv.text(x0 + w - 1, y, "not available", fontsize=4.1, va="center", ha="right", color=S.GREY)
    ly = bottom + 8.4
    xx = x0 + 3
    cv.plot(xx, ly, "o", ms=2.5, mfc=DER, mec="white", mew=0.3)
    cv.text(xx + 1.6, ly, "Derivation, All-10", fontsize=4.5, va="center"); xx += S.text_w_mm("Derivation, All-10", 4.5) + 5
    cv.plot([xx, xx], [ly - 0.75, ly + 0.75], color="#8FB9A8", lw=0.55)
    cv.text(xx + 1.2, ly, "MRI + M13 and MRI-only models", fontsize=4.5, va="center")
    xx += S.text_w_mm("MRI + M13 and MRI-only models", 4.5) + 5
    cv.plot(xx, ly, "D", ms=2.4, mfc="white", mec=IND, mew=0.6)
    cv.text(xx + 1.6, ly, "Independent sample (n = 23\u2009974)", fontsize=4.5, va="center")
    cv.text(x0 + 3, ly + 3.0, "Adjusted for age, sex, and clinical risk factors (model 2); anxiety not analysed "
            "in the independent sample.", fontsize=4.0, va="center", color=S.GREY)
    return bottom + 13


# ----------------------------------------------------------------------------------------- D
CAT = B4.CAT; CATCOL = B4.CATCOL; CATMK = B4.CATMK
ICON = {"Heart only": (True, False), "Brain only": (False, True), "Heart and brain": (True, True)}


def panel_d(F, cv, x0, y0, w=80):
    pre = pd.read_csv(V / "independent_involvement_preimaging_imaging.csv"); pre = pre[pre.scheme == "obs_cat"]
    lm = pd.read_csv(V / "independent_involvement_landmark_imaging.csv"); lm = lm[lm.scheme == "obs_cat"]
    S.panel_letter(cv, x0 + 1, y0 + 1, "D", "Organ pattern of abnormality, before and after imaging")
    outs = [("stroke", "Stroke"), ("cerebrovascular", "Cerebro-\nvascular"), ("hf", "Heart\nfailure"),
            ("af", "Atrial\nfibrillation")]
    LBL, AW, G = 16.5, 26.0, 6.5
    a0 = x0 + LBL; b0 = a0 + AW + G
    lo_, hi_ = 0.5, 8.0

    def X(base, v):
        return base + (np.log(np.clip(v, lo_, hi_)) - np.log(lo_)) / (np.log(hi_) - np.log(lo_)) * AW
    top = y0 + 15.0; rh = 2.7; gh = 1.4
    for base, t1, t2 in ((a0, "At imaging", "prevalent disease, OR"), (b0, "After imaging", "new events, HR")):
        cv.text(base + AW / 2, top - 6.4, t1, fontsize=4.9, fontweight="bold", ha="center", color=S.INK)
        cv.text(base + AW / 2, top - 3.9, t2, fontsize=4.3, ha="center", color=S.GREY)
    yy = top; ys = []
    for k, (o, lab) in enumerate(outs):
        if k % 2 == 0:
            cv.add_patch(Rectangle((x0 + 1, yy - rh / 2 - 0.3), w - 2, 3 * rh + 0.6, fc=S.BAND, ec="none", zorder=-1))
        cv.text(x0 + 2, yy + rh, lab, fontsize=4.6, fontweight="bold", va="center", color=S.INK, linespacing=0.95)
        for c in CAT:
            ys.append((o, c, yy)); yy += rh
        yy += gh
    bottom = yy - gh
    for base in (a0, b0):
        for v in [0.5, 1, 2, 4, 8]:
            cv.plot([X(base, v), X(base, v)], [top - 1.7, bottom - 0.6], color="#8E949C" if v == 1 else S.RULE,
                    lw=0.5 if v == 1 else 0.3, zorder=0, ls="-" if v == 1 else (0, (2, 1.4)))
            cv.text(X(base, v), bottom + 0.2, S.md(v, 1) if v < 1 else str(int(v)), fontsize=4.4, ha="center",
                    va="top", color=S.GREY)
    for o, c, y in ys:
        for base, tab in ((a0, pre), (b0, lm)):
            r = tab[(tab.outcome == o) & (tab.level == c)]
            if not len(r):
                continue
            r = r.iloc[0]
            est = r.OR if "OR" in r and not np.isnan(r.get("OR", np.nan)) else r.HR
            sig = r.lo > 1
            cv.plot([X(base, r.lo), X(base, r.hi)], [y, y], color=CATCOL[c], lw=0.7, zorder=2)
            cnt = int(r.cases) if base == a0 else int(r.events)
            cv.text(base + AW + 0.6, y, str(cnt), fontsize=3.9, va="center", ha="left", color=S.GREY)
            cv.plot(X(base, est), y, CATMK[c], ms=2.3, mfc=CATCOL[c] if sig else "white", mec=CATCOL[c], mew=0.55,
                    zorder=3)
    ly = bottom + 6.4
    xx = x0 + 2.5
    for c in CAT:
        cv.plot(xx, ly, CATMK[c], ms=2.4, mfc=CATCOL[c], mec=CATCOL[c], mew=0.5)
        xe = organ_icon(cv, xx + 1.6, ly, *ICON[c], s=3.6)
        cv.text(xe + 0.8, ly, c, fontsize=4.4, va="center", color=S.INK)
        xx = xe + S.text_w_mm(c, 4.4) + 3.6
    cv.text(x0 + 2.5, ly + 3.6, "Independent sample; reference, no MRI marker at z \u2265 1. Open symbols, 95% CI "
            "includes 1;", fontsize=4.0, va="center", color=S.GREY)
    cv.text(x0 + 2.5, ly + 5.8, "grey numbers, prevalent cases (left) and new events (right). Associations, not "
            "mechanisms.", fontsize=4.0, va="center", color=S.GREY)
    return ly + 7.5


# ----------------------------------------------------------------------------------------- E
def panel_e(F, cv, x0, y0, w=80, h=36):
    inv = pd.read_csv(RES / "out_involvement.csv"); inv = inv[inv.scheme == "model_all10"].copy()
    inv["level"] = inv.level.fillna("None")
    S.panel_letter(cv, x0 + 1, y0 + 1, "E", "Model-implied organ involvement and event rates (derivation)")
    outs = [("hbve", "Primary\ncomposite"), ("cardiac", "Cardiac"), ("cerebrovascular", "Cerebro-\nvascular")]
    levels = ["None", "Heart only", "Heart and brain"]
    col = {"None": "#C9CED6", "Heart only": "#E9A27A", "Heart and brain": "#6B4C9A"}
    icon = {"None": (False, False), "Heart only": (True, False), "Heart and brain": (True, True)}
    xx = x0 + 3.0; yy = y0 + 7.4
    for lv in levels:
        lab = {"None": "Stage 0", "Heart only": "Heart only", "Heart and brain": "Heart + brain"}[lv]
        cv.add_patch(Rectangle((xx, yy - 0.9), 1.8, 1.8, fc=col[lv], ec="none"))
        xe = organ_icon(cv, xx + 2.4, yy, *icon[lv], s=3.6)
        cv.text(xe + 0.8, yy, lab, fontsize=4.4, va="center", color=S.INK)
        xx = xe + S.text_w_mm(lab, 4.4) + 4.0
    ax = F.add(x0 + 12, y0 + 12.0, w - 15, h - 20)
    bw = 0.26
    for i, (o, lab) in enumerate(outs):
        for j, lv in enumerate(levels):
            r = inv[(inv.outcome == o) & (inv.level == lv)].iloc[0]
            ev, py = r.events, r.py
            lo = stats.chi2.ppf(0.025, 2 * ev) / 2 / py * 1000 if ev else 0
            hi = stats.chi2.ppf(0.975, 2 * ev + 2) / 2 / py * 1000
            xb = i + (j - 1) * (bw + 0.04)
            ax.bar(xb, r.rate, bw, color=col[lv], lw=0, zorder=2)
            ax.plot([xb, xb], [lo, hi], color="#3C4046", lw=0.45, zorder=3)
            ax.text(xb, hi + 0.6, f"{int(ev)}", fontsize=3.9, ha="center", va="bottom", color=S.GREY)
    ax.set_xticks(range(len(outs))); ax.set_xticklabels([l for _, l in outs], fontsize=4.5, linespacing=0.95)
    ax.set_ylabel("per 1000 person-years", fontsize=4.7, labelpad=1.5)
    ax.set_ylim(0, 36); ax.yaxis.set_major_locator(FixedLocator([0, 10, 20, 30])); ax.tick_params(axis="y", labelsize=4.4)
    ax.grid(axis="y", color=S.RULE, lw=0.3); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=1.5)
    cv.text(x0 + w - 2.5, y0 + 11.6, "numbers above bars, events; whiskers, 95% CI", fontsize=3.8, color=S.GREY,
            va="center", ha="right")


# ----------------------------------------------------------------------------------------- driver
Y2 = 69


def render(which="all"):
    S.set_style()
    if which in ("a", "all"):
        F = S.MMFig(88, 64); cv = F.canvas(zorder=5); km_panel(F, cv, 0, 0, "derivation", "A"); F.save(f"{OUT}/Fig4a")
        print("saved a")
    if which in ("b", "all"):
        F = S.MMFig(88, 64); cv = F.canvas(zorder=5); km_panel(F, cv, 0, 0, "independent", "B"); F.save(f"{OUT}/Fig4b")
        print("saved b")
    if which in ("c", "all"):
        F = S.MMFig(98, 98); cv = F.canvas(zorder=5); panel_c(F, cv, 0, 0); F.save(f"{OUT}/Fig4c"); print("saved c")
    if which in ("d", "all"):
        F = S.MMFig(80, 67); cv = F.canvas(zorder=5); panel_d(F, cv, 0, 0); F.save(f"{OUT}/Fig4d"); print("saved d")
    if which in ("e", "all"):
        F = S.MMFig(80, 36); cv = F.canvas(zorder=5); panel_e(F, cv, 0, 0); F.save(f"{OUT}/Fig4e"); print("saved e")
    if which in ("composite", "all"):
        F = S.MMFig(W, 171); cv = F.canvas(zorder=5)
        km_panel(F, cv, 0, 0, "derivation", "A"); km_panel(F, cv, 92, 0, "independent", "B")
        cv.plot([1, W - 1], [Y2 - 2.0, Y2 - 2.0], color=S.RULE, lw=0.4)
        cv.plot([90.0, 90.0], [2, Y2 - 4], color=S.RULE, lw=0.35)
        panel_c(F, cv, 0, Y2)
        ye = panel_d(F, cv, 100, Y2)
        panel_e(F, cv, 100, max(ye + 1.0, Y2 + 64))
        F.save(f"{OUT}/Figure4_composite"); print("saved composite", W, 171)


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "all")
