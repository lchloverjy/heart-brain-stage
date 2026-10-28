"""Figure 5 (v6) - Clinical value beyond SCORE2 (restrained presentation).

A  event rates (per 1000 person-years, exact Poisson 95% CI) by stage group within SCORE2 risk groups; derivation and
   independent sample on a shared axis; reference line = high-risk stage 0; events and n tabulated underneath
B  change in C-index over SCORE2: main estimates = derivation All-10 (solid circle; red tick = optimism-corrected) and
   independent sample (open diamond); MRI + M13 and MRI-only models as faint ticks; right column ΔC (95% CI)
C  decision curve, derivation: Δ net benefit (SCORE2 + stage minus SCORE2) at 3 years; primary outcome raw (thin) and
   3-point moving average (thick); other outcomes smoothed and muted; shaded threshold range with gain
D  benchmark against simpler burden summaries (number of abnormal markers, mean abnormality z)

Usage: python3 fig5_value_v6.py [a|b|c|d|composite|all]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator

import style4 as S
from fig2v4 import mix
import fig4v4 as B5            # shared constants of the v4 figure (stage palette, outcome labels)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; V = RES / "validation"
OUT = Path(__file__).resolve().parent / "output_v6" / "Figure5"
OUT.mkdir(parents=True, exist_ok=True)
W = 180
STG, STG_LAB, STG_COL = B5.STG, B5.STG_LAB, B5.STG_COL
DER = "#0F4A38"; IND = B5.INDEP; SENS = "#8FB9A8"; OPT = "#C0392B"
OUTLAB = dict(B5.OUTLAB, hbve="Primary composite outcome")
GROUPS = ["Low-to-moderate", "High or very high"]
GLAB = {"Low-to-moderate": "SCORE2 low to moderate", "High or very high": "SCORE2 high or very high"}


def pval(p):
    if p < 0.001:
        return "p<0\u00b7001"
    return f"p={S.md(p, 3 if p < 0.1 else 2)}"


# ----------------------------------------------------------------------------------------- A
def strata_block(F, cv, x, y, w, h, src, title, tcol, ymax=40, show_ylab=True, row_labels=True):
    st = pd.read_csv(src); st = st[st.outcome == "hbve"]
    cv.text(x + 9.5, y + 1.6, title, fontsize=5.0, fontweight="bold", va="center", color=tcol)
    axx, axy, axw, axh = x + 9.5, y + 5.0, w - 10.5, h - 17.0
    ax = F.add(axx, axy, axw, axh)
    ref = st[(st.score2_group == "High or very high") & (st.stage_group == "Stage 0")].iloc[0]
    ax.axhline(ref.rate, color="#8E949C", lw=0.45, ls=(0, (2.2, 1.6)), zorder=1)
    xs, cells = [], []
    for gi, g in enumerate(GROUPS):
        ax.add_patch(Rectangle((gi * 3.6 - 0.55, 0), 3.1, ymax, fc=S.BAND if gi == 0 else "#F6F1EC", ec="none", zorder=0))
        for si, sg in enumerate(STG):
            r = st[(st.score2_group == g) & (st.stage_group == sg)].iloc[0]
            xx = gi * 3.6 + si
            xs.append(xx); cells.append(r)
            ax.plot([xx, xx], [r.rate_lo, min(r.rate_hi, ymax * 0.995)], color=STG_COL[sg], lw=0.9, zorder=2,
                    solid_capstyle="round")
            ax.plot(xx, r.rate, "o", ms=3.6, mfc=STG_COL[sg], mec="white", mew=0.5, zorder=3)
    ax.set_xlim(-0.55, 3.6 + 2.55); ax.set_ylim(0, ymax)
    ax.set_xticks(xs); ax.set_xticklabels([STG_LAB[s] for s in STG] * 2, fontsize=4.6)
    ax.tick_params(axis="x", length=0, pad=1.2); ax.tick_params(axis="y", labelsize=4.5, length=1.6, pad=1.2)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, ymax, 5))); ax.yaxis.set_major_formatter(S.md_formatter(0))
    if show_ylab:
        ax.set_ylabel("Events per 1000 person-years", fontsize=4.8, labelpad=1.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for gi, g in enumerate(GROUPS):
        ax.text(gi * 3.6 + 1, ymax * 0.975, ["Low to\nmoderate", "High or\nvery high"][gi], fontsize=4.3, ha="center",
                va="top", color=S.INK, linespacing=0.95)
    ax.text(6.0, ref.rate + ymax * 0.012, "high risk, stage 0", fontsize=3.9, ha="right", va="bottom", color=S.GREY)
    # events / n mini-table
    ty = axy + axh + 4.6
    if row_labels:
        cv.text(x + 1.0, ty, "events", fontsize=3.9, color=S.GREY, va="center")
        cv.text(x + 1.0, ty + 2.4, "n", fontsize=3.9, color=S.GREY, va="center")
    for xx, r in zip(xs, cells):
        X = axx + (xx + 0.55) / 6.15 * axw
        cv.text(X, ty, f"{int(r.events)}", fontsize=3.9, ha="center", va="center", color=S.INK)
        cv.text(X, ty + 2.4, S.md_int(int(r.n)), fontsize=3.8, ha="center", va="center", color=S.GREY)
    cv.text(axx + axw / 2, ty + 5.2, "Heart\u2013brain stage", fontsize=4.6, ha="center", va="center", color=S.INK)


def panel_a(F, cv, x0, y0, w=104, h=62):
    S.panel_letter(cv, x0 + 1, y0 + 1, "A", "Event rates by heart\u2013brain stage within SCORE2 risk groups")
    strata_block(F, cv, x0, y0 + 7, w / 2 - 1, h - 7, RES / "out_score2_strata.csv", "Derivation (All-10)", DER)
    strata_block(F, cv, x0 + w / 2 + 1, y0 + 7, w / 2 - 1, h - 7, V / "independent_score2_strata_imaging.csv",
                 "Independent (MRI only)", IND, row_labels=False)


# ----------------------------------------------------------------------------------------- B
MODELS_SENS = ["mri_plus_m13", "mri_only_matched_all10"]


def panel_b(F, cv, x0, y0, w=74, h=62):
    inc = pd.read_csv(RES / "out_score2_increment.csv")
    iinc = pd.read_csv(V / "independent_score2_increment_imaging.csv").set_index("outcome")
    S.panel_letter(cv, x0 + 1, y0 + 1, "B", "Discrimination added to SCORE2")
    outs = ["hbve", "mace", "cardiac", "hf", "af", "cerebrovascular", "stroke"]
    LBL, AX = 23.5, 28.0
    a0 = x0 + LBL
    lo_, hi_ = -0.02, 0.20

    def X(v):
        return a0 + (np.clip(v, lo_, hi_) - lo_) / (hi_ - lo_) * AX
    top = y0 + 12.5; rh = 5.4
    cv.text(x0 + w - 0.5, top - 5.6, "\u0394C (95% CI)", fontsize=4.5, ha="right", va="center", color=S.INK,
            fontweight="bold")
    bottom = top + rh * len(outs) - rh / 2
    for v in [0, 0.05, 0.10, 0.15, 0.20]:
        cv.plot([X(v), X(v)], [top - rh / 2 - 0.4, bottom], color="#8E949C" if v == 0 else S.RULE,
                lw=0.5 if v == 0 else 0.3, ls="-" if v == 0 else (0, (2, 1.4)), zorder=0)
        cv.text(X(v), bottom + 0.8, S.md(v, 2) if v else "0", fontsize=4.4, ha="center", va="top", color=S.GREY)
    cv.text(a0 + AX / 2, bottom + 3.8, "\u0394C index vs SCORE2", fontsize=4.8, ha="center", va="top")
    for i, o in enumerate(outs):
        y = top + i * rh
        if i % 2 == 0:
            cv.add_patch(Rectangle((x0 + 1, y - rh / 2), w - 1.5, rh, fc=S.BAND, ec="none", zorder=-1))
        r0 = inc[(inc.outcome == o) & (inc.model == "all10")].iloc[0]
        lab = OUTLAB[o].replace("Primary composite outcome", "Primary composite").replace(" composite", "") \
            if o != "hbve" else "Primary composite"
        cv.text(x0 + 2.2, y - 0.95, lab, fontsize=4.6, va="center", color=S.INK)
        evt = f"{int(r0.events)} | {S.md_int(int(iinc.loc[o].events))} events" if o in iinc.index else f"{int(r0.events)} events"
        cv.text(x0 + 2.2, y + 1.25, evt, fontsize=3.9, va="center", color=S.GREY)
        yd, yi = y - 0.95, y + 0.95
        cv.plot([X(r0.dC_lo), X(r0.dC_hi)], [yd, yd], color=DER, lw=0.75, zorder=2, solid_capstyle="butt")
        for m in MODELS_SENS:
            q = inc[(inc.outcome == o) & (inc.model == m)]
            if len(q):
                cv.plot([X(q.dC.iloc[0])] * 2, [yd - 0.7, yd + 0.7], color=SENS, lw=0.55, zorder=2.5)
        cv.plot([X(r0.dC_corrected)] * 2, [yd - 0.75, yd + 0.75], color=OPT, lw=0.65, zorder=4)
        cv.plot(X(r0.dC), yd, "o", ms=2.4, mfc=DER, mec="white", mew=0.3, zorder=3)
        cv.text(x0 + w - 0.5, yd, f"{S.md(r0.dC, 3)} ({S.md(r0.dC_lo, 3)} to {S.md(r0.dC_hi, 3)})", fontsize=3.9,
                ha="right", va="center", color=DER)
        if o in iinc.index:
            q = iinc.loc[o]
            cv.plot([X(q.dC_lo), X(q.dC_hi)], [yi, yi], color=IND, lw=0.75, zorder=2, solid_capstyle="butt")
            cv.plot(X(q.dC), yi, "D", ms=2.2, mfc="white", mec=IND, mew=0.55, zorder=3)
            cv.text(x0 + w - 0.5, yi, f"{S.md(q.dC, 3)} ({S.md(q.dC_lo, 3)} to {S.md(q.dC_hi, 3)})", fontsize=3.9,
                    ha="right", va="center", color=IND)
    ly = bottom + 8.0; xx = x0 + 2.2
    items = [("o", DER, "Derivation, All-10"), ("D", IND, "Independent, MRI only")]
    for mk, c, lab in items:
        cv.plot(xx, ly, mk, ms=2.3, mfc=c if mk == "o" else "white", mec=c if mk == "D" else "white",
                mew=0.55 if mk == "D" else 0.3)
        cv.text(xx + 1.5, ly, lab, fontsize=4.3, va="center", color=S.INK)
        xx += S.text_w_mm(lab, 4.3) + 4.2
    ly2 = ly + 3.0; xx = x0 + 2.2
    cv.plot([xx, xx], [ly2 - 0.7, ly2 + 0.7], color=OPT, lw=0.65)
    cv.text(xx + 1.0, ly2, "optimism-corrected", fontsize=4.3, va="center", color=S.INK)
    xx += S.text_w_mm("optimism-corrected", 4.3) + 4.2
    cv.plot([xx, xx], [ly2 - 0.7, ly2 + 0.7], color=SENS, lw=0.55)
    cv.text(xx + 1.0, ly2, "MRI + M13 and MRI-only models", fontsize=4.3, va="center", color=S.INK)


# ----------------------------------------------------------------------------------------- C
def panel_c(F, cv, x0, y0, w=86, h=58):
    d = pd.read_csv(RES / "out_dca.csv")
    S.panel_letter(cv, x0 + 1, y0 + 1, "C", "Net benefit added to SCORE2 at 3 years (derivation)")
    ax = F.add(x0 + 11.5, y0 + 9.0, w - 33.5, h - 21.0)
    col = {"hbve": DER, "mace": "#B4534A", "cardiac": "#D9873F", "cerebrovascular": "#7A64B4"}
    lab = {"hbve": "Primary composite", "mace": "Stroke, MI, or HF", "cardiac": "Cardiac", "cerebrovascular": "Cerebrovascular"}
    h0 = d[d.outcome == "hbve"].sort_values("threshold")
    gain = h0.threshold[(h0.nb_score2_stage - h0.nb_score2) >= 0.001]
    g_lo, g_hi = 100 * gain.min(), 100 * gain.max()
    ax.axvspan(g_lo, g_hi, color=DER, alpha=0.07, lw=0, zorder=0)
    ax.axhline(0, color="#8E949C", lw=0.5, zorder=1)
    for o in ["cardiac", "cerebrovascular", "mace", "hbve"]:
        g = d[d.outcome == o].sort_values("threshold")
        y = 1000 * (g.nb_score2_stage - g.nb_score2)
        ys = pd.Series(y.to_numpy()).rolling(3, center=True, min_periods=1).mean()
        if o == "hbve":
            ax.plot(100 * g.threshold, y, color=mix(DER, "#FFFFFF", 0.55), lw=0.5, zorder=2.5)
            ax.plot(100 * g.threshold, ys, color=DER, lw=1.3, zorder=3)
        else:
            ax.plot(100 * g.threshold, ys, color=col[o], lw=0.7, alpha=0.75, zorder=2)
    ax.set_xlim(0.5, 10); ax.set_ylim(-3, 6)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 6, 8, 10])); ax.yaxis.set_major_locator(FixedLocator([-2, 0, 2, 4, 6]))
    ax.xaxis.set_major_formatter(S.md_formatter(0)); ax.yaxis.set_major_formatter(S.md_formatter(0))
    ax.tick_params(labelsize=4.5, length=1.6, pad=1.2)
    ax.set_xlabel("3-year risk threshold (%)", fontsize=4.8, labelpad=1.2)
    ax.set_ylabel("\u0394 net benefit (per 1000 people)", fontsize=4.8, labelpad=1.5)
    ax.grid(axis="y", color=S.RULE, lw=0.3); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    gh = S.md(g_hi, 1) if g_hi % 1 else S.md(g_hi, 0)
    ax.text((g_lo + g_hi) / 2, 5.7, f"gain at {S.md(g_lo, 0)}\u2013{gh}%", fontsize=4.3, ha="center", va="top", color=DER)
    lx = x0 + w - 20.5
    for i, o in enumerate(["hbve", "mace", "cardiac", "cerebrovascular"]):
        yy = y0 + 12.5 + i * 3.3
        cv.plot([lx, lx + 3], [yy, yy], color=col[o], lw=1.3 if o == "hbve" else 0.7, alpha=1 if o == "hbve" else 0.75)
        cv.text(lx + 3.8, yy, lab[o], fontsize=4.3, va="center", color=S.INK)
    yy = y0 + 12.5 + 4 * 3.3
    cv.plot([lx, lx + 3], [yy, yy], color=mix(DER, "#FFFFFF", 0.55), lw=0.5)
    cv.text(lx + 3.8, yy, "primary, unsmoothed", fontsize=4.3, va="center", color=S.GREY)
    cv.text(lx, y0 + 31.5, "SCORE2 + stage minus\nSCORE2; thick lines,\n3-point moving\naverage", fontsize=4.0,
            va="top", color=S.GREY, linespacing=1.1)


# ----------------------------------------------------------------------------------------- D
def panel_d(F, cv, x0, y0, w=90, h=58):
    cp = pd.read_csv(RES / "out_compare_predictors.csv")
    iv = pd.read_csv(V / "independent_discrimination_vs_counts_imaging.csv")
    S.panel_letter(cv, x0 + 1, y0 + 1, "D", "Benchmark against simpler burden summaries")
    PRED = [("stage", "Heart\u2013brain stage", DER, "o"), ("count", "Number of abnormal markers", "#9AA0A8", "s"),
            ("meanz", "Mean abnormality z", "#C9A227", "D")]
    rows = [("Derivation (All-10), over SCORE2", ["hbve", "mace", "cardiac", "cerebrovascular"]),
            ("Independent (MRI only), over clinical factors", ["hbve", "cardiac", "stroke", "af", "hf"])]
    LBL, AX = 30.0, 38.0
    a0 = x0 + LBL
    lo_, hi_ = -0.02, 0.08

    def X(v):
        return a0 + (np.clip(v, lo_, hi_) - lo_) / (hi_ - lo_) * AX
    top = y0 + 11.0; rh = 3.5
    yy = top; ys = []
    for gname, outs in rows:
        cv.text(x0 + 2, yy, gname, fontsize=4.7, fontweight="bold", va="center", color=DER if "Deriv" in gname else IND)
        yy += 3.1
        for o in outs:
            ys.append((gname, o, yy)); yy += rh
        yy += 0.8
    bottom = yy - 0.8 - rh / 2
    for v in [-0.02, 0, 0.02, 0.04, 0.06, 0.08]:
        cv.plot([X(v), X(v)], [top + 1.4, bottom], color="#8E949C" if v == 0 else S.RULE, lw=0.5 if v == 0 else 0.3,
                ls="-" if v == 0 else (0, (2, 1.4)), zorder=0)
        cv.text(X(v), bottom + 0.8, S.md(v, 2) if v else "0", fontsize=4.4, ha="center", va="top", color=S.GREY)
    cv.text(a0 + AX / 2, bottom + 3.8, "\u0394C index over reference model", fontsize=4.8, ha="center", va="top")
    k = 0
    for gname, o, y in ys:
        if k % 2 == 0:
            cv.add_patch(Rectangle((x0 + 1, y - rh / 2), w - 1.5, rh, fc=S.BAND, ec="none", zorder=-1))
        k += 1
        cv.text(x0 + 4, y, OUTLAB[o], fontsize=4.5, va="center", color=S.INK)
        if "Deriv" in gname:
            g = cp[cp.outcome == o].set_index("predictor")
            base = g.C_base.dropna().iloc[0]
            vals = {"stage": g.loc["all10_sd", "C_new"] - base, "count": g.loc["n_abnormal_z1_sd", "C_new"] - base,
                    "meanz": g.loc["mean_abn_z_sd", "C_new"] - base}
            pbc = g.loc["all10_sd|count", "LR_p"]
        else:
            r = iv[iv.outcome == o].iloc[0]
            vals = {"stage": r.C_rf_p10 - r.C_rf, "count": r.C_rf_n_abn - r.C_rf, "meanz": r.C_rf_mean_z - r.C_rf}
            pbc = None
        xs = [X(v) for v in vals.values()]
        cv.plot([min(xs), max(xs)], [y, y], color="#C9CED6", lw=0.9, zorder=1, solid_capstyle="round")
        for kk, lab, c, mk in PRED:
            cv.plot(X(vals[kk]), y, mk, ms=2.6 if kk == "stage" else 2.3, mfc=c, mec="white", mew=0.35,
                    zorder=3 if kk == "stage" else 2)
        if pbc is not None:
            cv.text(x0 + w - 0.5, y, pval(pbc), fontsize=4.1, ha="right", va="center", color=S.GREY)
    cv.text(x0 + w - 0.5, top, "stage beyond\ncount", fontsize=4.0, ha="right", va="center", color=S.GREY,
            linespacing=1.0)
    ly = bottom + 8.0; xx = x0 + 2
    for kk, lab, c, mk in PRED:
        cv.plot(xx, ly, mk, ms=2.5, mfc=c, mec="white", mew=0.35)
        cv.text(xx + 1.5, ly, lab, fontsize=4.3, va="center", color=S.INK)
        xx += S.text_w_mm(lab, 4.3) + 4.0
    cv.text(x0 + 2, ly + 3.0, "The stage is not the strongest single summary; its value is an ordered, organ-specific "
            "description.", fontsize=4.0, va="center", color=S.GREY)


# ----------------------------------------------------------------------------------------- driver
def render(which="all"):
    S.set_style()
    if which in ("a", "all"):
        F = S.MMFig(104, 63); cv = F.canvas(zorder=5); panel_a(F, cv, 0, 0); F.save(f"{OUT}/Fig5a"); print("saved a")
    if which in ("b", "all"):
        F = S.MMFig(74, 63); cv = F.canvas(zorder=5); panel_b(F, cv, 0, 0); F.save(f"{OUT}/Fig5b"); print("saved b")
    if which in ("c", "all"):
        F = S.MMFig(86, 60); cv = F.canvas(zorder=5); panel_c(F, cv, 0, 0); F.save(f"{OUT}/Fig5c"); print("saved c")
    if which in ("d", "all"):
        F = S.MMFig(90, 64); cv = F.canvas(zorder=5); panel_d(F, cv, 0, 0); F.save(f"{OUT}/Fig5d"); print("saved d")
    if which in ("composite", "all"):
        F = S.MMFig(W, 132); cv = F.canvas(zorder=5)
        panel_a(F, cv, 0, 0); panel_b(F, cv, 106, 0)
        cv.plot([1, W - 1], [65.5, 65.5], color=S.RULE, lw=0.4)
        panel_c(F, cv, 0, 68); panel_d(F, cv, 90, 68)
        F.save(f"{OUT}/Figure5_composite"); print("saved composite", W, 132)


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "all")
