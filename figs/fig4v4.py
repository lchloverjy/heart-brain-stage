import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Figure 4 (v4) - clinical value beyond SCORE2.
a, event rates by SCORE2 risk group x heart-brain stage; b, change in C-index over SCORE2; c, net-benefit gain at 3 years;
d, benchmark against simpler burden summaries (derivation and independent sample)."""
import sys, os
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
import style4 as S

RES = _os.path.join(HB_ROOT, "results"); V = f"{RES}/validation"
OUT = _os.path.join(HB_ROOT, "figs/output/Figure4")
W = 180
STG = ["Stage 0", "Stage 1-2", "Stage >=3"]
STG_LAB = {"Stage 0": "0", "Stage 1-2": "1\u20132", "Stage >=3": "\u22653"}
STG_COL = {"Stage 0": "#A9B2BC", "Stage 1-2": "#5FAE90", "Stage >=3": "#0F4A38"}
MODELS = ["all10", "mri_plus_m13", "mri_only_matched_all10"]
MCOL = {"all10": "#0F4A38", "mri_plus_m13": "#3E8E73", "mri_only_matched_all10": "#8FB9A8"}
MMK = {"all10": "o", "mri_plus_m13": "s", "mri_only_matched_all10": "D"}
OUTLAB = {"hbve": "Heart\u2013brain vascular events", "mace": "Stroke, MI, or heart failure", "cardiac": "Cardiac composite",
          "cerebrovascular": "Cerebrovascular composite", "stroke": "Stroke", "af": "Atrial fibrillation", "hf": "Heart failure",
          "ihd": "Ischaemic heart disease"}
INDEP = "#3D5A80"


def strata_block(F, cv, x, y, w, h, o, ymax, title, src=None, tcol=None):
    st = pd.read_csv(src or f"{RES}/out_score2_strata.csv"); st = st[st.outcome == o]
    ax = F.add(x + 9, y + 8, w - 10, h - 18)
    groups = ["Low-to-moderate", "High or very high"]
    ref = st[(st.score2_group == "High or very high") & (st.stage_group == "Stage 0")].iloc[0]
    ax.axhline(ref.rate, color="#8E949C", lw=0.5, ls=(0, (2.2, 1.6)), zorder=1)
    xs = []
    for gi, g in enumerate(groups):
        for si, sg in enumerate(STG):
            r = st[(st.score2_group == g) & (st.stage_group == sg)].iloc[0]
            xx = gi * 3.8 + si
            xs.append(xx)
            ax.plot([xx, xx], [r.rate_lo, min(r.rate_hi, ymax * 0.995)], color=STG_COL[sg], lw=0.9, zorder=2, solid_capstyle="butt")
            ax.plot(xx, r.rate, "o", ms=4.0, mfc=STG_COL[sg], mec="white", mew=0.5, zorder=3)
            ax.text(xx, -ymax * 0.10, f"{int(r.events)}\n{S.md_int(int(r.n))}", fontsize=4.2, ha="center", va="top", color=S.GREY,
                    linespacing=1.05)
        if gi == 0:
            ax.add_patch(Rectangle((-0.6, 0), 3.2, ymax, fc=S.BAND, ec="none", zorder=0))
    ax.set_xlim(-0.6, 3.8 + 2.6); ax.set_ylim(0, ymax)
    ax.set_xticks(xs); ax.set_xticklabels([STG_LAB[s] for s in STG] * 2, fontsize=5.0)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, ymax, 5)))
    ax.yaxis.set_major_formatter(S.md_formatter(0))
    ax.set_ylabel("Events per 1000 person-years", fontsize=5.2, labelpad=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.text(1, ymax * 0.975, "Low\u2013moderate", fontsize=4.7, ha="center", va="top", color=S.INK)
    ax.text(4.8, ymax * 0.975, "High or very high", fontsize=4.7, ha="center", va="top", color=S.INK)
    ax.text(2.9, ymax * 1.06, "SCORE2 risk group", fontsize=4.5, ha="center", va="bottom", color=S.GREY)
    ax.text(6.4, ref.rate + ymax * 0.015, "high risk, stage 0", fontsize=4.4, ha="right", va="bottom", color=S.GREY)
    cv.text(x + 9 + (w - 10) / 2, y + h - 0.6, "Heart\u2013brain stage (events, n)", fontsize=5.0, ha="center", va="center", color=S.INK)
    cv.text(x + 9, y + 3.2, title, fontsize=5.6, fontweight="bold", va="center", color=tcol or S.INK)
    return ax


def panel_a(F, cv, x0, y0, w=104, h=60):
    S.panel_letter(cv, x0 + 1, y0 + 1, "a", "Stage re-stratifies heart\u2013brain vascular risk within SCORE2 groups")
    strata_block(F, cv, x0, y0 + 6, w / 2 - 1, h - 6, "hbve", 40, "Derivation (All-10)")
    strata_block(F, cv, x0 + w / 2 + 1, y0 + 6, w / 2 - 1, h - 6, "hbve", 40, "Independent (MRI only)",
                 src=f"{V}/independent_score2_strata_imaging.csv", tcol=INDEP)


def panel_b(F, cv, x0, y0, w=74, h=60):
    inc = pd.read_csv(f"{RES}/out_score2_increment.csv")
    ind2 = pd.read_csv(f"{V}/independent_score2_increment_imaging.csv")
    iinc = pd.read_csv(f"{V}/independent_score2_increment_imaging.csv").set_index("outcome")
    S.panel_letter(cv, x0 + 1, y0 + 1, "b", "Discrimination added to SCORE2")
    outs = ["hbve", "mace", "cardiac", "hf", "af", "cerebrovascular", "stroke"]
    LBL = 26.0; AX = 28.0; a0 = x0 + LBL
    lo_, hi_ = -0.02, 0.20
    def X(v): return a0 + (np.clip(v, lo_, hi_) - lo_) / (hi_ - lo_) * AX
    top = y0 + 13.0; rh = 5.3
    cv.text(x0 + w - 0.5, top - 5.2, "All-10 \u0394C", fontsize=4.9, ha="right", va="center", color=S.INK, fontweight="bold")
    bottom = top + rh * len(outs) - rh / 2
    for v in [0, 0.05, 0.10, 0.15, 0.20]:
        cv.plot([X(v), X(v)], [top - rh / 2 - 0.5, bottom], color="#8E949C" if v == 0 else S.RULE, lw=0.5 if v == 0 else 0.35,
                ls="-" if v == 0 else (0, (2, 1.4)), zorder=0)
        cv.text(X(v), bottom + 0.9, S.md(v, 2) if v else "0", fontsize=4.7, ha="center", va="top", color=S.GREY)
    cv.text(a0 + AX / 2, bottom + 4.0, "\u0394C-index vs SCORE2", fontsize=5.2, ha="center", va="top")
    for i, o in enumerate(outs):
        y = top + i * rh
        if i % 2 == 0:
            cv.add_patch(Rectangle((x0 + 1, y - rh / 2), w - 1.5, rh, fc=S.BAND, ec="none", zorder=-1))
        r0 = inc[(inc.outcome == o) & (inc.model == "all10")].iloc[0]
        cv.text(x0 + 2.5, y - 0.9, OUTLAB[o], fontsize=4.9, va="center", color=S.INK)
        ri = ind2[ind2.outcome == o]
        evtxt = f"{int(r0.events)} | {int(ri.events.iloc[0])} events" if len(ri) else f"{int(r0.events)} events"
        cv.text(x0 + 2.5, y + 1.3, evtxt, fontsize=4.3, va="center", color=S.GREY)
        for k, m in enumerate(MODELS):
            r = inc[(inc.outcome == o) & (inc.model == m)].iloc[0]
            yo = y + (k - 1.5) * 1.05
            cv.plot([X(r.dC_lo), X(r.dC_hi)], [yo, yo], color=MCOL[m], lw=0.75, zorder=2)
            if r.dC_hi > hi_:
                cv.plot(X(hi_), yo, ">", ms=1.9, color=MCOL[m], zorder=2)
            cv.plot(X(r.dC), yo, MMK[m], ms=2.5, mfc=MCOL[m], mec="white", mew=0.35, zorder=3)
            cv.plot([X(r.dC_corrected)] * 2, [yo - 0.55, yo + 0.55], color="#C0392B" if m == "all10" else "none", lw=0.6, zorder=4)
        if o in iinc.index:
            q = iinc.loc[o]; yo = y + 1.5 * 1.05
            cv.plot([X(q.dC_lo), X(q.dC_hi)], [yo, yo], color=INDEP, lw=0.75, zorder=2)
            cv.plot(X(q.dC), yo, "^", ms=2.7, mfc="white", mec=INDEP, mew=0.6, zorder=3)
        cv.text(x0 + w - 0.5, y, f"{S.md(r0.dC, 3)}", fontsize=4.8, ha="right", va="center", color=S.INK)
        cv.text(x0 + w - 0.5, y + 1.9, f"({S.md(r0.dC_lo, 3)} to {S.md(r0.dC_hi, 3)})", fontsize=3.9, ha="right", va="center",
                color=S.GREY)
    ly = bottom + 8.2; xx = x0 + 2.5
    for m, lab in [("all10", "All-10"), ("mri_plus_m13", "MRI + M13"), ("mri_only_matched_all10", "MRI only")]:
        cv.plot(xx, ly, MMK[m], ms=2.5, mfc=MCOL[m], mec="white", mew=0.35)
        cv.text(xx + 1.5, ly, lab, fontsize=4.6, va="center", color=S.INK)
        xx += S.text_w_mm(lab, 4.6) + 4.0
    cv.plot(xx, ly, "^", ms=2.7, mfc="white", mec=INDEP, mew=0.6)
    cv.text(xx + 1.5, ly, "Independent", fontsize=4.6, va="center", color=INDEP)
    xx += S.text_w_mm("Independent", 4.6) + 4.0
    cv.plot([x0 + 2.5, x0 + 2.5], [ly + 3.4 - 0.6, ly + 3.4 + 0.6], color="#C0392B", lw=0.6)
    cv.text(x0 + 3.5, ly + 3.4, "All-10 optimism-corrected \u0394C", fontsize=4.6, va="center", color=S.INK)


def panel_c(F, cv, x0, y0, w=86, h=58):
    d = pd.read_csv(f"{RES}/out_dca.csv")
    S.panel_letter(cv, x0 + 1, y0 + 1, "c", "Net-benefit gain at 3 years")
    ax = F.add(x0 + 12, y0 + 9, w - 33, h - 21)
    col = {"hbve": "#0F4A38", "mace": "#C0392B", "cardiac": "#D9692F", "cerebrovascular": "#6E4FB0"}
    lab = {"hbve": "Heart\u2013brain vascular", "mace": "Stroke, MI, or HF", "cardiac": "Cardiac", "cerebrovascular": "Cerebrovascular"}
    h0 = d[d.outcome == "hbve"].sort_values("threshold")
    dnb = 1000 * (h0.nb_score2_stage - h0.nb_score2)
    pos = h0.threshold[dnb > 0]
    ax.axhline(0, color="#8E949C", lw=0.5, zorder=1)
    for o in ["cardiac", "cerebrovascular", "mace", "hbve"]:
        g = d[d.outcome == o].sort_values("threshold")
        y = 1000 * (g.nb_score2_stage - g.nb_score2)
        ys = pd.Series(y.to_numpy()).rolling(3, center=True, min_periods=1).mean()
        ax.plot(100 * g.threshold, ys, color=col[o], lw=1.25 if o == "hbve" else 0.85, zorder=3 if o == "hbve" else 2,
                alpha=1 if o == "hbve" else 0.85)
    gain = h0.threshold[(h0.nb_score2_stage - h0.nb_score2) >= 0.001]
    g_lo, g_hi = 100 * gain.min(), 100 * gain.max()
    ax.axvspan(g_lo, g_hi, color="#0F4A38", alpha=0.07, lw=0, zorder=0)
    ax.set_xlim(0.5, 10); ax.set_ylim(-3, 6)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 6, 8, 10]))
    ax.yaxis.set_major_locator(FixedLocator([-2, 0, 2, 4, 6]))
    ax.xaxis.set_major_formatter(S.md_formatter(0)); ax.yaxis.set_major_formatter(S.md_formatter(0))
    ax.set_xlabel("3-year risk threshold (%)", fontsize=5.3, labelpad=1.5)
    ax.set_ylabel("\u0394 net benefit (per 1000 people)", fontsize=5.2, labelpad=2)
    ax.grid(axis="y", color=S.RULE, lw=0.35); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.text((g_lo + g_hi) / 2, 5.6, f"gain at\n{S.md(g_lo, 0)}\u2013{S.md(g_hi, 1) if g_hi % 1 else S.md(g_hi, 0)}%", fontsize=4.5,
            ha="center", va="top", color="#0F4A38", linespacing=1.0)
    lx = x0 + w - 19.5
    for i, o in enumerate(["hbve", "mace", "cardiac", "cerebrovascular"]):
        yy = y0 + 13 + i * 3.4
        cv.plot([lx, lx + 3], [yy, yy], color=col[o], lw=1.2 if o == "hbve" else 0.85)
        cv.text(lx + 3.8, yy, lab[o], fontsize=4.7, va="center", color=S.INK)
    cv.text(lx, y0 + 29.5, "SCORE2 + stage\nminus SCORE2;\n3-point moving\naverage", fontsize=4.4, va="top", color=S.GREY,
            linespacing=1.1)


def panel_d(F, cv, x0, y0, w=90, h=58):
    cp = pd.read_csv(f"{RES}/out_compare_predictors.csv")
    iv = pd.read_csv(f"{V}/independent_discrimination_vs_counts_imaging.csv")
    S.panel_letter(cv, x0 + 1, y0 + 1, "d", "Benchmark against simpler burden summaries")
    PRED = [("stage", "Heart\u2013brain stage", "#0F4A38", "o"), ("count", "Number of abnormal markers", "#9AA0A8", "s"),
            ("meanz", "Mean abnormality z", "#C9A227", "D")]
    rows = [("Derivation (All-10; vs SCORE2)", [("hbve", None), ("mace", None), ("cardiac", None), ("cerebrovascular", None)]),
            ("Independent (MRI only; vs clinical)", [("hbve", None), ("cardiac", None), ("stroke", None), ("af", None), ("hf", None)])]
    LBL = 30.0; AX = 34.0; a0 = x0 + LBL
    lo_, hi_ = -0.02, 0.08
    def X(v): return a0 + (np.clip(v, lo_, hi_) - lo_) / (hi_ - lo_) * AX
    top = y0 + 11.5; rh = 3.55
    yy = top; ys = []
    for gname, outs in rows:
        cv.text(x0 + 2, yy, gname, fontsize=5.0, fontweight="bold", va="center", color=S.INK if "Deriv" in gname else INDEP)
        yy += 3.2
        for o, _ in outs:
            ys.append((gname, o, yy)); yy += rh
        yy += 0.8
    bottom = yy - 0.8 - rh / 2
    for v in [-0.02, 0, 0.02, 0.04, 0.06, 0.08]:
        cv.plot([X(v), X(v)], [top + 1.4, bottom], color="#8E949C" if v == 0 else S.RULE, lw=0.5 if v == 0 else 0.35,
                ls="-" if v == 0 else (0, (2, 1.4)), zorder=0)
        cv.text(X(v), bottom + 0.9, S.md(v, 2) if v else "0", fontsize=4.6, ha="center", va="top", color=S.GREY)
    cv.text(a0 + AX / 2, bottom + 3.9, "\u0394C-index over reference model", fontsize=5.1, ha="center", va="top")
    for gname, o, y in ys:
        cv.text(x0 + 4, y, OUTLAB[o], fontsize=4.8, va="center", color=S.INK)
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
        cv.plot([min(xs), max(xs)], [y, y], color="#C9CED6", lw=0.6, zorder=1)
        for k, lab, c, mk in PRED:
            cv.plot(X(vals[k]), y, mk, ms=2.7 if k == "stage" else 2.4, mfc=c, mec="white", mew=0.35, zorder=3 if k == "stage" else 2)
        if pbc is not None:
            cv.text(x0 + w - 0.5, y, S.pval(pbc), fontsize=4.5, ha="right", va="center", color=S.GREY)
    cv.text(x0 + w - 0.5, top, "stage vs\ncount, P", fontsize=4.4, ha="right", va="center", color=S.GREY, linespacing=1.0)
    ly = bottom + 8.4; xx = x0 + 2
    for k, lab, c, mk in PRED:
        cv.plot(xx, ly, mk, ms=2.6, mfc=c, mec="white", mew=0.35)
        cv.text(xx + 1.5, ly, lab, fontsize=4.6, va="center", color=S.INK)
        xx += S.text_w_mm(lab, 4.6) + 4.2


def render(which="all"):
    os.makedirs(OUT, exist_ok=True)
    S.set_style()
    if which in ("a", "all"):
        F = S.MMFig(104, 62); cv = F.canvas(zorder=5); panel_a(F, cv, 0, 0); F.save(f"{OUT}/Fig4a")
    if which in ("b", "all"):
        F = S.MMFig(74, 62); cv = F.canvas(zorder=5); panel_b(F, cv, 0, 0); F.save(f"{OUT}/Fig4b")
    if which in ("c", "all"):
        F = S.MMFig(86, 60); cv = F.canvas(zorder=5); panel_c(F, cv, 0, 0); F.save(f"{OUT}/Fig4c")
    if which in ("d", "all"):
        F = S.MMFig(90, 62); cv = F.canvas(zorder=5); panel_d(F, cv, 0, 0); F.save(f"{OUT}/Fig4d")
    if which in ("composite", "all"):
        F = S.MMFig(W, 128); cv = F.canvas(zorder=5)
        panel_a(F, cv, 0, 0); panel_b(F, cv, 106, 0); panel_c(F, cv, 0, 66); panel_d(F, cv, 90, 66)
        F.save(f"{OUT}/Figure4_composite")


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "all")
