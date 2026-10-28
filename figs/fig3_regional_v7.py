"""Figure 3 (v7) - Where the heart-brain stage is expressed: subtype-specific and shared regional anatomy.
A  stage-matched regional difference between the two subtypes (Subtype 1 minus Subtype 2), derivation sample
   (All-10) and independent sample (transferred MRI-only model), same anatomical layout as figure 2B
B  brain regions coupled with the cardiac composite (unchanged, figure 3 v6 B)
C  cardiac correlates of white-matter burden (unchanged, figure 3 v6 C)
D  white-matter predominance of coupling, extended with out-of-model measures (FA, MD, subcortical nuclei)
E  cardiac abnormality co-occurs with white-matter injury (unchanged, figure 3 v6 E)
Usage: python3 fig3_regional_v7.py [a|b|c|d|e|all|composite]
"""
import sys
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import style4 as S
import base_style as BS
import anat3d as A
import regional_data as R
import regional_subtype_data as RS
import fig3_regional_v6 as F3
import fig2_sequence_v7 as F2

OUT = Path(__file__).resolve().parent / "output_v7" / "Figure3"
OUT.mkdir(parents=True, exist_ok=True)
W = 180
H_A, H_BC, H_DE = 57, 62, 56
Y_BC = H_A + 2
Y_DE = Y_BC + H_BC + 2
H = Y_DE + H_DE
DAG = F2.DAG
CMAP_D = LinearSegmentedColormap.from_list("s2_s1", [F2.SUBC[2], "#E9D3A8", "#F7F6F4", "#A9D2CC", F2.SUBC[1]], 512)
NORM_D = Normalize(-0.8, 0.8)


def fmt(v, nd=2):
    return S.md(v, nd)


# ----------------------------------------------------------------------------------------- A
def panel_a(F, cv, x0, y0):
    S.panel_letter(cv, x0 + 1, y0 + 1, "A",
                   "Stage-matched regional difference between subtypes (Subtype 1 minus Subtype 2)")
    F2.header(cv, x0, y0, y0 + 9.3)
    A.colorbar_mm(F, x0 + 2.0, y0 + 12.6, 25.0, 1.4, CMAP_D, NORM_D, [-0.8, -0.4, 0, 0.4, 0.8], "", fontsize=4.0)
    cv.text(x0 + 2.0, y0 + 9.4, "\u03b2, S1 \u2212 S2 (z units)", fontsize=4.5, color=S.INK, va="bottom")
    cv.text(x0 + 2.0, y0 + 11.9, "S2 more abnormal \u2190  \u2192 S1", fontsize=4.0, color=S.GREY, va="bottom")
    rep = RS.replication().set_index("set")
    rows = [("derivation", "Derivation", "All-10 model"), ("independent", "Independent", "MRI-only model, transferred")]
    for i, (samp, lab, sub) in enumerate(rows):
        y = y0 + 24.0 + 19.0 * i
        m, _ = RS.contrast_maps(samp)
        if i:
            cv.plot([x0 + 1, x0 + W - 1], [y - 10.0, y - 10.0], color=S.RULE, lw=0.45)
        F2.draw_row(cv, x0, y, m, CMAP_D, NORM_D, masks=True)
        cv.text(x0 + 1.5, y - 4.4, lab, fontsize=5.8, fontweight="bold", color=S.INK, va="center")
        cv.text(x0 + 1.5, y - 1.3, sub, fontsize=4.3, color=S.GREY, va="center")
        cv.text(x0 + 1.5, y + 1.6, f"S1 n = {S.md_int(m['n_s1'])}", fontsize=4.3, color=F2.SUBC[1], va="center")
        cv.text(x0 + 1.5, y + 4.3, f"S2 n = {S.md_int(m['n_s2'])}", fontsize=4.3, color=F2.SUBC[2], va="center")
    ra, ro = rep.loc["all regions"], rep.loc["out-of-model"]
    cv.text(x0 + 1.5, y0 + 51.6,
            f"Linear models at stage \u22651: regional z ~ subtype + stage + age + sex; faded, q \u2265 0\u00b705 "
            f"(Benjamini\u2013Hochberg). Replication of the difference map: Spearman \u03c1 = {fmt(ra.spearman_rho)} "
            f"across {int(ra.n_regions)} regions ({fmt(ro.spearman_rho)} for the {int(ro.n_regions)} out-of-model "
            f"regions); direction concordant in {fmt(100 * ra.sign_concordance, 0)}%. {DAG} Out-of-model measure.",
            fontsize=4.1, color=S.GREY, va="top")


# ----------------------------------------------------------------------------------------- D (extended)
CLASSES = [("Cortical thickness", "Cortical thickness", "#B9A9DD", "old"),
           ("Hippocampal volume", "Hippocampal volume", "#9C82CC", "old"),
           ("Subcortical volume", f"Subcortical nuclei{DAG}", "#C7B9E4", "new"),
           ("White-matter free water", "WM free water", "#6E4FB0", "old"),
           ("White-matter FA", f"WM FA{DAG}", "#56399C", "new"),
           ("White-matter MD", f"WM MD{DAG}", "#47308A", "new"),
           ("White-matter hyperintensity", "WMH volume", "#3B2667", "wmh")]


def panel_d(F, cv, x0, y0, w=70):
    S.panel_letter(cv, x0 + 1, y0 + 1, "D", "White-matter predominance of heart\u2013brain coupling")
    old = R.coupling()
    old = old[(old["sample"] == "independent") & (old.exposure == "cardiac_comp") & (old.organ == "brain")]
    ext = RS.dictionary()  # noqa: F841 (kept for clarity: classes defined in dictionary_ext)
    import pandas as pd
    ce = pd.read_csv(RS.RES / "coupling_ext.csv")
    ce = ce[ce["sample"] == "independent"]
    ts_old, ts_new = R.tissue_summary(), RS.tissue_summary()
    cv.text(x0 + 5.0, y0 + 5.6, "Each point is one brain region (independent sample); \u03b2 per 1 SD cardiac "
            "composite", fontsize=4.4, color=S.GREY, va="top")
    ax = F.add(x0 + 24.0, y0 + 8.6, w - 26.0, 31.0)
    rng = np.random.default_rng(7)
    for i, (k, lab, col, src) in enumerate(CLASSES):
        g = ce[(ce.model == "A") & (ce.tissue == k)] if src != "wmh" else old[(old.model == "A") & (old.tissue == k)]
        jit = rng.uniform(-0.2, 0.2, len(g)) if len(g) > 2 else np.zeros(len(g))
        for (_, r), j in zip(g.iterrows(), jit):
            mk = "D" if r.method == "ProbtrackX" else "o"
            ax.plot(r.beta, i + j, mk, ms=1.5 if mk == "o" else 1.4, mfc=col, mec="white", mew=0.15,
                    alpha=0.85 if r.q < 0.05 else 0.35, zorder=2)
        for mdl, off, fill in (("A", 0, True), ("B", 0.36, False)):
            if src == "wmh":
                rr = old[(old.model == mdl) & (old.tissue == k)].iloc[0]
                med, lo, hi = rr.beta, rr.lo, rr.hi
            else:
                ts = ts_old if src == "old" else ts_new
                rr = ts[(ts["sample"] == "independent") & (ts.model == mdl) & (ts.contrast == k)].iloc[0]
                med, lo, hi = rr.median_beta, rr.lo, rr.hi
            c = S.INK if fill else S.GREY
            ax.plot([lo, hi], [i + off, i + off], color=c, lw=0.9 if fill else 0.6, zorder=4, solid_capstyle="butt")
            ax.plot(med, i + off, "|" if fill else "D", ms=5.0 if fill else 2.0, mew=1.1 if fill else 0.6, color=c,
                    mfc="white", zorder=5)
    ax.axvline(0, color=S.GREY, lw=0.4, zorder=0)
    ax.axhline(2.5, color=S.RULE, lw=0.4, ls=(0, (2, 1.5)), zorder=0)
    ax.set_ylim(len(CLASSES) - 0.4, -0.5)
    ax.set_yticks(range(len(CLASSES)))
    nreg = {k: (ce[(ce.model == "A") & (ce.tissue == k)].shape[0] if s != "wmh" else 1) for k, _, _, s in CLASSES}
    ax.set_yticklabels([f"{lab} ({nreg[k]})" for k, lab, _, _ in CLASSES], fontsize=4.4)
    for t, (_, _, col, _) in zip(ax.get_yticklabels(), CLASSES):
        t.set_color(S.INK)
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.05, 0.125)
    ax.set_xticks([0, 0.05, 0.10])
    ax.xaxis.set_major_formatter(S.md_formatter(2))
    ax.tick_params(axis="x", labelsize=4.3)
    ax.set_xlabel("\u03b2 (regional z per 1 SD cardiac composite)", fontsize=4.5, labelpad=0.6)
    ax.text(-0.048, 1.0, "grey\nmatter", fontsize=3.9, color=S.GREY, ha="left", va="center", linespacing=0.95)
    ax.text(-0.048, 4.5, "white\nmatter", fontsize=3.9, color=S.GREY, ha="left", va="center", linespacing=0.95)
    d_fw = ts_old[(ts_old["sample"] == "independent") & (ts_old.model == "A") & ts_old.contrast.str.contains("minus")].iloc[0]
    tn = ts_new[(ts_new["sample"] == "independent") & (ts_new.model == "A")].set_index("contrast")
    d_fa = tn.loc["White-matter FA minus cortical thickness"]
    d_sc = tn.loc["Subcortical volume minus cortical thickness"]
    cv.text(x0 + 1.0, y0 + 44.2,
            f"WM minus cortex (median \u03b2): free water {fmt(d_fw.median_beta, 3)} ({fmt(d_fw.lo, 3)}\u2013"
            f"{fmt(d_fw.hi, 3)}); FA{DAG} {fmt(d_fa.median_beta, 3)} ({fmt(d_fa.lo, 3)}\u2013{fmt(d_fa.hi, 3)}); "
            f"both p < 0\u00b7001.\nSubcortical nuclei{DAG} minus cortex {fmt(d_sc.median_beta, 3)} "
            f"({fmt(d_sc.lo, 3)} to {fmt(d_sc.hi, 3)}).", fontsize=4.0, color=S.INK, va="top", linespacing=1.15)
    h = [Line2D([], [], marker="o", ls="", ms=2.6, mfc="#6E4FB0", mec="white", label="TBSS tract or region"),
         Line2D([], [], marker="D", ls="", ms=2.3, mfc="#6E4FB0", mec="white", label="probabilistic tract"),
         Line2D([], [], marker="|", ls="-", ms=4, mew=1.1, color=S.INK, lw=0.8, label="median (95% CI), model A"),
         Line2D([], [], marker="D", ls="-", ms=2, mew=0.6, mfc="white", color=S.GREY, lw=0.6,
                label="model B (+ vascular risk factors)")]
    cv.legend(handles=h, loc="upper left", bbox_to_anchor=((x0 + 1.0) / cv.get_xlim()[1], 1 - (y0 + 49.6) / cv.get_ylim()[0]),
              fontsize=4.0, ncol=2, handletextpad=0.25, columnspacing=0.8, labelspacing=0.2, borderaxespad=0)


# ----------------------------------------------------------------------------------------- driver
SINGLE = {"a": (W, H_A, panel_a), "b": (108, H_BC, F3.panel_b), "c": (70, H_BC, F3.panel_c),
          "d": (72, H_DE, panel_d), "e": (108, H_DE, F3.panel_e)}


def render(which):
    S.set_style()
    keys = list(SINGLE) if which in ("all", "panels") else ([which] if which in SINGLE else [])
    for k in keys:
        w, h, fn = SINGLE[k]
        Fm = BS.MMFig(w, h); cv = Fm.canvas()
        fn(Fm, cv, 0, 0)
        Fm.save(str(OUT / f"Fig3{k}"))
        print("saved", k)
    if which in ("all", "composite"):
        Fm = BS.MMFig(W, H); cv = Fm.canvas()
        panel_a(Fm, cv, 0, 0)
        F3.panel_b(Fm, cv, 0, Y_BC)
        F3.panel_c(Fm, cv, 110, Y_BC)
        panel_d(Fm, cv, 0, Y_DE)
        F3.panel_e(Fm, cv, 72, Y_DE + 3)
        cv.plot([1, W - 1], [Y_BC - 1.0, Y_BC - 1.0], color=S.RULE, lw=0.4)
        cv.plot([1, W - 1], [Y_DE - 1.0, Y_DE - 1.0], color=S.RULE, lw=0.4)
        Fm.save(str(OUT / "Figure3_composite"))
        print("saved composite", W, H)


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "all")
