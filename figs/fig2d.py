import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Figure 2d - heart-brain co-abnormality in the independent imaging sample (n = 23,974)."""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
import style4 as S

V = _os.path.join(HB_ROOT, "results/validation")
HEART = [("LV_remodeling_Z", "LV remodelling"), ("LV_myocardial_dysfunction_Z", "LV dysfunction"), ("Aortic_stiffness_Z", "Aortic stiffness")]
BRAIN = [("WMH_burden_Z", "WMH"), ("FreeWater_ISOVF_Z", "Free\nwater"), ("Hippocampal_atrophy_Z", "Hippo-\ncampus"),
         ("Cortical_thinning_Z", "Cortical\nthickness")]


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * float(np.clip(t, 0, 1)))


def orcol(v):
    """Odds-ratio colour on the shared v5 diverging map (log2 scale, 0.5-2 mapped to the full range)."""
    import anat3d
    t = float(np.clip(np.log(v) / np.log(2.0), -1, 1))
    return tuple(anat3d.ZMAP(0.5 + 0.5 * t)[:3])


def panel_d(cv, x0, y0, w=106, letter="d", title="Cardiac abnormality travels with white-matter injury"):
    d = pd.read_csv(f"{V}/heart_brain_coabnormality.csv"); d = d[d["sample"] == "independent"]
    S.panel_letter(cv, x0 + 1, y0 + 1, letter, title)
    cw, ch = 6.5, 4.6
    LBL = 17.5
    for b, z in enumerate([1, 2]):
        mx = x0 + LBL + b * (4 * cw + 3.6); my = y0 + 18.5
        cv.text(mx + 2 * cw, y0 + 7.4, f"Both abnormal at z \u2265 {z}", fontsize=5.2, fontweight="bold", ha="center", color=S.INK)
        # column header
        for j, (bk, bl) in enumerate(BRAIN):
            cv.text(mx + (j + 0.5) * cw, my - 1.0, bl, fontsize=4.2, ha="center", va="bottom", color=S.ORG["Brain"],
                    linespacing=0.95)
        cv.plot([mx + 0.3, mx + 2 * cw - 0.3], [my - 6.3, my - 6.3], color=S.ORG["Brain"], lw=0.5)
        cv.plot([mx + 2 * cw + 0.3, mx + 4 * cw - 0.3], [my - 6.3, my - 6.3], color="#A995D3", lw=0.5)
        cv.text(mx + cw, my - 6.9, "white matter", fontsize=4.3, ha="center", va="bottom", color=S.GREY)
        cv.text(mx + 3 * cw, my - 6.9, "grey matter", fontsize=4.3, ha="center", va="bottom", color=S.GREY)
        for i, (hk, hl) in enumerate(HEART):
            if b == 0:
                cv.text(mx - 1.0, my + (i + 0.5) * ch, hl, fontsize=4.7, ha="right", va="center", color=S.ORG["Heart"])
            for j, (bk, bl) in enumerate(BRAIN):
                r = d[(d.heart == hk) & (d.brain == bk) & (d.z == z)].iloc[0]
                fc = "#E6E8EB" if r.n11 < 5 else orcol(r.OR)
                cv.add_patch(Rectangle((mx + j * cw, my + i * ch), cw - 0.35, ch - 0.35, fc=fc, ec="none"))
                sig = (r.lo > 1) or (r.hi < 1)
                tc = "white" if (np.log(r.OR) / np.log(2) > 0.62 and r.n11 >= 5) else S.INK
                lab = S.md(r.OR, 2) + ("*" if sig else "")
                if r.n11 < 5:
                    lab = "<5"
                cv.text(mx + j * cw + (cw - 0.35) / 2, my + i * ch + (ch - 0.35) / 2, lab, fontsize=4.2, ha="center",
                        va="center", color=tc, fontweight="bold" if sig else "normal")
    # dose-response bars (any heart abnormal count -> any brain abnormal)
    bx = x0 + LBL + 2 * (4 * cw + 3.6) + 1.5; by = y0 + 13.0; bw_ = x0 + w - bx - 0.5
    g = d[d.heart.str.startswith("n_heart_abnormal=")].copy()
    g["k"] = g.heart.str.split("=").str[1].astype(int)
    g = g.sort_values("k")
    cv.text(bx, y0 + 7.4, "Any brain abnormal", fontsize=5.0, fontweight="bold", color=S.INK)
    hh = 15.5
    for i, r in enumerate(g.itertuples()):
        yy = by + 2.5 + i * 3.9
        val = r.p_brain_given_heart
        BL = bw_ - 15.5
        cv.add_patch(Rectangle((bx + 3.6, yy - 1.2), BL * val / 0.6, 2.4, fc=mix("#ECE7F6", S.ORG["Brain"], 0.25 + 0.25 * i),
                               ec="none"))
        cv.text(bx + 2.6, yy, str(r.k), fontsize=4.8, ha="right", va="center", color=S.ORG["Heart"], fontweight="bold")
        cv.text(bx + 3.6 + BL * val / 0.6 + 0.5, yy, f"{S.md(100 * val, 0)}%", fontsize=4.4, va="center", color=S.INK)
        cv.text(bx + bw_, yy, S.md_int(int(r.n)), fontsize=4.2, ha="right", va="center", color=S.GREY)
    cv.text(bx + 0.2, by - 0.6, "cardiac\nmarkers", fontsize=4.2, color=S.ORG["Heart"], va="bottom", linespacing=0.95)
    cv.text(bx + bw_, by - 0.6, "n", fontsize=4.2, ha="right", va="bottom", color=S.GREY)
    # colour key + footnote
    ky = y0 + 18.5 + 3 * ch + 3.0; kx = x0 + LBL
    for k in range(40):
        v = np.exp(np.log(0.5) + (np.log(2) - np.log(0.5)) * k / 39)
        cv.add_patch(Rectangle((kx + k * 0.55, ky), 0.56, 1.6, fc=orcol(v), ec="none"))
    for v, lab in [(0.5, "0\u00b75"), (1, "1"), (2, "2")]:
        xx = kx + (np.log(v) - np.log(0.5)) / (np.log(2) - np.log(0.5)) * 22
        cv.text(xx, ky + 2.4, lab, fontsize=4.3, ha="center", va="top", color=S.GREY)
    cv.text(kx + 24, ky + 0.8, "odds ratio of co-abnormality; * 95% CI excludes 1; grey, <5 co-abnormal people",
            fontsize=4.4, va="center", color=S.GREY)
    cv.text(kx, ky + 6.0, "Independent imaging sample (n = 23\u2009974); z scores from age- and sex-specific normative models.",
            fontsize=4.3, va="center", color=S.GREY)
