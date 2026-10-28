"""Figure 1 (v8) - study overview / graphical abstract, Nature-portfolio style, icon-led.

A  participants and time zero            B  multi-region heart and brain MRI (3D anatomy + pinned markers)
C  model-implied order (metro lines) and measured regional signature at stage >=3
D  risk after imaging (rates with 95% CI, HR, C index, SCORE2 strata)
E  interpretation and intended use (bottom-right box; unlettered in the composite)

Every number is read from source_data/ (facts.json, sustain_*.csv and the Source Data workbook).
3D anatomy: v7 PyVista renders (Rodero 2021 heart template, CC BY 4.0; fsaverage5, JHU tracts, ENIGMA meshes),
re-used unchanged (scripts/split_renders.py). Icons: Health Icons v2 (MIT).
Usage: python fig1v8.py [a|b|c|d|e|composite|all]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image
from scipy.stats import chi2

import icons as I
import style8 as S
from style8 import (ABBR, BAND, EVENT, GREY, INK, LIGHT, MID, OK, ORG, ORGAN_OF, RULE, STAGE4, SUBC, T, md,
                    md_int)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "source_data"
REN = ROOT / "assets" / "renders"
OUT = ROOT / "output" / "Figure1"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 180.0, 122.0
COLX = (0.0, 41.0, 113.0)          # left edges of the three grid columns
YB = 78.0                          # top of the bottom row
ICON = {"Heart": "body/heart_organ", "Brain": "body/neurology", "Protein": "body/blood_drop"}


# ======================================================================================== data layer
class Data:
    def __init__(self):
        self.f = json.load(open(SRC / "facts.json"))
        cache = SRC / "_sheets.pkl"
        sheets = ["Fig1D_rates_by_stage", "Fig1D_cox_primary", "Fig5A_score2_strata", "Fig5B_score2_increment",
                  "Fig2C_heart_first", "Fig2A_PVD_All10", "Fig2B_3A_dictionary_ext", "Fig2B_cell_meta",
                  "Fig2D_denovo_similarity"]
        if cache.exists():
            self.x = pickle.load(open(cache, "rb"))
        else:
            xl = pd.ExcelFile(SRC / "Source_Data_Figures_1-5_S1-S4.xlsx")
            self.x = {s: pd.read_excel(xl, s) for s in sheets}
            pickle.dump(self.x, open(cache, "wb"))
        x = self.x
        meta = x["Fig2B_cell_meta"]
        dm = meta[meta["sample"] == "derivation"]
        self.n_der = int(dm.n.sum())                                     # 3770
        self.n_ind = int(self.f["n_ind"].replace(" ", "").replace("\u2009", ""))
        cox = x["Fig1D_cox_primary"]
        d2 = cox[(cox["sample"] == "derivation") & (cox.adj == "M2") & (cox.exposure == "per10pp")].iloc[0]
        i1 = cox[(cox["sample"] == "independent") & (cox.adj == "M1")].iloc[0]
        i2 = cox[(cox["sample"] == "independent") & (cox.adj == "M2")].iloc[0]
        self.risk = {"der": (int(d2.n), int(d2.events)), "ind": (int(i1.n), int(i1.events))}
        self.hr = {"der": (d2.HR, d2.lo, d2.hi), "ind": (i2.HR, i2.lo, i2.hi)}
        self.hr_n = {"der": (int(d2.n), int(d2.events)), "ind": (int(i2.n), int(i2.events))}
        # rates by stage group (derivation CI from the sheet; independent exact Poisson CI, same method)
        r = x["Fig1D_rates_by_stage"]
        r = r[r.outcome == "hbve"]
        rd = r[r["sample"] == "derivation"].set_index("stage_grp")
        ri = r[r["sample"] == "independent"].set_index("grp")
        self.groups = ["0", "1", "2", ">=3"]
        self.rate = {"der": [], "ind": []}
        for g in self.groups:
            a = rd.loc[g]
            self.rate["der"].append((a.rate, a.rate_lo, a.rate_hi, int(a.events), int(a.n)))
            b = ri.loc[g]
            lo, hi = poisson_ci(int(b.events), b.py)
            self.rate["ind"].append((b.rate, lo, hi, int(b.events), int(b.n)))
        # check: our Poisson CI reproduces the derivation CI in the sheet
        for g in self.groups:
            a = rd.loc[g]
            lo, hi = poisson_ci(int(a.events), a.person_years)
            assert abs(lo - a.rate_lo) < 1e-3 and abs(hi - a.rate_hi) < 1e-3, g
        inc = x["Fig5B_score2_increment"]
        inc = inc[inc.outcome == "hbve"]
        a = inc[(inc["sample"] == "derivation") & (inc.model == "all10")].iloc[0]
        b = inc[inc["sample"] == "independent"].iloc[0]
        self.cidx = {"der": (a.C_score2, a.C_score2_stage, a.dC, a.dC_lo, a.dC_hi),
                     "ind": (b.C_score2, b.C_score2_stage, b.dC, b.dC_lo, b.dC_hi)}
        s2 = x["Fig5A_score2_strata"]
        s2 = s2[(s2.outcome == "hbve") & (s2["sample"] == "independent")]
        self.s2 = {}
        for grp in ("Low-to-moderate", "High or very high"):
            q = s2[s2.score2_group == grp].set_index("stage_group")
            self.s2[grp] = [(q.loc[s].rate, q.loc[s].rate_lo, q.loc[s].rate_hi)
                            for s in ("Stage 0", "Stage 1-2", "Stage >=3")]
        # sequences (All-10, first abnormality z = 1, maximum-likelihood order)
        p = x["Fig2A_PVD_All10"]
        p = p[(p.model == "all10") & (p.z == 1)]
        self.seq = {st: p[p.subtype == st].sort_values("ml_position").label.tolist() for st in (1, 2)}
        hf = x["Fig2C_heart_first"]
        self.p_hf = {int(rw.subtype): rw.p_heart_before_brain for _, rw in hf[hf.model == "all10"].iterrows()}
        allm = hf[hf.model.isin(["all10", "mri_plus_m13", "mri_only_matched_all10"])]
        self.p_hf_min = allm.p_heart_before_brain.min()
        self.n_models = allm.model.nunique()
        self.frac = {1: self.f["fracS1"], 2: self.f["fracS2"]}
        self.fracv = {int(rw.subtype): rw.fraction for _, rw in hf[hf.model == "all10"].iterrows()}
        # regional measure dictionary
        dic = x["Fig2B_3A_dictionary_ext"]
        self.nreg = {(o, bool(m)): int(((dic.organ == o) & (dic.in_model == m)).sum())
                     for o in ("heart", "brain") for m in (True, False)}
        self.n_fw = int((dic.tissue == "White-matter free water").sum())
        self.n_ct = int((dic.tissue == "Cortical thickness").sum())
        self.wmh = {}
        cells = None
        try:
            cells = pickle.load(open(SRC / "_cells.pkl", "rb"))
        except Exception:
            xl = pd.ExcelFile(SRC / "Source_Data_Figures_1-5_S1-S4.xlsx")
            cells = pd.read_excel(xl, "Fig2B_subtype_stage_cells")
            cells = cells[cells["var"] == "WMH_burden_Z_raw"]
            pickle.dump(cells, open(SRC / "_cells.pkl", "wb"))
        for st in (1, 2):
            self.wmh[st] = float(cells[(cells["sample"] == "derivation") &
                                       (cells.cell == f"Subtype {st}, stage >=3")].mean_z.iloc[0])

    def report(self):
        f = self.f
        print("n_der", self.n_der, "n_ind", self.n_ind, "at risk/events", self.risk, "HR n/events", self.hr_n)
        print("HR", {k: tuple(round(v, 3) for v in t) for k, t in self.hr.items()})
        print("rates der", [tuple(round(v, 1) for v in t[:3]) for t in self.rate["der"]])
        print("rates ind", [tuple(round(v, 1) for v in t[:3]) for t in self.rate["ind"]])
        print("C", {k: tuple(round(v, 3) for v in t) for k, t in self.cidx.items()})
        print("SCORE2", {k: [tuple(round(v, 1) for v in t) for t in q] for k, q in self.s2.items()})
        print("seq", {k: [ABBR[s] for s in v] for k, v in self.seq.items()}, "P", self.p_hf, "minP", self.p_hf_min)
        print("regional", self.nreg, "FW", self.n_fw, "CT", self.n_ct, "WMH", self.wmh)
        print("facts: fu", f["fu_med"], f["ind_fu_med"], "gap", f["gap_med"], "rho", f["rep_rho1"], f["rep_rho2"])


def poisson_ci(k, py, per=1000.0):
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return lo / py * per, hi / py * per


D = None
R = {}


def render(name):
    if name not in R:
        R[name] = np.asarray(Image.open(REN / f"{name}.png"))
    return R[name]


def F_(k):
    return str(D.f[k])


# ======================================================================================== A
def panel_a(F, cv, x0=0.0, y0=0.0):
    S.panel_letter(cv, x0 + 0.2, y0 + 0.5, "A", "Participants and time zero")
    rows = [("Derivation", D.n_der, True, [("Heart", 3), ("Brain", 4), ("Protein", 3)], "SuStaIn fitted"),
            ("Independent", D.n_ind, False, [("Heart", 3), ("Brain", 4)], "staged, not refitted")]
    xr = x0 + 38.8
    for i, (name, n, filled, mk, role) in enumerate(rows):
        y = y0 + 8.4 + i * 12.4
        bx = x0 + 3.5
        if filled:
            S.icon_badge(cv, "people/people", bx, y + 3.3, 6.4, INK, filled=True)
        else:
            S.icon_badge(cv, "people/people", bx, y + 3.3, 6.4, GREY, ring=GREY, ring_lw=0.7, bg="white")
        tx = x0 + 8.3
        T(cv, tx, y + 0.7, name, 6.0, INK, "bold")
        S.runs(cv, xr - S.text_w(md_int(n), 8.6, "bold") - S.text_w("n = ", 5.6), y + 0.75,
               [("n = ", 5.6, "normal", GREY), (md_int(n), 8.6, "bold", INK)])
        bw = xr - tx
        cv.add_patch(Rectangle((tx, y + 3.0), bw, 1.15, fc=S.light(INK, 0.9), ec="none", zorder=3))
        cv.add_patch(Rectangle((tx, y + 3.0), bw * n / D.n_ind, 1.15, fc=INK if filled else MID, ec="none",
                               zorder=4))
        xx = tx
        for org, k in mk:
            I.draw(cv, ICON[org], xx + 1.15, y + 6.75, 2.5, ORG[org], zorder=8)
            T(cv, xx + 2.65, y + 6.85, f"\u00d7{k}", 5.4, ORG[org], "bold")
            xx += 6.3
        T(cv, xr, y + 6.85, role, 5.0, GREY, ha="right", style="italic")
    # ------------------------------------------------------------ timeline
    ty = y0 + 40.6
    x1, x2, x3 = x0 + 3.6, x0 + 19.6, x0 + 36.2
    S.icon_badge(cv, "devices/medical_sample", x1, ty, 5.0, GREY, icon_frac=0.6)
    cv.plot([x1 + 2.8, x2 - 4.5], [ty, ty], color=MID, lw=0.75, ls=(0, (0.9, 1.1)), zorder=4)
    S.halo(cv, x2, ty, 8.8, INK, alpha=0.10)
    S.icon_badge(cv, "devices/tac", x2, ty, 6.6, INK, filled=True, icon_frac=0.64)
    S.arrow(cv, (x2 + 4.5, ty), (x3 - 0.25, ty), color=INK, lw=0.9, ms=4.2)
    S.flag(cv, x3 + 0.35, ty + 1.9, h=4.4, color=INK)
    # tier above: dates / visit
    T(cv, x0 + 0.3, ty - 4.9, "2006\u201310", 5.2, GREY, ha="left")
    T(cv, x2, ty - 5.9, "imaging visit", 5.6, INK, "bold", ha="center")
    T(cv, x3 + 2.4, ty - 4.9, "30 Sep 2021", 5.2, GREY, ha="right")
    # durations on the segments
    T(cv, (x1 + x2) / 2 + 0.6, ty - 1.5, f"{F_('gap_med')} y", 5.0, GREY, ha="center", style="italic")
    T(cv, (x2 + x3) / 2 + 2.0, ty - 1.5, f"{F_('fu_med')} | {F_('ind_fu_med')} y", 5.2, INK, "bold",
      ha="center")
    # tier below: roles
    T(cv, x1, ty + 3.4, "blood\nsample", 5.0, GREY, ha="center", va="top", linespacing=1.0)
    T(cv, x2, ty + 4.6, "time zero", 5.6, INK, "bold", ha="center", va="top")
    T(cv, x3 + 2.4, ty + 3.4, "end of\nfollow-up", 5.0, GREY, ha="right", va="top", linespacing=1.0)
    T(cv, x0 + 0.4, ty + 9.6, "medians; follow-up: derivation | independent", 5.0, MID, style="italic")
    # ------------------------------------------------------------ primary outcome
    yo = y0 + 55.8
    cv.plot([x0 + 0.4, xr], [yo - 2.6, yo - 2.6], color=RULE, lw=0.4, zorder=2)
    T(cv, x0 + 0.4, yo, "Primary outcome", 6.0, INK, "bold")
    outs = [("body/neurology", "stroke/TIA"), ("body/blood_vessel", "IHD"), ("body/heart_organ", "HF"),
            ("symbols/heartbeat", "AF")]
    for k, (ic, lab) in enumerate(outs):
        cx = x0 + 4.6 + k * 9.75
        S.icon_badge(cv, ic, cx, yo + 4.6, 4.8, EVENT, icon_frac=0.64)
        T(cv, cx, yo + 8.15, lab, 5.0, GREY, ha="center")
    cx_risk, cx_ev = x0 + 20.0, x0 + 33.0
    T(cv, cx_risk, yo + 11.1, "at risk", 5.0, GREY, ha="right")
    T(cv, cx_ev, yo + 11.1, "events", 5.0, GREY, ha="right")
    for i, key in enumerate(("der", "ind")):
        y = yo + 14.3 + i * 3.4
        if key == "der":
            cv.add_patch(Circle((x0 + 1.4, y), 0.95, fc=INK, ec="none", zorder=6))
        else:
            cv.add_patch(Circle((x0 + 1.4, y), 0.85, fc="white", ec=GREY, lw=0.7, zorder=6))
        T(cv, x0 + 3.2, y, "derivation" if key == "der" else "independent", 5.0, GREY)
        n, ev = D.risk[key]
        T(cv, cx_risk, y, md_int(n), 6.0, INK, "bold", ha="right")
        T(cv, cx_ev, y, str(ev), 6.0, INK, "bold", ha="right")


# ======================================================================================== B
HEART_SITES = {"AoS": (0.235, 0.300), "LVR": (0.885, 0.585), "LVD": (0.700, 0.860)}
BRAIN_SITES = {"CT": (0.815, 0.115), "FW": (0.865, 0.470), "WMH": (0.610, 0.600), "Hip": (0.480, 0.690)}


def panel_b(F, cv, x0=COLX[1], y0=0.0):
    S.panel_letter(cv, x0 + 0.2, y0 + 0.5, "B", "Multi-region heart and brain MRI")
    heart, brain = render("heart_lao"), render("glass_brain")
    hx, hy, hw, hh = S.place(cv, heart, x0 + 15.2, y0 + 11.6, h=39.0, anchor="tc", z=3)
    bx, by, bw, bh = S.place(cv, brain, x0 + 50.6, y0 + 13.4, w=38.6, anchor="tc", z=3)
    # organ headers
    for cx, org, lab in ((x0 + 15.2, "Heart", "Cardiac MRI"), (x0 + 50.6, "Brain", "Brain MRI")):
        wl = S.text_w(lab, 6.2, "bold")
        sx = cx - (wl + 4.2) / 2
        S.icon_badge(cv, ICON[org], sx + 1.6, y0 + 8.0, 3.4, ORG[org], filled=True, icon_frac=0.66)
        T(cv, sx + 4.0, y0 + 8.05, lab, 6.2, ORG[org], "bold")
    S.gradient_arc(cv, (x0 + 25.4, y0 + 15.4), (x0 + 34.6, y0 + 16.6), (0, -5.2), ORG["Heart"], ORG["Brain"],
                   lw=1.25)
    T(cv, x0 + 30.2, y0 + 19.6, "heart\u2013brain\naxis", 5.0, GREY, ha="center", va="top", style="italic",
      linespacing=1.0)

    def hp(k):
        fx, fy = HEART_SITES[k]
        return hx + fx * hw, hy + fy * hh

    def bp(k):
        fx, fy = BRAIN_SITES[k]
        return bx + fx * bw, by + fy * bh

    d = 4.6
    pins = [("AoS", "Heart", hp("AoS"), (x0 + 2.4, y0 + 22.0)),
            ("LVR", "Heart", hp("LVR"), (x0 + 28.6, y0 + 33.0)),
            ("LVD", "Heart", hp("LVD"), (x0 + 28.6, y0 + 44.4)),
            ("CT", "Brain", bp("CT"), (x0 + 67.9, y0 + 12.4)),
            ("FW", "Brain", bp("FW"), (x0 + 67.9, y0 + 30.4)),
            ("WMH", "Brain", bp("WMH"), (x0 + 62.6, y0 + 48.4)),
            ("Hip", "Brain", bp("Hip"), (x0 + 44.6, y0 + 49.0))]
    for ab, org, site, at in pins:
        S.leader(cv, site, at, color=S.dark(ORG[org], 0.15), lw=0.4)
        S.abbr_badge(cv, at[0], at[1], d, ORG[org], ab, fs=5.0, z=9)
    # legend (doubles as the key to the station labels in C)
    heart_m = [("LVR", "LV remodelling", "volumes, mass"), ("LVD", "LV dysfunction", "segmental strain"),
               ("AoS", "Aortic stiffness", "distensibility")]
    brain_m = [("CT", "Cortical thinning", f"{D.n_ct} regions"), ("FW", "WM free water", f"{D.n_fw} tracts"),
               ("Hip", "Hippocampal atrophy", "volume"), ("WMH", "WMH burden", "volume")]
    ly = y0 + 53.6
    for col, items, org, noff in ((x0 + 0.6, heart_m, "Heart", 6.1), (x0 + 36.6, brain_m, "Brain", 6.7)):
        for k, (ab, name, sub) in enumerate(items):
            y = ly + k * 3.6
            cv.add_patch(Circle((col + 0.55, y), 0.55, fc=ORG[org], ec="none", zorder=8))
            T(cv, col + 1.6, y, ab, 5.2, ORG[org], "bold")
            e = S.runs(cv, col + noff, y, [(name, 5.4, "bold", INK)])
            T(cv, e + 0.8, y, sub, 5.0, GREY)
    # bottom strip: plasma proteins | regional measures
    yb = y0 + 70.4
    cv.plot([x0 + 0.4, x0 + 69.6], [yb - 3.6, yb - 3.6], color=RULE, lw=0.4, zorder=2)
    S.icon_badge(cv, "devices/medical_sample", x0 + 2.5, yb + 0.6, 4.8, ORG["Protein"], icon_frac=0.62)
    T(cv, x0 + 5.8, yb - 1.6, "Plasma protein modules", 5.6, ORG["Protein"], "bold")
    parts = []
    for k, (m, npr) in enumerate((("M13", 71), ("M20", 34), ("M5", 148))):
        parts += [(m, 5.2, "bold", ORG["Protein"]), (f" {npr}" + ("   " if k < 2 else ""), 5.0, "normal", GREY)]
    parts += [(" proteins", 5.0, "normal", GREY)]
    S.runs(cv, x0 + 5.8, yb + 1.95, parts)
    # 313 regional measures, proportional bar
    hx0, hw0 = x0 + 35.6, 34.0
    tot = sum(D.nreg.values())
    T(cv, hx0, yb - 1.6, f"{tot} regional measures", 5.6, INK, "bold")
    segs = [(D.nreg[("heart", True)], ORG["Heart"]), (D.nreg[("heart", False)], S.light(ORG["Heart"], 0.6)),
            (D.nreg[("brain", True)], ORG["Brain"]), (D.nreg[("brain", False)], S.light(ORG["Brain"], 0.6))]
    xx = hx0
    for n, c in segs:
        w = hw0 * n / tot
        cv.add_patch(Rectangle((xx, yb + 0.2), w, 1.7, fc=c, ec="white", lw=0.25, zorder=5))
        xx += w
    nh = D.nreg[("heart", True)] + D.nreg[("heart", False)]
    nb = D.nreg[("brain", True)] + D.nreg[("brain", False)]
    T(cv, hx0, yb + 3.45, f"{nh}", 5.2, ORG["Heart"], "bold")
    T(cv, hx0 + hw0 * nh / tot + 1.2, yb + 3.45, f"{nb}", 5.2, ORG["Brain"], "bold")
    nin = D.nreg[("heart", True)] + D.nreg[("brain", True)]
    nout = tot - nin
    # shading key: dark = model input, light = display only
    t1, t2 = f"in model", f"display only"
    lx2 = hx0 + hw0 - S.text_w(t2, 5.0) - 1.7
    lx = lx2 - 1.1 - S.text_w(t1, 5.0) - 1.7
    for xx_, t, fc in ((lx, t1, GREY), (lx2, t2, S.light(GREY, 0.6))):
        cv.add_patch(Rectangle((xx_, yb + 2.85), 1.3, 1.2, fc=fc, ec="none", zorder=5))
        T(cv, xx_ + 1.7, yb + 3.45, t, 5.0, GREY)


# ======================================================================================== C
def panel_c(F, cv, x0=COLX[2], y0=0.0):
    S.panel_letter(cv, x0 + 0.2, y0 + 0.5, "C", "Model-implied order of abnormality")
    names = {1: "LV dysfunction-led", 2: "LV remodelling-led"}
    sp = 6.5
    xs0 = x0 + 3.4
    for i, st in enumerate((1, 2)):
        yb = y0 + 8.3 + i * 14.5
        c = SUBC[st]
        S.pill(cv, x0 + 6.1, yb, 11.6, 3.3, fc=c, z=6)
        T(cv, x0 + 6.1, yb + 0.05, f"Subtype {st}", 5.6, "white", "bold", ha="center")
        S.donut(cv, x0 + 14.6, yb, 1.6, D.fracv[st], c, w=0.46)
        e = S.runs(cv, x0 + 16.9, yb, [(D.frac[st], 5.8, "bold", c)])
        T(cv, e + 1.4, yb, names[st], 5.6, GREY, style="italic")
        seq = D.seq[st]
        yl = yb + 8.9
        xs = [xs0 + k * sp for k in range(len(seq))]
        cv.plot([xs[0], xs[-1] + 2.8], [yl, yl], color=S.light(c, 0.5), lw=2.0, zorder=3,
                solid_capstyle="round")
        fh = next(k for k, l in enumerate(seq) if ORGAN_OF[l] == "Heart")
        fb = next(k for k, l in enumerate(seq) if ORGAN_OF[l] == "Brain")
        for k, lab in enumerate(seq):
            org = ORGAN_OF[lab]
            big = k in (fh, fb)
            if big:
                S.halo(cv, xs[k], yl, 6.6, ORG[org], alpha=0.22, z=4)
            S.abbr_badge(cv, xs[k], yl, 4.9 if big else 4.4, ORG[org], ABBR[lab], fs=5.0, z=6, ring_lw=0.5)
        # heart -> brain marker row
        yi = yb + 4.3
        I.draw(cv, ICON["Heart"], xs[fh], yi, 2.9, ORG["Heart"], zorder=8)
        I.draw(cv, ICON["Brain"], xs[fb], yi, 2.9, ORG["Brain"], zorder=8)
        S.gradient_arc(cv, (xs[fh] + 2.0, yi), (xs[fb] - 2.3, yi), (0, -0.01), ORG["Heart"], ORG["Brain"],
                       lw=0.7, head=0.62)
        T(cv, xs[fb] + 2.4, yi + 0.05, "P(heart first) = ", 5.2, GREY)
        T(cv, xs[fb] + 2.4 + S.text_w("P(heart first) = ", 5.2), yi + 0.05, md(D.p_hf[st], 2), 5.6, INK,
          "bold")
    ya = y0 + 36.0
    S.arrow(cv, (xs0 - 1.4, ya), (xs0 + 9 * sp + 2.8, ya), color=MID, lw=0.5, ms=3.6)
    T(cv, xs0 - 1.4, ya + 1.6, "earlier", 5.0, MID, va="center")
    T(cv, xs0 + 9 * sp + 2.8, ya + 1.6, "later", 5.0, MID, ha="right", va="center")
    T(cv, (xs0 + xs0 + 9 * sp) / 2, ya + 1.6, "rank of first abnormality (z = 1)", 5.0, MID, ha="center")
    # robustness checks
    yk = y0 + 41.2
    S.check(cv, x0 + 1.9, yk, 2.5)
    S.runs(cv, x0 + 3.8, yk, [("Heart first in all ", 5.2, "normal", INK), (f"{D.n_models}", 5.2, "bold", INK),
                              (" SuStaIn models", 5.2, "normal", INK),
                              (f"  (P \u2265 {md(np.floor(D.p_hf_min * 100) / 100, 2)})", 5.2, "normal", GREY)])
    S.check(cv, x0 + 1.9, yk + 3.5, 2.5)
    S.runs(cv, x0 + 3.8, yk + 3.5, [("De novo refit reproduces the order", 5.2, "normal", INK),
                                    (f"  (n = {F_('rep_n')}; \u03c1 = {F_('rep_rho1')}, {F_('rep_rho2')})",
                                     5.2, "normal", GREY)])
    signature(F, cv, x0, y0 + 48.4)


def signature(F, cv, x0, y0):
    """Measured regional signature at stage >=3 (derivation sample, mean regional z; v7 renders)."""
    cmap, norm = S.z_cmap()
    T(cv, x0 + 0.4, y0, "Measured signature at stage \u22653", 6.0, INK, "bold")
    # colour bar
    cbx, cbw = x0 + 49.0, 16.6
    cbar = np.asarray(cmap(np.linspace(0, 1, 256)))[None, :, :]
    cv.imshow(cbar, extent=(cbx, cbx + cbw, y0 + 0.55, y0 - 0.45), aspect="auto", zorder=5, interpolation="bilinear")
    cv.add_patch(Rectangle((cbx, y0 - 0.45), cbw, 1.0, fc="none", ec=GREY, lw=0.3, zorder=6))
    for v in (0, 0.5, 1.0):
        xv = cbx + cbw * norm(v)
        cv.plot([xv, xv], [y0 + 0.55, y0 + 1.0], color=GREY, lw=0.3, zorder=6)
        T(cv, xv, y0 + 1.95, md(v, 1) if v else "0", 5.0, GREY, ha="center")
    T(cv, cbx - 1.0, y0 + 0.05, "z", 5.4, GREY, ha="right", style="italic")
    cols = [("LV wall\u2020", 9.4), ("aorta", 18.4), ("cortex", 28.6), ("subcortical\u2020", 40.0),
            ("WM free\nwater", 51.0), ("WMH", 61.4)]
    yc = y0 + 5.4
    for lab, cx in cols:
        T(cv, x0 + cx, yc, lab, 5.0, GREY, ha="center", va="center", linespacing=0.95)
    for i, st in enumerate((1, 2)):
        cy = y0 + 11.7 + i * 7.8
        T(cv, x0 + 1.6, cy, f"S{st}", 6.2, SUBC[st], "bold", ha="center")
        S.place(cv, render(f"sig_s{st}_0"), x0 + cols[0][1], cy, h=7.2, anchor="c", z=3)
        S.place(cv, render(f"sig_s{st}_1"), x0 + cols[1][1], cy, h=7.0, anchor="c", z=3)
        S.place(cv, render(f"sig_s{st}_2"), x0 + cols[2][1], cy, w=9.6, anchor="c", z=3)
        S.place(cv, render(f"sig_s{st}_3"), x0 + cols[3][1], cy, w=8.8, anchor="c", z=3)
        S.place(cv, render(f"sig_s{st}_4"), x0 + cols[4][1], cy, h=7.6, anchor="c", z=3)
        v = D.wmh[st]
        col = cmap(norm(v))
        S.card(cv, x0 + cols[5][1] - 3.3, cy - 1.7, 6.6, 3.4, fc=col, r=0.9, z=4)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        T(cv, x0 + cols[5][1], cy + 0.05, md(v, 2), 5.4, "white" if lum < 0.6 else INK, "bold", ha="center")
    T(cv, x0 + 0.4, y0 + 25.3, "Derivation sample, mean regional z.  \u2020 Not a model input (display only).",
      5.0, MID)


# ======================================================================================== D
def stage_chips(cv, x_of, y, labels, cols, w=2.9, h=1.9, fs=5.0):
    for xv, lab, c in zip(x_of, labels, cols):
        S.card(cv, xv - w / 2, y - h / 2, w, h, fc=c, r=0.5, z=6)
        lum = sum(np.array(S.to_rgb(c)) * [0.299, 0.587, 0.114])
        T(cv, xv, y + 0.03, lab, fs, "white" if lum < 0.6 else INK, "bold", ha="center")


def style_axes(ax):
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.45)
    ax.tick_params(length=1.6, width=0.45, pad=1.0, labelsize=5.0, colors="#3A3F47")
    ax.patch.set_alpha(0)


def panel_d(F, cv, x0=0.0, y0=YB):
    S.panel_letter(cv, x0 + 0.2, y0 + 0.5, "D", "Risk after imaging")
    # sample legend (shared with A)
    lx = x0 + 34.0
    cv.add_patch(Circle((lx, y0 + 2.4), 0.95, fc=INK, ec="none", zorder=6))
    e = S.runs(cv, lx + 1.6, y0 + 2.45, [("derivation", 5.4, "normal", INK)])
    cv.add_patch(Circle((e + 2.4, y0 + 2.4), 0.85, fc="white", ec=GREY, lw=0.7, zorder=6))
    S.runs(cv, e + 4.0, y0 + 2.45, [("independent", 5.4, "normal", INK),
                                    ("   primary outcome; 95% CI", 5.0, "normal", MID, "italic")])
    # ---------------------------------------------------------------- D1 rates by stage
    T(cv, x0 + 0.4, y0 + 7.6, "Event rate by stage", 6.0, INK, "bold")
    axx, axy, axw, axh = x0 + 7.4, y0 + 10.6, 27.8, 24.0
    ax = F.add(axx, axy, axw, axh)
    style_axes(ax)
    off = 0.13
    for key, dx, ls in (("der", -off, "-"), ("ind", off, (0, (2.2, 1.4)))):
        r = D.rate[key]
        xs = np.arange(4) + dx
        ax.plot(xs, [t[0] for t in r], color=MID, lw=0.7, ls=ls, zorder=2)
        for i, t in enumerate(r):
            ec = S.dark(STAGE4[i], 0.28) if i else "#9AA0A8"
            ax.plot([xs[i], xs[i]], [t[1], t[2]], color=ec, lw=0.75, zorder=3, solid_capstyle="butt")
            if key == "der":
                ax.scatter(xs[i], t[0], s=13, c=[STAGE4[i]], edgecolors=ec, linewidths=0.6, zorder=5)
            else:
                ax.scatter(xs[i], t[0], s=12, c="white", edgecolors=ec, linewidths=0.85, zorder=5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(True)
    ax.set_ylabel("events per 1000 PY", fontsize=5.2, labelpad=1.2)
    rd, ri = D.rate["der"], D.rate["ind"]
    ax.text(3 - off, rd[3][2] + 0.9, md(rd[3][0]), fontsize=5.4, fontweight="bold", ha="center", va="bottom",
            color=INK, clip_on=False)
    ax.text(3 + off + 0.2, ri[3][0], md(ri[3][0]), fontsize=5.4, fontweight="bold", ha="left", va="center",
            color=INK, clip_on=False)
    xm = [axx + axw * (v + 0.5) / 4 for v in range(4)]
    stage_chips(cv, xm, axy + axh + 1.9, ["0", "1", "2", "\u22653"], STAGE4)
    T(cv, axx + axw / 2, axy + axh + 4.6, "heart\u2013brain stage", 5.2, GREY, ha="center")
    # ---------------------------------------------------------------- D2 HR and D3 C index
    gx = x0 + 40.2
    fx0, fx1 = x0 + 52.0, x0 + 65.0
    T(cv, gx, y0 + 7.6, "Hazard ratio per 10 pp", 6.0, INK, "bold")
    T(cv, gx + S.text_w("Hazard ratio per 10 pp", 6.0, "bold") + 1.0, y0 + 7.65, "adjusted*", 5.0, MID,
      style="italic")
    lo_, hi_ = np.log(0.9), np.log(2.0)

    def fxv(v):
        return fx0 + (np.log(v) - lo_) / (hi_ - lo_) * (fx1 - fx0)

    yr = {"der": y0 + 11.9, "ind": y0 + 15.7}
    cv.plot([fxv(1), fxv(1)], [y0 + 10.1, y0 + 17.6], color=LIGHT, lw=0.45, ls=(0, (1.5, 1.2)), zorder=3)
    for key, lab in (("der", "derivation"), ("ind", "independent")):
        y = yr[key]
        T(cv, gx, y, lab, 5.0, GREY)
        h, l, u = D.hr[key]
        cv.plot([fxv(l), fxv(u)], [y, y], color=INK, lw=0.85, zorder=5, solid_capstyle="butt")
        if key == "der":
            cv.add_patch(Circle((fxv(h), y), 0.85, fc=INK, ec="white", lw=0.3, zorder=6))
        else:
            cv.add_patch(Circle((fxv(h), y), 0.8, fc="white", ec=INK, lw=0.75, zorder=6))
        S.runs(cv, fx1 + 1.2, y, [(md(h, 2), 5.6, "bold", INK), (f" ({md(l, 2)}\u2013{md(u, 2)})", 5.0,
                                                                   "normal", GREY)])
    ytk = y0 + 18.6
    cv.plot([fx0, fx1], [ytk, ytk], color="#3A3F47", lw=0.45, zorder=3)
    for v, lab in ((1.0, "1"), (1.5, "1\u00b75"), (2.0, "2")):
        cv.plot([fxv(v), fxv(v)], [ytk, ytk + 0.7], color="#3A3F47", lw=0.45, zorder=3)
        T(cv, fxv(v), ytk + 1.75, lab, 5.0, "#3A3F47", ha="center")
    # C index dumbbell
    T(cv, gx, y0 + 25.4, "C index: SCORE2 \u2192 + stage", 6.0, INK, "bold")
    clo, chi = 0.67, 0.72

    def cxv(v):
        return fx0 + (v - clo) / (chi - clo) * (fx1 - fx0)

    yc = {"der": y0 + 29.6, "ind": y0 + 33.4}
    green = STAGE4[3]
    for key, lab in (("der", "derivation"), ("ind", "independent")):
        y = yc[key]
        T(cv, gx, y, lab, 5.0, GREY)
        c0, c1, dc, dlo, dhi = D.cidx[key]
        cv.plot([cxv(c0), cxv(c1)], [y, y], color=S.light(green, 0.45), lw=1.6, zorder=4, solid_capstyle="butt")
        cv.add_patch(Circle((cxv(c0), y), 0.8, fc="white", ec=MID, lw=0.7, zorder=6))
        if key == "der":
            cv.add_patch(Circle((cxv(c1), y), 0.88, fc=green, ec="white", lw=0.3, zorder=6))
        else:
            cv.add_patch(Circle((cxv(c1), y), 0.82, fc="white", ec=green, lw=0.85, zorder=6))
        S.runs(cv, fx1 + 1.2, y, [(f"+{md(dc, 3)}", 5.6, "bold", green),
                                  (f" ({md(dlo, 3)}\u2013{md(dhi, 3)})", 5.0, "normal", GREY)])
    ytk = y0 + 36.3
    cv.plot([fx0, fx1], [ytk, ytk], color="#3A3F47", lw=0.45, zorder=3)
    for v in (0.68, 0.70, 0.72):
        cv.plot([cxv(v), cxv(v)], [ytk, ytk + 0.7], color="#3A3F47", lw=0.45, zorder=3)
        T(cv, cxv(v), ytk + 1.75, md(v, 2), 5.0, "#3A3F47", ha="center")
    T(cv, gx, y0 + 41.2, "* M2: age, sex, vascular risk factors; complete cases", 5.0, MID, style="italic")
    # ---------------------------------------------------------------- D4 SCORE2 strata (independent)
    sx = x0 + 84.6
    T(cv, sx, y0 + 7.6, "Within SCORE2 risk", 6.0, INK, "bold")
    T(cv, sx + S.text_w("Within SCORE2 risk", 6.0, "bold") + 1.0, y0 + 7.65, "indep.", 5.0, MID,
      style="italic")
    axx2, axw2 = x0 + 89.2, 15.6
    ax2 = F.add(axx2, axy, axw2, axh)
    style_axes(ax2)
    gcol = [STAGE4[0], S.mix(STAGE4[1], STAGE4[2], 0.5), STAGE4[3]]
    for grp, ls, lc in (("Low-to-moderate", (0, (2.2, 1.4)), MID), ("High or very high", "-", INK)):
        r = D.s2[grp]
        ax2.plot(range(3), [t[0] for t in r], color=lc, lw=0.75, ls=ls, zorder=2)
        for i, t in enumerate(r):
            ec = S.dark(gcol[i], 0.28) if i else "#9AA0A8"
            ax2.plot([i, i], [t[1], t[2]], color=ec, lw=0.75, zorder=3, solid_capstyle="butt")
            ax2.scatter(i, t[0], s=12, c="white", edgecolors=ec, linewidths=0.85, zorder=5)
        ax2.text(2.22, r[2][0], md(r[2][0]), fontsize=5.4 if grp.startswith("High") else 5.0,
                 fontweight="bold" if grp.startswith("High") else "normal", ha="left", va="center",
                 color=INK if grp.startswith("High") else GREY)
    ax2.set_xlim(-0.4, 2.4)
    ax2.set_ylim(0, 30)
    ax2.set_yticks([0, 10, 20, 30])
    ax2.set_xticks([])
    # direct labels in the empty space next to each line
    ax2.text(-0.28, 21.4, "high /\nvery high", fontsize=5.0, ha="left", va="bottom", color=INK,
             fontweight="bold", linespacing=0.95)
    ax2.text(1.3, 3.3, "low\u2013moderate", fontsize=5.0, ha="center", va="top", color=GREY)
    xm2 = [axx2 + axw2 * (v + 0.4) / 2.8 for v in range(3)]
    stage_chips(cv, xm2, axy + axh + 1.9, ["0", "1\u20132", "\u22653"], gcol, w=3.6)
    T(cv, axx2 + axw2 / 2, axy + axh + 4.6, "stage", 5.2, GREY, ha="center")


# ======================================================================================== E (interpretation)
def panel_e(F, cv, x0=COLX[2], y0=YB, w=67.0, h=43.6):
    S.card(cv, x0, y0, w, h, fc=BAND, r=1.8, z=1)
    cv.add_patch(FancyBboxPatch((x0, y0 + 2.0), 1.0, h - 4.0, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=ORG["Brain"], ec="none", zorder=2))
    T(cv, x0 + 3.4, y0 + 3.6, "Interpretation and intended use", 6.6, INK, "bold")
    rows = [("hb", "Heart changes come first", "model-implied order, inferred from cross-sectional data"),
            ("sub", "Two subtypes, distinct signatures", "S1 hippocampus and cortex; S2 LV wall and aorta"),
            ("wm", "Heart\u2013brain coupling in white matter", "stronger in white matter than in cortex"),
            ("risk", "Higher stage, more events", f"HR {md(D.hr['ind'][0], 2)}\u2013{md(D.hr['der'][0], 2)} "
                                                     "per 10 pp; replicated independently"),
            ("bal", "Modest gain beyond SCORE2", "for people already scanned; not a screening tool")]
    for k, (ic, claim, qual) in enumerate(rows):
        y = y0 + 10.2 + k * 7.05
        cx = x0 + 7.4
        S.pill(cv, cx, y, 9.0, 5.4, fc="white", z=4)
        if ic == "hb":
            I.draw(cv, ICON["Heart"], cx - 2.4, y, 3.0, ORG["Heart"], zorder=6)
            I.draw(cv, ICON["Brain"], cx + 2.4, y, 3.0, ORG["Brain"], zorder=6)
            S.arrow(cv, (cx - 0.6, y), (cx + 0.75, y), color=GREY, lw=0.5, ms=2.6, z=6)
        elif ic == "sub":
            cv.add_patch(Circle((cx - 1.1, y), 1.65, fc=SUBC[1], ec="white", lw=0.4, zorder=6))
            cv.add_patch(Circle((cx + 1.1, y), 1.65, fc=SUBC[2], ec="white", lw=0.4, zorder=6, alpha=0.95))
            T(cv, cx - 1.35, y + 0.05, "1", 5.0, "white", "bold", ha="center", z=7)
            T(cv, cx + 1.35, y + 0.05, "2", 5.0, "white", "bold", ha="center", z=7)
        elif ic == "wm":
            I.draw(cv, "body/nerve", cx, y, 3.8, ORG["Brain"], zorder=6)
        elif ic == "risk":
            I.draw(cv, "graphs/chart_line", cx, y, 3.6, STAGE4[3], zorder=6)
        else:
            I.draw(cv, "objects/justice", cx, y, 3.8, EVENT, zorder=6)
        T(cv, x0 + 13.0, y - 1.35, claim, 6.0, INK, "bold")
        T(cv, x0 + 13.0, y + 1.55, qual, 5.0, GREY)


# ======================================================================================== chrome
def chrome(cv):
    for x in (COLX[1] - 1.0, COLX[2] - 1.0):
        cv.plot([x, x], [2.0, 74.0], color=RULE, lw=0.4, zorder=1)
    cv.plot([0.4, W - 0.4], [YB - 2.2, YB - 2.2], color=RULE, lw=0.4, zorder=1)


SINGLE = {"a": (39.6, 75.0, panel_a), "b": (70.6, 75.0, panel_b), "c": (67.4, 75.0, panel_c),
          "d": (111.6, 44.0, panel_d), "e": (67.4, 44.0, panel_e)}
STEM = {"a": "Fig1a", "b": "Fig1b", "c": "Fig1c", "d": "Fig1d", "e": "Fig1e_interpretation"}


def main(which="all"):
    global D
    S.set_style()
    D = Data()
    D.report()
    keys = list(SINGLE) if which in ("all", "panels") else ([which] if which in SINGLE else [])
    for k in keys:
        w, h, fn = SINGLE[k]
        Fm = S.MMFig(w, h)
        cv = Fm.canvas()
        if k == "e":
            fn(Fm, cv, 0.2, 0.2, w=w - 0.4, h=h - 0.4)
        else:
            fn(Fm, cv, 0.0, 0.0)
        Fm.save(str(OUT / STEM[k]))
        print("saved", STEM[k], w, h)
    if which in ("all", "composite"):
        Fm = S.MMFig(W, H)
        cv = Fm.canvas()
        panel_a(Fm, cv, COLX[0], 0.0)
        panel_b(Fm, cv, COLX[1], 0.0)
        panel_c(Fm, cv, COLX[2], 0.0)
        panel_d(Fm, cv, 0.0, YB)
        panel_e(Fm, cv, COLX[2], YB, w=W - COLX[2], h=H - YB - 0.4)
        chrome(cv)
        Fm.save(str(OUT / "Figure1"))
        print("saved Figure1", W, H)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
