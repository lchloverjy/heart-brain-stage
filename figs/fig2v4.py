import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Figure 2 (v4) - the heart-brain abnormality sequence.
A, positional variance diagrams (All-10, two subtypes); B, anatomical stage snapshots; C, organ-onset ladder across
models and the de novo refit; D, observed cardiac-brain co-abnormality in the independent sample (fig2d.py)."""
import sys, os
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.colors import to_rgb
import style4 as S

RES = _os.path.join(HB_ROOT, "results")
OUT = _os.path.join(HB_ROOT, "figs/output/Figure2")
W = 180
LAB = {"Protein_M13_Z": "Protein M13", "Protein_M20_Z": "Protein M20", "Protein_M5_Z": "Protein M5", "LV_remodeling_Z": "LV remodelling",
       "LV_myocardial_dysfunction_Z": "LV dysfunction", "Aortic_stiffness_Z": "Aortic stiffness", "Hippocampal_atrophy_Z": "Hippocampal atrophy",
       "Cortical_thinning_Z": "Cortical thinning", "WMH_burden_Z": "WMH burden", "FreeWater_ISOVF_Z": "Free water"}
SUB_FULL = {1: "S1  LV dysfunction-led", 2: "S2  LV remodelling-led"}
FRAC = {}


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * t)


def zcol(org, z):
    if z <= 0:
        return "#EEF0F3"
    return {1: S.ORG_MID[org], 2: S.ORG[org], 3: mix(S.ORG[org], "#1A1A1A", 0.38)}[int(z)]


def pcol(org, p):
    p = float(np.clip(p, 0, 1)) ** 0.75
    return mix("#FFFFFF", mix(S.ORG[org], "#1A1A1A", 0.18), p)


# ------------------------------------------------------------------ a
def panel_a(ax, x0=0, y0=0):
    pvd = pd.read_csv(f"{RES}/sustain_pvd_all10.csv")
    reach = pd.read_csv(f"{RES}/fig2_stage_reach.csv"); reach = reach[reach.model == "all10"]
    hf = pd.read_csv(f"{RES}/sustain_heart_first.csv"); hf = hf[hf.model == "all10"].set_index("subtype")
    S.panel_letter(ax, x0 + 1, y0 + 1, "a", "Model-implied order of heart, plasma-protein, and brain abnormality (All-10)")
    cell = 1.78; LW = 23.0; NE = 30
    for b, sub in enumerate([1, 2]):
        bx = x0 + 2 + b * 80.5
        mx = bx + LW; my = y0 + 18.5
        g = pvd[pvd.subtype == sub].sort_values("ml_position")
        frac = hf.loc[sub, "fraction"]; FRAC[sub] = frac
        ax.text(bx, y0 + 8.6, SUB_FULL[sub].split("  ")[0], fontsize=6.4, fontweight="bold", color=S.INK, va="center")
        ax.text(bx + 5.2, y0 + 8.6, SUB_FULL[sub].split("  ")[1], fontsize=6.2, color=S.INK, va="center")
        ax.text(mx + NE * cell, y0 + 8.6, f"model fraction {S.md(100 * frac, 1)}%", fontsize=5.0, color=S.GREY, va="center",
                ha="right")
        # reach strip
        r = reach[reach.subtype == sub].set_index("stage").frac_reached
        sh = 4.6; sy = my - 1.4
        for st in range(1, NE + 1):
            v = r.get(st, 0.0)
            if v > 0:
                ax.add_patch(Rectangle((mx + (st - 1) * cell + 0.15, sy - sh * v), cell - 0.3, sh * v, fc="#9BA3AD", ec="none", zorder=2))
        ax.plot([mx, mx + NE * cell], [sy, sy], color="#9BA3AD", lw=0.35)
        ax.text(mx - 0.8, sy - sh * 0.62, "share", fontsize=4.5, color=S.GREY, ha="right", va="center")
        ax.text(mx - 0.8, sy - sh * 0.62 + 1.9, "reaching", fontsize=4.5, color=S.GREY, ha="right", va="center")
        ax.text(mx + 2 * cell, sy - sh + 0.4, "100%", fontsize=4.2, color=S.GREY, ha="left", va="center")
        last = int(r[r > 0].index.max())
        ax.text(mx + last * cell + 0.6, sy - 1.2, f"max stage {last}", fontsize=4.4, color=S.GREY, va="center")
        # matrix
        ax.add_patch(Rectangle((mx, my), NE * cell, NE * cell, fc="white", ec="none", zorder=1))
        for i, row in enumerate(g.itertuples()):
            org = row.organ
            for p in range(NE):
                v = getattr(row, f"p{p + 1}")
                if v > 0.004:
                    ax.add_patch(Rectangle((mx + p * cell, my + i * cell), cell, cell, fc=pcol(org, v), ec="none", zorder=2))
            yy = my + (i + 0.5) * cell
            bio = [k for k, v in LAB.items() if v == row.label][0] if row.label in LAB.values() else row.label
            ax.add_patch(Rectangle((bx, yy - 0.62), 1.24, 1.24, fc=zcol(org, row.z), ec="none", zorder=3))
            ax.text(bx + 2.0, yy, row.label, fontsize=4.75, va="center", ha="left", color=S.INK)
            ax.text(mx - 0.7, yy, f"z{int(row.z)}", fontsize=4.5, va="center", ha="right", color=S.GREY)
        # grid frame and ticks
        ax.add_patch(Rectangle((mx, my), NE * cell, NE * cell, fc="none", ec="#B7BCC4", lw=0.4, zorder=4))
        for t in [1, 5, 10, 15, 20, 25, 30]:
            xx = mx + (t - 0.5) * cell
            ax.plot([xx, xx], [my + NE * cell, my + NE * cell + 0.8], color="#6B7078", lw=0.4)
            ax.text(xx, my + NE * cell + 1.3, str(t), fontsize=4.9, ha="center", va="top", color=S.GREY)
        ax.text(mx + NE * cell / 2, my + NE * cell + 4.6, "Position in sequence (stage)", fontsize=5.3, ha="center", va="top",
                color=S.INK)
    # legend: posterior probability ramps
    lx = x0 + 163.5; ly = y0 + 21
    ax.text(lx - 0.5, ly - 3.2, "Posterior", fontsize=5.0, color=S.INK, fontweight="bold")
    ax.text(lx - 0.5, ly - 1.0, "probability", fontsize=5.0, color=S.INK, fontweight="bold")
    for j, org in enumerate(["Heart", "Protein", "Brain"]):
        cx = lx + j * 4.6
        n = 40; hh = 26
        for k in range(n):
            ax.add_patch(Rectangle((cx, ly + 1.5 + hh * (1 - (k + 1) / n)), 2.6, hh / n + 0.02, fc=pcol(org, (k + 0.5) / n), ec="none"))
        ax.add_patch(Rectangle((cx, ly + 1.5), 2.6, hh, fc="none", ec="#B7BCC4", lw=0.35))
        ax.text(cx + 1.3, ly + 1.5 + hh + 1.3, org[0] if org != "Protein" else "P", fontsize=5.0, ha="center", va="top",
                color=S.ORG[org], fontweight="bold")
    for v, lab in [(0, "0"), (0.5, "0\u00b75"), (1, "1")]:
        ax.text(lx + 13.6, ly + 1.5 + 26 * (1 - v), lab, fontsize=4.6, va="center", color=S.GREY)
    ax.text(lx - 0.5, ly + 33.5, "H heart\nP plasma protein\nB brain", fontsize=4.5, color=S.GREY, va="top", linespacing=1.15)
    ax.text(lx - 0.5, ly + 42.0, "Row chip =\nabnormality\nthreshold z", fontsize=4.5, color=S.GREY, va="top", linespacing=1.15)


# ------------------------------------------------------------------ b
SNAP = [1, 2, 3, 5, 8, 12]


def snapshot(ax, x, y, levels, prot=True):
    """heart SAX + brain coronal coloured by z level reached; levels: biomarker -> z."""
    S.draw_heart_sax_mm(ax, x + 6.6, y + 0.4, 10.5, col_myo=zcol("Heart", levels["LV_myocardial_dysfunction_Z"]),
                        col_cav=zcol("Heart", levels["LV_remodeling_Z"]) if levels["LV_remodeling_Z"] > 0 else "#FFFFFF",
                        col_rv="#F4F1EE", col_ao=zcol("Heart", levels["Aortic_stiffness_Z"]), outline="#8E8A86", lw=0.35)
    S.draw_brain_mm(ax, x + 18.6, y + 0.0, 12.6, {"cortex": zcol("Brain", levels["Cortical_thinning_Z"]),
                                                   "hippocampus": zcol("Brain", levels["Hippocampal_atrophy_Z"]),
                                                   "wm_pv": zcol("Brain", levels["WMH_burden_Z"]) if levels["WMH_burden_Z"] > 0 else "#F6F5F8",
                                                   "wm_deep": zcol("Brain", levels["FreeWater_ISOVF_Z"]) if levels["FreeWater_ISOVF_Z"] > 0 else "#FBFAFC",
                                                   "subcortical": "#ECEAEF"})
    if prot:
        for i, b in enumerate(["Protein_M13_Z", "Protein_M20_Z", "Protein_M5_Z"]):
            ax.add_patch(Circle((x + 3.4 + i * 2.6, y + 7.4), 0.9, fc=zcol("Protein", levels[b]), ec="#8E8A86", lw=0.3, zorder=5))


def panel_b(ax, x0=0, y0=84):
    if not FRAC:
        hf = pd.read_csv(f"{RES}/sustain_heart_first.csv"); hf = hf[hf.model == "all10"]
        FRAC.update(dict(zip(hf.subtype, hf.fraction)))
    sn = pd.read_csv(f"{RES}/fig2_stage_snapshots.csv"); sn = sn[sn.model == "all10"]
    reach = pd.read_csv(f"{RES}/fig2_stage_reach.csv"); reach = reach[reach.model == "all10"]
    S.panel_letter(ax, x0 + 1, y0 + 1, "b", "Stage snapshots: abnormality accumulated along each subtype sequence")
    LX = 30.0; SW = 25.0
    for r_, sub in enumerate([1, 2]):
        yy = y0 + 18.0 + r_ * 25.5
        ax.text(x0 + 2, yy - 1.6, SUB_FULL[sub].split("  ")[0], fontsize=6.4, fontweight="bold", color=S.INK, va="center")
        ax.text(x0 + 2, yy + 1.6, SUB_FULL[sub].split("  ")[1], fontsize=5.4, color=S.INK, va="center")
        ax.text(x0 + 2, yy + 4.5, f"model fraction {S.md(100 * FRAC.get(sub, np.nan), 0)}%", fontsize=4.6, color=S.GREY,
                va="center")
        rr = reach[reach.subtype == sub].set_index("stage")
        for j, st in enumerate(SNAP):
            sx = x0 + LX + j * SW
            lv = sn[(sn.subtype == sub) & (sn.stage == st)].set_index("biomarker").z_reached.to_dict()
            if j % 2 == 0:
                ax.add_patch(Rectangle((sx - 0.6, yy - 10.6), SW, 23.4, fc=S.BAND, ec="none", zorder=0))
            snapshot(ax, sx, yy - 1.0, lv)
            new = sn[(sn.subtype == sub) & (sn.stage == st)].set_index("biomarker").z_reached
            prev = sn[(sn.subtype == sub) & (sn.stage == st - 1)].set_index("biomarker").z_reached
            ch = (new - prev)[(new - prev) > 0]
            evlab = f"{LAB[ch.index[0]]} z{int(new[ch.index[0]])}" if len(ch) else ""
            fr_ = rr.frac_reached.get(st, 0); nr_ = rr.n_reached.get(st, 0)
            rtxt = (f"{S.md(100 * fr_, 0)}% reached" if fr_ >= 0.005 else ("<1% reached" if nr_ > 0 else "model only"))
            ax.text(sx + 10.4, yy + 10.4, f"Stage {st}", fontsize=5.4, fontweight="bold", ha="right", va="center", color=S.INK)
            ax.text(sx + 11.2, yy + 10.4, rtxt, fontsize=4.6, ha="left", va="center", color=S.GREY)
            ax.text(sx + 12.0, yy - 8.6, f"+ {evlab}", fontsize=4.5, ha="center", va="center", color=S.darken(S.ORG[sn[(sn.biomarker == ch.index[0])].organ.iloc[0]], 0.1) if len(ch) else S.GREY)
    # legend
    ly = y0 + 63.5; lx = x0 + 30
    ax.text(lx - 0.8, ly, "Abnormality reached", fontsize=5.1, va="center", ha="right", color=S.INK)
    for i, org in enumerate(["Heart", "Protein", "Brain"]):
        bx = lx + 2 + i * 33
        ax.text(bx, ly, {"Heart": "Heart", "Protein": "Plasma protein", "Brain": "Brain"}[org], fontsize=5.0, va="center",
                color=S.ORG[org], fontweight="bold")
        ox = bx + (8.5 if org != "Protein" else 15.5)
        for k, lab in enumerate(["none", "z1", "z2", "z3"]):
            ax.add_patch(Rectangle((ox + k * 4.4, ly - 1.1), 2.2, 2.2, fc=zcol(org, k), ec="#B7BCC4" if k == 0 else "none", lw=0.3))
            ax.text(ox + k * 4.4 + 1.1, ly + 2.4, lab, fontsize=4.3, ha="center", va="center", color=S.GREY)
    ax.text(x0 + 178.5, ly, "Dots under the heart: M13, M20, M5", fontsize=4.5, ha="right", va="center", color=S.GREY)


# ------------------------------------------------------------------ c
def panel_c(ax, x0=0, y0=150):
    on = pd.read_csv(f"{RES}/fig2_organ_onset.csv")
    S.panel_letter(ax, x0 + 1, y0 + 1, "c", "First cardiac event precedes first brain event across models")
    rows = []
    for m in ["all10", "mri_plus_m13", "mri_only_matched_all10"]:
        subs = sorted(on[on.model == m].subtype.unique())
        for s in subs:
            rows.append((m, s))
    rep = f"{RES}/replication_sustain/rep6000_onset.csv"
    if os.path.exists(rep):
        rp = pd.read_csv(rep)
        on = pd.concat([on, rp])
        for s in sorted(rp.subtype.unique()):
            rows.append(("replication", s))
    L = x0 + 30; R = x0 + 60; xmax = 6
    def X(p): return L + (p - 0.5) / xmax * (R - L)
    top = y0 + 12.0; dy = 4.15
    for p in range(1, xmax + 1):
        ax.plot([X(p), X(p)], [top - 2.4, top + dy * (len(rows) - 0.4)], color=S.RULE, lw=0.35, zorder=0)
        ax.text(X(p), top + dy * (len(rows) - 0.4) + 1.0, str(p), fontsize=4.9, ha="center", va="top", color=S.GREY)
    ax.text((L + R) / 2, top + dy * (len(rows) - 0.4) + 4.3, "Position of first event", fontsize=5.3, ha="center", va="top")
    ax.text(R + 3.5, top - 4.3, "P(heart\nfirst)", fontsize=4.7, ha="center", va="center", color=S.GREY, linespacing=1.0)
    prev_m = None
    for i, (m, s) in enumerate(rows):
        yy = top + i * dy
        g = on[(on.model == m) & (on.subtype == s)].set_index("organ")
        if m != prev_m:
            nm = {"all10": "All-10", "mri_plus_m13": "MRI + M13", "mri_only_matched_all10": "MRI only",
                  "replication": "Independent\nrefit (n = 6000)"}[m]
            mc = {"all10": "#0F4A38", "mri_plus_m13": "#2F7A60", "mri_only_matched_all10": "#4E8E76"}.get(m, "#3D5A80")
            ax.text(x0 + 2, yy, nm, fontsize=5.2, fontweight="bold", va="center", color=mc)
            if prev_m is not None:
                ax.plot([x0 + 2, R + 7], [yy - dy / 2, yy - dy / 2], color=S.RULE, lw=0.35)
            prev_m = m
        ax.text(L - 2.0, yy, f"S{s}", fontsize=5.0, va="center", ha="right", color=S.INK)
        hx = X(g.loc["Heart", "median"]); bx = X(g.loc["Brain", "median"])
        ax.plot([hx, bx], [yy, yy], color="#9AA0A8", lw=0.9, zorder=1, solid_capstyle="round")
        for org, mk, sz in [("Heart", "o", 3.4), ("Protein", "s", 2.9), ("Brain", "D", 2.8)]:
            if org in g.index:
                lo, hi = g.loc[org, "lo"], g.loc[org, "hi"]
                off = {"Heart": -0.0, "Protein": 0.0, "Brain": 0.0}[org]
                if hi > lo:
                    ax.plot([X(lo), X(hi)], [yy + off, yy + off], color=S.ORG[org], lw=0.6, alpha=0.7, zorder=2)
                ax.plot(X(g.loc[org, "median"]), yy + off, mk, ms=sz, mfc=S.ORG[org], mec="white", mew=0.4, zorder=3)
        ax.text(R + 3.5, yy, S.md(g.loc["P(heart<brain)", "median"], 2), fontsize=5.0, ha="center", va="center", color=S.INK)
    ly = top + dy * (len(rows) - 0.4) + 8.2
    for i, (org, mk, lab) in enumerate([("Heart", "o", "first cardiac"), ("Protein", "s", "first protein"), ("Brain", "D", "first brain")]):
        ax.plot(x0 + 4 + i * 20, ly, mk, ms=3.0, mfc=S.ORG[org], mec="white", mew=0.4)
        ax.text(x0 + 5.6 + i * 20, ly, lab, fontsize=4.8, va="center", color=S.INK)


def render(which="all"):
    os.makedirs(OUT, exist_ok=True)
    S.set_style()
    if which in ("a", "all"):
        F = S.MMFig(W, 82); cv = F.canvas(); panel_a(cv, 0, 0); F.save(f"{OUT}/Fig2a")
    if which in ("b", "all"):
        F = S.MMFig(W, 70); cv = F.canvas(); panel_b(cv, 0, 0); F.save(f"{OUT}/Fig2b")
    if which in ("c", "all"):
        F = S.MMFig(72, 66); cv = F.canvas(); panel_c(cv, 0, 0); F.save(f"{OUT}/Fig2c")
    if which in ("d", "all"):
        import fig2d
        F = S.MMFig(108, 42); cv = F.canvas(); fig2d.panel_d(cv, 0, 0); F.save(f"{OUT}/Fig2d")
    if which in ("composite", "all"):
        F = S.MMFig(W, 216); cv = F.canvas()
        panel_a(cv, 0, 0); panel_b(cv, 0, 84); panel_c(cv, 0, 152)
        import fig2d  # panel D; fail loudly rather than silently omit it if its inputs are missing
        fig2d.panel_d(cv, 74, 152)
        F.save(f"{OUT}/Figure2_composite")


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "all")
