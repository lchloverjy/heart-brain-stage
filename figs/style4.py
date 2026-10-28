"""v4 figure style (eClinicalMedicine / Lancet conventions) shared by all main and supplementary figures."""
import math
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch, FancyBboxPatch, Ellipse
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import FuncFormatter

from base_style import MMFig, text_w_mm, lighten, text_on, despine  # noqa: E402,F401
import brainslice as BS  # noqa: E402

MM = 1 / 25.4
ORG = {"Heart": "#D9692F", "Brain": "#6E4FB0", "Protein": "#2F78B7"}
ORG_LIGHT = {"Heart": "#FBE9DF", "Brain": "#ECE7F6", "Protein": "#E3EEF8"}
ORG_MID = {"Heart": "#F2B795", "Brain": "#C3B5E3", "Protein": "#A9CBE8"}
STAGE4 = ["#C9CED6", "#8FB9A8", "#3E8E73", "#0F4A38"]          # stage 0, 1, 2, >=3
STAGE4_LABEL = ["Stage 0", "Stage 1", "Stage 2", "Stage \u22653"]
MODEL_COLOR = {"all10": "#0F4A38", "mri_plus_m13": "#3E8E73", "mri_only_matched_all10": "#8FB9A8"}
MODEL_MARK = {"all10": "o", "mri_plus_m13": "s", "mri_only_matched_all10": "D"}
MODEL_LABEL = {"all10": "All-10 (protein + heart + brain)", "mri_plus_m13": "MRI + protein M13", "mri_only_matched_all10": "MRI only"}
MODEL_SHORT = {"all10": "All-10", "mri_plus_m13": "MRI + M13", "mri_only_matched_all10": "MRI only"}
INV_COLOR = {"None": "#C9CED6", "Heart only": "#E9A27A", "Heart and brain": "#7A4E9C", "Brain only": "#A995D3"}
INK = "#1F2226"; GREY = "#6B7078"; LIGHT = "#A9AEB6"; RULE = "#D9DCE1"; BAND = "#F4F5F7"
SUBTYPE_NAME = {1: "S1 \u00b7 LV dysfunction-led", 2: "S2 \u00b7 LV remodelling-led"}
MIDDOT = "\u00b7"


def set_style():
    plt.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6,
        "legend.fontsize": 5.6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.minor.width": 0.35, "ytick.minor.width": 0.35, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.labelpad": 2.0, "lines.linewidth": 0.9, "lines.solid_capstyle": "round",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#33363B",
        "xtick.color": "#33363B", "ytick.color": "#33363B", "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600, "figure.dpi": 150,
        "mathtext.default": "regular", "legend.frameon": False, "axes.unicode_minus": True, "hatch.linewidth": 0.4,
        "axes.formatter.use_mathtext": False,
    })


# ------------------------------------------------------------------ Lancet number formatting
def md(x, nd=2):
    """Number with a midline decimal point (Lancet style)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = f"{x:.{nd}f}"
    return s.replace("-", "\u2212").replace(".", MIDDOT)


def md_int(n):
    """Lancet: no separator below 10 000, thin space above."""
    n = int(n)
    return f"{n:,}".replace(",", "\u2009") if n >= 10000 else str(n)


def hr_ci(hr, lo, hi, nd=2):
    return f"{md(hr, nd)} ({md(lo, nd)}\u2013{md(hi, nd)})"


def pval(p):
    """Lancet: two significant figures; p<0.0001 otherwise."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.0001:
        return "p<0" + MIDDOT + "0001"
    s = f"{p:.2g}"
    if "e" in s:
        s = f"{p:.6f}".rstrip("0")
    return "p=" + s.replace(".", MIDDOT)


def md_formatter(nd=None):
    def f(x, pos=None):
        if nd is None:
            s = f"{x:g}"
        else:
            s = f"{x:.{nd}f}"
        return s.replace("-", "\u2212").replace(".", MIDDOT)
    return FuncFormatter(f)


def pct_formatter(nd=0):
    return FuncFormatter(lambda x, pos=None: (f"{100 * x:.{nd}f}").replace(".", MIDDOT))


def panel_letter(ax, x, y, letter, title=None, dx=3.8, size=8):
    ax.text(x, y, letter.upper(), fontsize=size, fontweight="bold", va="top", ha="left", color=INK)
    if title:
        ax.text(x + dx, y + 0.45, title, fontsize=6.6, fontweight="bold", va="top", ha="left", color=INK)


def seq_cmap(color, n=256, white="#FFFFFF"):
    return LinearSegmentedColormap.from_list("seq", [white, lighten(color, 0.55), color, _darken(color, 0.35)], N=n)


def _darken(c, f):
    r, g, b = to_rgb(c)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


darken = _darken


# ------------------------------------------------------------------ schematic heart (mm canvas, y down)
def _bez(P0, P1, P2, P3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3


def heart_parts(cx, cy, w):
    """Polygons (mm, y down) for a stylised anterior-view heart: body, LV wall, LV cavity, aortic arch (+branches).
    w = overall width in mm. Schematic geometry (not to anatomical scale)."""
    s = w / 20.0
    A = lambda x, y: np.array([cx + x * s, cy + y * s])
    body = np.vstack([
        _bez(A(-5.6, -4.6), A(-9.4, -3.2), A(-9.6, 3.2), A(-6.0, 6.4)),
        _bez(A(-6.0, 6.4), A(-3.0, 9.0), A(2.0, 10.6), A(5.4, 10.4)),
        _bez(A(5.4, 10.4), A(8.6, 10.0), A(9.6, 5.0), A(8.8, 0.6)),
        _bez(A(8.8, 0.6), A(8.2, -3.2), A(6.2, -5.6), A(3.0, -5.9)),
        _bez(A(3.0, -5.9), A(0.0, -6.2), A(-3.0, -5.6), A(-5.6, -4.6)),
    ])
    lv_out = np.vstack([
        _bez(A(1.6, -5.2), A(5.6, -5.0), A(8.0, -2.6), A(8.5, 0.8)),
        _bez(A(8.5, 0.8), A(9.2, 5.0), A(8.2, 9.6), A(5.3, 10.0)),
        _bez(A(5.3, 10.0), A(3.2, 10.2), A(1.6, 7.0), A(0.9, 3.4)),
        _bez(A(0.9, 3.4), A(0.3, 0.4), A(0.4, -3.4), A(1.6, -5.2)),
    ])
    c = lv_out.mean(0)
    lv_in = c + (lv_out - c) * np.array([0.50, 0.62])
    lv_in = lv_in + np.array([0.35 * s, 0.6 * s])
    t = np.linspace(np.pi, 0, 70)
    ac = A(2.2, -8.6); rx, ry = 3.5 * s, 2.7 * s
    arch = np.column_stack([ac[0] + rx * np.cos(t), ac[1] - ry * np.sin(t)])
    aorta = np.vstack([A(-1.3, -4.4), arch, A(5.7, -3.0)])
    branches = []
    for ang in (np.pi * 0.66, np.pi * 0.52, np.pi * 0.38):
        p0 = np.array([ac[0] + rx * np.cos(ang), ac[1] - ry * np.sin(ang)])
        d = np.array([np.cos(ang) * 0.35, -1.0]); d = d / np.linalg.norm(d)
        branches.append(np.vstack([p0, p0 + d * 2.1 * s]))
    return dict(body=body, lv_out=lv_out, lv_in=lv_in, aorta=aorta, branches=branches, scale=s)


def mm_to_pt(ax, L):
    """Convert a length in data units (mm canvas) to points."""
    p0 = ax.transData.transform((0, 0)); p1 = ax.transData.transform((L, 0))
    return abs(p1[0] - p0[0]) * 72 / ax.figure.dpi


def draw_heart_mm(ax, cx, cy, w, col_body="#F7E6DC", col_myo="#E39A70", col_cav="#FFFFFF", col_ao="#EDB896",
                  outline="#9A6A50", lw=0.45, ao_lw_mm=1.5, zorder=3):
    P = heart_parts(cx, cy, w)
    lwpt = mm_to_pt(ax, ao_lw_mm); olw = mm_to_pt(ax, lw / 72 * 25.4) if False else lw
    for br in P["branches"]:
        ax.plot(br[:, 0], br[:, 1], color=outline, lw=lwpt * 0.42 + 2 * olw, solid_capstyle="round", zorder=zorder)
        ax.plot(br[:, 0], br[:, 1], color=col_ao, lw=lwpt * 0.42, solid_capstyle="round", zorder=zorder + 0.05)
    ax.plot(P["aorta"][:, 0], P["aorta"][:, 1], color=outline, lw=lwpt + 2 * olw, solid_capstyle="butt", zorder=zorder + 0.06)
    ax.plot(P["aorta"][:, 0], P["aorta"][:, 1], color=col_ao, lw=lwpt, solid_capstyle="butt", zorder=zorder + 0.1)
    ax.add_patch(Polygon(P["body"], closed=True, fc=col_body, ec=outline, lw=lw, zorder=zorder + 0.2, joinstyle="round"))
    ax.add_patch(Polygon(P["lv_out"], closed=True, fc=col_myo, ec=outline, lw=lw * 0.8, zorder=zorder + 0.3, joinstyle="round"))
    ax.add_patch(Polygon(P["lv_in"], closed=True, fc=col_cav, ec=outline, lw=lw * 0.6, zorder=zorder + 0.4, joinstyle="round"))
    return P


def heart_anchor(P, part):
    if part == "aorta":
        a = P["aorta"]; return a[int(len(a) * 0.62)]
    if part == "lv_in":
        return P["lv_in"].mean(0)
    if part == "lv_out":
        o = P["lv_out"]; i = P["lv_in"]
        return 0.5 * (o[np.argmax(o[:, 0])] + i[np.argmax(i[:, 0])])
    if part == "body":
        return P["body"][np.argmin(P["body"][:, 0])] * 0.35 + P["body"].mean(0) * 0.65
    return P[part].mean(0)


BRAIN_PART = {"Hippocampal_atrophy_Z": "hippocampus", "Cortical_thinning_Z": "cortex", "WMH_burden_Z": "wm_pv",
              "FreeWater_ISOVF_Z": "wm_deep"}
HEART_PART = {"LV_remodeling_Z": "lv_in", "LV_myocardial_dysfunction_Z": "lv_out", "Aortic_stiffness_Z": "aorta"}


def draw_brain_mm(ax, cx, cy, width, colors, y_mm=-20, zorder=3):
    base = {"cortex": "#E6E1EF", "wm_deep": "#F3F1F7", "wm_pv": "#ECE8F3", "subcortical": "#E9E7EC",
            "ventricle": "#FFFFFF", "hippocampus": "#DCD5EA"}
    base.update(colors)
    return BS.draw_slice_mm(ax, base, cx, cy, width, y_mm=y_mm, zorder=zorder)


def brain_anchor(part, cx, cy, width, y_mm=-20):
    return BS.anchor_mm(part, cx, cy, width, y_mm=y_mm)


def brain_aspect(y_mm=-20):
    return BS.slice_aspect(y_mm)


# ------------------------------------------------------------------ cardiac MRI short-axis glyph (preferred)
def _circle(cx, cy, r, n=160):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def heart_sax_parts(cx, cy, w):
    """Short-axis cardiac MRI schematic in a w-wide box: LV myocardium ring, LV cavity, RV crescent, aortic ring."""
    from shapely.geometry import Point
    R = 0.30 * w; r = 0.185 * w
    lvc = (cx + 0.06 * w, cy + 0.05 * w)
    lv_epi = Point(*lvc).buffer(R, 128)
    rv = Point(lvc[0] - 0.13 * w, lvc[1] - 0.01 * w).buffer(0.345 * w, 128).difference(lv_epi.buffer(0.004 * w, 64))
    if rv.geom_type == "MultiPolygon":
        rv = max(rv.geoms, key=lambda g: g.area)
    aoc = (cx + 0.36 * w, cy - 0.33 * w)
    return dict(lv_epi=_circle(*lvc, R), lv_endo=_circle(*lvc, r), lvc=np.array(lvc), R=R, r=r,
                rv=np.asarray(rv.exterior.coords), ao_out=_circle(*aoc, 0.105 * w), ao_in=_circle(*aoc, 0.068 * w),
                aoc=np.array(aoc))


def draw_heart_sax_mm(ax, cx, cy, w, col_myo="#F2B795", col_cav="#FFFFFF", col_rv="#F7E6DC", col_ao="#F2B795",
                      col_ao_lumen="#FFFFFF", outline="#9A6A50", lw=0.45, zorder=3, bg="#FFFFFF"):
    P = heart_sax_parts(cx, cy, w)
    ax.add_patch(Polygon(P["rv"], closed=True, fc=col_rv, ec=outline, lw=lw * 0.8, zorder=zorder, joinstyle="round"))
    ax.add_patch(Polygon(P["lv_epi"], closed=True, fc=col_myo, ec=outline, lw=lw, zorder=zorder + 0.1))
    ax.add_patch(Polygon(P["lv_endo"], closed=True, fc=col_cav, ec=outline, lw=lw * 0.7, zorder=zorder + 0.2))
    ax.add_patch(Polygon(P["ao_out"], closed=True, fc=col_ao, ec=outline, lw=lw, zorder=zorder + 0.1))
    ax.add_patch(Polygon(P["ao_in"], closed=True, fc=col_ao_lumen, ec=outline, lw=lw * 0.7, zorder=zorder + 0.2))
    return P


def heart_sax_anchor(P, part):
    if part == "lv_in":
        return P["lvc"]
    if part == "lv_out":
        a = np.deg2rad(-35)
        rr = 0.5 * (P["R"] + P["r"])
        return P["lvc"] + rr * np.array([np.cos(a), -np.sin(a)])
    if part == "aorta":
        return P["aoc"]
    if part == "rv":
        return P["rv"].mean(0)
