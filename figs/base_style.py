"""Shared style for Figures 2, 3, 5 and 6 (Nature Communications spec; consistent with Figure 1 v2)."""
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

MM = 1 / 25.4

# ------------------------------------------------------------------ palettes
DOMAIN = {"Protein": "#2F78B7", "Cardiac": "#D9692F", "Brain": "#6E4FB0", "Cognition": "#3C3F46"}
DOMAIN_LIGHT = {"Protein": "#E3EEF8", "Cardiac": "#FBE9DF", "Brain": "#ECE7F6", "Cognition": "#ECEDEF"}
DOMAIN_MID = {"Protein": "#A9CBE8", "Cardiac": "#F2B795", "Brain": "#C3B5E3", "Cognition": "#C4C7CD"}

MODEL_ORDER = ["All-10", "MRI + M13", "MRI matched"]
MODEL_COLOR = {"All-10": "#1B6B52", "MRI + M13": "#4FA284", "MRI matched": "#98CDB7"}
MODEL_EDGE = {"All-10": "#0F4A38", "MRI + M13": "#2F7A60", "MRI matched": "#4E8E76"}
MODEL_MARKER = {"All-10": "o", "MRI + M13": "s", "MRI matched": "D"}
MODEL_ALIAS = {"All-10": "All-10", "MRI+M13": "MRI + M13", "MRI + M13": "MRI + M13",
               "MRI matched": "MRI matched", "Matched MRI": "MRI matched"}

ADJ_COLOR = {"Core": "#2C3E50", "Extended": "#8B98AB"}
PRED_COLOR = {"Clinical": "#9AA0A8", "Stage": "#2E9C76", "Stage + subtype": "#0B3D2E"}

POS = "#B2182B"
NEG = "#2166AC"
INK = "#222222"
GREY = "#6B7078"
LIGHT = "#A9AEB6"
RULE = "#D9DCE1"
BAND = "#F4F5F7"
NA_FILL = "#EEEFF2"

RDBU = LinearSegmentedColormap.from_list(
    "rdbu_nc", ["#1D4E89", "#4A86BF", "#9CC3E0", "#E3EEF6", "#F7F7F7", "#F8E3D8", "#EBA487", "#C9503E", "#8E1B24"])


def set_style():
    plt.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6,
        "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.35, "ytick.minor.width": 0.35,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
        "lines.linewidth": 0.9, "lines.solid_capstyle": "round",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150, "mathtext.default": "regular",
        "legend.frameon": False, "axes.unicode_minus": True, "hatch.linewidth": 0.4,
    })


class MMFig:
    """Figure whose axes are positioned in millimetres from the top-left corner."""

    def __init__(self, w_mm, h_mm):
        self.W, self.H = w_mm, h_mm
        self.fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))

    def add(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def canvas(self, x=0, y=0, w=None, h=None, zorder=0):
        w = self.W if w is None else w
        h = self.H if h is None else h
        ax = self.add(x, y, w, h)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_zorder(zorder)
        ax.patch.set_alpha(0)
        return ax

    def save(self, stem, png_dpi=600):
        self.fig.savefig(f"{stem}.pdf", facecolor="white")
        self.fig.savefig(f"{stem}.svg", facecolor="white")
        self.fig.savefig(f"{stem}.png", dpi=png_dpi, facecolor="white")
        plt.close(self.fig)


def panel_letter(ax, x, y, letter, title=None, title_dx=4.2):
    ax.text(x, y, letter.upper(), fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
    if title:
        ax.text(x + title_dx, y + 0.35, title, fontsize=6.5, fontweight="bold", va="top", ha="left", color=INK)


def text_w_mm(t, size, weight="normal", style="normal"):
    fp = FontProperties(family=mpl.rcParams["font.sans-serif"], size=size, weight=weight, style=style)
    ext = TextPath((0, 0), t, prop=fp).get_extents()
    return ext.width * 25.4 / 72


def sci(p, prefix="q", digits=2, thresh=1e-3):
    """Format a P/q value: q = 0.012 or q = 1.2 × 10⁻⁵ (mathtext)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    lead = f"{prefix} = " if prefix else ""
    if p >= thresh:
        if p >= 0.1:
            return f"{lead}{p:.2f}"
        return f"{lead}{p:.3f}" if p < 0.01 else f"{lead}{p:.3f}".rstrip("0")
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    return f"{lead}{m:.{digits - 1}f}$\\times$10$^{{{e}}}$"


def despine(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def lighten(hex_color, f=0.5):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def div_norm(vmax):
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def text_on(color_rgba):
    """Black or white text depending on background luminance."""
    r, g, b = color_rgba[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK
