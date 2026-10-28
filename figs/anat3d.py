"""Anatomical rendering toolkit for v5 figures (heart-brain regional atlas).

All drawing functions take a millimetre canvas axis (y downward, as produced by
base_style.MMFig.canvas) plus a centre and a width in millimetres.

Assets (figs/assets):
  fsa5_lh/rh.gii, aparc_fsa5.csv, dk68_order.txt, sctx_lh/rh.gii  - ENIGMA Toolbox (BSD-3)
  JHU-ICBM-labels-1mm, JHU-ICBM-FA-1mm, JHU-labels.xml            - FSL atlases (JHU DTI-81)
"""
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib as mpl
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import Wedge, FancyBboxPatch, Circle

ASSET = Path(__file__).resolve().parent / "assets"
NA_COLOR = "#E4E6EA"
EDGE = "#FFFFFF"

# ----------------------------------------------------------------- colour maps
ZMAP = LinearSegmentedColormap.from_list(
    "hb_div", ["#1D4F73", "#2C6E9B", "#8DB8D3", "#F6F4F1", "#F2C27A", "#E9A23B", "#C8462B", "#7A1F2B"], N=512)
ZMAP.set_bad(NA_COLOR)


def znorm(vmin, vmax):
    return TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)


def divmap(vmin, vmax, n=512):
    """Diverging map truncated from a symmetric [-M, M] range (M = max |limit|) with a linear norm, so that
    colour saturation is proportional to |value| and colour-bar ticks are evenly spaced."""
    M = max(abs(vmin), abs(vmax))
    xs = np.linspace(vmin, vmax, n)
    cm = ListedColormap(ZMAP((xs + M) / (2 * M)))
    cm.set_bad(NA_COLOR)
    return cm, Normalize(vmin, vmax)


def shade(rgb, inten, spec=0.0):
    rgb = np.asarray(rgb)[..., :3]
    out = rgb * inten[:, None] + spec[:, None] * (1 - rgb) * 0.55
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------- generic mesh renderer
def _rot(azim, elev):
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def project(verts, view):
    """view = 3x3 matrix mapping world -> camera (x right, y up, z towards viewer)."""
    return verts @ view.T


def render_mesh(ax, verts, faces, face_rgb, view, cx, cy, width, light=(-0.45, 0.55, 0.70),
                ambient=0.40, diffuse=0.62, specular=0.18, edge_pairs=None, edge_color=EDGE, edge_lw=0.12,
                zorder=3, height=None, outline=True, outline_color="#9AA0A8", outline_lw=0.25, alpha=1.0,
                return_xform=False, two_sided=False, smooth=True):
    P = project(verts, view)
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    span = hi - lo
    s = width / span[0]
    if height is not None:
        s = min(s, height / span[1])
    ctr = (lo + hi) / 2

    def to_mm(Q):
        return np.column_stack([cx + (Q[:, 0] - ctr[0]) * s, cy - (Q[:, 1] - ctr[1]) * s])

    tri = P[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    if two_sided:
        n = np.where(n[:, 2:3] < 0, -n, n)
    front = n[:, 2] > 0 if not two_sided else np.ones(len(n), bool)
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    H = L + np.array([0, 0, 1.0]); H /= np.linalg.norm(H)
    if smooth and not two_sided:
        vn = np.zeros_like(P)
        for k in range(3):
            np.add.at(vn, faces[:, k], n)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        lam = np.clip(vn @ L, 0, 1)[faces].mean(1)
        spec = (specular * np.clip(vn @ H, 0, 1) ** 24)[faces].mean(1)
    else:
        lam = np.clip(n @ L, 0, 1)
        spec = specular * np.clip(n @ H, 0, 1) ** 24
    inten = ambient + diffuse * lam
    depth = tri[:, :, 2].mean(1)
    order = np.argsort(depth)
    order = order[front[order]]
    rgb = shade(np.asarray(face_rgb)[order], inten[order], spec[order])
    polys = [to_mm(t[:, :2]) for t in tri[order]]
    pc = PolyCollection(polys, facecolors=np.column_stack([rgb, np.full(len(rgb), alpha)]),
                        edgecolors=np.column_stack([rgb, np.full(len(rgb), alpha)]), linewidths=0.05,
                        zorder=zorder, rasterized=True)
    ax.add_collection(pc)
    if edge_pairs is not None and len(edge_pairs):
        e = edge_pairs
        # keep borders on front-facing geometry only (both adjacent faces visible from the camera)
        segs = to_mm(P[e.ravel(), :2]).reshape(-1, 2, 2)
        zmid = P[e, 2].mean(1)
        keep = zmid >= np.percentile(P[:, 2], 2)
        ax.add_collection(LineCollection(segs[keep], colors=edge_color, linewidths=edge_lw, zorder=zorder + 0.1,
                                         rasterized=True, alpha=0.9))
    if return_xform:
        return to_mm, P
    return None


# ----------------------------------------------------------------- cortex (fsaverage5, DK)
@lru_cache(None)
def cortex(hemi):
    g = nib.load(ASSET / f"fsa5_{hemi}.gii")
    v = g.darrays[0].data.astype(float)
    f = g.darrays[1].data.astype(int)
    lab = np.loadtxt(ASSET / "aparc_fsa5.csv", dtype=int)
    nv = len(v)
    lab = lab[:nv] if hemi == "lh" else lab[nv:2 * nv]
    names = [s.strip() for s in open(ASSET / "dk68_order.txt").read().split("\n") if s.strip()]
    idx = list(range(1, 4)) + list(range(5, 39)) + list(range(40, 71))
    lab2name = {idx[i]: names[i] for i in range(68)}
    vname = np.array([lab2name.get(int(x), "") for x in lab], dtype=object)
    # region name without hemisphere prefix
    vreg = np.array([s[2:] if s else "" for s in vname], dtype=object)
    # face label = label of first vertex (fsaverage5 parcels are contiguous)
    flab = lab[f[:, 0]]
    # parcel border faces: faces sharing an edge with a face of a different label
    edges = {}
    for fi, tri in enumerate(f):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            k = (a, b) if a < b else (b, a)
            edges.setdefault(k, []).append(fi)
    border = np.zeros(len(f), bool)
    for k, fs in edges.items():
        if len(fs) == 2 and flab[fs[0]] != flab[fs[1]]:
            border[fs[0]] = border[fs[1]] = True
    return v, f, vreg, border


VIEWS = {  # camera matrices (world -> camera); fsaverage: x = left-right, y = posterior-anterior, z = inferior-superior
    ("lh", "lateral"): np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float),
    ("lh", "medial"): np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float),
    ("rh", "lateral"): np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float),
    ("rh", "medial"): np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float),
}


def draw_cortex(ax, hemi, view, values, cx, cy, width, cmap=ZMAP, norm=None, alpha_mask=None, zorder=3,
                base_color=None, border=True, height=None, border_mix=0.22):
    """values: dict region (e.g. 'superiorfrontal') -> value; missing -> NA colour.
    alpha_mask: dict region -> bool (False = desaturate)."""
    v, f, vreg, bface = cortex(hemi)
    freg = vreg[f[:, 0]]
    rgb = np.zeros((len(f), 3))
    for i, r in enumerate(freg):
        if base_color is not None:
            rgb[i] = to_rgb(base_color if r else NA_COLOR)
            continue
        val = values.get(r, np.nan) if r else np.nan
        if np.isfinite(val):
            c = np.array(cmap(norm(val))[:3])
            if alpha_mask is not None and not alpha_mask.get(r, True):
                c = c * 0.35 + np.array(to_rgb("#EEF0F2")) * 0.65
            rgb[i] = c
        else:
            rgb[i] = to_rgb(NA_COLOR)
    if border:
        rgb[bface] = rgb[bface] * (1 - border_mix) + border_mix
    return render_mesh(ax, v, f, rgb, VIEWS[(hemi, view)], cx, cy, width, zorder=zorder, height=height)


def cortex_aspect(hemi="lh", view="lateral"):
    v, *_ = cortex(hemi)
    P = project(v, VIEWS[(hemi, view)])
    span = P[:, :2].max(0) - P[:, :2].min(0)
    return span[1] / span[0]


# ----------------------------------------------------------------- hippocampus (ENIGMA subcortical mesh)
SCTX_N = [867, 1419, 3012, 3784, 1446, 4003, 3726, 7653]
SCTX_NAMES = ["accumbens", "amygdala", "caudate", "hippocampus", "pallidum", "putamen", "thalamus", "ventricle"]


@lru_cache(None)
def subcortical(hemi, structure="hippocampus"):
    g = nib.load(ASSET / f"sctx_{hemi}.gii")
    v = g.darrays[0].data.astype(float)
    f = g.darrays[1].data.astype(int)
    k = SCTX_NAMES.index(structure)
    start = int(np.sum(SCTX_N[:k]))
    stop = start + SCTX_N[k]
    keep = np.all((f >= start) & (f < stop), 1)
    return v, f[keep]


def draw_hippocampus(ax, hemi, value, cx, cy, width, cmap=ZMAP, norm=None, color=None, zorder=3, view="lateral"):
    v, f = subcortical(hemi)
    c = np.array(to_rgb(color)) if color is not None else (
        np.array(cmap(norm(value))[:3]) if np.isfinite(value) else np.array(to_rgb(NA_COLOR)))
    rgb = np.tile(c, (len(f), 1))
    return render_mesh(ax, v, f, rgb, VIEWS[(hemi, view)], cx, cy, width, zorder=zorder, specular=0.25)


# ----------------------------------------------------------------- 3D left ventricle with AHA segments
def _spheroid(a, c, phis, ts):
    """Prolate spheroid surface, apex at z=0, long axis +z. phi in [0, phimax].
    Circumferential angle runs clockwise when viewed from the base so that, seen from the apex,
    90 deg = anterior (up), 180 deg = septum (left), 0 deg = lateral (right), as in the AHA bull's-eye."""
    P, T = np.meshgrid(phis, ts, indexing="ij")
    r = a * np.sin(P)
    return np.stack([r * np.cos(T), -r * np.sin(T), c * (1 - np.cos(P))], -1)


def lv_mesh(nu=48, nt=120, a=0.50, c=1.40, phimax=0.58 * np.pi, wall=0.16):
    """Truncated prolate-spheroid LV: epicardial shell + basal myocardial annulus + cavity.
    Returns verts, faces, per-face AHA segment (1-17; 0 = annulus, -1 = cavity)."""
    phis = np.linspace(0.015, phimax, nu)
    ts = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    epi = _spheroid(a, c, phis, ts)
    zbase = epi[-1, 0, 2]
    zap = epi[0, 0, 2]
    V = [epi.reshape(-1, 3)]
    faces, seg = [], []
    for i in range(nu - 1):
        for j in range(nt):
            p, q = i * nt + j, i * nt + (j + 1) % nt
            r_, s_ = (i + 1) * nt + j, (i + 1) * nt + (j + 1) % nt
            faces += [[p, q, r_], [q, s_, r_]]
            zm = (epi[i, j, 2] + epi[i + 1, j, 2]) / 2
            lev = (zm - zap) / (zbase - zap)
            tm = np.rad2deg((ts[j] + np.pi / nt) % (2 * np.pi))
            s = aha_segment(lev, tm)
            seg += [s, s]
    # basal annulus (outer ring = last epi row, inner ring = endocardial radius)
    off = len(V[0])
    ro = epi[-1, :, :2]
    ri = ro * (1 - wall / a)
    ring_i = np.column_stack([ri, np.full(nt, zbase)])
    V.append(ring_i)
    o0 = (nu - 1) * nt
    for j in range(nt):
        p, q = o0 + j, o0 + (j + 1) % nt
        r_, s_ = off + j, off + (j + 1) % nt
        faces += [[p, r_, q], [q, r_, s_]]
        seg += [0, 0]
    # cavity: shallow endocardial cup seen through the base
    off2 = off + nt
    endo = _spheroid(a - wall, c - wall * 0.8, phis[::3], ts)
    endo[..., 2] += wall * 0.8
    endo[..., 2] = np.minimum(endo[..., 2], zbase)
    V.append(endo.reshape(-1, 3))
    m = endo.shape[0]
    for i in range(m - 1):
        for j in range(nt):
            p, q = off2 + i * nt + j, off2 + i * nt + (j + 1) % nt
            r_, s_ = off2 + (i + 1) * nt + j, off2 + (i + 1) * nt + (j + 1) % nt
            faces += [[p, r_, q], [q, r_, s_]]
            seg += [-1, -1]
    return np.concatenate(V), np.array(faces), np.array(seg)


def aha_segment(level, ang):
    """level: 0 apex .. 1 base; ang: degrees, 90 = anterior, 180 = septal (bull's-eye convention)."""
    ang = ang % 360
    if level < 0.10:
        return 17
    if level < 0.40:  # apical 13-16
        if 45 <= ang < 135:
            return 13
        if 135 <= ang < 225:
            return 14
        if 225 <= ang < 315:
            return 15
        return 16
    base = 7 if level < 0.70 else 1
    k = int(((ang - 60) % 360) // 60)  # 60-120 -> 0 (anterior) ... counterclockwise
    return base + k


def lv_view(azim=-15, elev=18, tilt=32):
    """World: LV long axis +z (apex -> base); anterior wall = -y, septum = -x, lateral = +x.
    Default camera: anterior view (septum on the viewer's left), slightly from above the base,
    rolled so that the apex points down and to the viewer's right as in situ."""
    t = np.deg2rad(tilt)
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    # map world (x, y, z) -> camera: camera-right = world x, camera-up = world z, towards viewer = -world y
    C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rr = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return Rr @ Rx @ C @ Rz


def draw_lv3d(ax, seg_values, cx, cy, width, cmap=ZMAP, norm=None, seg_colors=None, azim=-15, elev=18,
              zorder=3, border=True, tilt=32, annulus="#F3D9C8", cavity="#7A3B2E", height=None, return_base=False):
    """3D LV seen from anterolateral, base open (myocardial annulus + cavity); seg_values: dict 1..16 -> value."""
    v, f, seg = lv_mesh()
    view = lv_view(azim, elev, tilt)
    rgb = np.zeros((len(f), 3))
    for i, s in enumerate(seg):
        if s == 0:
            rgb[i] = to_rgb(annulus)
        elif s == 17:
            rgb[i] = to_rgb("#EADFD8")
        elif s == -1:
            rgb[i] = to_rgb(cavity)
        elif seg_colors is not None:
            rgb[i] = to_rgb(seg_colors.get(int(s), NA_COLOR))
        else:
            val = seg_values.get(int(s), np.nan)
            rgb[i] = cmap(norm(val))[:3] if np.isfinite(val) else to_rgb(NA_COLOR)
    if border:
        e = {}
        for fi, tri in enumerate(f):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                k = (a, b) if a < b else (b, a)
                e.setdefault(k, []).append(fi)
        bf = np.zeros(len(f), bool)
        for k, fs in e.items():
            if len(fs) == 2 and seg[fs[0]] != seg[fs[1]] and seg[fs[0]] > 0 and seg[fs[1]] > 0:
                bf[fs[0]] = bf[fs[1]] = True
        rgb[bf] = rgb[bf] * 0.62 + 0.38
    res = render_mesh(ax, v, f, rgb, view, cx, cy, width, zorder=zorder, ambient=0.44, diffuse=0.58,
                      specular=0.22, height=height, two_sided=True, return_xform=True)
    if return_base:
        to_mm, _ = res
        zb = v[:, 2].max()
        return to_mm(project(np.array([[0.0, 0.0, zb]]), view)[:, :2])[0]
    return None


# ----------------------------------------------------------------- AHA bull's-eye (16 segments)
def aha_wedges():
    """Return list of (segment, r_in, r_out, theta1, theta2) in unit radius."""
    out = []
    rings = [(1, 0.72, 1.00), (7, 0.44, 0.72)]
    for base, ri, ro in rings:
        for k in range(6):
            t1 = 60 + 60 * k
            out.append((base + k, ri, ro, t1, t1 + 60))
    for k, s in enumerate([13, 14, 15, 16]):
        t1 = 45 + 90 * k
        out.append((s, 0.18, 0.44, t1, t1 + 90))
    return out


def draw_bullseye(ax, cx, cy, r, values, cmap=ZMAP, norm=None, edge=EDGE, lw=0.35, zorder=3, apex_color="#EEF0F2",
                  mask=None, labels=False, fontsize=3.6):
    """values: dict seg(1-16) -> value. mm canvas with y downward: angles mirrored so anterior is at top."""
    for s, ri, ro, t1, t2 in aha_wedges():
        val = values.get(s, np.nan)
        c = cmap(norm(val)) if np.isfinite(val) else NA_COLOR
        if mask is not None and not mask.get(s, True) and np.isfinite(val):
            c = np.array(to_rgb(c)) * 0.35 + np.array(to_rgb("#EEF0F2")) * 0.65
        # y axis points down: use negative angles so that 90 deg (anterior) is at the top
        w = Wedge((cx, cy), ro * r, -t2, -t1, width=(ro - ri) * r, facecolor=c, edgecolor=edge, lw=lw, zorder=zorder)
        ax.add_patch(w)
        if labels:
            tm = np.deg2rad((t1 + t2) / 2)
            rm = (ri + ro) / 2 * r
            ax.text(cx + rm * np.cos(tm), cy - rm * np.sin(tm), str(s), fontsize=fontsize, ha="center", va="center",
                    color="#3A3D42", zorder=zorder + 1)
    ax.add_patch(Circle((cx, cy), 0.18 * r, facecolor=apex_color, edgecolor=edge, lw=lw, zorder=zorder))


# ----------------------------------------------------------------- JHU white-matter slices
@lru_cache(None)
def jhu():
    lab = nib.load(ASSET / "JHU-ICBM-labels-1mm.nii.gz")
    fa = nib.load(ASSET / "JHU-ICBM-FA-1mm.nii.gz")
    names = {}
    for e in ET.parse(ASSET / "JHU-labels.xml").getroot().iter("label"):
        names[int(e.get("index"))] = e.text.strip()
    return np.asarray(lab.dataobj).astype(int), np.asarray(fa.dataobj).astype(float), lab.affine, names


def _norm_name(s):
    s = s.lower().replace("+", " ").replace("/", " ")
    s = re.sub(r"\(.*?\)", lambda m: m.group(0), s)
    return re.sub(r"[^a-z]", "", s)


def jhu_label_for(region, hemi):
    """Map UKB TBSS tract name (e.g. 'Anterior_corona_radiata', hemi 'left') to a JHU label index."""
    _, _, _, names = jhu()
    key = _norm_name(region.replace("_", " "))
    suf = {"left": " L", "right": " R"}.get(hemi, "")
    best = None
    for idx, nm in names.items():
        if idx == 0:
            continue
        if suf and not nm.endswith(suf):
            continue
        if not suf and (nm.endswith(" L") or nm.endswith(" R")):
            continue
        base = nm[:-2] if suf else nm
        nb = _norm_name(base)
        if nb.startswith(key[:14]) or key.startswith(nb[:14]):
            if best is None or abs(len(nb) - len(key)) < best[1]:
                best = (idx, abs(len(nb) - len(key)))
    return None if best is None else best[0]


def draw_wm_slice(ax, z_mni, values, cx, cy, width, cmap=ZMAP, norm=None, mask=None, zorder=3, crop=8,
                  outline=True):
    """values: dict JHU label index -> value. Neurological orientation (left on left), anterior up."""
    lab, fa, aff, _ = jhu()
    k = int(round(np.linalg.solve(aff, [0, 0, z_mni, 1])[2]))
    L = lab[:, :, k].T[::-1, :]          # rows = y (anterior top), cols = x
    F = fa[:, :, k].T[::-1, :]
    if aff[0, 0] < 0:                     # voxel x increases towards subject's left -> flip for neurological view
        L, F = L[:, ::-1], F[:, ::-1]
    brain = F > 0.02
    ys, xs = np.where(brain)
    y0, y1, x0, x1 = ys.min() - crop, ys.max() + crop, xs.min() - crop, xs.max() + crop
    L, F, brain = L[y0:y1, x0:x1], F[y0:y1, x0:x1], brain[y0:y1, x0:x1]
    rgba = np.ones(L.shape + (4,))
    g = np.clip(0.94 - 0.50 * np.clip(F / 0.65, 0, 1) ** 0.8, 0, 1)
    rgba[..., :3] = np.where(brain[..., None], np.stack([g, g, g], -1) * np.array([1, 1, 1.02]), 1)
    rgba[..., 3] = 1.0
    # brain outline
    eb = np.zeros_like(brain)
    eb[1:, :] |= brain[1:, :] != brain[:-1, :]
    eb[:, 1:] |= brain[:, 1:] != brain[:, :-1]
    rgba[eb, :3] = 0.72
    for idx in np.unique(L):
        if idx == 0:
            continue
        m = L == idx
        val = values.get(int(idx), np.nan)
        if np.isfinite(val):
            c = np.array(cmap(norm(val))[:3])
            if mask is not None and not mask.get(int(idx), True):
                c = c * 0.35 + np.array(to_rgb("#EEF0F2")) * 0.65
        else:
            c = np.array(to_rgb("#D3D6DB"))
        rgba[m, :3] = c
    if outline:
        e = np.zeros_like(L, bool)
        e[1:, :] |= (L[1:, :] != L[:-1, :]) & ((L[1:, :] > 0) | (L[:-1, :] > 0))
        e[:, 1:] |= (L[:, 1:] != L[:, :-1]) & ((L[:, 1:] > 0) | (L[:, :-1] > 0))
        rgba[e, :3] = rgba[e, :3] * 0.55 + 0.45
    h = width * L.shape[0] / L.shape[1]
    ax.imshow(rgba, extent=(cx - width / 2, cx + width / 2, cy + h / 2, cy - h / 2), interpolation="nearest",
              zorder=zorder, rasterized=True)
    return h


def wm_slice_aspect(z_mni, crop=8):
    lab, fa, aff, _ = jhu()
    k = int(round(np.linalg.solve(aff, [0, 0, z_mni, 1])[2]))
    F = fa[:, :, k].T
    ys, xs = np.where(F > 0.02)
    return (ys.max() - ys.min() + 2 * crop) / (xs.max() - xs.min() + 2 * crop)


def wm_label_xy(z_mni, idx, cx, cy, width, crop=8):
    """Millimetre-canvas centroid of JHU label `idx` on the slice drawn by draw_wm_slice with the same args."""
    lab, fa, aff, _ = jhu()
    k = int(round(np.linalg.solve(aff, [0, 0, z_mni, 1])[2]))
    L = lab[:, :, k].T[::-1, :]
    F = fa[:, :, k].T[::-1, :]
    if aff[0, 0] < 0:
        L, F = L[:, ::-1], F[:, ::-1]
    ys, xs = np.where(F > 0.02)
    y0, y1, x0, x1 = ys.min() - crop, ys.max() + crop, xs.min() - crop, xs.max() + crop
    L = L[y0:y1, x0:x1]
    rr, cc = np.where(L == idx)
    if len(rr) == 0:
        return None
    h = width * L.shape[0] / L.shape[1]
    return (cx - width / 2 + (cc.mean() + 0.5) * width / L.shape[1],
            cy - h / 2 + (rr.mean() + 0.5) * h / L.shape[0])


# ----------------------------------------------------------------- value chip (single global measures)
def chip(ax, x, y, w, h, value, label, cmap=ZMAP, norm=None, fontsize=4.6, zorder=4, txt_value=True, nd=2):
    c = cmap(norm(value)) if np.isfinite(value) else NA_COLOR
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%.2f" % (h * 0.45),
                                facecolor=c, edgecolor="white", lw=0.4, zorder=zorder))
    lum = np.dot(to_rgb(c), [0.299, 0.587, 0.114])
    col = "white" if lum < 0.55 else "#1F2226"
    t = label
    if txt_value and np.isfinite(value):
        t = f"{label} {value:.{nd}f}".replace(".", "\u00b7").replace("-", "\u2212")
    ax.text(x + w / 2, y + h / 2, t, fontsize=fontsize, ha="center", va="center", color=col, zorder=zorder + 1)


def colorbar_mm(fig_mm, x, y, w, h, cmap, norm, ticks, label, orientation="horizontal", fontsize=5.2, nd=None):
    ax = fig_mm.add(x, y, w, h)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(0.3)
    ax.tick_params(labelsize=fontsize - 0.4, width=0.3, length=1.4, pad=1)
    tl = [(f"{t:.{nd}f}" if (nd is not None and t != 0) else f"{t:g}").replace(".", "\u00b7").replace("-", "\u2212") for t in ticks]
    if orientation == "horizontal":
        ax.set_xticklabels(tl)
        ax.set_xlabel(label, fontsize=fontsize, labelpad=1.2)
    else:
        ax.set_yticklabels(tl)
        ax.set_ylabel(label, fontsize=fontsize, labelpad=1.2)
    return ax


def draw_vessel(ax, x, y, w, h, color=None, zorder=3, n=16, rim="#7A3B2E"):
    """Shaded cylinder icon for an arterial segment (open lumen at top); (x, y) = top centre in mm (y down).
    color=None gives the default aortic orange; otherwise shading is derived from `color`."""
    from matplotlib.patches import Ellipse, Rectangle
    if color is None:
        lo, hi = np.array(to_rgb("#B9542A")), np.array(to_rgb("#F6C9AE"))
    else:
        c = np.array(to_rgb(color))
        lo, hi = c * 0.72, c + (1 - c) * 0.55
    for k in range(n):
        t = k / (n - 1)
        prof = np.clip(1 - abs(t - 0.38) / 0.62, 0, 1) ** 0.8
        ax.add_patch(Rectangle((x - w / 2 + k * w / n, y), w / n + 0.02, h, fc=lo + (hi - lo) * prof, ec="none",
                               zorder=zorder))
    ax.add_patch(Ellipse((x, y + h), w, w * 0.36, fc=lo * 0.85 + 0.15 * hi, ec="none", zorder=zorder - 0.1))
    ax.add_patch(Ellipse((x, y), w, w * 0.36, fc=hi * 0.9 + 0.1, ec=lo * 0.8, lw=0.3, zorder=zorder + 0.1))
    ax.add_patch(Ellipse((x, y), w * 0.62, w * 0.20, fc=lo * 0.75, ec="none", zorder=zorder + 0.2))
