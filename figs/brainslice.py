"""Vector rendering of a schematic-yet-anatomical coronal brain slice.

Anatomy: Harvard-Oxford subcortical probabilistic atlas (FSL, MNI152 1 mm).
Regions mapped to SuStaIn brain-MRI biomarkers:
    cortex                      -> Cortical thinning
    hippocampus                 -> Hippocampal atrophy
    periventricular white matter-> WMH burden
    deep white matter           -> Free-water / ISOVF
"""
import os
import pickle
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
from matplotlib.patches import Polygon

CACHE = os.path.join(os.path.dirname(__file__), "_coronal_cache.pkl")

# label ids in HO subcortical atlas (0 = background in maxprob; prob volume is 0-based w/o bg)
HO = {
    "L_WM": 1, "L_CTX": 2, "L_LV": 3, "L_THA": 4, "L_CAU": 5, "L_PUT": 6, "L_PAL": 7,
    "BS": 8, "L_HIP": 9, "L_AMY": 10, "L_ACC": 11, "R_WM": 12, "R_CTX": 13, "R_LV": 14,
    "R_THA": 15, "R_CAU": 16, "R_PUT": 17, "R_PAL": 18, "R_HIP": 19, "R_AMY": 20, "R_ACC": 21,
}

CLASSES = ["bg", "cortex", "wm_deep", "wm_pv", "ventricle", "hippocampus", "subcortical", "brainstem"]


def _build(y_mm=-24, up=4, pv_mm=6.0):
    from nilearn import datasets
    prob = datasets.fetch_atlas_harvard_oxford("sub-prob-1mm", verbose=0)
    img = prob.maps if hasattr(prob.maps, "get_fdata") else nib.load(prob.maps)
    P = np.asarray(img.get_fdata(), dtype=float) / 100.0  # x, y, z, label(1..21 -> idx 0..20)
    j = int(round(y_mm + 126))
    S = P[:, j, :, :]  # x, z, labels
    S = np.transpose(S, (1, 0, 2))  # z (rows), x (cols), labels

    def g(*names):
        return np.sum([S[..., HO[n] - 1] for n in names], axis=0)

    cls = {
        "cortex": g("L_CTX", "R_CTX"),
        "wm": g("L_WM", "R_WM"),
        "ventricle": g("L_LV", "R_LV"),
        "hippocampus": g("L_HIP", "R_HIP") * 1.25,
        "subcortical": g("L_THA", "R_THA", "L_CAU", "R_CAU", "L_PUT", "R_PUT", "L_PAL", "R_PAL",
                         "L_AMY", "R_AMY", "L_ACC", "R_ACC"),
    }
    total = np.sum(list(cls.values()), axis=0)
    # upsample + smooth for crisp, anatomical contours
    zoom = lambda a: ndi.gaussian_filter(ndi.zoom(a, up, order=1), sigma=up * 0.55)
    Z = {k: zoom(v) for k, v in cls.items()}
    T = zoom(total)
    names = list(Z.keys())
    stack = np.stack([Z[k] for k in names], 0)
    lab = np.argmax(stack, 0) + 1
    lab[T < 0.45] = 0
    # split WM into periventricular (WMH) vs deep (free water)
    vent = lab == (names.index("ventricle") + 1)
    dist = ndi.distance_transform_edt(~vent) / up  # mm
    wm = lab == (names.index("wm") + 1)
    out = np.zeros_like(lab)
    m = {n: lab == (names.index(n) + 1) for n in names}
    out[m["cortex"]] = CLASSES.index("cortex")
    out[wm & (dist > pv_mm)] = CLASSES.index("wm_deep")
    out[wm & (dist <= pv_mm)] = CLASSES.index("wm_pv")
    out[m["ventricle"]] = CLASSES.index("ventricle")
    out[m["hippocampus"]] = CLASSES.index("hippocampus")
    out[m["subcortical"]] = CLASSES.index("subcortical")
    # internal holes (3rd ventricle / unlabeled CSF) -> CSF class
    filled = ndi.binary_fill_holes(out > 0)
    out[filled & (out == 0)] = CLASSES.index("ventricle")
    # remove tiny islands
    lab_cc, ncc = ndi.label(out > 0)
    if ncc > 1:
        sizes = ndi.sum(np.ones_like(lab_cc), lab_cc, index=np.arange(1, ncc + 1))
        keep = np.isin(lab_cc, 1 + np.where(sizes > 0.02 * sizes.max())[0])
        out[~keep] = 0
    # crop to brain bounding box
    rows, cols = np.where(out > 0)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    pad = 2 * up
    out = out[max(r0 - pad, 0): r1 + pad, max(c0 - pad, 0): c1 + pad]
    # polygons per class
    polys = {}
    for ci, cname in enumerate(CLASSES):
        if ci == 0:
            continue
        mask = ndi.binary_fill_holes(out == ci) if cname in ("hippocampus",) else (out == ci)
        mask = ndi.binary_opening(mask, iterations=1)
        cs = measure.find_contours(np.pad(mask.astype(float), 1), 0.5)
        keep = []
        for c in cs:
            c = c - 1
            if len(c) < 12:
                continue
            area = 0.5 * np.abs(np.dot(c[:, 0], np.roll(c[:, 1], 1)) - np.dot(c[:, 1], np.roll(c[:, 0], 1)))
            if area < 6 * up * up:
                continue
            keep.append(c[:, ::-1].copy())  # (x, y) = (col, row)
        polys[cname] = keep
    outline = measure.find_contours(np.pad((out > 0).astype(float), 1), 0.5)
    outline = [o[:, ::-1] - 1 for o in outline if len(o) > 50]
    return {"label": out, "polys": polys, "outline": outline, "shape": out.shape, "up": up}


def get_slice(y_mm=-24, rebuild=False):
    key = f"y{y_mm}"
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            cache = pickle.load(fh)
    if rebuild or key not in cache:
        cache[key] = _build(y_mm)
        with open(CACHE, "wb") as fh:
            pickle.dump(cache, fh)
    return cache[key]


def draw_slice(ax, colors, y_mm=-24, edge="white", lw=0.35, outline_color="#8C8C96", outline_lw=0.45,
               zorder=1):
    """colors: dict class->facecolor for cortex, wm_deep, wm_pv, ventricle, hippocampus, subcortical, brainstem"""
    sl = get_slice(y_mm)
    H, W = sl["shape"]
    order = ["cortex", "wm_deep", "wm_pv", "subcortical", "brainstem", "ventricle", "hippocampus"]
    for cname in order:
        for poly in sl["polys"].get(cname, []):
            ax.add_patch(Polygon(poly, closed=True, facecolor=colors[cname], edgecolor=colors[cname],
                                 lw=0.25, joinstyle="round", zorder=zorder))
    # thin white separators drawn on top for a refined look
    for cname in ["wm_deep", "wm_pv", "ventricle", "hippocampus", "subcortical", "brainstem"]:
        for poly in sl["polys"].get(cname, []):
            ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=edge, lw=lw,
                                 joinstyle="round", zorder=zorder + 1))
    for o in sl["outline"]:
        ax.plot(o[:, 0], o[:, 1], color=outline_color, lw=outline_lw, solid_joinstyle="round",
                zorder=zorder + 2)
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(-2, H + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    return sl


# ----------------------------------------------------------------------------------------------
# Drawing on a millimetre canvas (y axis pointing down), used for the composite figure
# ----------------------------------------------------------------------------------------------
def slice_aspect(y_mm=-20):
    H, W = get_slice(y_mm)["shape"]
    return H / W


def _tf(poly, cx, cy, width, H, W):
    s = width / W
    x = cx - width / 2 + poly[:, 0] * s
    y = cy + (H * s) / 2 - poly[:, 1] * s
    return np.column_stack([x, y])


def draw_slice_mm(ax, colors, cx, cy, width, y_mm=-20, sep_color="white", sep_lw=0.3,
                  vent_edge="#B4B6C0", outline_color="#8E9099", outline_lw=0.4, zorder=3):
    sl = get_slice(y_mm)
    H, W = sl["shape"]
    order = ["cortex", "wm_deep", "wm_pv", "subcortical", "ventricle", "hippocampus"]
    for cname in order:
        for poly in sl["polys"].get(cname, []):
            P = _tf(poly, cx, cy, width, H, W)
            ax.add_patch(Polygon(P, closed=True, facecolor=colors[cname], edgecolor=colors[cname],
                                 lw=0.2, joinstyle="round", zorder=zorder))
    for cname in ["wm_deep", "wm_pv", "subcortical", "hippocampus"]:
        for poly in sl["polys"].get(cname, []):
            P = _tf(poly, cx, cy, width, H, W)
            ax.add_patch(Polygon(P, closed=True, facecolor="none", edgecolor=sep_color, lw=sep_lw,
                                 joinstyle="round", zorder=zorder + 1))
    for poly in sl["polys"].get("ventricle", []):
        P = _tf(poly, cx, cy, width, H, W)
        ax.add_patch(Polygon(P, closed=True, facecolor="none", edgecolor=vent_edge, lw=0.25,
                             joinstyle="round", zorder=zorder + 2))
    for o in sl["outline"]:
        P = _tf(o, cx, cy, width, H, W)
        ax.plot(P[:, 0], P[:, 1], color=outline_color, lw=outline_lw, solid_joinstyle="round",
                zorder=zorder + 3)
    return width * H / W


def anchor_points(y_mm=-20):
    """Interior representative point (pixel coords x, y-up) of each class in the left hemisphere
    of the image (radiological right is irrelevant for a schematic key)."""
    sl = get_slice(y_mm)
    lab = sl["label"]
    H, W = lab.shape
    pts = {}
    for cname in ["cortex", "wm_deep", "wm_pv", "hippocampus"]:
        m = lab == CLASSES.index(cname)
        m[:, W // 2:] = False
        if cname == "cortex":  # prefer superolateral convexity
            m[: int(H * 0.55), :] = False
        if cname == "wm_deep":
            m[: int(H * 0.45), :] = False
        dt = ndi.distance_transform_edt(m)
        r, c = np.unravel_index(np.argmax(dt), dt.shape)
        pts[cname] = (c, r)
    return pts, (H, W)


def anchor_mm(cname, cx, cy, width, y_mm=-20):
    pts, (H, W) = anchor_points(y_mm)
    c, r = pts[cname]
    P = _tf(np.array([[c, r]], dtype=float), cx, cy, width, H, W)
    return P[0, 0], P[0, 1]
