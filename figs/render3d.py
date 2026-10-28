"""Off-screen PyVista renders of the heart template and brain surfaces, returned as cropped RGBA arrays
and placed on millimetre canvases (base_style.MMFig).  v6.

Heart: figs/assets/heart/heart_surface.vtp + vessels.vtp (built by heart_assets.py from the CC BY 4.0
Rodero et al. 2021 average four-chamber mesh).  Brain: ENIGMA fsaverage5 pial surfaces + nilearn sulc.
Renders are cached as PNG in figs/cache3d keyed by a hash of all arguments (delete the folder to refresh).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

pv.OFF_SCREEN = True
HERE = Path(__file__).resolve().parent
ASSET = HERE / "assets"
CACHE = HERE / "cache3d"
CACHE.mkdir(exist_ok=True)
RENDER_VERSION = "v7.0"

# ------------------------------------------------------------------ palettes
ANAT = {  # anatomical illustration colours (figure 1)
    1: "#B9483F", 2: "#C9675C", 3: "#D68A7E", 4: "#D68A7E", 5: "#B23A33", 6: "#5873A8",
    "valve": "#D9A196", "la_in": "#B23A33", "ra_in": "#5873A8",
    "asc": "#B23A33", "arch": "#B23A33", "desc": "#B23A33", "branch": "#B23A33", "pa": "#5873A8",
    "svc": "#5873A8", "ivc": "#5873A8"}
NEUTRAL = "#E4DFD9"      # unmeasured anatomy in data figures
NEUTRAL_VES = "#EAE6E1"
VIEWS = {  # camera direction (focus -> camera) and view-up, mesh axes +x left, +y posterior, +z superior
    "anterior": ((0.0, -1.0, 0.08), (0, 0, 1)),
    "lao": ((0.62, -0.78, 0.12), (0, 0, 1)),
    "lao_steep": ((0.85, -0.50, 0.15), (0, 0, 1)),
    "lateral": ((1.0, -0.12, 0.05), (0, 0, 1)),
    "inferior": ((0.25, -0.35, -0.90), (0, -1, 0)),
    "posterior": ((0.1, 1.0, 0.10), (0, 0, 1)),
    "rao": ((-0.55, -0.83, 0.10), (0, 0, 1)),
}
PART = {1: "asc", 2: "arch", 3: "desc", 4: "branch", 5: "pa", 6: "svc", 7: "ivc"}
_MESH = {}


def _hex(c):
    from matplotlib.colors import to_rgb
    return np.asarray(to_rgb(c), float)


def _load(name):
    if name not in _MESH:
        _MESH[name] = pv.read(ASSET / "heart" / f"{name}.vtp") if name in ("heart_surface", "vessels") else None
    return _MESH[name]


def _key(**kw):
    s = json.dumps(kw, sort_keys=True, default=lambda o: np.asarray(o).round(5).tolist())
    return hashlib.md5((RENDER_VERSION + s).encode()).hexdigest()[:16]


def _crop(img, pad=4):
    a = img[..., 3]
    ys, xs = np.where(a > 8)
    if len(ys) == 0:
        return img
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def _plotter(px):
    p = pv.Plotter(window_size=(px, px), off_screen=True, lighting="none")
    p.set_background("white")
    # studio three-point lighting (camera-relative)
    for pos, inten in [((-0.6, 0.9, 1.0), 0.75), ((0.9, 0.2, 0.7), 0.35), ((0.2, -0.6, -0.5), 0.18), ((0.0, 0.0, 1.0), 0.25)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), intensity=inten)
        lt.positional = False
        lt.set_camera_light()
        p.add_light(lt)
    p.enable_anti_aliasing("ssaa")
    return p


def _shoot(p, center, view, zoom=1.0, dist=None, views=None):
    d, up = (views or VIEWS)[view] if isinstance(view, str) else view
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    b = np.asarray(p.bounds, float).reshape(3, 2)
    c = b.mean(1) if center is None else np.asarray(center, float)
    R = 0.5 * np.linalg.norm(b[:, 1] - b[:, 0])
    dist = R / np.sin(np.radians(11)) * 1.04 if dist is None else dist
    p.camera_position = [tuple(c + dist * d), tuple(c), up]
    p.camera.view_angle = 22
    p.camera.zoom(zoom)
    img = p.screenshot(None, transparent_background=True, return_img=True)
    p.close()
    return _crop(img)


# ------------------------------------------------------------------ heart
def lv_point_rgb(surf, seg_rgb, base_rgb, boundary=True, bcol=(1, 1, 1), bw=1.0):
    """Per-point RGB for the LV epicardium from a dict seg -> rgb (missing / 17 -> base)."""
    aha = np.asarray(surf.point_data["aha"])
    rgb = np.tile(base_rgb, (surf.n_points, 1))
    for k, c in seg_rgb.items():
        rgb[aha == int(k)] = c
    if boundary:
        Z = np.asarray(surf.point_data["Z"]); F = np.asarray(surf.point_data["PHI"])
        lv = aha > 0
        bd = (np.abs(Z - 1 / 3) < 0.010 * bw) | (np.abs(Z - 2 / 3) < 0.010 * bw) | (np.abs(Z - 0.08) < 0.008 * bw)
        wrap = lambda a: np.abs((a + np.pi) % (2 * np.pi) - np.pi)
        b6 = np.min([wrap(F - a) for a in (0, np.pi / 3, 2 * np.pi / 3, np.pi, -np.pi / 3, -2 * np.pi / 3)], axis=0)
        b4 = np.min([wrap(F - a) for a in (np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4)], axis=0)
        r = 1.0 / np.maximum(0.25, 1 - Z)          # keep apparent line width constant towards the apex
        bd |= (Z > 1 / 3) & (b6 < 0.030 * bw)
        bd |= (Z <= 1 / 3) & (Z >= 0.08) & (b4 < 0.040 * bw * r)
        rgb[bd & lv] = np.asarray(bcol, float)
    return rgb


def render_heart(view="lao", px=1400, style="data", seg_rgb=None, lv_rgb=None, aorta_rgb=None,
                 desc_rgb=None, boundary=True, bcol="#FFFFFF", vessels=True, zoom=1.0, crop_branches=False,
                 specular=0.22, neutral=NEUTRAL):
    """Return an RGBA uint8 image of the heart.
    style 'anatomical': illustration colours; 'data': neutral anatomy with coloured LV / aorta.
    seg_rgb: {AHA seg: colour} paints LV segments; lv_rgb: single colour for the whole LV (overrides base);
    aorta_rgb / desc_rgb: colour for ascending / descending aorta (others neutral)."""
    key = _key(f="heart", view=view, px=px, style=style, seg=seg_rgb, lv=lv_rgb, ao=aorta_rgb, de=desc_rgb,
               b=boundary, bc=bcol, v=vessels, z=zoom, sp=specular, n=neutral)
    fn = CACHE / f"heart_{key}.png"
    if fn.exists():
        return np.asarray(Image.open(fn))
    s = _load("heart_surface")
    tag = np.asarray(s.point_data["tag"])
    anat = style == "anatomical"
    base = {t: _hex(ANAT[t]) if anat else _hex(neutral) for t in range(1, 7)}
    rgb = np.tile(_hex(ANAT["valve"]) if anat else _hex(neutral), (s.n_points, 1))
    for t in range(1, 7):
        rgb[tag == t] = base[t]
    if anat:
        rgb[np.isin(tag, [11, 12, 13, 14, 15, 18, 19, 20, 21, 22])] = _hex(ANAT[3])
        rgb[np.isin(tag, [16, 17, 23, 24])] = _hex(ANAT["ra_in"])
    lvb = _hex(lv_rgb) if lv_rgb is not None else base[1]
    srgb = {k: _hex(c) for k, c in (seg_rgb or {}).items()}
    lvrgb = lv_point_rgb(s, srgb, lvb, boundary=boundary, bcol=_hex(bcol))
    lvmask = np.asarray(s.point_data["aha"]) > 0
    rgb[lvmask] = lvrgb[lvmask]
    if lv_rgb is not None:
        rgb[(tag == 1) & ~lvmask] = lvb
    if aorta_rgb is not None:
        rgb[tag == 5] = _hex(aorta_rgb)
    p = _plotter(px)
    s2 = s.copy(); s2.point_data["rgb"] = rgb
    p.add_mesh(s2, scalars="rgb", rgb=True, smooth_shading=True, specular=specular, specular_power=18,
               ambient=0.18, diffuse=0.82)
    if vessels:
        v = _load("vessels")
        part = np.asarray(v.point_data["part"])
        vr = np.zeros((v.n_points, 3))
        for k, nm in PART.items():
            if anat:
                c = _hex(ANAT[nm])
            else:
                c = _hex(NEUTRAL_VES)
                if nm == "asc" and aorta_rgb is not None:
                    c = _hex(aorta_rgb)
                if nm == "desc" and desc_rgb is not None:
                    c = _hex(desc_rgb)
            vr[part == k] = c
        v2 = v.copy(); v2.point_data["rgb"] = vr
        p.add_mesh(v2, scalars="rgb", rgb=True, smooth_shading=True, specular=specular, specular_power=18,
                   ambient=0.18, diffuse=0.82)
    img = _shoot(p, None, view, zoom)
    Image.fromarray(img).save(fn)
    return img


def render_aorta(view="lao_steep", px=1000, asc_rgb="#E4DFD9", desc_rgb="#E4DFD9", ghost_heart=True,
                 arch_rgb=NEUTRAL_VES, zoom=1.0):
    """Aortic arch (ascending, arch, descending) with an optional faint heart silhouette."""
    key = _key(f="aorta", view=view, px=px, a=asc_rgb, d=desc_rgb, g=ghost_heart, ar=arch_rgb, z=zoom)
    fn = CACHE / f"aorta_{key}.png"
    if fn.exists():
        return np.asarray(Image.open(fn))
    s = _load("heart_surface"); v = _load("vessels")
    tag = np.asarray(s.point_data["tag"])
    part = np.asarray(v.point_data["part"])
    p = _plotter(px)
    keep = np.isin(part, [1, 2, 3, 4])
    av = v.extract_points(np.where(keep)[0], adjacent_cells=False).extract_surface(algorithm="dataset_surface")
    ap = np.asarray(av.point_data["part"])
    vr = np.zeros((av.n_points, 3))
    vr[ap == 1] = _hex(asc_rgb); vr[ap == 2] = _hex(arch_rgb); vr[ap == 3] = _hex(desc_rgb); vr[ap == 4] = _hex(arch_rgb)
    av.point_data["rgb"] = vr
    p.add_mesh(av, scalars="rgb", rgb=True, smooth_shading=True, specular=0.22, specular_power=18, ambient=0.2, diffuse=0.8)
    ao = s.extract_points(np.where(tag == 5)[0], adjacent_cells=False).extract_surface(algorithm="dataset_surface")
    p.add_mesh(ao, color=asc_rgb, smooth_shading=True, specular=0.22, ambient=0.2, diffuse=0.8)
    if ghost_heart:
        p.add_mesh(s, color="#D9D4CE", opacity=0.22, smooth_shading=True, specular=0.0, ambient=0.35, diffuse=0.65)
    img = _shoot(p, None, view, zoom)
    Image.fromarray(img).save(fn)
    return img


# ------------------------------------------------------------------ brain (fsaverage5 pial; sulc shading)
_BR = {}


def _cortex(hemi):
    if hemi not in _BR:
        import nibabel as nib
        from nilearn import datasets
        g = nib.load(str(ASSET / f"fsa5_{hemi}.gii"))
        v = g.darrays[0].data.astype(float); f = g.darrays[1].data.astype(int)
        fs = datasets.fetch_surf_fsaverage("fsaverage5")
        sulc = nib.load(fs["sulc_left" if hemi == "lh" else "sulc_right"]).darrays[0].data.astype(float)
        mesh = pv.PolyData(v, np.c_[np.full(len(f), 3), f].ravel())
        _BR[hemi] = (mesh, sulc)
    return _BR[hemi]


BRAIN_VIEWS = {  # fsaverage axes: +x right, +y anterior, +z superior
    ("lh", "lateral"): ((-1, 0, 0), (0, 0, 1)), ("lh", "medial"): ((1, 0, 0), (0, 0, 1)),
    ("rh", "lateral"): ((1, 0, 0), (0, 0, 1)), ("rh", "medial"): ((-1, 0, 0), (0, 0, 1)),
    ("lh", "dorsal"): ((0, 0, 1), (0, 1, 0)), ("lh", "oblique"): ((-0.85, 0.35, 0.40), (0, 0, 1)),
}


def render_cortex(hemi="lh", view="lateral", px=1200, vertex_rgb=None, base="#E9E5E0", sulc_depth=0.20,
                  zoom=1.0, specular=0.12):
    """vertex_rgb: (n_vertices, 3) colours (NaN rows -> base). Sulcal depth darkens base and data alike."""
    key = _key(f="cortex", h=hemi, v=view, px=px, rgb=None if vertex_rgb is None else hashlib.md5(
        np.nan_to_num(np.asarray(vertex_rgb, float), nan=-1).round(4).tobytes()).hexdigest(), b=base, s=sulc_depth, z=zoom)
    fn = CACHE / f"cortex_{key}.png"
    if fn.exists():
        return np.asarray(Image.open(fn))
    mesh, sulc = _cortex(hemi)
    rgb = np.tile(_hex(base), (mesh.n_points, 1))
    if vertex_rgb is not None:
        vr = np.asarray(vertex_rgb, float)
        ok = ~np.isnan(vr).any(1)
        rgb[ok] = vr[ok]
    sh = 1 - sulc_depth * np.clip((sulc - sulc.min()) / (np.ptp(sulc) + 1e-9), 0, 1) ** 0.9
    rgb = np.clip(rgb * sh[:, None], 0, 1)
    m2 = mesh.copy(); m2.point_data["rgb"] = rgb
    p = _plotter(px)
    p.add_mesh(m2, scalars="rgb", rgb=True, smooth_shading=True, specular=specular, specular_power=12, ambient=0.22, diffuse=0.78)
    img = _shoot(p, None, BRAIN_VIEWS[(hemi, view)], zoom)
    Image.fromarray(img).save(fn)
    return img


# ------------------------------------------------------------------ placement on mm canvases
def place(ax, img, x, y, w=None, h=None, anchor="tl", zorder=3, alpha=1.0):
    """Draw RGBA image on a mm canvas (y downwards). Give w or h (mm); aspect preserved.
    anchor: 'tl' top-left, 'c' centre, 'tc' top-centre, 'bc' bottom-centre."""
    ih, iw = img.shape[:2]
    if w is None and h is None:
        raise ValueError("w or h required")
    if w is None:
        w = h * iw / ih
    if h is None:
        h = w * ih / iw
    if anchor == "c":
        x, y = x - w / 2, y - h / 2
    elif anchor == "tc":
        x = x - w / 2
    elif anchor == "bc":
        x, y = x - w / 2, y - h
    ax.imshow(img, extent=(x, x + w, y + h, y), interpolation="lanczos", zorder=zorder, alpha=alpha)
    return x, y, w, h


def fit(img, wmax, hmax):
    """(w, h) in mm that fits inside wmax x hmax keeping aspect."""
    ih, iw = img.shape[:2]
    s = min(wmax / iw, hmax / ih)
    return iw * s, ih * s


# ------------------------------------------------------------------ glass brain with JHU white-matter tracts
_TR = {}


def tract_meshes(sigma=1.0):
    """Marching-cubes surfaces of the 20 JHU probabilistic tracts (maxprob thr25, MNI 1 mm); cached in memory."""
    if "tracts" not in _TR:
        import nibabel as nib
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        im = nib.load(str(ASSET / "JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz"))
        lab = np.asarray(im.get_fdata()).astype(int); A = im.affine
        out = {}
        for t in range(1, 21):
            m = gaussian_filter((lab == t).astype(float), sigma)
            if m.max() < 0.5:
                continue
            v, f, _, _ = marching_cubes(m, 0.5)
            vw = (A @ np.c_[v, np.ones(len(v))].T).T[:, :3]
            mesh = pv.PolyData(vw, np.c_[np.full(len(f), 3), f].ravel()).smooth_taubin(n_iter=30, pass_band=0.1)
            out[t - 1] = mesh          # key = JHU-tracts.xml index (0-19)
        _TR["tracts"] = out
    return _TR["tracts"]


GLASS_VIEWS = {"oblique": ((-0.78, 0.30, 0.55), (0, 0, 1)), "lateral": ((-1, 0, 0.05), (0, 0, 1)),
               "dorsal": ((0, -0.05, 1), (0, 1, 0)), "front_oblique": ((-0.55, 0.70, 0.45), (0, 0, 1))}


def render_glass_brain(view="oblique", px=1400, tract_rgb=None, tract_default="#7A5BC0", cortex_rgb="#D9D2CB",
                       cortex_opacity=0.16, hippo_rgb="#8C6BC4", hemis=("lh", "rh"), zoom=1.0):
    """Translucent pial cortex with opaque JHU tracts and hippocampi. tract_rgb: {jhu tract index: colour}."""
    key = _key(f="glass", v=view, px=px, t=tract_rgb, td=tract_default, c=cortex_rgb, o=cortex_opacity, h=hippo_rgb, hs=hemis, z=zoom)
    fn = CACHE / f"glass_{key}.png"
    if fn.exists():
        return np.asarray(Image.open(fn))
    import anat3d
    p = _plotter(px)
    for t, m in tract_meshes().items():
        c = (tract_rgb or {}).get(t, tract_default)
        p.add_mesh(m, color=c, smooth_shading=True, specular=0.25, specular_power=16, ambient=0.2, diffuse=0.8)
    if hippo_rgb is not None:
        for h in hemis:
            v, f = anat3d.subcortical(h, "hippocampus")
            hm = pv.PolyData(np.asarray(v, float), np.c_[np.full(len(f), 3), f].ravel())
            p.add_mesh(hm, color=hippo_rgb, smooth_shading=True, specular=0.25, ambient=0.2, diffuse=0.8)
    for h in hemis:
        mesh, _ = _cortex(h)
        p.add_mesh(mesh, color=cortex_rgb, opacity=cortex_opacity, smooth_shading=True, specular=0.05,
                   ambient=0.3, diffuse=0.7, culling="back")
    img = _shoot(p, None, GLASS_VIEWS[view] if isinstance(view, str) else view, zoom)
    Image.fromarray(img).save(fn)
    return img


# ------------------------------------------------------------------ data helpers
def dk_vertex_rgb(hemi, values, cmap, norm, mask=None, border=0.30, fade=0.65, na="#E9E6E2"):
    """Per-vertex RGB for fsaverage5 from Desikan-Killiany region values (anat3d naming).
    mask: region -> bool (False = faded, as in the 2D maps); border: white mix on region boundaries."""
    import anat3d
    v, f, vreg, _ = anat3d.cortex(hemi)
    rgb = np.tile(_hex(na), (len(v), 1))
    for r in set(vreg):
        if not r:
            continue
        val = values.get(r, np.nan)
        if not np.isfinite(val):
            continue
        c = np.asarray(cmap(norm(val))[:3], float)
        if mask is not None and not mask.get(r, True):
            c = c * (1 - fade) + _hex("#EEF0F2") * fade
        rgb[vreg == r] = c
    if border:
        fr = np.asarray(vreg)[f]
        mixed = ~((fr[:, 0] == fr[:, 1]) & (fr[:, 1] == fr[:, 2]))
        bv = np.unique(f[mixed])
        rgb[bv] = rgb[bv] * (1 - border) + border
    return rgb


def seg_colours(values, cmap, norm, mask=None, fade=0.65):
    out = {}
    for k, v in values.items():
        if not np.isfinite(v):
            continue
        c = np.asarray(cmap(norm(v))[:3], float)
        if mask is not None and not mask.get(k, True):
            c = c * (1 - fade) + _hex("#EEF0F2") * fade
        out[int(k)] = c
    return out


# ------------------------------------------------------------------ subcortical nuclei (ENIGMA sctx meshes; v7)
SCTX = ["thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "accumbens"]
SUBC_VIEWS = {"lateral": ((-1, 0.10, 0.18), (0, 0, 1)), "medial": ((1, 0.10, 0.18), (0, 0, 1)),
              "inferomedial": ((0.80, 0.30, -0.50), (0, 0, 1)), "oblique": ((-0.80, 0.42, 0.45), (0, 0, 1)),
              "dorsolateral": ((-0.70, 0.05, 0.72), (0, 1, 0.2))}


_SC = {}


def sctx_surface(hemi, structure, h=0.6):
    """Watertight surface for one ENIGMA subcortical structure. The distributed meshes have inconsistently
    oriented, non-manifold faces (dark seams when shaded), so the triangles are densely sampled, voxelised at
    h mm, morphologically closed and filled, Gaussian-smoothed, and re-extracted with marching cubes."""
    k = (hemi, structure, h)
    if k not in _SC:
        import anat3d
        from scipy import ndimage as ndi
        v, f = anat3d.subcortical(hemi, structure)
        v = np.asarray(v, float); f = np.asarray(f, int)
        tri = v[f]
        w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3], [.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                      [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        pts = np.einsum("kj,ijd->ikd", w, tri).reshape(-1, 3)
        lo = pts.min(0) - 4 * h
        shape = np.ceil((pts.max(0) + 4 * h - lo) / h).astype(int) + 1
        vol = np.zeros(shape, bool)
        idx = np.round((pts - lo) / h).astype(int)
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        vol = ndi.binary_dilation(vol, iterations=2)
        vol = ndi.binary_fill_holes(vol)
        vol = ndi.binary_erosion(vol, iterations=2)
        sm = ndi.gaussian_filter(vol.astype(float), 1.0)
        g = pv.ImageData(dimensions=sm.shape, spacing=(h, h, h), origin=tuple(lo))
        g.point_data["v"] = sm.ravel(order="F")
        surf = g.contour([0.5], scalars="v").smooth_taubin(n_iter=20, pass_band=0.1)
        _SC[k] = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)
    return _SC[k]


def render_subcortical(hemi="lh", view="lateral", px=1000, rgb=None, glass=False, glass_rgb="#D9D2CB",
                       glass_opacity=0.08, zoom=1.0):
    """Seven FIRST structures of one hemisphere (ENIGMA subcortical meshes) coloured by `rgb`
    {structure: colour}; optional translucent cortex for spatial context."""
    import anat3d
    rgb = rgb or {}
    key = _key(f="subc", h=hemi, v=view, px=px, rgb={k: np.round(_hex(c), 4).tolist() for k, c in rgb.items()},
               g=glass, go=glass_opacity, z=zoom)
    fn = CACHE / f"subc_{key}.png"
    if fn.exists():
        return np.asarray(Image.open(fn))
    p = _plotter(px)
    pts = []
    for s in SCTX:
        m = sctx_surface(hemi, s)
        pts.append(m.points)
        p.add_mesh(m, color=_hex(rgb.get(s, NEUTRAL)), smooth_shading=True, specular=0.18, specular_power=14,
                   ambient=0.25, diffuse=0.80)
    allp = np.vstack(pts)
    c = allp.mean(0)
    R = 0.5 * np.linalg.norm(allp.max(0) - allp.min(0))
    if glass:
        mesh, _ = _cortex(hemi)
        p.add_mesh(mesh, color=_hex(glass_rgb), opacity=glass_opacity, smooth_shading=True, ambient=0.4)
        b = np.asarray(mesh.bounds, float).reshape(3, 2)
        R = 0.5 * np.linalg.norm(b[:, 1] - b[:, 0])
        c = b.mean(1)
    img = _shoot(p, c, SUBC_VIEWS[view], zoom, dist=R / np.sin(np.radians(11)) * 1.04)
    Image.fromarray(img).save(fn)
    return img
