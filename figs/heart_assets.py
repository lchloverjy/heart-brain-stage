"""Build the 3D heart assets used by figures 1-4 (v6).

Source template (CC BY 4.0): Rodero C, Strocchi M, Marciniak M, et al. Virtual cohort of extreme and average
four-chamber heart meshes from statistical shape model. Zenodo 2021. doi:10.5281/zenodo.4593739 (file average.tar.gz;
md5 992f31e20c1aa73c10c5d9a6b6ac903a). Mean shape of a statistical shape model built from 19 adult cardiac CT scans
(Rodero et al. PLoS Comput Biol 2021;17:e1008851). Not a UK Biobank participant.

Outputs (figs/assets/heart/):
  heart_surface.vtp  outer epicardial/atrial/great-vessel-stump surface, Taubin-smoothed, with point arrays
                     tag (dominant mesh label), aha (AHA segment 1-17 for LV points, 0 otherwise), Z, PHI, RHO, V
  vessels.vtp        schematic continuation of great vessels (spline tubes) with point array 'part'
                     1 ascending aorta, 2 arch, 3 descending aorta, 4 arch branches, 5 pulmonary trunk/arteries,
                     6 SVC, 7 IVC
Mesh axes: +x patient left, +y posterior, +z superior (verified from chamber centroids).

Usage: python3 heart_assets.py /path/to/average.vtk   (see assets/fetch_heart_template.sh)
"""
import sys
from pathlib import Path
import numpy as np
import pyvista as pv

OUT = Path(__file__).resolve().parent / "assets" / "heart"
PART = {"asc": 1, "arch": 2, "desc": 3, "branch": 4, "pa": 5, "svc": 6, "ivc": 7}


def aha_segment(z, f):
    """AHA 17-segment label from universal ventricular coordinates (Z 0 apex - 1 base; PHI 0 septum,
    +/-pi LV free wall, >0 anterior side)."""
    seg = np.zeros(len(z), int)
    apex = z < 0.08
    api = (z >= 0.08) & (z <= 1 / 3)
    mid = (z > 1 / 3) & (z <= 2 / 3)
    bas = z > 2 / 3
    seg[apex] = 17
    q = np.pi / 4
    seg[api & (np.abs(f) < q)] = 14
    seg[api & (f >= q) & (f < 3 * q)] = 13
    seg[api & (f <= -q) & (f > -3 * q)] = 15
    seg[api & (np.abs(f) >= 3 * q)] = 16
    s = np.pi / 3
    k = np.full(len(z), -1)
    k[(f >= 0) & (f < s)] = 1          # anteroseptal
    k[(f >= s) & (f < 2 * s)] = 0      # anterior
    k[f >= 2 * s] = 5                  # anterolateral
    k[(f < 0) & (f >= -s)] = 2         # inferoseptal
    k[(f < -s) & (f >= -2 * s)] = 3    # inferior
    k[f < -2 * s] = 4                  # inferolateral
    seg[bas] = 1 + k[bas]
    seg[mid] = 7 + k[mid]
    return seg


def cut_face(surf, tag, t, axis):
    """Centre, normal, and outer radius of the planar cut annulus of a vessel stump."""
    sub = surf.extract_cells(np.where(tag == t)[0]).extract_surface(algorithm="dataset_surface")
    sub = sub.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False)
    n = np.asarray(sub.cell_data["Normals"])
    ctr = np.asarray(sub.cell_centers().points)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    for _ in range(3):
        top = ctr @ ax > np.quantile(ctr @ ax, 0.80)
        sel = top & (np.abs(n @ ax) > 0.90)
        if sel.sum() < 20:
            sel = top
        P = ctr[sel]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        nn = vt[2] if vt[2] @ ax > 0 else -vt[2]
        ax = nn
    r = np.linalg.norm((P - c) - np.outer((P - c) @ ax, ax), axis=1)
    return c, ax, float(np.quantile(r, 0.97))


def tube(points, radii, n=220, sides=48):
    sp = pv.Spline(np.asarray(points, float), n)
    rr = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(radii)), radii)
    sp.point_data["r"] = rr
    t = sp.tube(scalars="r", absolute=True, n_sides=sides, capping=True)
    return t


def tube_parts(points, radii, labels, n=220, sides=48):
    """Continuous spline tube with per-point radius and integer part label (assigned by nearest control span)."""
    P = np.asarray(points, float)
    sp = pv.Spline(P, n)
    # parameter of each spline point along the control polygon
    seg_len = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u_ctrl = seg_len / seg_len[-1]
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sp.points, axis=0), axis=1))]
    u = sl / sl[-1]
    sp.point_data["r"] = np.interp(u, u_ctrl, radii)
    lab = np.asarray(labels, float)
    sp.point_data["part"] = lab[np.clip(np.searchsorted(u_ctrl, u, side="right") - 1, 0, len(lab) - 1)]
    t = sp.tube(scalars="r", absolute=True, n_sides=sides, capping=True)
    t.point_data["part"] = np.rint(np.asarray(t.point_data["part"])).astype(int)
    for k in list(t.point_data.keys()):
        if k != "part":
            del t.point_data[k]
    t.cell_data.clear()
    return t


def build(vtk_path):
    OUT.mkdir(parents=True, exist_ok=True)
    m = pv.read(vtk_path)
    s = m.extract_surface(algorithm="dataset_surface", pass_pointid=True, pass_cellid=True)
    ctag = np.asarray(m.cell_data["ID"])[np.asarray(s.cell_data["vtkOriginalCellIds"])]
    s.cell_data["tag"] = ctag
    conn = s.connectivity("largest")          # outer surface only (cavity surfaces discarded)
    tag = np.asarray(conn.cell_data["tag"])
    # vessel stump geometry measured on the unsmoothed surface
    geo = {}
    for nm, t, ax in [("ao", 5, (-0.2, -0.2, 0.96)), ("pa", 6, (0.06, 0.33, 0.94)),
                      ("svc", 16, (0.0, 0.0, 1.0)), ("ivc", 17, (0.0, 0.0, -1.0))]:
        geo[nm] = cut_face(conn, tag, t, ax)
        print(nm, "centre", np.round(geo[nm][0], 1), "normal", np.round(geo[nm][1], 2), "r", round(geo[nm][2], 1))
    # smooth (geometry only; point data preserved)
    sm = conn.triangulate().smooth_taubin(n_iter=60, pass_band=0.06, boundary_smoothing=False)
    # dominant tag per point (from incident faces)
    faces = sm.faces.reshape(-1, 4)[:, 1:]
    ptag = np.zeros(sm.n_points, int)
    cnt = np.zeros((sm.n_points, 25), int)
    for j in range(3):
        np.add.at(cnt, (faces[:, j], np.asarray(sm.cell_data["tag"])), 1)
    ptag = cnt.argmax(1)
    sm.point_data["tag"] = ptag
    Z = np.asarray(sm.point_data["Z.dat"]); PHI = np.asarray(sm.point_data["PHI.dat"]); V = np.asarray(sm.point_data["V.dat"])
    aha = aha_segment(Z, PHI)
    aha[(V > -0.5) | (ptag != 1)] = 0
    sm.point_data["aha"] = aha
    for k_old, k_new in [("Z.dat", "Z"), ("PHI.dat", "PHI"), ("RHO.dat", "RHO"), ("V.dat", "V")]:
        sm.point_data[k_new] = np.asarray(sm.point_data[k_old]); del sm.point_data[k_old]
    for k in list(sm.point_data.keys()):
        if k not in ("tag", "aha", "Z", "PHI", "RHO", "V"):
            del sm.point_data[k]
    for k in list(sm.cell_data.keys()):
        if k != "tag":
            del sm.cell_data[k]
    sm.save(OUT / "heart_surface.vtp")
    print("surface", sm.n_cells, "faces;", "AHA points", {int(k): int((aha == k).sum()) for k in range(1, 18)})

    # ---- schematic great-vessel continuation (anatomical background only); one continuous spline per vessel
    parts = []
    ca, na, ra = geo["ao"]
    r0 = min(ra, 14.0) + 0.3
    ctrl = [ca - 4 * na, ca + 8 * na, ca + 18 * na + np.array([1.5, 2.5, 0]),            # ascending (0-2)
            ca + np.array([7, 13, 31]), ca + np.array([16, 28, 37]), ca + np.array([24, 45, 32]),  # arch (3-5)
            ca + np.array([27, 60, 16]), ca + np.array([27, 68, -4]),                    # arch -> descending (6-7)
            ca + np.array([26, 71, -40]), ca + np.array([25, 72, -78]), ca + np.array([24, 72, -104])]
    rad = [r0, r0, r0 * 0.98, r0 * 0.94, r0 * 0.89, r0 * 0.85, r0 * 0.83, r0 * 0.82, r0 * 0.80, r0 * 0.78, r0 * 0.76]
    lab = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    parts.append(tube_parts(ctrl, rad, lab, n=520))
    arch_ctrl = ctrl[2:8]
    archline = pv.Spline(np.asarray(arch_ctrl), 400).points
    for frac, d, rf, L in [(0.30, (-0.40, -0.10, 1.0), 0.50, 34), (0.45, (-0.02, 0.05, 1.0), 0.33, 36),
                           (0.58, (0.38, 0.12, 1.0), 0.40, 32)]:
        b0 = archline[int(frac * len(archline))]
        d = np.asarray(d) / np.linalg.norm(d)
        pts = [b0 - 2 * d, b0 + 0.5 * L * d + np.array([0, 0, 2]), b0 + L * d]
        parts.append(tube_parts(pts, [r0 * rf, r0 * rf * 0.95, r0 * rf * 0.9], [4, 4, 4], n=80, sides=32))
    # pulmonary trunk -> left pulmonary artery (continuous) + right pulmonary artery from the bifurcation
    cp, npa, rp = geo["pa"]
    rpa = 12.5
    B = cp + 15 * npa + np.array([3, 5, 1])
    parts.append(tube_parts([cp - 5 * npa, cp + 6 * npa, B, B + np.array([14, 11, 2]), B + np.array([27, 23, -3]),
                             B + np.array([36, 33, -8])], [rpa, rpa, rpa * 0.92, rpa * 0.70, rpa * 0.63, rpa * 0.60],
                            [5] * 6, n=200))
    parts.append(tube_parts([B + np.array([2, 1, 0]), B + np.array([-14, 12, 3]), B + np.array([-31, 21, 2]),
                             B + np.array([-46, 25, -2])], [rpa * 0.72, rpa * 0.66, rpa * 0.62, rpa * 0.58], [5] * 4, n=120))
    cs, ns, rs = geo["svc"]
    parts.append(tube_parts([cs - 6 * ns, cs + 8 * ns, cs + np.array([-2, -2, 38]), cs + np.array([-3, -3, 60])],
                            [7.5, 8.0, 8.5, 8.5], [6] * 4, n=100, sides=40))
    ci, ni, ri = geo["ivc"]
    parts.append(tube_parts([ci - 6 * ni, ci + 6 * ni, ci + np.array([-2, 4, -30])], [8.0, 8.5, 8.8], [7] * 3, n=80, sides=40))
    allv = [t.triangulate() for t in parts]
    ves = allv[0].merge(allv[1:]) if len(allv) > 1 else allv[0]
    ves = ves.extract_surface(algorithm="dataset_surface")
    ves.save(OUT / "vessels.vtp")
    np.save(OUT / "stump_geometry.npy", np.array([np.r_[geo[k][0], geo[k][1], geo[k][2]] for k in ("ao", "pa", "svc", "ivc")]))
    print("vessels", ves.n_cells, "faces")


if __name__ == "__main__":
    # default: the template downloaded by assets/fetch_heart_template.sh (or set HEART_TEMPLATE)
    import os
    default = os.environ.get("HEART_TEMPLATE", str(Path(__file__).resolve().parent / "assets" / "heart_template" / "average.vtk"))
    build(sys.argv[1] if len(sys.argv) > 1 else default)
