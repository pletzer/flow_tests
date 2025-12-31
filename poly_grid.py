import numpy as np
from collections import defaultdict


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def ensure_ccw(poly):
    area = 0.0
    for k in range(len(poly)):
        x0, y0 = poly[k]
        x1, y1 = poly[(k+1) % len(poly)]
        area += x0*y1 - x1*y0
    if area < 0:
        poly = poly[::-1]
    return poly


def clip_segment_to_box(p0, p1, xmin, xmax, ymin, ymax):
    """
    Liang–Barsky clipping.
    Returns (q0, q1) or None if no intersection.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    t0, t1 = 0.0, 1.0

    def clip(p, q, t0, t1):
        if abs(p) < 1e-14:
            if q < 0:
                return None
            return t0, t1
        r = q / p
        if p < 0:
            if r > t1: return None
            if r > t0: t0 = r
        else:
            if r < t0: return None
            if r < t1: t1 = r
        return t0, t1

    for p, q in [
        (-dx, x0 - xmin),
        ( dx, xmax - x0),
        (-dy, y0 - ymin),
        ( dy, ymax - y0),
    ]:
        res = clip(p, q, t0, t1)
        if res is None:
            return None
        t0, t1 = res

    q0 = np.array([x0 + t0*dx, y0 + t0*dy])
    q1 = np.array([x0 + t1*dx, y0 + t1*dy])
    return q0, q1

def polygon_edge_lengths(polygon):
    polygon = np.asarray(polygon)
    lengths = []
    for k in range(len(polygon)):
        p0 = polygon[k]
        p1 = polygon[(k+1) % len(polygon)]
        lengths.append(np.linalg.norm(p1 - p0))
    return np.array(lengths)

def clipped_segment_length(i, j, seg, dx, dy):
    xi0, eta0, xi1, eta1 = seg
    x0 = (i + xi0) * dx
    y0 = (j + eta0) * dy
    x1 = (i + xi1) * dx
    y1 = (j + eta1) * dy
    return np.hypot(x1 - x0, y1 - y0)

def polygon_cell_segments_parametric(
    polygon, Nx, Ny, dx, dy, debug=False
):
    """
    Returns:
        dict[(i,j)] = [(xi0,eta0,xi1,eta1), ...]
    """
    GEOM_TOL = 1e-10 * min(dx, dy)

    polygon = ensure_ccw(np.asarray(polygon))
    segments = defaultdict(list)

    for k in range(len(polygon)):
        p0 = polygon[k]
        p1 = polygon[(k+1) % len(polygon)]

        # bounding box in index space
        xmin = min(p0[0], p1[0])
        xmax = max(p0[0], p1[0])
        ymin = min(p0[1], p1[1])
        ymax = max(p0[1], p1[1])

        i0 = max(0, int(np.floor(xmin / dx)))
        i1 = min(Nx-1, int(np.floor(xmax / dx)))
        j0 = max(0, int(np.floor(ymin / dy)))
        j1 = min(Ny-1, int(np.floor(ymax / dy)))

        for i in range(i0, i1+1):
            for j in range(j0, j1+1):

                cx0 = i * dx
                cx1 = (i+1) * dx
                cy0 = j * dy
                cy1 = (j+1) * dy

                clipped = clip_segment_to_box(
                    p0, p1,
                    cx0, cx1, cy0, cy1
                )
                if clipped is None:
                    continue

                q0, q1 = clipped

                # parametric coordinates
                xi0  = (q0[0] - cx0) / dx
                eta0 = (q0[1] - cy0) / dy
                xi1  = (q1[0] - cx0) / dx
                eta1 = (q1[1] - cy0) / dy

                seg = (xi0, eta0, xi1, eta1)
                subsegment_length = clipped_segment_length(i, j, seg, dx, dy)
                if subsegment_length > GEOM_TOL:
                    segments[(i,j)].append(seg)
                elif debug:
                    print(f"Dropped degenerate segment in cell {(i,j)}: {seg}")

    return dict(segments)


def check_polygon_coverage_length(polygon, cell_segments, dx, dy, tol=1e-10):
    poly = ensure_ccw(np.asarray(polygon))

    # original length
    L_poly = polygon_edge_lengths(poly).sum()

    # reconstructed length
    L_clip = 0.0
    for (i, j), segs in cell_segments.items():
        for seg in segs:
            L_clip += clipped_segment_length(i, j, seg, dx, dy)

    rel_err = abs(L_clip - L_poly) / max(L_poly, 1e-14)

    return {
        "L_polygon": L_poly,
        "L_clipped": L_clip,
        "relative_error": rel_err,
        "ok": rel_err < tol
    }

def check_edge_coverage(poly, cell_segments, dx, dy, tol=1e-12):
    poly = ensure_ccw(np.asarray(poly))
    ok = True
    reports = []

    for k in range(len(poly)):
        p0 = poly[k]
        p1 = poly[(k+1) % len(poly)]
        edge_vec = p1 - p0
        L = np.linalg.norm(edge_vec)
        if L < tol:
            continue

        intervals = []

        for (i, j), segs in cell_segments.items():
            for seg in segs:
                # reconstruct physical points
                xi0, eta0, xi1, eta1 = seg
                q0 = np.array([(i + xi0)*dx, (j + eta0)*dy])
                q1 = np.array([(i + xi1)*dx, (j + eta1)*dy])

                # check colinearity
                v0 = q0 - p0
                v1 = q1 - p0
                cross = abs(np.cross(edge_vec, v0)) / L
                if cross > 1e-10:
                    continue

                s0 = np.dot(v0, edge_vec) / (L*L)
                s1 = np.dot(v1, edge_vec) / (L*L)
                intervals.append((min(s0,s1), max(s0,s1)))

        if not intervals:
            ok = False
            reports.append((k, "no coverage"))
            continue

        intervals.sort()
        s = 0.0
        for a, b in intervals:
            if a > s + tol:
                ok = False
                reports.append((k, f"gap at s={s:.3e}"))
                break
            s = max(s, b)

        if s < 1.0 - tol:
            ok = False
            reports.append((k, f"ends at s={s:.3e}"))

    return ok, reports


def flux_poly_matrices(polygon, Nx, Ny, dx, dy):
    """
    Compute the sparse matrices to estimate the flux on the surface of a polygon
    return umatrix, vmatrix
    """
    umatrix = {}
    vmatrix = {}
    # compute the intersections between the polygon and the grid
    # {(i,j): [(xsi0, eta0, xsi1, eta1), ...]}
    cell_segments = polygon_cell_segments_parametric(polygon, Nx, Ny, dx, dy)

    # iterate over all the intersected cells
    for cell, segments in cell_segments.items():
        # cell indices
        i, j = cell
        # iterate over the segments in the cell
        for seg in segments:
            # get the start/end parametric coordinates of the subsegment
            xsi0, eta0, xsi1, eta1 = seg

            # use weights of Eq (10) in https://journals.ametsoc.org/view/journals/mwre/147/1/mwr-d-18-0146.1.xml
            dxsi, deta = xsi1 - xsi0, eta1 - eta0
            axsi, aeta = 0.5*(xsi0 + xsi1), 0.5*(eta0 + eta1)
            # x flux
            umatrix[(i  ,j)] = umatrix.get((i  ,j), 0.0) + deta*(1. - axsi)
            umatrix[(i+1,j)] = umatrix.get((i+1,j), 0.0) + deta*axsi
            # y flux
            vmatrix[(i,j  )] = vmatrix.get((i,j  ), 0.0) + dxsi*(1. - aeta)
            vmatrix[(i,j+1)] = vmatrix.get((i,j+1), 0.0) + dxsi*aeta

    return umatrix, vmatrix


def flux(umatrix, vmatrix, u, v, dx, dy):
    """
    Compute the total flux across a polygon for a uniform grid
    umatrix, vmatrix matrices returned by flux_poly_matrices
    u, v Arakawa staggered velocity field in x and y directions. u has dimensions (Nx+1, Ny) and v has dimensions (Nx, Ny+1)
    returns the total flux
    """
    # turn the velocities into face fluxes
    uflux = u*dy
    vflux = v*dx
    # sparse matrix multiplication
    tot_flux = 0.0
    for cell in umatrix:
        i, j = cell
        tot_flux += umatrix[cell]*uflux[i, j]
    for cell in vmatrix:
        i, j = cell
        tot_flux += vmatrix[cell]*vflux[i, j]
    return tot_flux


#################################################################
def test1():
    Lx, Ly = 20.0, 10.0
    Nx, Ny = 20, 10
    dx, dy = Lx/Nx, Ly/Ny
    polygon = [(0., 0.), (1.0, 0.), (1.0, 1.0)]
    cell_segments = polygon_cell_segments_parametric(polygon, Nx, Ny, dx, dy)
    print(cell_segments)
    print(check_polygon_coverage_length(polygon, cell_segments, dx, dy))
    print(check_edge_coverage(polygon, cell_segments, dx, dy))
    umatrix, vmatrix = flux_poly_matrices(polygon, Nx, Ny, dx, dy)
    rng = np.random.default_rng(seed=42)
    u = rng.random((Nx+1, Ny))
    v = rng.random((Nx, Ny+1))
    total_flux = flux(umatrix, vmatrix, u, v, dx, dy)
    print(f'total_flux = {total_flux}')

if __name__ == '__main__':
    test1()
