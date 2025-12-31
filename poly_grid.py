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

def polygon_cell_segments_parametric(
    polygon, Nx, Ny, dx, dy
):
    """
    Returns:
        dict[(i,j)] = [(xi0,eta0,xi1,eta1), ...]
    """

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

                segments[(i,j)].append(
                    (xi0, eta0, xi1, eta1)
                )

    return dict(segments)
