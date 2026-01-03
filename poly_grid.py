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

def is_inside_polygon(poly, point):
    """
    Determines if a point is inside a polygon using Ray Casting.
    
    Args:
        poly: List of (x, y) tuples representing the polygon vertices.
        point: Tuple (x, y) representing the point to check.
    Returns:
        bool: True if inside, False otherwise.
    """
    poly = ensure_ccw(np.asarray(poly))

    n = len(poly)
    inside = False
    x, y = point

    # Loop through each edge of the polygon
    for i in range(n):
        # Current vertex and the next vertex (looping back to the start)
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Check if the ray intersects the edge
        # 1. Is the point's Y-coordinate between the edge's Y-coordinates?
        # 2. Is the point's X-coordinate to the left of the intersection point?
        if min(y1, y2) < y <= max(y1, y2):
            if x <= max(x1, x2):
                # Calculate the actual x-intersection of the edge with the ray
                if y1 != y2:
                    x_inters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                
                # If the ray is to the left of the intersection, toggle state
                if x1 == x2 or x <= x_inters:
                    inside = not inside

    return inside

# ------------------------------------------------------------

class PolyGrid:
    """
    A class to compute the intersection of a polygon with a 2D rectilinear grid
    """

    def __init__(self, poly, Nx, Ny, dx, dy, debug=False):
        """
        Constructor
        Args:
            poly: List of (x, y) tuples representing the polygon vertices.
            Nx, Ny are the number of x and y cells
            dx, dy is the grid space
        """

        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy

        # make sure the polygon vertices are in counterclockwise direction
        self.poly = ensure_ccw(poly)

        # find the intersections of the polygon segments with the grid. cell_segments 
        # associates a list of segments to each (i, j) cell, i.e {(i, j): [(xi0, eta0, xi1, eta1), ...]}
        # where 0 <= xi, eta <= 1 are the start/end parametric coordinates of a sub-segment of the polygon. 
        # Note that if the sub-segment falls onto a cell edge then the sub-segment may be associated to 
        # one or the other cell.
        self.cell_segments = polygon_cell_segments_parametric(self.poly, self.Nx, self.Ny, self.dx, self.dy, debug=debug)

        # compute the flux matrices A and B
        self.flux_poly_matrices()

        # compute the map betwen a flat index and the sparse (i, j) indexing. This is required to construct the constraint 
        # matrix M which uses a flat cell indexing 

        # collect all the i,j (degrees of freedoms involved in the constraint)
        ijs = set()
        for ijij in list(self.A.keys()) + list(self.B.keys()):
            ijs.add((ijij[0], ijij[1]))
            ijs.add((ijij[2], ijij[3]))

        self.k2ij = dict()
        k = 0
        for ij in ijs:
            self.k2ij[k] = ij
            k += 1
    
    def update_fluxes(self, uflux, vflux):
        """
        Update the fluxes (u.dx, v.dy) to enforce the impermeability condition on the surface of the polygon
        u*dx = u*dx - A^T lambda_
        v*dy = v*dy - B^T lambda_
        """

        # compute the residuls
        g = self.get_residuals(uflux, vflux)

        # compute A A^T + B B^T
        M = self.get_M()

        # solve the (small) matrix system
        lambda_ = np.linalg.solve(M, g)

        # update the velocities
        for k1, (i1, j1) in self.k2ij.items():
            for k2, (i2, j2) in self.k2ij.items():
                uflux[i1, j1] -= self.A.get((i2, j2, i1, j1), 0.0) * lambda_[k2]
                vflux[i1, j1] -= self.B.get((i2, j2, i1, j1), 0.0) * lambda_[k2]


    def get_flux_residuals(self, uflux, vflux):
        """
        Compute the residul vector g = A.u + B.v
        """
        # number of intersected cells
        Nc = len(self.k2ij)
        g = np.zeros((Nc,), float)
        for k1, (i1, j1) in self.k2ij.items():
            for k2, (i2, j2) in self.k2ij.items():
                g[k1] += self.A.get((i1, j1, i2, j2), 0.0) * uflux[i2, j2] \
                       + self.B.get((i1, j1, i2, j2), 0.0) * vflux[i2, j2]
        return g


    def get_M(self):
        """
        Compute the matrix A A^T + B B^T in flat index space (k)
        """
        # number of intersected cells
        Nc = len(self.k2ij)
        M = np.zeros((Nc, Nc), float)
        for k1, (i1, j1) in self.k2ij.items():
            for k2, (i2, j2) in self.k2ij.items():
                for k3, (i3, j3) in self.k2ij.items():
                    M[k1, k2] += self.A.get((i1, j1, i3, j3), 0.0) * self.A.get((i2, j2, i3, j3), 0.0) \
                               + self.B.get((i1, j1, i3, j3), 0.0) * self.B.get((i2, j2, i3, j3), 0.0)
        return M



    def flux_poly_matrices(self):
        """
        Compute the sparse matrices to estimate the flux on the surface of a polygon
        """
        self.A = {}
        self.B = {}
        # iterate over all the intersected cells
        for cell, segments in self.cell_segments.items():
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
                self.A[(i, j, i  , j)] = self.A.get((i, j, i  , j), 0.0) + deta*(1. - axsi)
                self.A[(i, j, i+1, j)] = self.A.get((i, j, i+1, j), 0.0) + deta*axsi

                # y flux
                self.B[(i, j, i, j  )] = self.B.get((i, j, i, j  ), 0.0) + dxsi*(1. - aeta)
                self.B[(i, j, i, j+1)] = self.B.get((i, j, i, j+1), 0.0) + dxsi*aeta

    def get_flux_in_cell(self, uflux, vflux):
        """
        Compute the flux across the obstacle for each intersected cell
        u, v Arakawa C staggered velocity field in x and y directions. u has dimensions (Nx+1, Ny) and v has dimensions (Nx, Ny+1)
        """

        flux_per_cell = {}

        for (ic, jc, iu, ju), weight in self.A.items():
            flux_per_cell[(ic, jc)] = flux_per_cell.get((ic, jc), 0.0) + weight*uflux[iu, ju]
        
        for (ic, jc, iv, jv), weight in self.B.items():
            flux_per_cell[(ic, jc)] = flux_per_cell.get((ic, jc), 0.0) + weight*vflux[iv, jv]

        return flux_per_cell


    def get_total_flux(self, uflux, vflux):
        """
        Compute the total flux across a polygon for a uniform grid
        u, v Arakawa C staggered velocity field in x and y directions. u has dimensions (Nx+1, Ny) and v has dimensions (Nx, Ny+1)
        returns the total flux across the obstacle
        """
        flux_per_cell = self.get_flux_in_cell(uflux=uflux, vflux=vflux)

        # sum the contributions from each cell
        tot_flux = 0.0
        for (i, j), flx in flux_per_cell.items():
            tot_flux += flx

        return tot_flux


    def check_polygon_coverage_length(self, tol=1e-10):

        # original length
        L_poly = polygon_edge_lengths(self.poly).sum()

        # reconstructed length
        L_clip = 0.0
        for (i, j), segs in self.cell_segments.items():
            for seg in segs:
                L_clip += clipped_segment_length(i, j, seg, self.dx, self.dy)

        rel_err = abs(L_clip - L_poly) / max(L_poly, 1e-14)

        return {
            "L_polygon": L_poly,
            "L_clipped": L_clip,
            "relative_error": rel_err,
            "ok": rel_err < tol
        }

    def check_edge_coverage(self, tol=1e-12):
        ok = True
        reports = []

        for k in range(len(self.poly)):
            p0 = self.poly[k]
            p1 = self.poly[(k+1) % len(self.poly)]
            edge_vec = np.array(p1) - np.array(p0)
            L = np.linalg.norm(edge_vec)
            if L < tol:
                continue

            intervals = []

            for (i, j), segs in self.cell_segments.items():
                for seg in segs:
                    # reconstruct physical points
                    xi0, eta0, xi1, eta1 = seg
                    q0 = np.array([(i + xi0)*self.dx, (j + eta0)*self.dy])
                    q1 = np.array([(i + xi1)*self.dx, (j + eta1)*self.dy])

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


#################################################################
def test1():
    Lx, Ly = 20.0, 10.0
    Nx, Ny = 20, 10
    dx, dy = Lx/Nx, Ly/Ny
    polygon = [(0., 0.), (1.0, 0.), (1.0, 1.0)]
    pg = PolyGrid(poly=polygon, Nx=Nx, Ny=Ny, dx=dx, dy=dy, debug=True)
    print(pg.cell_segments)
    print(pg.check_polygon_coverage_length())
    print(pg.check_edge_coverage())
    u = np.ones((Nx+1, Ny), float)
    v = np.zeros((Nx, Ny+1), float)
    uflux = u * dy
    vflux = v * dx
    tot_flux = pg.get_total_flux(uflux=uflux, vflux=vflux)
    print(f'tot_flux = {tot_flux}')
    
def test2():
    Lx, Ly = 2.0, 1.0
    Nx, Ny = 20, 10
    dx, dy = Lx/Nx, Ly/Ny
    polygon = poly = [(0.3*Lx, 0.0*Ly), (0.5*Lx, 0.0*Ly), (0.5*Lx, 0.6*Ly), (0.3*Lx, 0.6*Ly)]
    pg = PolyGrid(poly=polygon, Nx=Nx, Ny=Ny, dx=dx, dy=dy, debug=True)
    print(pg.cell_segments)
    print(pg.check_polygon_coverage_length())
    print(pg.check_edge_coverage())
    u = np.ones((Nx+1, Ny), float)
    v = np.zeros((Nx, Ny+1), float)
    uflux = u * dy
    vflux = v * dx
    tot_flux = pg.get_total_flux(uflux=uflux, vflux=vflux)
    print(f'tot_flux = {tot_flux}')

if __name__ == '__main__':
    test1()
    test2()
