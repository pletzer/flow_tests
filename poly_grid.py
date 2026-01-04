import numpy as np
from collections import defaultdict

"""
Flux conserving impermeability enforcement based on mimetic interpolation
"""

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
        poly = poly[::-1].copy()
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

def polygon_cell_segments_parametric(polygon, Nx, Ny, dx, dy, debug=False, closed=True):
    """
    Compute the parametric segments of a polygon within each grid cell.

    Returns:
        dict[(i,j)] = [(xi0, eta0, xi1, eta1), ...]
    where xi, eta are parametric coordinates in [0,1] for the subsegment in the cell.
    """
    GEOM_TOL = 1e-14 * min(dx, dy)  # minimal length to keep segment
    EPS = 1e-12                       # numerical tolerance

    if closed:
        polygon.append(polygon[0])

    polygon = ensure_ccw(np.asarray(polygon))
    segments = defaultdict(list)

    
    # number of segments
    nseg = len(polygon) - 1

    if debug:
        print(f'Number of segments = {nseg} closed = {closed}')
    
    for k in range(nseg):
        p0 = polygon[k]
        p1 = polygon[k + 1]

        # Bounding box in index space
        xmin, xmax = min(p0[0], p1[0]), max(p0[0], p1[0])
        ymin, ymax = min(p0[1], p1[1]), max(p0[1], p1[1])

        # Determine which cells this segment can intersect
        i0 = max(0, int(np.floor(xmin / dx)))
        i1 = min(Nx - 1, int(np.floor(xmax / dx)))
        j0 = max(0, int(np.floor(ymin / dy)))
        j1 = min(Ny - 1, int(np.floor(ymax / dy)))

        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                # Cell boundaries
                cx0, cx1 = i * dx, (i + 1) * dx
                cy0, cy1 = j * dy, (j + 1) * dy

                # Clip the segment to the cell
                clipped = clip_segment_to_box(p0, p1, cx0, cx1, cy0, cy1)
                if clipped is None:
                    continue

                q0, q1 = clipped

                # parametric coordinates within the cell
                xi0  = (q0[0] - cx0) / dx
                eta0 = (q0[1] - cy0) / dy
                xi1  = (q1[0] - cx0) / dx
                eta1 = (q1[1] - cy0) / dy

                # Compute segment length in physical space
                length = np.hypot(q1[0] - q0[0], q1[1] - q0[1])
                if length < GEOM_TOL:
                    if debug:
                        print(f"Skipped degenerate segment in cell {(i,j)}: {(xi0, eta0, xi1, eta1)}")
                    continue

                # Keep only segments that actually lie (at least partially) inside the cell
                if (max(xi0, xi1) < -EPS or min(xi0, xi1) > 1.0 + EPS or
                    max(eta0, eta1) < -EPS or min(eta0, eta1) > 1.0 + EPS):
                    if debug:
                        print(f"Skipped out-of-cell segment in cell {(i,j)}: {(xi0, eta0, xi1, eta1)}")
                    continue

                seg = (xi0, eta0, xi1, eta1)
                segments[(i, j)].append(seg)
                if debug:
                    print(f"Found segment in cell {(i,j)}: {seg}")

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

    def __init__(self, poly, Nx, Ny, dx, dy, debug=False, closed=True):
        """
        Constructor
        Args:
            poly: List of (x, y) tuples representing the polygon vertices.
            Nx, Ny are the number of x and y cells
            dx, dy is the grid space
            closed whether the contour closes or not
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
        self.cell_segments = polygon_cell_segments_parametric(self.poly, self.Nx, self.Ny, self.dx, self.dy, debug=debug, closed=closed)

        # compute the flux matrices A and B
        self.flux_poly_matrices()

        # compute the map betwen a flat index and the sparse (i, j) indexing. This is required to construct the constraint 
        # matrix M which uses a flat cell indexing 

        # collect all the i,j (degrees of freedoms involved in the constraint)
        ijs = set()
        for (ic, jc, _, _) in self.A.keys():
            ijs.add((ic, jc))
        for (ic, jc, _, _) in self.B.keys():
            ijs.add((ic, jc))

        self.k2ij = dict()
        self.ij2k = dict()
        k = 0
        for ij in sorted(ijs): # not really necessary to sort but helps with debugging
            self.k2ij[k] = ij
            self.ij2k[ij] = k
            k += 1

        # compute A A^T + B B^T
        self.M = self.get_M()
        self.compute_edge_fractions()

    
    def update_fluxes(self, uflux, vflux):
        """
        Update the fluxes (u.dx, v.dy) to enforce the impermeability condition on the surface of the polygon
        u*dx = u*dx - A^T lambda_
        v*dy = v*dy - B^T lambda_
        """

        # compute the residuals
        g = self.get_flux_residuals(uflux, vflux)

        # solve the (small) matrix system
        eps = 1e-14 * np.trace(self.M)
        lambda_ = np.linalg.solve(self.M + eps*np.eye(self.M.shape[0]), g) # guard against singular M

        # update the fluxes

        # uflux -= A^T @ lambda
        for (ic, jc, iu, ju), a_elem in self.A.items():
            kc = self.ij2k[(ic, jc)]
            uflux[iu, ju] -= a_elem * lambda_[kc]

        # vflux -= B^T @ lambda
        for (ic, jc, iv, jv), b_elem in self.B.items():
            kc = self.ij2k[(ic, jc)]
            vflux[iv, jv] -= b_elem * lambda_[kc]


    def get_flux_residuals(self, uflux, vflux):
        """
        Compute the residul vector g = A @ uflux + B @ vflux
        """
        # number of intersected cells
        Nc = len(self.k2ij)
        g = np.zeros((Nc,), float)
        for (i1, j1, i2, j2), a_elem in self.A.items():
            k1 = self.ij2k[(i1, j1)]
            g[k1] += a_elem * uflux[i2, j2]
        for (i1, j1, i2, j2), b_elem in self.B.items():
            k1 = self.ij2k[(i1, j1)]
            g[k1] += b_elem * vflux[i2, j2]
        return g


    def get_M(self):
        """
        Compute the matrix A A^T + B B^T in flat index space (k)
        """
        # number of intersected cells
        Nc = len(self.k2ij)
        M = np.zeros((Nc, Nc), float)

        # A contribution
        for (ic1, jc1, if1, jf1), a1 in self.A.items():
            k1 = self.ij2k[(ic1, jc1)]
            for (ic2, jc2, if2, jf2), a2 in self.A.items():
                if (if1, jf1) == (if2, jf2):
                    k2 = self.ij2k[(ic2, jc2)]
                    M[k1, k2] += a1 * a2

        # B contribution
        for (ic1, jc1, if1, jf1), b1 in self.B.items():
            k1 = self.ij2k[(ic1, jc1)]
            for (ic2, jc2, if2, jf2), b2 in self.B.items():
                if (if1, jf1) == (if2, jf2):
                    k2 = self.ij2k[(ic2, jc2)]
                    M[k1, k2] += b1 * b2

        return M



    def flux_poly_matrices(self):
        """
        Compute the sparse matrices A and B to estimate the flux on the surface of a polygon.

        This version checks that contributions are within the valid staggered grid bounds:
            - uflux has shape (Nx+1, Ny)
            - vflux has shape (Nx, Ny+1)
        """
        self.A = {}
        self.B = {}

        for cell, segments in self.cell_segments.items():

            i, j = cell

            # Skip cells fully outside the valid cell index range
            if not (0 <= i <= self.Nx-1 and 0 <= j <= self.Ny-1):
                continue

            # Initialize sparse matrix entries to zero
            self.A[(i, j, i, j)] = 0.0
            self.A[(i, j, i+1, j)] = 0.0
            self.B[(i, j, i, j)] = 0.0
            self.B[(i, j, i, j+1)] = 0.0

            for seg in segments:

                xsi0, eta0, xsi1, eta1 = seg
                dxsi, deta = xsi1 - xsi0, eta1 - eta0
                axsi, aeta = 0.5*(xsi0 + xsi1), 0.5*(eta0 + eta1)

                # ---------------------------
                # x-flux contribution (u*dx)
                # ---------------------------
                # uflux has shape (Nx+1, Ny)
                if 0 <= i <= self.Nx-1 and 0 <= j <= self.Ny-1:
                    if 0 <= i+1 <= self.Nx:
                        self.A[(i, j, i, j)] += deta * (1.0 - axsi)
                        self.A[(i, j, i+1, j)] += deta * axsi

                # ---------------------------
                # y-flux contribution (v*dy)
                # ---------------------------
                # vflux has shape (Nx, Ny+1)
                if 0 <= i <= self.Nx-1 and 0 <= j <= self.Ny-1:
                    if 0 <= j+1 <= self.Ny:
                        self.B[(i, j, i, j)] += dxsi * (1.0 - aeta)
                        self.B[(i, j, i, j+1)] += dxsi * aeta


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
    

    def compute_edge_fractions(self):
        """
        Compute the valid fractions for each edge
        """
        self.dxfrac = np.ones((self.Nx, self.Ny + 1), float)
        self.dyfrac = np.ones((self.Nx + 1, self.Ny), float)

        tol = 1.e-10

        # iterate over the cut-cells
        print(f'cell_segments = {self.cell_segments}')
        for cell, segments in self.cell_segments.items():
            i, j = cell
            for seg in segments:
                xsi0, eta0, xsi1, eta1 = seg
                isx0 = 1 - xsi0
                isx1 = 1 - xsi1
                ate0 = 1 - eta0
                ate1 = 1 - eta1
                # iterate over left/right and bottom/top
                for side in 0, 1:
                    edis = 1 - side

                    # dxfrac
                    if abs(eta0 - side) < tol:
                        # eta0 is 0 or 1
                        self.dxfrac[i, j + side] = isx0*edis + xsi0*side
                    elif abs(eta1 - side) < tol:
                        # eta1 is 0 or 1
                        self.dxfrac[i, j + side] = isx1*side + xsi1*edis
                    else: 
                        # no interesection on this edge but still need to check if the edge is 
                        # fully inside the polygon
                        p0 = ((i + 0)*self.dx, (j + side)*self.dy)
                        p1 = ((i + 1)*self.dx, (j + side)*self.dy)
                        if is_inside_polygon(self.poly, p0) and \
                           is_inside_polygon(self.poly, p1):
                            self.dxfrac[i, j + side] = 0.0

                    # dyfrac
                    if abs(xsi0 - side) < tol:
                        # xsi0 is 0 or 1
                        self.dyfrac[i + side, j] = ate0*side + eta0*edis
                    elif abs(xsi1 - side) < tol:
                        # xsi1 is 0 or 1
                        self.dyfrac[i + side, j] = ate1*edis + eta1*side
                    else: 
                        # no interesection on this edge but still need to check if the edge is 
                        # fully inside the polygon
                        p0 = ((i + side)*self.dx, (j + 0)*self.dy)
                        p1 = ((i + side)*self.dx, (j + 1)*self.dy)
                        if is_inside_polygon(self.poly, p0) and \
                           is_inside_polygon(self.poly, p1):
                            self.dyfrac[i + side, j] = 0.0



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

def test_edge_fracs():
    Lx, Ly = 2.0, 2.0
    Nx, Ny = 2, 2
    dx, dy = Lx/Nx, Ly/Ny
    polygon = [(1.0*dx, 0.3*dy), (1.3*dx, 1.0*dy), (1.7*dx, 1.8*dy), (1.2*dx, 2.0*dy), (0.7*dx, 2.0*dy), (0.2*dx, 1.0*dy),]
    pg = PolyGrid(poly=polygon, Nx=Nx, Ny=Ny, dx=dx, dy=dy, debug=True, closed=True)
    dxfrac, dyfrac = pg.dxfrac, pg.dyfrac
    print(f'dxfrac = {dxfrac}')
    print(f'dyfrac = {dyfrac}')
    tol = 1.e-10
    assert abs(dxfrac[0, 0] - 1) < tol
    assert abs(dxfrac[1, 0] - 1) < tol
    assert abs(dxfrac[0, 1] - 0.2) < tol
    assert abs(dxfrac[1, 1] - 0.7) < tol
    assert abs(dxfrac[0, 2] - 0.7) < tol
    assert abs(dxfrac[1, 2] - 0.8) < tol

    assert abs(dyfrac[0, 0] - 1) < tol
    assert abs(dyfrac[0, 1] - 1) < tol
    assert abs(dyfrac[1, 0] - 0.3) < tol
    assert abs(dyfrac[1, 1] - 0) < tol # completely inside the polygon
    assert abs(dyfrac[2, 0] - 1) < tol
    assert abs(dyfrac[2, 1] - 1) < tol
     

def test4():
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 1, 1
    dx, dy = Lx/Nx, Ly/Ny
    polygon = [(1.0*dx, 1.0*dy), (0.0*dx, 0.0*dy)]
    pg = PolyGrid(poly=polygon, Nx=Nx, Ny=Ny, dx=dx, dy=dy, debug=True, closed=False)
    uflux = np.zeros((Nx+1, Ny), float)
    vflux = np.zeros((Nx, Ny+1), float)
    uflux[0, 0] = 0
    uflux[1, 0] = 2
    vflux[0, 0] = 1
    vflux[0, 1] = 3
    uflux_out = uflux.copy()
    vflux_out = vflux.copy()
    pg.update_fluxes(uflux=uflux_out, vflux=vflux_out)
    # since the polygon's segment runs parallel to the v flux and there is no uflux, 
    # no update is expected
    print('test4 in:')
    print(f'uflux = {uflux}')
    print(f'vflux = {vflux}')
    print('test4 out:')
    print(f'uflux = {uflux_out}')
    print(f'vflux = {vflux_out}')

    tol = 1.e-10
    # assert abs(vflux_in[0, 0] - vflux_out[0, 0]) < tol
    # assert abs(vflux_in[0, 1] - vflux_out[0, 1]) < tol
    # assert abs(uflux_in[0, 0] - uflux_out[0, 0]) < tol
    # assert abs(uflux_in[1, 0] - uflux_out[1, 0]) < tol


def test3():
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 1, 1
    dx, dy = Lx/Nx, Ly/Ny
    polygon = [(0.5*dx, 1*dy), (0.5*dx, 0*dy)]
    pg = PolyGrid(poly=polygon, Nx=Nx, Ny=Ny, dx=dx, dy=dy, debug=True, closed=False)

    uflux_in = np.zeros((Nx+1, Ny), float)
    vflux_in = np.zeros((Nx, Ny+1), float)
    vflux_in[0, 0] = 1.0 # lower face
    vflux_in[0, 1] = 1.0 # upper face
    uflux_out = uflux_in.copy()
    vflux_out = vflux_in.copy()
    pg.update_fluxes(uflux=uflux_out, vflux=vflux_out)
    # since the polygon's segment runs parallel to the v flux and there is no uflux, 
    # no update is expected
    print('test3 in:')
    print(f'uflux0 = {uflux_in[0, 0]} uflux1 = {uflux_in[1, 0]}')
    print(f'vflux0 = {vflux_in[0, 0]} vflux1 = {vflux_in[0, 1]}')
    print('test 3 out:')
    print(f'uflux0 = {uflux_out[0, 0]} uflux1 = {uflux_out[1, 0]}')
    print(f'vflux0 = {vflux_out[0, 0]} vflux1 = {vflux_out[0, 1]}')

    tol = 1.e-10
    assert abs(vflux_in[0, 0] - vflux_out[0, 0]) < tol
    assert abs(vflux_in[0, 1] - vflux_out[0, 1]) < tol
    assert abs(uflux_in[0, 0] - uflux_out[0, 0]) < tol
    assert abs(uflux_in[1, 0] - uflux_out[1, 0]) < tol


    uflux_in = np.zeros((Nx+1, Ny), float)
    vflux_in = np.zeros((Nx, Ny+1), float)
    uflux_in[0, 0] = 1.0 # left face
    uflux_out = uflux_in.copy()
    vflux_out = vflux_in.copy()
    pg.update_fluxes(uflux=uflux_out, vflux=vflux_out)
    # since the polygon's segment runs parallel to the v flux and there is no uflux, 
    # no update is expected
    print('test3 in:')
    print(f'uflux0 = {uflux_in[0, 0]} uflux1 = {uflux_in[1, 0]}')
    print(f'vflux0 = {vflux_in[0, 0]} vflux1 = {vflux_in[0, 1]}')
    print('test3 out:')
    print(f'uflux0 = {uflux_out[0, 0]} uflux1 = {uflux_out[1, 0]}')
    print(f'vflux0 = {vflux_out[0, 0]} vflux1 = {vflux_out[0, 1]}')

    tol = 1.e-10
    assert abs(uflux_out[0, 0] + uflux_out[1, 0]) < tol

   

if __name__ == '__main__':
    #test1()
    #test2()
    test3() 
    test4()
    #test5()
    #test6()
    test_edge_fracs()
