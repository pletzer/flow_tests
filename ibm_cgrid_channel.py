
"""
2D incompressible Navier-Stokes solver on an Arakawa C-grid
Channel flow with inflow/outflow BCs, free-slip walls, and a polygonal immersed obstacle.

Stability/enforcement:
  - Red–Black Gauss–Seidel (RBGS) pressure solver (fast convergence).
  - Clamp–Project–Clamp sequence for no-penetration (u·n=0) at obstacle boundary.
  - Thin band of faces around polygon where the clamp is applied.
  - Semi-implicit diffusion (Helmholtz solve via Jacobi).

Performance:
  - numba-accelerated stencil kernels for speed.
  - small interpolation kernels are JIT (serial) to avoid nested parallel issues.

"""

import numpy as np

# VTK writer
import vtk
from vtk.util import numpy_support
from poly_grid import is_inside_polygon, PolyGrid


# -----------------------------
# Optional Numba acceleration
# -----------------------------
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    # Fallback decorators: no-op
    def njit(*args, **kwargs):
        def wrapper(f): return f
        return wrapper
    def prange(*args):
        return range(*args)

# ============================================================
# VTK writer (masked interior to avoid plotting inside obstacle)
# ============================================================

def write_vtr(fname, u, v, p, Lx, Ly, chi_p=None):
    Nx, Ny = p.shape

    # staggered -> cell centers
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])

    # Mask interior cells so we don't visualize vectors/pressure inside the obstacle
    if chi_p is not None:
        mask = (1.0 - chi_p)  # 1=fluid, 0=solid
        uc = uc * mask
        vc = vc * mask
        p  = p  * mask

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(Nx + 1, Ny + 1, 1)

    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    z = np.array([0.0])

    grid.SetXCoordinates(numpy_support.numpy_to_vtk(x))
    grid.SetYCoordinates(numpy_support.numpy_to_vtk(y))
    grid.SetZCoordinates(numpy_support.numpy_to_vtk(z))

    # Pressure at cells
    p_vtk = numpy_support.numpy_to_vtk(p.ravel(order="F"), deep=True)
    p_vtk.SetName("pressure")
    grid.GetCellData().AddArray(p_vtk)

    # Velocity at cells
    vel = np.zeros((Nx, Ny, 3))
    vel[:, :, 0] = uc
    vel[:, :, 1] = vc
    vel_vtk = numpy_support.numpy_to_vtk(vel.reshape(-1, 3, order="F"), deep=True)
    vel_vtk.SetName("velocity")
    grid.GetCellData().AddArray(vel_vtk)

    # Optional: fluid mask for visualization
    if chi_p is not None:
        mask_vtk = numpy_support.numpy_to_vtk((1.0 - chi_p).ravel(order="F"), deep=True)
        mask_vtk.SetName("fluid_mask")
        grid.GetCellData().AddArray(mask_vtk)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.Write()


# ============================================================
# Geometry & masking utilities
# ============================================================

def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cond = ((y0 > y) != (y1 > y))
        if cond:
            x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-16)
            if x_int > x:
                inside = not inside
    return inside

def build_masks(Lx, Ly, Nx, Ny, poly):
    dx, dy = Lx / Nx, Ly / Ny

    # p centers
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    chi_p = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if point_in_polygon(xc[i], yc[j], poly):
                chi_p[i, j] = 1.0

    # u faces (i*dx, (j+0.5)*dy)
    xu = (np.arange(Nx + 1)) * dx
    yu = (np.arange(Ny) + 0.5) * dy
    chi_u = np.zeros((Nx + 1, Ny))
    for iu in range(Nx + 1):
        for j in range(Ny):
            if point_in_polygon(xu[iu], yu[j], poly):
                chi_u[iu, j] = 1.0

    # v faces ((i+0.5)*dx, j*dy)
    xv = (np.arange(Nx) + 0.5) * dx
    yv = (np.arange(Ny + 1)) * dy
    chi_v = np.zeros((Nx, Ny + 1))
    for i in range(Nx):
        for jv in range(Ny + 1):
            if point_in_polygon(xv[i], yv[jv], poly):
                chi_v[i, jv] = 1.0

    return chi_p, chi_u, chi_v

def polygon_edges(poly):
    segs = []
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        segs.append((x0, y0, x1, y1))
    return segs

def nearest_edge_normal(x, y, segs):
    best_d2 = 1e300
    nx, ny = 0.0, 0.0
    for (x0, y0, x1, y1) in segs:
        ex, ey = x1 - x0, y1 - y0
        L2 = ex*ex + ey*ey + 1e-16
        t = ((x - x0) * ex + (y - y0) * ey) / L2
        if t < 0.0: t = 0.0
        if t > 1.0: t = 1.0
        xc = x0 + t * ex
        yc = y0 + t * ey
        dx, dy = x - xc, y - yc
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            nrm = np.sqrt(ex*ex + ey*ey) + 1e-16
            nx, ny = ey / nrm, -ex / nrm  # unit normal (sign arbitrary for slip)
    return nx, ny, np.sqrt(best_d2)

def build_face_normals(Lx, Ly, Nx, Ny, poly):
    segs = polygon_edges(poly)
    dx, dy = Lx / Nx, Ly / Ny

    # u faces
    xu = (np.arange(Nx + 1)) * dx
    yu = (np.arange(Ny) + 0.5) * dy
    n_u_x = np.zeros((Nx + 1, Ny))
    n_u_y = np.zeros((Nx + 1, Ny))
    for iu in range(Nx + 1):
        for j in range(Ny):
            nx, ny, _ = nearest_edge_normal(xu[iu], yu[j], segs)
            n_u_x[iu, j] = nx
            n_u_y[iu, j] = ny

    # v faces
    xv = (np.arange(Nx) + 0.5) * dx
    yv = (np.arange(Ny + 1)) * dy
    n_v_x = np.zeros((Nx, Ny + 1))
    n_v_y = np.zeros((Nx, Ny + 1))
    for i in range(Nx):
        for jv in range(Ny + 1):
            nx, ny, _ = nearest_edge_normal(xv[i], yv[jv], segs)
            n_v_x[i, jv] = nx
            n_v_y[i, jv] = ny

    return (n_u_x, n_u_y), (n_v_x, n_v_y)

# ---------- Thin band around polygon boundary (faces close to edges) ----------
def build_face_band(Lx, Ly, Nx, Ny, poly, band_thickness):
    segs = polygon_edges(poly)
    dx, dy = Lx / Nx, Ly / Ny

    # u faces
    xu = (np.arange(Nx + 1)) * dx
    yu = (np.arange(Ny) + 0.5) * dy
    band_u = np.zeros((Nx + 1, Ny), dtype=np.bool_)
    for iu in range(Nx + 1):
        for j in range(Ny):
            _, _, dist = nearest_edge_normal(xu[iu], yu[j], segs)
            if dist <= band_thickness:
                band_u[iu, j] = True

    # v faces
    xv = (np.arange(Nx) + 0.5) * dx
    yv = (np.arange(Ny + 1)) * dy
    band_v = np.zeros((Nx, Ny + 1), dtype=np.bool_)
    for i in range(Nx):
        for jv in range(Ny + 1):
            _, _, dist = nearest_edge_normal(xv[i], yv[jv], segs)
            if dist <= band_thickness:
                band_v[i, jv] = True

    return band_u, band_v

# ============================================================
# Staggered interpolations (Numba JIT, serial to avoid nested parfors)
# ============================================================

@njit(cache=True, fastmath=True)
def v_at_u_from_v(v):
    Nx = v.shape[0]
    Ny = v.shape[1] - 1
    v_u = np.empty((Nx + 1, Ny), dtype=v.dtype)
    for iu in range(Nx + 1):
        iL = iu - 1
        if iL < 0: iL = 0
        iR = iu
        if iR > Nx - 1: iR = Nx - 1
        for j in range(Ny):
            v_u[iu, j] = 0.25 * (v[iL, j] + v[iR, j] + v[iL, j+1] + v[iR, j+1])
    return v_u

@njit(cache=True, fastmath=True)
def u_at_v_from_u(u):
    Nx = u.shape[0] - 1
    Ny = u.shape[1]
    u_v = np.empty((Nx, Ny + 1), dtype=u.dtype)
    for i in range(Nx):
        for jv in range(Ny + 1):
            jD = jv - 1
            if jD < 0: jD = 0
            jU = jv
            if jU > Ny - 1: jU = Ny - 1
            u_v[i, jv] = 0.25 * (u[i, jD] + u[i+1, jD] + u[i, jU] + u[i+1, jU])
    return u_v

# ============================================================
# Differential operators (Numba)
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def divergence(u, v, dx, dy):
    Nx = u.shape[0] - 1
    Ny = u.shape[1]
    div = np.empty((Nx, Ny))
    for i in prange(Nx):
        for j in range(Ny):
            div[i, j] = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy
    return div

@njit(cache=True, fastmath=True, parallel=True)
def grad_p_to_u(p, dx):
    Nx, Ny = p.shape
    gp = np.zeros((Nx + 1, Ny))
    for j in prange(Ny):
        for i in range(1, Nx):
            gp[i, j] = (p[i, j] - p[i-1, j]) / dx
        gp[0, j] = 0.0
        gp[Nx, j] = 0.0
    return gp

@njit(cache=True, fastmath=True, parallel=True)
def grad_p_to_v(p, dy):
    Nx, Ny = p.shape
    gp = np.zeros((Nx, Ny + 1))
    for i in prange(Nx):
        gp[i, 0] = 0.0
        for j in range(1, Ny):
            gp[i, j] = (p[i, j] - p[i, j-1]) / dy
        gp[i, Ny] = 0.0
    return gp

@njit(cache=True, fastmath=True, parallel=True)
def laplacian_u(u, dx, dy):
    Nx_p1, Ny = u.shape
    lap = np.zeros_like(u)
    for iu in prange(Nx_p1):
        iuL = iu - 1 if iu > 0 else 0
        iuR = iu + 1 if iu < Nx_p1 - 1 else Nx_p1 - 1
        for j in range(Ny):
            jD = j - 1 if j > 0 else 0
            jU = j + 1 if j < Ny - 1 else Ny - 1
            lap[iu, j] = (u[iuR, j] - 2.0*u[iu, j] + u[iuL, j]) / dx**2 \
                       + (u[iu, jU] - 2.0*u[iu, j] + u[iu, jD]) / dy**2
    return lap

@njit(cache=True, fastmath=True, parallel=True)
def laplacian_v(v, dx, dy):
    Nx, Ny_p1 = v.shape
    lap = np.zeros_like(v)
    for i in prange(Nx):
        iL = i - 1 if i > 0 else 0
        iR = i + 1 if i < Nx - 1 else Nx - 1
        for jv in range(Ny_p1):
            jD = jv - 1 if jv > 0 else 0
            jU = jv + 1 if jv < Ny_p1 - 1 else Ny_p1 - 1
            lap[i, jv] = (v[iR, jv] - 2.0*v[i, jv] + v[iL, jv]) / dx**2 \
                       + (v[i, jU] - 2.0*v[i, jv] + v[i, jD]) / dy**2
    return lap

# ============================================================
# Advection (donor-cell upwind, NumPy → Numba)
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def advect_u(u, v, dx, dy):
    Nx_p1, Ny = u.shape
    adv = np.zeros_like(u)

    # x-flux donor-cell
    for iu in prange(Nx_p1):
        iuL = iu - 1 if iu > 0 else 0
        iuR = iu + 1 if iu < Nx_p1 - 1 else Nx_p1 - 1
        for j in range(Ny):
            u_up = u[iuL, j] if u[iu, j] >= 0.0 else u[iuR, j]
            F_i  = u[iu, j] * u_up
            up_im = u[iuL, j] if u[iuL, j] >= 0.0 else u[iu, j]
            F_im = u[iuL, j] * up_im
            adv[iu, j] -= (F_i - F_im) / dx

    # y-flux donor-cell using v at u
    v_u = v_at_u_from_v(v)  # serial JIT, safe to call
    for iu in prange(Nx_p1):
        for j in range(Ny):
            jD = j - 1 if j > 0 else 0
            jU = j + 1 if j < Ny - 1 else Ny - 1
            uv_ij  = u[iu, j]  * v_u[iu, j]
            uv_up  = u[iu, jU] * v_u[iu, jU]
            uv_dn  = u[iu, jD] * v_u[iu, jD]
            if v_u[iu, j] >= 0.0:
                adv[iu, j] -= (uv_ij - uv_dn) / dy
            else:
                adv[iu, j] -= (uv_up - uv_ij) / dy

    return adv

@njit(cache=True, fastmath=True, parallel=True)
def advect_v(u, v, dx, dy):
    Nx, Ny_p1 = v.shape
    adv = np.zeros_like(v)

    # x-flux donor-cell using u at v
    u_v = u_at_v_from_u(u)  # serial JIT
    for i in prange(Nx):
        iL = i - 1 if i > 0 else 0
        iR = i + 1 if i < Nx - 1 else Nx - 1
        for jv in range(Ny_p1):
            uv_ij = u_v[i, jv] * v[i, jv]
            if u_v[i, jv] >= 0.0:
                uv_im = u_v[iL, jv] * v[iL, jv]
                adv[i, jv] -= (uv_ij - uv_im) / dx
            else:
                uv_ip = u_v[iR, jv] * v[iR, jv]
                adv[i, jv] -= (uv_ip - uv_ij) / dx

    # y-flux donor-cell (v^2)
    for i in prange(Nx):
        for jv in range(Ny_p1):
            jD = jv - 1 if jv > 0 else 0
            jU = jv + 1 if jv < Ny_p1 - 1 else Ny_p1 - 1
            v_up = v[i, jD] if v[i, jv] >= 0.0 else v[i, jU]
            K_j  = v[i, jv] * v_up
            v_upm = v[i, jD] if v[i, jD] >= 0.0 else v[i, jv]
            K_jm = v[i, jD] * v_upm
            adv[i, jv] -= (K_j - K_jm) / dy
    return adv

# ============================================================
# Helmholtz (semi-implicit diffusion) via Jacobi (Numba)
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def helmholtz_u(u_rhs, dt, nu, dx, dy, iters):
    Nx_p1, Ny = u_rhs.shape
    u = u_rhs.copy()
    coef = 1.0 + 2.0 * dt * nu * (1.0/dx**2 + 1.0/dy**2)
    for _ in range(iters):
        u_new = np.empty_like(u)
        for iu in prange(Nx_p1):
            iuL = iu - 1 if iu > 0 else 0
            iuR = iu + 1 if iu < Nx_p1 - 1 else Nx_p1 - 1
            for j in range(Ny):
                jD = j - 1 if j > 0 else 0
                jU = j + 1 if j < Ny - 1 else Ny - 1
                u_new[iu, j] = (u_rhs[iu, j] +
                                dt * nu * ((u[iuL, j] + u[iuR, j]) / dx**2 +
                                           (u[iu, jD] + u[iu, jU]) / dy**2)) / coef
        u = u_new
    return u

@njit(cache=True, fastmath=True, parallel=True)
def helmholtz_v(v_rhs, dt, nu, dx, dy, iters):
    Nx, Ny_p1 = v_rhs.shape
    v = v_rhs.copy()
    coef = 1.0 + 2.0 * dt * nu * (1.0/dx**2 + 1.0/dy**2)
    for _ in range(iters):
        v_new = np.empty_like(v)
        for i in prange(Nx):
            iL = i - 1 if i > 0 else 0
            iR = i + 1 if i < Nx - 1 else Nx - 1
            for jv in range(Ny_p1):
                jD = jv - 1 if jv > 0 else 0
                jU = jv + 1 if jv < Ny_p1 - 1 else Ny_p1 - 1
                v_new[i, jv] = (v_rhs[i, jv] +
                                dt * nu * ((v[iL, jv] + v[iR, jv]) / dx**2 +
                                           (v[i, jD] + v[i, jU]) / dy**2)) / coef
        v = v_new
    return v

# ============================================================
# IBM: implicit normal-only penalization inside obstacle
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def implicit_penalize_u(u_star, v_on_u, n_x, n_y, alpha_dt, chi_u):
    Nx_p1, Ny = u_star.shape
    out = u_star.copy()
    for iu in prange(Nx_p1):
        for j in range(Ny):
            proj_n    = u_star[iu, j] * n_x[iu, j] + v_on_u[iu, j] * n_y[iu, j]
            damp      = 1.0 / (1.0 + alpha_dt * chi_u[iu, j])
            proj_new  = proj_n * damp
            delta_un  = (proj_new - proj_n) * n_x[iu, j]
            out[iu, j] = u_star[iu, j] + delta_un
    return out

@njit(cache=True, fastmath=True, parallel=True)
def implicit_penalize_v(v_star, u_on_v, n_x, n_y, alpha_dt, chi_v):
    Nx, Ny_p1 = v_star.shape
    out = v_star.copy()
    for i in prange(Nx):
        for jv in range(Ny_p1):
            proj_n    = u_on_v[i, jv] * n_x[i, jv] + v_star[i, jv] * n_y[i, jv]
            damp      = 1.0 / (1.0 + alpha_dt * chi_v[i, jv])
            proj_new  = proj_n * damp
            delta_vn  = (proj_new - proj_n) * n_y[i, jv]
            out[i, jv] = v_star[i, jv] + delta_vn
    return out

# ---------- Explicit no-penetration clamp in band ----------
@njit(cache=True, fastmath=True)
def enforce_no_penetration_u(u, v_on_u, n_x, n_y, band_u):
    Nx_p1, Ny = u.shape
    for iu in range(Nx_p1):
        for j in range(Ny):
            if band_u[iu, j]:
                vn = u[iu, j] * n_x[iu, j] + v_on_u[iu, j] * n_y[iu, j]
                u[iu, j] -= vn * n_x[iu, j]
    return u

@njit(cache=True, fastmath=True)
def enforce_no_penetration_v(v, u_on_v, n_x, n_y, band_v):
    Nx, Ny_p1 = v.shape
    for i in range(Nx):
        for jv in range(Ny_p1):
            if band_v[i, jv]:
                vn = u_on_v[i, jv] * n_x[i, jv] + v[i, jv] * n_y[i, jv]
                v[i, jv] -= vn * n_y[i, jv]
    return v

#@reject_nans
def enforce_slip_obstacle(u, v, dx, dy, poly_grid):

    # fluxes from velocity field, taking into account the fact that edges
    # intersected by the obstacle are only partially valid
    uflux = u * poly_grid.dyfrac * dy
    vflux = v * poly_grid.dxfrac * dx

    poly_grid.update_fluxes(uflux=uflux, vflux=vflux)

    # back to velocity
    tol = 1.e-8 # need to avoid leakeage

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if poly_grid.dyfrac[i, j] > 1.e-3:
                u[i, j] = uflux[i, j] / (poly_grid.dyfrac[i, j] * dy)
            else:
                u[i, j] = 0.0
    #u = np.where(poly_grid.dyfrac * dy > tol, uflux / (poly_grid.dyfrac * dy), 0.0)

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if poly_grid.dxfrac[i, j] > 1.e-3:
                v[i, j] = vflux[i, j] / (poly_grid.dxfrac[i, j] * dx)
            else:
                v[i, j] = 0.0
    #v = np.where(poly_grid.dxfrac * dx > tol, vflux / (poly_grid.dxfrac * dx), 0.0)

    return u, v


# ============================================================
# Pressure Poisson: Red–Black Gauss–Seidel (RBGS)
# ============================================================

@njit(cache=True, fastmath=True)
def poisson_pressure_rbgs(rhs, chi_p, dx, dy, max_iter=5000, tol=1e-8):
    Nx, Ny = rhs.shape
    p = np.zeros_like(rhs)
    cdx2 = dx*dx
    cdy2 = dy*dy
    denom = 2.0 * (cdx2 + cdy2)

    for it in range(max_iter):
        max_res = 0.0

        # RED pass (i+j even)
        for i in range(Nx):
            iL = i - 1 if i > 0 else 0
            iR = i + 1 if i < Nx - 1 else Nx - 1
            for j in range(Ny):
                if ((i + j) & 1) != 0:  # not red
                    continue
                if chi_p[i, j] > 0.5:
                    continue
                jD = j - 1 if j > 0 else 0
                jU = j + 1 if j < Ny - 1 else Ny - 1

                p_new = ((p[iR, j] + p[iL, j]) * cdy2 +
                         (p[i, jU] + p[i, jD]) * cdx2 -
                         rhs[i, j] * cdx2 * cdy2) / denom
                r = abs(p_new - p[i, j])
                if r > max_res:
                    max_res = r
                p[i, j] = p_new

        # BLACK pass (i+j odd)
        for i in range(Nx):
            iL = i - 1 if i > 0 else 0
            iR = i + 1 if i < Nx - 1 else Nx - 1
            for j in range(Ny):
                if ((i + j) & 1) == 0:  # not black
                    continue
                if chi_p[i, j] > 0.5:
                    continue
                jD = j - 1 if j > 0 else 0
                jU = j + 1 if j < Ny - 1 else Ny - 1

                p_new = ((p[iR, j] + p[iL, j]) * cdy2 +
                         (p[i, jU] + p[i, jD]) * cdx2 -
                         rhs[i, j] * cdx2 * cdy2) / denom
                r = abs(p_new - p[i, j])
                if r > max_res:
                    max_res = r
                p[i, j] = p_new

        if max_res < tol:
            break

    # Remove mean pressure (anchor)
    sum_p = 0.0
    cnt   = 0
    for i in range(Nx):
        for j in range(Ny):
            if chi_p[i, j] < 0.5:
                sum_p += p[i, j]
                cnt += 1
    if cnt > 0:
        p_mean = sum_p / cnt
        for i in range(Nx):
            for j in range(Ny):
                p[i, j] -= p_mean

    return p

# ============================================================
# Boundary Conditions (Python helpers)
# ============================================================

def apply_wall_slip(u, v):
    # Free-slip walls: v = 0 (no-penetration), du/dy = 0 (mirror)
    v[:, 0]  = 0.0
    v[:, -1] = 0.0
    u[:, 0]  = u[:, 1]
    u[:, -1] = u[:, -2]

def apply_inflow(u, v, uin_fun, Ly, Ny):
    y_u = (np.arange(Ny) + 0.5) * (Ly / Ny)
    u[0, :] = uin_fun(y_u)
    v[0, :] = 0.0

def apply_outflow(u, v, uin_fun, Ly, Ny):
    y_u = (np.arange(Ny) + 0.5) * (Ly / Ny)
    u[-1, :] = uin_fun(y_u)
    v[-1, :] = 0.0

def apply_outflow_convective(u, v, dt, dx, u_out):
    """
    Convective outlet:
    φ_N = φ_{N-1} - (dt*U_out/dx) (φ_{N-1} - φ_{N-2})
    """
    Nx_p1, _ = u.shape
    Nx = Nx_p1 - 1
    if Nx_p1 >= 3:
        u[Nx, :] = u[Nx - 1, :] - (dt * u_out / dx) * (u[Nx - 1, :] - u[Nx - 2, :])
    Nx_v, _ = v.shape
    if Nx_v >= 3:
        v[Nx_v - 1, :] = v[Nx_v - 2, :] - (dt * u_out / dx) * (v[Nx_v - 2, :] - v[Nx_v - 3, :])

# ============================================================
# Adaptive time step (Python)
# ============================================================

def pick_dt(u, v, dx, dy, nu, cfl_target=0.4, diff_safety=0.3, dt_floor=1e-6, dt_ceiling=0.5):
    umax = max(float(np.max(np.abs(u))), 1e-12)
    vmax = max(float(np.max(np.abs(v))), 1e-12)
    dt_adv  = cfl_target * min(dx / umax, dy / vmax)
    dt_diff = diff_safety * min(dx*dx, dy*dy) / max(nu, 1e-12)
    return max(dt_floor, min(dt_adv, dt_diff, dt_ceiling))

# ============================================================
# Monitoring: max |u·n| in the band
# ============================================================

def max_normal_penetration(u, v, n_u_x, n_u_y, n_v_x, n_v_y, band_u, band_v):
    v_on_u = v_at_u_from_v(v)
    u_on_v = u_at_v_from_u(u)
    vn_u = np.abs(u * n_u_x + v_on_u * n_u_y)
    vn_v = np.abs(u_on_v * n_v_x + v * n_v_y)
    mnu = np.max(vn_u[band_u]) if np.any(band_u) else 0.0
    mnv = np.max(vn_v[band_v]) if np.any(band_v) else 0.0
    return float(mnu), float(mnv)

# ============================================================
# One time step (Python orchestration calling numba kernels)
# ============================================================

def step(u, v, p, params, masks, normals, bands, uin_fun, poly_grid):
    nu   = params["nu"]
    rho  = params["rho"]
    dx   = params["dx"]
    dy   = params["dy"]
    fx   = params.get("fx", 0.0)
    alpha= params.get("alpha", 50.0)
    Ly   = params["Ly"]
    Ny   = u.shape[1]

    # Adaptive dt
    dt = pick_dt(u, v, dx, dy, nu,
                 cfl_target=params.get("cfl_target", 0.4),
                 diff_safety=params.get("diff_safety", 0.3),
                 dt_floor=params.get("dt_floor", 1e-6),
                 dt_ceiling=params.get("dt_ceiling", 0.5))
    params["dt"] = dt

    chi_p, chi_u, chi_v = masks
    (n_u_x, n_u_y), (n_v_x, n_v_y) = normals
    band_u, band_v = bands

    # BCs before operators
    apply_wall_slip(u, v)
    apply_inflow(u, v, uin_fun, Ly, Ny)
    # u_out = float(np.mean(u[-2, :])) if u.shape[0] > 2 else 0.0
    # apply_outflow_convective(u, v, dt, dx, u_out)
    apply_outflow(u, v, uin_fun, Ly, Ny)

    # Advection
    adv_u = advect_u(u, v, dx, dy)
    adv_v = advect_v(u, v, dx, dy)

    # Explicit non-diffusive RHS
    f_u = fx * np.ones_like(u)
    rhs_u = u + dt * (-adv_u + f_u)
    rhs_v = v + dt * (-adv_v)

    # Semi-implicit diffusion
    u_star = helmholtz_u(rhs_u, dt, nu, dx, dy, iters=params.get("helmholtz_iters", 60))
    v_star = helmholtz_v(rhs_v, dt, nu, dx, dy, iters=params.get("helmholtz_iters", 60))

    # Implicit IBM (normal-only) inside solid
    v_on_u = v_at_u_from_v(v_star)
    u_on_v = u_at_v_from_u(u_star)
    alpha_dt = alpha * dt
    u_star = implicit_penalize_u(u_star, v_on_u, n_u_x, n_u_y, alpha_dt, chi_u)
    v_star = implicit_penalize_v(v_star, u_on_v, n_v_x, n_v_y, alpha_dt, chi_v)

    # Re-enforce BCs
    apply_wall_slip(u_star, v_star)
    apply_inflow(u_star, v_star, uin_fun, Ly, Ny)
    # u_out = float(np.mean(u_star[-2, :])) if u_star.shape[0] > 2 else 0.0
    # apply_outflow_convective(u_star, v_star, dt, dx, u_out)
    apply_outflow(u_star, v_star, uin_fun, Ly, Ny)

    # First projection (RBGS)
    div_star = divergence(u_star, v_star, dx, dy)
    rhs = (rho / dt) * (div_star * (1.0 - chi_p))
    p_new = poisson_pressure_rbgs(rhs, chi_p, dx, dy, max_iter=5000, tol=1e-8)

    dpdx_u = grad_p_to_u(p_new, dx)
    dpdy_v = grad_p_to_v(p_new, dy)
    u_next = u_star - dt / rho * dpdx_u
    v_next = v_star - dt / rho * dpdy_v

    # Clamp–Project–Clamp loop
    n_ib_iter = params.get("n_ib_iter", 2)
    for _ in range(n_ib_iter):
        # clamp u·n=0 in band
        v_on_u_corr = v_at_u_from_v(v_next)
        u_on_v_corr = u_at_v_from_u(u_next)
        # u_next = enforce_no_penetration_u(u_next, v_on_u_corr, n_u_x, n_u_y, band_u)
        # v_next = enforce_no_penetration_v(v_next, u_on_v_corr, n_v_x, n_v_y, band_v)
        u_next, v_next = enforce_slip_obstacle(u_next, v_next, dx, dy, poly_grid)

        # re-project (RBGS)
        div_fix = divergence(u_next, v_next, dx, dy)
        rhs_fix = (rho / dt) * (div_fix * (1.0 - chi_p))
        p_fix = poisson_pressure_rbgs(rhs_fix, chi_p, dx, dy, max_iter=3000, tol=1e-8)
        u_next = u_next - dt / rho * grad_p_to_u(p_fix, dx)
        v_next = v_next - dt / rho * grad_p_to_v(p_fix, dy)

    # Final clamp so last operation preserves no-penetration
    v_on_u_corr = v_at_u_from_v(v_next)
    u_on_v_corr = u_at_v_from_u(u_next)
    # u_next = enforce_no_penetration_u(u_next, v_on_u_corr, n_u_x, n_u_y, band_u)
    # v_next = enforce_no_penetration_v(v_next, u_on_v_corr, n_v_x, n_v_y, band_v)
    u_next, v_next = enforce_slip_obstacle(u_next, v_next, dx, dy, poly_grid)


    # Mask obstacle interior (robustness)
    u_next *= (1.0 - chi_u)
    v_next *= (1.0 - chi_v)

    # Final BCs
    apply_wall_slip(u_next, v_next)
    apply_inflow(u_next, v_next, uin_fun, Ly, Ny)
    # u_out = float(np.mean(u_next[-2, :])) if u_next.shape[0] > 2 else 0.0
    # apply_outflow_convective(u_next, v_next, dt, dx, u_out)
    apply_outflow(u_next, v_next, uin_fun, Ly, Ny)

    return u_next, v_next, p_new

# ============================================================
# Driver / example
# ============================================================

def run_channel_with_obstacle_inout():
    # Domain & resolution
    Lx, Ly = 2.0, 1.0
    Nx, Ny = 64, 32
    dx, dy = Lx / Nx, Ly / Ny

    # Fluid params
    rho = 1.0
    nu  = 0.01
    fx  = 0.0

    # Inflow profile (uniform)
    def uin_fun(y):
        U0 = 0.3
        return U0 * np.ones_like(y)

    # Polygon obstacle (diamond)
    cx, cy = 0.8, 0.5
    w, h = 0.2, 0.25
    poly = [(cx - w/2, cy), (cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2)]

    poly_grid = PolyGrid(poly, Nx=Nx, Ny=Ny, dx=dx, dy=dy, debug=False, closed=True)

    # Masks & normals
    chi_p, chi_u, chi_v = build_masks(Lx, Ly, Nx, Ny, poly)
    normals = build_face_normals(Lx, Ly, Nx, Ny, poly)

    # Thin band around obstacle (wider band helps at corners)
    band_thickness = 1.0 * min(dx, dy)
    band_u, band_v = build_face_band(Lx, Ly, Nx, Ny, poly, band_thickness)

    # Fields
    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))
    p = np.zeros((Nx, Ny))

    params = dict(
        nu=nu, rho=rho, dx=dx, dy=dy, fx=fx, alpha=50.0, Ly=Ly,
        cfl_target=0.4, diff_safety=0.3, dt_floor=1e-6, dt_ceiling=0.5,
        helmholtz_iters=60,
        n_ib_iter=2
    )

    nsteps = 1000
    for istep in range(nsteps):
        u, v, p = step(u, v, p, params,
                       (chi_p, chi_u, chi_v),
                       normals,
                       (band_u, band_v),
                       uin_fun, 
                       poly_grid)

        dt = params["dt"]
        umax = max(float(np.max(np.abs(u))), 1e-12)
        vmax = max(float(np.max(np.abs(v))), 1e-12)
        cfl = max(umax * dt / dx, vmax * dt / dy)

        if istep % 10 == 0:
            kinE = 0.5 * (np.mean(u**2) + np.mean(v**2))
            div_norm = np.linalg.norm(divergence(u, v, dx, dy)) / (Nx * Ny)
            mnu, mnv = max_normal_penetration(u, v,
                                              normals[0][0], normals[0][1],
                                              normals[1][0], normals[1][1],
                                              band_u, band_v)
            print(f"step={istep:4d} dt={dt:.3e} CFL={cfl:.3f} KE={kinE:.6e} "
                  f"divL2/N={div_norm:.3e}  max|u·n|_band: u={mnu:.3e}, v={mnv:.3e}")

            write_vtr(f'ibm_cgrid_channel_{istep:05d}.vtr', u, v, p, Lx, Ly, chi_p)

    meta = dict(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dx=dx, dy=dy)
    return u, v, p, (chi_p, chi_u, chi_v), normals, (band_u, band_v), meta


if __name__ == "__main__":
    print(f"Numba available: {NUMBA_AVAILABLE}")
    u, v, p, masks, normals, bands, meta = run_channel_with_obstacle_inout()
    print("Simulation finished.")
