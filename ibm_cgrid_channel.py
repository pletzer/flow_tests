
"""
2D incompressible Navier-Stokes solver on an Arakawa C-grid
Channel flow with inflow/outflow BCs, free-slip walls, and a polygonal immersed obstacle.
Slip over the obstacle is enforced via normal-only penalization (Brinkman forcing).
Author: (adapted for Alexander)
"""

import numpy as np
import vtk
from vtk.util import numpy_support


def write_vtr(fname, u, v, p, Lx, Ly):

    Nx, Ny = p.shape

    uc, vc = 0.5 * (u[:-1, :] + u[1:, :]), 0.5 * (v[:, :-1] + v[:, 1:])

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(Nx+1, Ny+1, 1)

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    z = np.array([0.0])

    grid.SetXCoordinates(numpy_support.numpy_to_vtk(x))
    grid.SetYCoordinates(numpy_support.numpy_to_vtk(y))
    grid.SetZCoordinates(numpy_support.numpy_to_vtk(z))

    # Pressure
    p_vtk = numpy_support.numpy_to_vtk(
        p.ravel(order="F"), deep=True
    )
    p_vtk.SetName("pressure")
    grid.GetCellData().AddArray(p_vtk)

    # Velocity
    vel = np.zeros((Nx, Ny, 3))
    vel[:, :, 0] = uc
    vel[:, :, 1] = vc

    vel_vtk = numpy_support.numpy_to_vtk(
        vel.reshape(-1, 3, order="F"), deep=True
    )
    vel_vtk.SetName("velocity")
    grid.GetCellData().AddArray(vel_vtk)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.Write()



# ============================================================
# Geometry & masking utilities
# ============================================================

def point_in_polygon(x, y, poly):
    """
    Ray-casting point-in-polygon test.
    poly: list of (x,y) vertices (closed or open list is fine).
    Returns True if (x,y) lies strictly inside polygon.
    """
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
    """
    Build masks for p (cell centers), u (vertical faces), v (horizontal faces).
    Returns chi_p, chi_u, chi_v with 1=solid, 0=fluid (useful for penalization).
    """
    dx = Lx / Nx
    dy = Ly / Ny

    # p centers at ((i+0.5)*dx, (j+0.5)*dy)
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    chi_p = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if point_in_polygon(Xc[i, j], Yc[i, j], poly):
                chi_p[i, j] = 1.0

    # u faces at (i*dx, (j+0.5)*dy)
    xu = (np.arange(Nx + 1)) * dx
    yu = (np.arange(Ny) + 0.5) * dy
    Xu, Yu = np.meshgrid(xu, yu, indexing='ij')
    chi_u = np.zeros((Nx + 1, Ny))
    for iu in range(Nx + 1):
        for j in range(Ny):
            if point_in_polygon(Xu[iu, j], Yu[iu, j], poly):
                chi_u[iu, j] = 1.0

    # v faces at ((i+0.5)*dx, j*dy)
    xv = (np.arange(Nx) + 0.5) * dx
    yv = (np.arange(Ny + 1)) * dy
    Xv, Yv = np.meshgrid(xv, yv, indexing='ij')
    chi_v = np.zeros((Nx, Ny + 1))
    for i in range(Nx):
        for jv in range(Ny + 1):
            if point_in_polygon(Xv[i, jv], Yv[i, jv], poly):
                chi_v[i, jv] = 1.0

    return chi_p, chi_u, chi_v


def polygon_edges(poly):
    """Return list of segments [(x0,y0,x1,y1), ...]."""
    n = len(poly)
    segs = []
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        segs.append((x0, y0, x1, y1))
    return segs


def nearest_edge_normal(x, y, segs):
    """
    Return unit normal of nearest polygon edge and the distance.
    Normal direction sign does not matter for penalization.
    """
    best_d2 = 1e300
    nx, ny = 0.0, 0.0
    for (x0, y0, x1, y1) in segs:
        ex, ey = x1 - x0, y1 - y0
        L2 = ex*ex + ey*ey + 1e-16
        t = ((x - x0) * ex + (y - y0) * ey) / L2
        t = max(0.0, min(1.0, t))
        xc = x0 + t * ex
        yc = y0 + t * ey
        dx, dy = x - xc, y - yc
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            nrm = np.sqrt(ex*ex + ey*ey) + 1e-16
            nx, ny = ey / nrm, -ex / nrm  # a perpendicular unit normal
    return nx, ny, np.sqrt(best_d2)


def build_face_normals(Lx, Ly, Nx, Ny, poly):
    """
    Compute normals at u- and v-face locations against nearest polygon edge.
    Returns (n_u_x, n_u_y), (n_v_x, n_v_y).
    """
    segs = polygon_edges(poly)
    dx = Lx / Nx
    dy = Ly / Ny

    # u faces at (i*dx, (j+0.5)*dy)
    xu = (np.arange(Nx + 1)) * dx
    yu = (np.arange(Ny) + 0.5) * dy
    n_u_x = np.zeros((Nx + 1, Ny))
    n_u_y = np.zeros((Nx + 1, Ny))
    for iu in range(Nx + 1):
        for j in range(Ny):
            nx, ny, _ = nearest_edge_normal(xu[iu], yu[j], segs)
            n_u_x[iu, j] = nx
            n_u_y[iu, j] = ny

    # v faces at ((i+0.5)*dx, j*dy)
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


# ============================================================
# Staggered interpolations
# ============================================================

def v_at_u_from_v(v):
    """
    Interpolate v (Nx, Ny+1) to u-face locations (Nx+1, Ny) using 4-point average.
    """
    Nx = v.shape[0]
    Ny = v.shape[1] - 1
    v_u = np.zeros((Nx + 1, Ny))
    for iu in range(Nx + 1):
        iL = (iu - 1) if iu > 0 else 0
        iR = iu if iu < Nx else Nx - 1
        for j in range(Ny):
            v_u[iu, j] = 0.25 * (v[iL, j] + v[iR, j] + v[iL, j+1] + v[iR, j+1])
    return v_u


def u_at_v_from_u(u):
    """
    Interpolate u (Nx+1, Ny) to v-face locations (Nx, Ny+1) using 4-point average.
    """
    Nx = u.shape[0] - 1
    Ny = u.shape[1]
    u_v = np.zeros((Nx, Ny + 1))
    for i in range(Nx):
        for jv in range(Ny + 1):
            jD = max(jv - 1, 0)
            jU = min(jv, Ny - 1)
            u_v[i, jv] = 0.25 * (u[i, jD] + u[i+1, jD] + u[i, jU] + u[i+1, jU])
    return u_v


# ============================================================
# Differential operators (C-grid)
# ============================================================

def divergence(u, v, dx, dy):
    """
    Div at p centers (Nx, Ny):
    div = (u[i+1,j] - u[i,j]) / dx + (v[i,j+1] - v[i,j]) / dy
    """
    Nx = u.shape[0] - 1
    Ny = u.shape[1]
    div = np.zeros((Nx, Ny))
    div += (u[1:, :] - u[:-1, :]) / dx
    div += (v[:, 1:] - v[:, :-1]) / dy
    return div


def grad_p_to_u(p, dx):
    """
    dp/dx at u faces: shape (Nx+1, Ny), Neumann at boundaries (gp[0]=gp[Nx]=0).
    """
    Nx, Ny = p.shape
    gp = np.zeros((Nx + 1, Ny))
    gp[1:Nx, :] = (p[1:, :] - p[:-1, :]) / dx
    gp[0, :] = 0.0
    gp[Nx, :] = 0.0
    return gp


def grad_p_to_v(p, dy):
    """
    dp/dy at v faces: shape (Nx, Ny+1), Neumann at top/bottom (gp[:,0]=gp[:,Ny]=0).
    """
    Nx, Ny = p.shape
    gp = np.zeros((Nx, Ny + 1))
    gp[:, 1:Ny] = (p[:, 1:] - p[:, :-1]) / dy
    gp[:, 0] = 0.0
    gp[:, Ny] = 0.0
    return gp


def laplacian_u(u, dx, dy):
    """
    Laplacian for u at faces (Nx+1, Ny), Neumann in x and y via mirrored neighbors.
    """
    Nx_p1, Ny = u.shape
    lap = np.zeros_like(u)

    # x-direction (Neumann via mirror)
    for iu in range(Nx_p1):
        iuL = max(iu - 1, 0)
        iuR = min(iu + 1, Nx_p1 - 1)
        lap[iu, :] += (u[iuR, :] - 2.0*u[iu, :] + u[iuL, :]) / dx**2

    # y-direction (Neumann via mirror)
    for j in range(Ny):
        jD = max(j - 1, 0)
        jU = min(j + 1, Ny - 1)
        lap[:, j] += (u[:, jU] - 2.0*u[:, j] + u[:, jD]) / dy**2

    return lap


def laplacian_v(v, dx, dy):
    """
    Laplacian for v at faces (Nx, Ny+1), Neumann in x and y via mirrored neighbors.
    """
    Nx, Ny_p1 = v.shape
    lap = np.zeros_like(v)

    # x-direction (Neumann via mirror)
    for i in range(Nx):
        iL = max(i - 1, 0)
        iR = min(i + 1, Nx - 1)
        lap[i, :] += (v[iR, :] - 2.0*v[i, :] + v[iL, :]) / dx**2

    # y-direction (Neumann via mirror)
    for jv in range(Ny_p1):
        jD = max(jv - 1, 0)
        jU = min(jv + 1, Ny_p1 - 1)
        lap[:, jv] += (v[:, jU] - 2.0*v[:, jv] + v[:, jD]) / dy**2

    return lap


# ============================================================
# Advection (donor-cell upwind)
# ============================================================

def advect_u(u, v, dx, dy):
    """
    Compute -∂(u^2)/∂x - ∂(uv)/∂y at u faces (Nx+1, Ny).
    Use central flux differences in x and upwind in y.
    """
    Nx_p1, Ny = u.shape
    adv = np.zeros_like(u)

    # d/dx (u^2)
    Fx = u*u
    for iu in range(Nx_p1):
        iuL = max(iu - 1, 0)
        iuR = min(iu + 1, Nx_p1 - 1)
        adv[iu, :] -= (Fx[iuR, :] - Fx[iuL, :]) / (2.0*dx)

    # d/dy (u*v) with upwind in y using v at u locations
    v_u = v_at_u_from_v(v)
    uv = u * v_u
    for j in range(Ny):
        jD = max(j - 1, 0)
        jU = min(j + 1, Ny - 1)
        # upwind selector per-column
        up = (v_u[:, j] >= 0.0)
        down = ~up
        adv[:, j] -= up  * (uv[:, j] - uv[:, jD]) / dy \
                     + down * (uv[:, jU] - uv[:, j]) / dy

    return adv


def advect_v(u, v, dx, dy):
    """
    Compute -∂(uv)/∂x - ∂(v^2)/∂y at v faces (Nx, Ny+1).
    Use upwind in x and central in y.
    """
    Nx, Ny_p1 = v.shape
    adv = np.zeros_like(v)

    # d/dx (u*v) with upwind in x using u at v locations
    u_v = u_at_v_from_u(u)
    uv = u_v * v
    for i in range(Nx):
        iL = max(i - 1, 0)
        iR = min(i + 1, Nx - 1)
        up = (u_v[i, :] >= 0.0)
        down = ~up
        adv[i, :] -= up  * (uv[i, :] - uv[iL, :]) / dx \
                     + down * (uv[iR, :] - uv[i, :]) / dx

    # d/dy (v^2) central
    Fy = v*v
    for jv in range(Ny_p1):
        jD = max(jv - 1, 0)
        jU = min(jv + 1, Ny_p1 - 1)
        adv[:, jv] -= (Fy[:, jU] - Fy[:, jD]) / (2.0*dy)

    return adv


# ============================================================
# IBM: normal-only penalization (slip on obstacle)
# ============================================================

def normal_only_penalty_on_u(u_face, v_interp, n_x, n_y, alpha, chi_u):
    """Penalize normal component at u faces."""
    proj_n = u_face * n_x + v_interp * n_y
    return -alpha * chi_u * (proj_n * n_x)

def normal_only_penalty_on_v(v_face, u_interp, n_x, n_y, alpha, chi_v):
    """Penalize normal component at v faces."""
    proj_n = u_interp * n_x + v_face * n_y
    return -alpha * chi_v * (proj_n * n_y)


# ============================================================
# Pressure Poisson (SOR, all-Neumann; fix mean to 0)
# ============================================================

def poisson_pressure(rhs, chi_p, dx, dy, omega=1.7, max_iter=2000, tol=1e-6):
    """
    Solve ∇²p = rhs on fluid cells (chi_p=0), Neumann on all boundaries (x & y).
    Skip solids (chi_p=1). We remove the nullspace by subtracting the mean after solve.
    """
    Nx, Ny = rhs.shape
    p = np.zeros_like(rhs)
    inv = 1.0 / (2.0*(dx*dx + dy*dy))
    for it in range(max_iter):
        max_res = 0.0
        for i in range(Nx):
            iL = max(i - 1, 0)
            iR = min(i + 1, Nx - 1)
            for j in range(Ny):
                if chi_p[i, j] > 0.5:
                    continue
                jD = max(j - 1, 0)
                jU = min(j + 1, Ny - 1)
                p_new = ((p[iR, j] + p[iL, j]) * dy*dy +
                         (p[i, jU] + p[i, jD]) * dx*dx -
                         rhs[i, j] * dx*dx * dy*dy) * inv
                res = abs(p_new - p[i, j])
                if res > max_res:
                    max_res = res
                p[i, j] = (1 - omega) * p[i, j] + omega * p_new
        if max_res < tol:
            break
    # remove nullspace (mean pressure)
    fluid_mask = (1.0 - chi_p)
    m = fluid_mask.sum()
    if m > 0:
        p_mean = (p * fluid_mask).sum() / m
        p -= p_mean
    return p


# ============================================================
# Boundary Conditions (inflow/outflow & free-slip walls)
# ============================================================

def apply_wall_slip(u, v):
    """
    Free-slip at y=0,Ly:
    v = 0 (no penetration), du/dy = 0 (copy from interior).
    """
    # v at walls
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    # u tangential slip: zero normal derivative -> mirror
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]


def apply_inflow(u, v, uin_fun, Ly, Ny):
    """
    Inflow at x=0 for u (iu=0) set to uin(y). v(i=0,:) = 0.
    uin_fun should accept array of y positions at u-face y locations (mid-cell).
    """
    y_u = (np.arange(Ny) + 0.5) * (Ly / Ny)  # u faces y at (j+0.5)*dy
    u[0, :] = uin_fun(y_u)
    v[0, :] = 0.0  # v at i=0 faces (includes j=0..Ny); no penetration at inflow


def apply_outflow_neumann(u, v):
    """
    Outflow at x=Lx: zero-gradient (Neumann).
    u(iu=Nx,:) = u(iu=Nx-1,:)
    v(i=Nx-1,:) = v(i=Nx-2,:)
    """
    Nx_p1, Ny = u.shape
    Nx = Nx_p1 - 1
    u[Nx, :] = u[Nx - 1, :]
    v[Nx - 1, :] = v[Nx - 2, :]


# ============================================================
# Time step
# ============================================================

def step(u, v, p, params, masks, normals, uin_fun):
    """
    One time-step with:
    - free-slip walls,
    - inflow at x=0 (uin_fun),
    - outflow (Neumann) at x=Lx,
    - normal-only IBM penalization for obstacle.
    """
    nu   = params["nu"]
    rho  = params["rho"]
    dt   = params["dt"]
    dx   = params["dx"]
    dy   = params["dy"]
    fx   = params.get("fx", 0.0)
    alpha = params.get("alpha", 50.0)
    Ly   = params["Ly"]
    Ny   = u.shape[1]

    chi_p, chi_u, chi_v = masks
    (n_u_x, n_u_y), (n_v_x, n_v_y) = normals

    # --- Enforce BCs BEFORE operators ---
    apply_wall_slip(u, v)
    apply_inflow(u, v, uin_fun, Ly, Ny)
    apply_outflow_neumann(u, v)

    # --- Advection (upwind) ---
    adv_u = advect_u(u, v, dx, dy)
    adv_v = advect_v(u, v, dx, dy)

    # --- Diffusion ---
    lap_u = laplacian_u(u, dx, dy)
    lap_v = laplacian_v(v, dx, dy)

    # --- Body force (drive x) ---
    f_u = fx * np.ones_like(u)
    f_v = np.zeros_like(v)

    # --- IBM: penalize ONLY normal component inside obstacle ---
    v_on_u = v_at_u_from_v(v)
    u_on_v = u_at_v_from_u(u)
    pen_u = normal_only_penalty_on_u(u, v_on_u, n_u_x, n_u_y, alpha, chi_u)
    pen_v = normal_only_penalty_on_v(v, u_on_v, n_v_x, n_v_y, alpha, chi_v)

    # --- Intermediate velocities (no pressure) ---
    u_star = u + dt * ( -adv_u + nu * lap_u + f_u + pen_u )
    v_star = v + dt * ( -adv_v + nu * lap_v + f_v + pen_v )

    # Re-enforce BCs on intermediates (important for stability)
    apply_wall_slip(u_star, v_star)
    apply_inflow(u_star, v_star, uin_fun, Ly, Ny)
    apply_outflow_neumann(u_star, v_star)

    # --- Poisson RHS ---
    div_star = divergence(u_star, v_star, dx, dy)
    rhs = (rho / dt) * (div_star * (1.0 - chi_p))  # zero RHS in solid

    # --- Pressure solve (Neumann all boundaries) ---
    p_new = poisson_pressure(rhs, chi_p, dx, dy)

    # --- Pressure correction ---
    dpdx_u = grad_p_to_u(p_new, dx)
    dpdy_v = grad_p_to_v(p_new, dy)

    u_next = u_star - dt / rho * dpdx_u
    v_next = v_star - dt / rho * dpdy_v

    # --- Mask obstacle (robustness) ---
    u_next *= (1.0 - chi_u)
    v_next *= (1.0 - chi_v)

    # --- Final BCs ---
    apply_wall_slip(u_next, v_next)
    apply_inflow(u_next, v_next, uin_fun, Ly, Ny)
    apply_outflow_neumann(u_next, v_next)

    return u_next, v_next, p_new


# ============================================================
# Driver / example
# ============================================================

def run_channel_with_obstacle_inout():
    """
    Run a demonstration: inflow/outflow with free-slip walls and slip over obstacle.
    Returns final fields and metadata.
    """
    # Domain & resolution
    Lx, Ly = 2.0, 1.0
    Nx, Ny = 32, 16
    dx, dy = Lx / Nx, Ly / Ny

    # Fluid params
    rho = 1.0
    nu  = 0.01
    fx  = 0.0  # set to small value if you prefer body-force driven flow

    # Time step (diffusive stability; adjust if advection dominates)
    dt  = 0.25 * min(dx, dy)**2 / max(nu, 1e-9)

    # Inflow profile: e.g., uniform or mild parabolic (slip walls so uniform is fine)
    def uin_fun(y):
        U0 = 0.5  # inflow magnitude
        return U0 * np.ones_like(y)
        # For a parabolic profile (Poiseuille-like):
        # return 4*U0*(y*(Ly - y))/Ly**2

    # Polygon obstacle (example: diamond centered in channel)
    cx, cy = 0.8, 0.5
    w, h = 0.2, 0.25
    poly = [
        (cx - w/2, cy),
        (cx, cy + h/2),
        (cx + w/2, cy),
        (cx, cy - h/2)
    ]

    # Masks and normals
    chi_p, chi_u, chi_v = build_masks(Lx, Ly, Nx, Ny, poly)
    normals = build_face_normals(Lx, Ly, Nx, Ny, poly)

    # Fields
    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))
    p = np.zeros((Nx, Ny))

    params = dict(nu=nu, rho=rho, dt=dt, dx=dx, dy=dy, fx=fx, alpha=50.0, Ly=Ly)

    nsteps = 1000
    for istep in range(nsteps):
        u, v, p = step(u, v, p, params, (chi_p, chi_u, chi_v), normals, uin_fun)

        # CFL monitor (advection); reduce dt if needed
        umax = max(np.max(np.abs(u)), 1e-12)
        vmax = max(np.max(np.abs(v)), 1e-12)
        cfl = max(umax * dt / dx, vmax * dt / dy)
        if cfl > 0.5:
            params["dt"] *= 0.5
            print(f"[step {istep}] CFL={cfl:.3f} -> dt={params['dt']:.3e}")

        # Diagnostics
        if istep % 10 == 0:
            kinE = 0.5 * (np.mean(u**2) + np.mean(v**2))
            div_norm = np.linalg.norm(divergence(u, v, dx, dy)) / (Nx * Ny)
            print(f"step={istep:4d}  KE={kinE:.6e}  divL2/N={div_norm:.3e}")
            write_vtr(f'ibm_cgrid_channel_{istep:05d}.vtr', u, v, p, Lx, Ly)

    meta = dict(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dx=dx, dy=dy)
    return u, v, p, (chi_p, chi_u, chi_v), normals, meta


if __name__ == "__main__":
    u, v, p, masks, normals, meta = run_channel_with_obstacle_inout()
    print("Simulation finished.")
