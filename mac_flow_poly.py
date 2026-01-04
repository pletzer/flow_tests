import numpy as np
import vtk
from vtk.util import numpy_support
from numba import njit, prange
from poly_grid import is_inside_polygon, PolyGrid 

# ============================================================
# Parameters
# ============================================================
Nx, Ny = 40, 20 # 100, 50 #40, 20
Lx, Ly = 2.0, 1.0
dx, dy = Lx / Nx, Ly / Ny

dt = 0.002
nu = 0.01

p_in  = 1.0     # inlet pressure
p_out = 0.0     # outlet pressure

nsteps  = 1001
p_iters = 80
vtk_stride = 100


# ============================================================
# Staggered fields (MAC grid)
# ============================================================
u = np.zeros((Nx+1, Ny))     # x-velocity (faces)
v = np.zeros((Nx, Ny+1))     # y-velocity (faces)
p = np.zeros((Nx, Ny))       # pressure (cells)

# ============================================================
# Obstacle represented as a polygon
# ============================================================
poly = [(0.3*Lx, 0.0*Ly), (0.5*Lx, 0.0*Ly), (0.5*Lx, 0.6*Ly), (0.3*Lx, 0.6*Ly)]

# poly_grid computes the intersection of an polygon with a grid
poly_grid = PolyGrid(poly, Nx=Nx, Ny=Ny, dx=dx, dy=dy, debug=False, closed=True)

# set the cell-centred mask
solid = np.zeros((Nx, Ny), dtype=bool)
for j in range(Ny):
    for i in range(Nx):
        xy = ((i + 0.5)*dx, (j + 0.5)*dy)
        if is_inside_polygon(poly, xy):
            solid[i, j] = True

# ============================================================
# Utility functions
# ============================================================
def cell_center_velocity(u, v):
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    return uc, vc

def apply_velocity_bc(u, v):
    # Bottom wall (slip)
    v[:, 0] = 0.0
    u[:, 0] = u[:, 1]

    # Top wall (slip)
    v[:, -1] = 0.0
    u[:, -1] = u[:, -2]

    # Inlet/outlet: velocity is free (pressure-driven)


def enforce_slip_obstacle(u, v, dx, dy, poly_grid):

    # fluxes from velocity field, taking into account the fact edges
    # intersected by the obstacle are only partially valid
    uflux = u * poly_grid.dyfrac * dy
    vflux = v * poly_grid.dxfrac * dx

    poly_grid.update_fluxes(uflux=uflux, vflux=vflux)

    tol = 1.e-8 # need to avoid leakeage

    denom = poly_grid.dyfrac * dy
    valid_mask = denom > tol
    u *= 0
    u[valid_mask] = uflux[valid_mask] / denom[valid_mask]

    denom = poly_grid.dxfrac * dx
    valid_mask = denom > tol
    v *= 0
    v[valid_mask] = vflux[valid_mask] / denom[valid_mask]

    return u, v

@njit
def predictor(Nx, Ny, dx, dy, dt, nu, u, v):
    u_star = u.copy()
    v_star = v.copy()

    # u-momentum
    for i in range(1, Nx):
        for j in range(1, Ny-1):
            u_c = u[i, j]
            v_c = 0.25 * (v[i-1, j] + v[i, j] +
                          v[i-1, j+1] + v[i, j+1])

            dudx = (u[i, j] - u[i-1, j]) / dx
            dudy = (u[i, j] - u[i, j-1]) / dy

            lapu = (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )

            u_star[i, j] = u[i, j] + dt * (
                -u_c * dudx - v_c * dudy + nu * lapu
            )

    # v-momentum
    for i in range(1, Nx-1):
        for j in range(1, Ny):
            u_c = 0.25 * (u[i, j-1] + u[i+1, j-1] +
                          u[i, j]   + u[i+1, j])
            v_c = v[i, j]

            dvdx = (v[i, j] - v[i-1, j]) / dx
            dvdy = (v[i, j] - v[i, j-1]) / dy

            lapv = (
                (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 +
                (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2
            )

            v_star[i, j] = v[i, j] + dt * (
                -u_c * dvdx - v_c * dvdy + nu * lapv
            )

    return u_star, v_star

@njit
def pressure_poisson(Nx, Ny, dx, dy, dt, p_iters, p_in, p_out, u_star, v_star, p, solid):
    rhs = np.zeros_like(p)
    for i in range(Nx):
        for j in range(Ny):
            rhs[i, j] = (
                (u_star[i+1, j] - u_star[i, j]) / dx +
                (v_star[i, j+1] - v_star[i, j]) / dy
            ) / dt

    for _ in range(p_iters):

        p_new = p.copy()

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):

                if solid[i, j]:
                    continue

                p_e = p[i+1, j] if not solid[i+1, j] else p[i, j]
                p_w = p[i-1, j] if not solid[i-1, j] else p[i, j]
                p_n = p[i, j+1] if not solid[i, j+1] else p[i, j]
                p_s = p[i, j-1] if not solid[i, j-1] else p[i, j]

                p_new[i, j] = 0.25 * (
                    p_e + p_w + p_n + p_s - dx * dy * rhs[i, j]
                )

        p = p_new

        # Dirichlet inlet/outlet
        p[0, :]  = p_in
        p[-1, :] = p_out

        # Neumann top/bottom walls
        p[:, 0]  = p[:, 1]
        p[:, -1] = p[:, -2]

    return p

@njit
def projection(Nx, Ny, dx, dy, dt, p, u_star, v_star, u, v):
    for i in range(1, Nx):
        for j in range(Ny):
            u[i, j] = u_star[i, j] - dt * (p[i, j] - p[i-1, j]) / dx

    for i in range(Nx):
        for j in range(1, Ny):
            v[i, j] = v_star[i, j] - dt * (p[i, j] - p[i, j-1]) / dy
    return u, v

def write_vtr(fname, u, v, p):
    uc, vc = cell_center_velocity(u, v)

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
# Time integration
# ============================================================
for step in range(nsteps):

    #print(f'starting step {step}')

    apply_velocity_bc(u, v)
    u, v = enforce_slip_obstacle(u, v, dx, dy, poly_grid)

    # --------------------------------------------------------
    # Predictor step (u*)
    # --------------------------------------------------------
    u_star, v_star = predictor(Nx, Ny, dx, dy, dt, nu, u, v)

    apply_velocity_bc(u_star, v_star)
    u_star, v_star = enforce_slip_obstacle(u_star, v_star, dx, dy, poly_grid)

    # --------------------------------------------------------
    # Pressure Poisson equation
    # --------------------------------------------------------
    p = pressure_poisson(Nx, Ny, dx, dy, dt, p_iters, p_in, p_out, u_star, v_star, p, solid)

    # --------------------------------------------------------
    # Projection step
    # --------------------------------------------------------
    u, v = projection(Nx, Ny, dx, dy, dt, p, u_star, v_star, u, v)

    apply_velocity_bc(u, v)
    u, v = enforce_slip_obstacle(u, v, dx, dy, poly_grid)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    if step % vtk_stride == 0:
        write_vtr(f"channel_poly_{step:05d}.vtr", u, v, p)
        print(f"Done step {step} checksum: {np.fabs(u).sum() + np.fabs(v).sum()}")

