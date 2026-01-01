import numpy as np
import vtk
from vtk.util import numpy_support

# ============================================================
# Parameters
# ============================================================
Nx, Ny = 100, 50
Lx, Ly = 2.0, 1.0
dx, dy = Lx / Nx, Ly / Ny

dt = 0.002
nu = 0.01

p_in  = 1.0     # inlet pressure
p_out = 0.0     # outlet pressure

nsteps  = 400
p_iters = 80
vtk_stride = 100

# ============================================================
# Staggered fields (MAC grid)
# ============================================================
u = np.zeros((Nx+1, Ny))     # x-velocity (faces)
v = np.zeros((Nx, Ny+1))     # y-velocity (faces)
p = np.zeros((Nx, Ny))       # pressure (cells)

# ============================================================
# Obstacle (cell-centered mask)
# ============================================================
solid = np.zeros((Nx, Ny), dtype=bool)

# i0, i1 = int(0.8 / dx), int(1.0 / dx)
# j0, j1 = int(0.0 / dy), int(0.4 / dy)
i0, i1 = int(0.3*Lx/dx), int(0.5*Lx/dx)
j0, j1 = int(0.0*Ly/dy), int(0.6*Ly/dy)
# the obstacle is aligned to the cells
solid[i0:i1, j0:j1] = True

umatrix, vmatrix = flux_poly_matrices(poly, Nx, Ny, dx, dy)

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

def enforce_slip_obstacle(u, v, solid):
    Nx, Ny = solid.shape

    # --------------------------------
    # u-velocity (vertical faces)
    # --------------------------------
    for i in range(1, Nx):
        for j in range(Ny):
            left_solid  = solid[i-1, j]
            right_solid = solid[i, j]

            if left_solid and not right_solid:
                # Normal face → no penetration
                u[i, j] = 0.0

            elif right_solid and not left_solid:
                u[i, j] = 0.0

            elif left_solid and right_solid:
                # Inside obstacle
                u[i, j] = 0.0

    # --------------------------------
    # v-velocity (horizontal faces)
    # --------------------------------
    for i in range(Nx):
        for j in range(1, Ny):
            bottom_solid = solid[i, j-1]
            top_solid    = solid[i, j]

            if bottom_solid and not top_solid:
                v[i, j] = 0.0

            elif top_solid and not bottom_solid:
                v[i, j] = 0.0

            elif bottom_solid and top_solid:
                v[i, j] = 0.0


def apply_pressure_bc(p):
    # Dirichlet inlet/outlet
    p[0, :]  = p_in
    p[-1, :] = p_out

    # Neumann top/bottom walls
    p[:, 0]  = p[:, 1]
    p[:, -1] = p[:, -2]

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

    apply_velocity_bc(u, v)
    enforce_slip_obstacle(u, v, solid)

    # --------------------------------------------------------
    # Predictor step (u*)
    # --------------------------------------------------------
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

    apply_velocity_bc(u_star, v_star)
    enforce_slip_obstacle(u_star, v_star, solid)

    # --------------------------------------------------------
    # Pressure Poisson equation
    # --------------------------------------------------------
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
        apply_pressure_bc(p)

    # --------------------------------------------------------
    # Projection step
    # --------------------------------------------------------
    for i in range(1, Nx):
        for j in range(Ny):
            u[i, j] = u_star[i, j] - dt * (p[i, j] - p[i-1, j]) / dx

    for i in range(Nx):
        for j in range(1, Ny):
            v[i, j] = v_star[i, j] - dt * (p[i, j] - p[i, j-1]) / dy

    apply_velocity_bc(u, v)
    enforce_slip_obstacle(u, v, solid)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    if step % vtk_stride == 0:
        write_vtr(f"channelslip_{step:05d}.vtr", u, v, p)
        print(f"Step {step}")
