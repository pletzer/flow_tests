import numpy as np
import vtk
from vtk.util import numpy_support

# ----------------------------
# Parameters
# ----------------------------
Nx, Ny = 128, 64           # number of pressure cells
Lx, Ly = 2.0, 1.0          # domain size
dx, dy = Lx / Nx, Ly / Ny
dt = 0.002
nu = 0.01                  # kinematic viscosity
Uin = 1.0                  # inflow/outflow velocity
nsteps = 500
p_iters = 50

# ----------------------------
# Staggered fields
# ----------------------------
u = np.zeros((Nx+1, Ny))   # u at vertical faces
v = np.zeros((Nx, Ny+1))   # v at horizontal faces
p = np.zeros((Nx, Ny))     # pressure at cell centers

# ----------------------------
# Obstacle mask (cell-centered)
# ----------------------------
solid = np.zeros((Nx, Ny), dtype=bool)
i0, i1 = int(0.8 / dx), int(1.0 / dx)
j0, j1 = int(0.3 / dy), int(0.7 / dy)
solid[i0:i1, j0:j1] = True

# ----------------------------
# Helper functions
# ----------------------------
def cell_center_velocity(u, v):
    """Reconstruct cell-centered velocity from staggered fields."""
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    return uc, vc

def apply_velocity_bc(u, v):
    """Apply velocity boundary conditions on MAC grid."""
    # Inlet (left)
    u[0, :] = Uin
    u[1, :] = Uin   # first interior face
    v[0, :] = 0.0

    # Outlet (right)
    u[-1, :] = Uin
    u[-2, :] = Uin
    v[-1, :] = 0.0

    # Top/bottom walls (no-slip)
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

def enforce_obstacle(u, v):
    """Set velocity to zero inside the obstacle."""
    for i in range(Nx):
        for j in range(Ny):
            if solid[i, j]:
                u[i, j] = 0.0
                u[i+1, j] = 0.0
                v[i, j] = 0.0
                v[i, j+1] = 0.0

def apply_pressure_bc(p):
    """Neumann BCs for pressure (dp/dn = 0 at boundaries)."""
    p[0, :] = p[1, :]       # inlet
    p[-1, :] = p[-2, :]     # outlet
    p[:, 0] = p[:, 1]       # bottom wall
    p[:, -1] = p[:, -2]     # top wall

def write_vtr(filename, u, v, p, dx, dy):
    """Write velocity and pressure to VTK RectilinearGrid."""
    Nx, Ny = p.shape
    uc, vc = cell_center_velocity(u, v)

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(Nx+1, Ny+1, 1)

    x = np.linspace(0, Nx*dx, Nx+1)
    y = np.linspace(0, Ny*dy, Ny+1)
    z = np.array([0.0])

    grid.SetXCoordinates(numpy_support.numpy_to_vtk(x))
    grid.SetYCoordinates(numpy_support.numpy_to_vtk(y))
    grid.SetZCoordinates(numpy_support.numpy_to_vtk(z))

    # Pressure (cell data)
    p_vtk = numpy_support.numpy_to_vtk(p.ravel(order="F"), deep=True)
    p_vtk.SetName("pressure")
    grid.GetCellData().AddArray(p_vtk)

    # Velocity (cell data, 3D vector)
    vel = np.zeros((Nx, Ny, 3))
    vel[:, :, 0] = uc
    vel[:, :, 1] = vc

    vel_vtk = numpy_support.numpy_to_vtk(vel.reshape(-1, 3, order="F"), deep=True)
    vel_vtk.SetName("velocity")
    grid.GetCellData().AddArray(vel_vtk)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

# ----------------------------
# Main time-stepping loop
# ----------------------------
for step in range(nsteps):
    apply_velocity_bc(u, v)
    enforce_obstacle(u, v)

    # ----------------------------
    # Advection + diffusion (u)
    # ----------------------------
    u_star = u.copy()
    for i in range(1, Nx):
        for j in range(1, Ny-1):
            u_c = u[i, j]
            v_c = 0.25*(v[i-1, j]+v[i, j]+v[i-1, j+1]+v[i, j+1])

            dudx = (u[i, j]-u[i-1, j])/dx
            dudy = (u[i, j]-u[i, j-1])/dy

            lapu = ((u[i+1, j]-2*u[i, j]+u[i-1, j])/dx**2 +
                    (u[i, j+1]-2*u[i, j]+u[i, j-1])/dy**2)

            u_star[i, j] = u[i, j] + dt*(-u_c*dudx - v_c*dudy + nu*lapu)

    # ----------------------------
    # Advection + diffusion (v)
    # ----------------------------
    v_star = v.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny):
            u_c = 0.25*(u[i,j-1]+u[i+1,j-1]+u[i,j]+u[i+1,j])
            v_c = v[i, j]

            dvdx = (v[i, j]-v[i-1, j])/dx
            dvdy = (v[i, j]-v[i, j-1])/dy

            lapv = ((v[i+1,j]-2*v[i,j]+v[i-1,j])/dx**2 +
                    (v[i,j+1]-2*v[i,j]+v[i,j-1])/dy**2)

            v_star[i,j] = v[i,j] + dt*(-u_c*dvdx - v_c*dvdy + nu*lapv)

    apply_velocity_bc(u_star, v_star)
    enforce_obstacle(u_star, v_star)

    # ----------------------------
    # Pressure Poisson equation
    # ----------------------------
    rhs = np.zeros_like(p)
    for i in range(Nx):
        for j in range(Ny):
            rhs[i,j] = ((u_star[i+1,j]-u_star[i,j])/dx +
                        (v_star[i,j+1]-v_star[i,j])/dy)/dt

    for _ in range(p_iters):
        p_new = p.copy()
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                if not solid[i,j]:
                    p_new[i,j] = 0.25*(p[i+1,j]+p[i-1,j]+p[i,j+1]+p[i,j-1]-dx*dy*rhs[i,j])
        p = p_new
        apply_pressure_bc(p)

    # ----------------------------
    # Projection step
    # ----------------------------
    for i in range(1, Nx):
        for j in range(Ny):
            u[i,j] = u_star[i,j] - dt*(p[i,j]-p[i-1,j])/dx
    for i in range(Nx):
        for j in range(1, Ny):
            v[i,j] = v_star[i,j] - dt*(p[i,j]-p[i,j-1])/dy

    # Reapply velocity BCs AFTER projection (critical)
    apply_velocity_bc(u, v)
    enforce_obstacle(u, v)

    # ----------------------------
    # Save VTK every 100 steps
    # ----------------------------
    if step % 100 == 0:
        write_vtr(f"channel_{step:05d}.vtr", u, v, p, dx, dy)
        print(f"Step {step} saved.")

