import numpy as np
import vtk
from vtk.util import numpy_support

# ============================================================
# Parameters
# ============================================================
Nx, Ny = 128, 64
Lx, Ly = 2.0, 1.0
dx, dy = Lx / Nx, Ly / Ny

dt = 0.002
nu = 0.01
p_in, p_out = 1.0, 0.0
nsteps = 1000
p_iters = 80
vtk_stride = 100

alpha_min = 0.1  # threshold below which cells are fully solid

# ============================================================
# Staggered fields
# ============================================================
u = np.zeros((Nx+1, Ny))
v = np.zeros((Nx, Ny+1))
p = np.zeros((Nx, Ny))

# ============================================================
# Circular obstacle level-set
# ============================================================
xc, yc = 1.0, 0.5      # circle center
R = 0.15               # radius

# cell centers
x = (np.arange(Nx) + 0.5) * dx
y = (np.arange(Ny) + 0.5) * dy
Xc, Yc = np.meshgrid(x, y, indexing='ij')

phi = np.sqrt((Xc - xc)**2 + (Yc - yc)**2) - R  # signed distance
alpha = np.clip(phi / (np.sqrt(2)*max(dx, dy)), 0, 1)  # rough volume fraction

# Cells with less than threshold become fully solid
alpha[alpha < alpha_min] = 0.0
alpha[alpha >= alpha_min] = 1.0

# Face fractions
Ax = np.zeros((Nx+1, Ny))
Ay = np.zeros((Nx, Ny+1))

# Simple approximation: faces open if neighboring cells are fluid
Ax[1:Nx, :] = 0.5 * (alpha[:-1, :] + alpha[1:, :])
Ax[0, :] = alpha[0, :]      # left
Ax[-1, :] = alpha[-1, :]    # right

Ay[:, 1:Ny] = 0.5 * (alpha[:, :-1] + alpha[:, 1:])
Ay[:, 0] = alpha[:, 0]       # bottom
Ay[:, -1] = alpha[:, -1]     # top

# ============================================================
# Utility functions
# ============================================================
def cell_center_velocity(u, v):
    uc = 0.5*(u[:-1,:]+u[1:,:])
    vc = 0.5*(v[:,:-1]+v[:,1:])
    return uc, vc

def apply_velocity_bc(u, v):
    # Bottom wall
    v[:,0] = 0.0
    u[:,0] = u[:,1]
    # Top wall
    v[:,-1] = 0.0
    u[:,-1] = u[:,-2]

def enforce_slip_obstacle(u, v, alpha):
    Nx, Ny = alpha.shape

    # u-velocity
    for i in range(1,Nx):
        for j in range(Ny):
            left  = alpha[i-1,j]
            right = alpha[i,j]
            if left==0 or right==0:
                u[i,j] = 0.0

    # v-velocity
    for i in range(Nx):
        for j in range(1,Ny):
            bottom = alpha[i,j-1]
            top    = alpha[i,j]
            if bottom==0 or top==0:
                v[i,j] = 0.0

def apply_pressure_bc(p):
    p[0,:] = p_in
    p[-1,:] = p_out
    p[:,0] = p[:,1]
    p[:,-1] = p[:,-2]

def write_vtr(fname, u, v, p):
    uc, vc = cell_center_velocity(u,v)
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(Nx+1, Ny+1, 1)
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    z = np.array([0.0])
    grid.SetXCoordinates(numpy_support.numpy_to_vtk(x))
    grid.SetYCoordinates(numpy_support.numpy_to_vtk(y))
    grid.SetZCoordinates(numpy_support.numpy_to_vtk(z))

    # pressure
    p_vtk = numpy_support.numpy_to_vtk(p.ravel(order='F'), deep=True)
    p_vtk.SetName('pressure')
    grid.GetCellData().AddArray(p_vtk)

    # velocity
    vel = np.zeros((Nx,Ny,3))
    vel[:,:,0] = uc
    vel[:,:,1] = vc
    vel_vtk = numpy_support.numpy_to_vtk(vel.reshape(-1,3,order='F'), deep=True)
    vel_vtk.SetName('velocity')
    grid.GetCellData().AddArray(vel_vtk)

    # alpha masking (cut-cell fraction)
    alpha_vtk = numpy_support.numpy_to_vtk(alpha.ravel(order='F'), deep=True)
    alpha_vtk.SetName('alpha')
    grid.GetCellData().AddArray(alpha_vtk)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.Write()

# ============================================================
# Time integration
# ============================================================
for step in range(nsteps):

    apply_velocity_bc(u,v)
    enforce_slip_obstacle(u,v,alpha)

    # ------------------------------
    # Predictor step (advection + diffusion)
    # ------------------------------
    u_star = u.copy()
    v_star = v.copy()

    # u-momentum
    for i in range(1,Nx):
        for j in range(1,Ny-1):
            u_c = u[i,j]
            v_c = 0.25*(v[i-1,j]+v[i,j]+v[i-1,j+1]+v[i,j+1])
            dudx = (u[i,j]-u[i-1,j])/dx
            dudy = (u[i,j]-u[i,j-1])/dy
            lapu = ((u[i+1,j]-2*u[i,j]+u[i-1,j])/dx**2 + 
                    (u[i,j+1]-2*u[i,j]+u[i,j-1])/dy**2)
            u_star[i,j] = u[i,j] + dt*(-u_c*dudx - v_c*dudy + nu*lapu) * Ax[i,j]

    # v-momentum
    for i in range(1,Nx-1):
        for j in range(1,Ny):
            u_c = 0.25*(u[i,j-1]+u[i+1,j-1]+u[i,j]+u[i+1,j])
            v_c = v[i,j]
            dvdx = (v[i,j]-v[i-1,j])/dx
            dvdy = (v[i,j]-v[i,j-1])/dy
            lapv = ((v[i+1,j]-2*v[i,j]+v[i-1,j])/dx**2 +
                    (v[i,j+1]-2*v[i,j]+v[i,j-1])/dy**2)
            v_star[i,j] = v[i,j] + dt*(-u_c*dvdx - v_c*dvdy + nu*lapv) * Ay[i,j]

    apply_velocity_bc(u_star, v_star)
    enforce_slip_obstacle(u_star, v_star, alpha)

    # ------------------------------
    # Pressure Poisson equation
    # ------------------------------
    rhs = np.zeros_like(p)
    for i in range(Nx):
        for j in range(Ny):
            rhs[i,j] = ((Ax[i+1,j]*u_star[i+1,j]-Ax[i,j]*u_star[i,j])/dx +
                        (Ay[i,j+1]*v_star[i,j+1]-Ay[i,j]*v_star[i,j])/dy) / alpha[i,j]

    for _ in range(p_iters):
        p_new = p.copy()
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                if alpha[i,j]==0: continue
                p_e = p[i+1,j] if alpha[i+1,j]>0 else p[i,j]
                p_w = p[i-1,j] if alpha[i-1,j]>0 else p[i,j]
                p_n = p[i,j+1] if alpha[i,j+1]>0 else p[i,j]
                p_s = p[i,j-1] if alpha[i,j-1]>0 else p[i,j]
                p_new[i,j] = (p_e+p_w+p_n+p_s - dx*dy*rhs[i,j])/4.0
        p = p_new
        apply_pressure_bc(p)

    # ------------------------------
    # Projection step
    # ------------------------------
    for i in range(1,Nx):
        for j in range(Ny):
            u[i,j] = u_star[i,j] - dt*(p[i,j]-p[i-1,j])/dx*Ax[i,j]

    for i in range(Nx):
        for j in range(1,Ny):
            v[i,j] = v_star[i,j] - dt*(p[i,j]-p[i,j-1])/dy*Ay[i,j]

    apply_velocity_bc(u,v)
    enforce_slip_obstacle(u,v,alpha)

    # ------------------------------
    # Output
    # ------------------------------
    if step % vtk_stride == 0:
        write_vtr(f"channel_cutcircle_{step:05d}.vtr", u,v,p)
        print(f"Step {step}")
