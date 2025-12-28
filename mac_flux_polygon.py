"""
This version uses fluxes instead of velocity updates
"""

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
nsteps = 800
p_iters = 80
vtk_stride = 100

# ============================================================
# Polygonal obstacle (triangle)
# ============================================================
def ensure_ccw(poly):
    area = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1) % len(poly)]
        area += x0*y1 - x1*y0
    if area < 0:
        poly = poly[::-1]
    return poly

polygon = np.array([
    [1.2, 0.0],
    [0.8, 0.7],
    [0.3, 0.0]
])
polygon = ensure_ccw(polygon)

def polygon_normals(poly):
    normals = []
    for k in range(len(poly)):
        p0 = poly[k]
        p1 = poly[(k+1) % len(poly)]
        edge = p1 - p0
        n = np.array([edge[1], -edge[0]])  # outward for CCW
        n /= np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

normals = polygon_normals(polygon)

def signed_distance_polygon(x, y):
    dmax = -1e20
    for p, n in zip(polygon, normals):
        d = (x - p[0]) * n[0] + (y - p[1]) * n[1]
        dmax = max(dmax, d)
    return dmax

def nearest_polygon_normal(x, y):
    dmin = 1e20
    nbest = None
    for p, n in zip(polygon, normals):
        d = abs((x - p[0]) * n[0] + (y - p[1]) * n[1])
        if d < dmin:
            dmin = d
            nbest = n
    return nbest

# ============================================================
# VTK output
# ============================================================
def write_vtr(fname, u, v, Nx, Ny, Lx, Ly):
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(Nx+1, Ny+1, 1)
    grid.SetXCoordinates(numpy_support.numpy_to_vtk(np.linspace(0, Lx, Nx+1)))
    grid.SetYCoordinates(numpy_support.numpy_to_vtk(np.linspace(0, Ly, Ny+1)))
    grid.SetZCoordinates(numpy_support.numpy_to_vtk(np.array([0.0])))

    vel = np.zeros((Nx, Ny, 3))
    vel[:, :, 0] = uc
    vel[:, :, 1] = vc
    vel_vtk = numpy_support.numpy_to_vtk(vel.reshape(-1, 3, order='F'), deep=True)
    vel_vtk.SetName("velocity")
    grid.GetCellData().AddArray(vel_vtk)

    p_vtk = numpy_support.numpy_to_vtk(p.ravel(order='F'), deep=True)
    p_vtk.SetName("pressure")
    grid.GetCellData().AddArray(p_vtk)

    a_vtk = numpy_support.numpy_to_vtk(alpha.ravel(order='F'), deep=True)
    a_vtk.SetName("alpha")
    grid.GetCellData().AddArray(a_vtk)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.Write()


# ============================================================
# Staggered fields (face fluxes)
# ============================================================
u = np.zeros((Nx+1, Ny))    # x-velocity at x-faces
v = np.zeros((Nx, Ny+1))    # y-velocity at y-faces
p = np.zeros((Nx, Ny))      # cell-centered pressure

Ax = np.ones((Nx+1, Ny))    # x-face fractions
Ay = np.ones((Nx, Ny+1))    # y-face fractions

# Normals at blocked faces (for slip)
nx_face = np.zeros((Nx+1, Ny, 2))
ny_face = np.zeros((Nx, Ny+1, 2))

# ============================================================
# Cell mask and face fractions
# ============================================================
alpha = np.ones((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        xc = (i + 0.5) * dx
        yc = (j + 0.5) * dy
        if signed_distance_polygon(xc, yc) < 0:
            alpha[i, j] = 0.0

# x-faces
for i in range(1, Nx):
    for j in range(Ny):
        if alpha[i-1,j]==0 or alpha[i,j]==0:
            Ax[i,j] = 0.0
            xface = i*dx
            yface = (j+0.5)*dy
            nx_face[i,j] = nearest_polygon_normal(xface, yface)

# y-faces
for i in range(Nx):
    for j in range(1, Ny):
        if alpha[i,j-1]==0 or alpha[i,j]==0:
            Ay[i,j] = 0.0
            xface = (i+0.5)*dx
            yface = j*dy
            ny_face[i,j] = nearest_polygon_normal(xface, yface)

# ============================================================
# Utility functions
# ============================================================
def apply_velocity_bc(u, v):
    v[:,0] = 0.0
    v[:,-1] = 0.0
    u[:,0] = u[:,1]
    u[:,-1] = u[:,-2]

def apply_pressure_bc(p):
    p[0,:] = p_in
    p[-1,:] = p_out
    p[:,0] = p[:,1]
    p[:,-1] = p[:,-2]

def enforce_slip_flux(Fx, Fy):
    # x-faces
    for i in range(1,Nx):
        for j in range(Ny):
            if Ax[i,j]==0:
                n = nx_face[i,j]
                vel = np.array([Fx[i,j]/Ax[i,j] if Ax[i,j]>0 else 0.0,
                                0.5*(Fy[i-1,j]+Fy[i,j])])
                vel -= np.dot(vel, n)*n
                Fx[i,j] = vel[0]*Ax[i,j]

    # y-faces
    for i in range(Nx):
        for j in range(1,Ny):
            if Ay[i,j]==0:
                n = ny_face[i,j]
                vel = np.array([0.5*(Fx[i,j-1]+Fx[i+1,j-1]),
                                Fy[i,j]/Ay[i,j] if Ay[i,j]>0 else 0.0])
                vel -= np.dot(vel, n)*n
                Fy[i,j] = vel[1]*Ay[i,j]

# ============================================================
# Time integration (flux-based MAC)
# ============================================================
for step in range(nsteps):

    # ------------------------------
    # Fluxes
    # ------------------------------
    Fx = u * Ax
    Fy = v * Ay

    apply_velocity_bc(u, v)
    enforce_slip_flux(Fx, Fy)

    # Predictor: simple diffusion-advection (can be improved)
    u_star = u.copy()
    v_star = v.copy()

    for i in range(1, Nx):
        for j in range(1, Ny-1):
            if Ax[i,j]==0: continue
            u_c = u[i,j]
            v_c = 0.25*(v[i-1,j]+v[i,j]+v[i-1,j+1]+v[i,j+1])
            lap = ((u[i+1,j]-2*u[i,j]+u[i-1,j])/dx**2 +
                   (u[i,j+1]-2*u[i,j]+u[i,j-1])/dy**2)
            u_star[i,j] += dt * (-u_c*(u[i,j]-u[i-1,j])/dx - v_c*(u[i,j]-u[i,j-1])/dy + nu*lap)

    for i in range(1, Nx-1):
        for j in range(1, Ny):
            if Ay[i,j]==0: continue
            u_c = 0.25*(u[i,j-1]+u[i+1,j-1]+u[i,j]+u[i+1,j])
            v_c = v[i,j]
            lap = ((v[i+1,j]-2*v[i,j]+v[i-1,j])/dx**2 +
                   (v[i,j+1]-2*v[i,j]+v[i,j-1])/dy**2)
            v_star[i,j] += dt * (-u_c*(v[i,j]-v[i-1,j])/dx - v_c*(v[i,j]-v[i,j-1])/dy + nu*lap)

    apply_velocity_bc(u_star, v_star)
    Fx = u_star*Ax
    Fy = v_star*Ay
    enforce_slip_flux(Fx, Fy)

    # Pressure Poisson
    rhs = np.zeros_like(p)
    for i in range(Nx):
        for j in range(Ny):
            if alpha[i,j]==0: continue
            rhs[i,j] = (Fx[i+1,j]-Fx[i,j])/dx + (Fy[i,j+1]-Fy[i,j])/dy

    for _ in range(p_iters):
        p_new = p.copy()
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                if alpha[i,j]==0: continue
                p_new[i,j] = 0.25*(p[i+1,j]+p[i-1,j]+p[i,j+1]+p[i,j-1]-dx*dy*rhs[i,j])
        p = p_new
        apply_pressure_bc(p)

    # Projection (correct fluxes)
    for i in range(1,Nx):
        for j in range(Ny):
            if Ax[i,j]>0:
                Fx[i,j] -= dt*(p[i,j]-p[i-1,j])/dx*Ax[i,j]
    for i in range(Nx):
        for j in range(1,Ny):
            if Ay[i,j]>0:
                Fy[i,j] -= dt*(p[i,j]-p[i,j-1])/dy*Ay[i,j]

    # Recover velocities
    u = Fx / Ax
    v = Fy / Ay
    #u = np.where(np.fabs(Ax) < 0.01, 0.0, Fx / Ax)
    #v = np.where(np.fabs(Ay) < 0.01, 0.0, Fy / Ay)

    apply_velocity_bc(u, v)
    enforce_slip_flux(Fx, Fy)

    # VTK output can be added here if needed
    if step % vtk_stride == 0:
        write_vtr(f"channel_flux_polygon_{step:05d}.vtr", u, v, Nx, Ny, Lx, Ly)
        print(f"Step {step}")
