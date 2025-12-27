import numpy as np

def interp_velocity_mimetic(x, y, u, v, Ax, Ay, xyi, alpha_min=0.0):
    """
    Mimetic interpolation of MAC velocities on a staggered grid with nodal coordinates.
    
    x: 1d array of nodal x coordinates, length Nx+1
    y: 1d array of nodal y coordinates, length Ny+1
    u: x-velocity, shape (Nx+1, Ny)
    v: y-velocity, shape (Nx, Ny+1)
    Ax, Ay: face fractions, same shapes as u and v
    xyi: target points, shape (Npoints, 2)
    alpha_min: fraction threshold for blocked faces
    """
    Nx = len(x) - 1
    Ny = len(y) - 1

    xi, yi = xyi[:,0], xyi[:,1]

    # Compute indices in face arrays
    ifloat_u = np.searchsorted(x, xi) - 1   # u has Nx+1 faces
    jfloat_u = np.searchsorted(y[:-1] + np.diff(y)/2, yi)  # interpolate along y
    ifloat_v = np.searchsorted(x[:-1] + np.diff(x)/2, xi)  # interpolate along x
    jfloat_v = np.searchsorted(y, yi) - 1   # v has Ny+1 faces

    # Clip to valid ranges
    i_u = np.clip(ifloat_u, 0, Nx-1)
    j_u = np.clip(jfloat_u, 0, Ny-1)
    i_v = np.clip(ifloat_v, 0, Nx-1)
    j_v = np.clip(jfloat_v, 0, Ny-1)

    # Local coordinates for linear interpolation
    xsi_u = (xi - x[i_u]) / (x[i_u+1] - x[i_u])
    eta_v = (yi - y[j_v]) / (y[j_v+1] - y[j_v])

    # --- Interpolate fluxes ---
    Fx = (1.0 - xsi_u) * (Ax[i_u,j_u]*u[i_u,j_u]) + xsi_u * (Ax[i_u+1,j_u]*u[i_u+1,j_u])
    Fy = (1.0 - eta_v) * (Ay[i_v,j_v]*v[i_v,j_v]) + eta_v * (Ay[i_v,j_v+1]*v[i_v,j_v+1])

    # Allocate outputs
    ui = np.zeros_like(Fx)
    vi = np.zeros_like(Fy)

    # Masks to avoid division by zero
    mask_u = Ax[i_u,j_u] > alpha_min
    mask_v = Ay[i_v,j_v] > alpha_min

    ui[mask_u] = Fx[mask_u] / Ax[i_u,j_u][mask_u]
    vi[mask_v] = Fy[mask_v] / Ay[i_v,j_v][mask_v]

    return ui, vi

