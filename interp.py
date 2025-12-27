import numpy as np

def iterp_velocity_mimetic(x, y, u, v, xyi):
    """
    Interpolate face centred u, v (Arakawa C) velocity
    @param x 1d x axis
    @param y 1d y axis
    @param u 2d x-velocity of size (Ny, Nx + 1)
    @param v 2d y-velocity of size (Ny + 1, Nx)
    @param xyz target points [(x0, y0), (x1, y1)...]
    """
    # assume uniform grid
    nx, ny = v.shape[1], u.shape[0] # number of cells
    nx1, ny1 = nx + 1, ny + 1
    dx, dy = x[1] - x[0], y[1] - y[0]
    xmin, ymin = x[0], y[0]

    # get the x, y target points
    xi, yi = xyi[:, 0], xyi[:, 1]

    # compute the target points in index space
    ifloat, jfloat = (xi - xmin)/dx, (yi - ymin)/dy

    # make sure the cells are inside the domain
    ifloat = np.maximum(0.0, np.minimum(ifloat, nx - 1.0))
    jfloat = np.maximum(0.0, np.minimum(jfloat, ny - 1.0))

    # locate the cell indices for the target points
    iint, jint = np.floor(ifloat).astype(int), np.floor(jfloat).astype(int)
    # ifloat and jfloat should always be one cell inside

    # parametric coordinates of the cell (range 0...1)
    xsi, eta = ifloat - iint, jfloat - jint

    # ui is piecewise constant in y and piecewise linear in x
    ui = (1.0 - xsi)*u[jint, iint] + xsi*u[jint, iint + 1]
    # vi is piecewise constant in x and piecewise linear in y
    vi = (1.0 - eta)*v[jint, iint] + eta*v[jint + 1, iint]

    return ui, vi