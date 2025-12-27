import numpy as np
from interp import interp_velocity_mimetic
import vtk

def trajectories_mimetic_rk4(x, y, u, v, Ax, Ay, p0, p1, ntraj, tintegr, alpha_min=0.0, dt=0.001):
    """
    Integrate trajectories using mimetic interpolation of u, v with RK4.
    
    Returns a vtkUnstructuredGrid containing all trajectories as polylines.
    """
    # Initialize seed points along the line from p0 to p1
    lambdas = np.linspace(0., 1., ntraj)
    xyi = np.zeros((ntraj, 2))
    xyi[:, 0] = p0[0] + lambdas*(p1[0] - p0[0])
    xyi[:, 1] = p0[1] + lambdas*(p1[1] - p0[1])

    # Store trajectories: list of arrays, one per trajectory
    traj_list = [ [xyi[i].copy()] for i in range(ntraj) ]

    # Number of time steps
    nsteps = int(np.ceil(tintegr / dt))

    for istep in range(nsteps):
        # RK4 steps
        k1 = interp_velocity_mimetic(x, y, u, v, Ax, Ay, xyi, alpha_min)
        k2 = interp_velocity_mimetic(x, y, u, v, Ax, Ay, xyi + 0.5*dt*np.column_stack(k1), alpha_min)
        k3 = interp_velocity_mimetic(x, y, u, v, Ax, Ay, xyi + 0.5*dt*np.column_stack(k2), alpha_min)
        k4 = interp_velocity_mimetic(x, y, u, v, Ax, Ay, xyi + dt*np.column_stack(k3), alpha_min)

        # Update positions
        xyi += (dt/6.0)*(np.column_stack(k1) + 2*np.column_stack(k2) + 2*np.column_stack(k3) + np.column_stack(k4))

        # Append new positions to each trajectory
        for i in range(ntraj):
            traj_list[i].append(xyi[i].copy())

    # -------------------------------
    # Build vtkUnstructuredGrid
    # -------------------------------
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    point_id = 0
    for traj in traj_list:
        npts = len(traj)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(npts)
        for k, pt in enumerate(traj):
            points.InsertNextPoint(pt[0], pt[1], 0.0)
            line.GetPointIds().SetId(k, point_id)
            point_id += 1
        lines.InsertNextCell(line)

    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    ugrid.SetCells(vtk.VTK_POLY_LINE, lines)

    return ugrid

