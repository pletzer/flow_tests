import staggered_io
from trajectories import trajectories_mimetic_rk4
import numpy as np
import vtk
import defopt

def main(*,
         input_filename: str, 
         p0: str="(0.3,0.1)", 
         p1: str="(0.3,0.9)", 
         ntraj: int=5, 
         tintegr: float=10.0, 
         alpha_min: float=0.0, 
         dt: float=0.001,
         output_vtk_filename: str="trajectories.vtu"):
    """
    Plot the mimetic stream lines
    input_filename: input file name, in npz format
    p0: starting point of the seed line, use "(x,y)" format
    p1: end point of the seed line, use "(x,y)" format
    ntraj: number of streamlines
    tintegr: integration time
    alpha_min: raction threshold for blocked faces
    dt: integration step
    output_vtk_filename: output file 
    """
    
    x, y, u, v, Ax, Ay = staggered_io.load_staggered_fields(input_filename)
    p0 = np.array(eval(p0))
    p1 = np.array(eval(p1))
    ugrid = trajectories_mimetic_rk4(x, y, u, v, Ax, Ay, p0, p1, ntraj, tintegr, alpha_min, dt)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_vtk_filename)
    writer.SetInputData(ugrid)
    writer.Write()

if __name__ == '__main__':
    defopt.run(main)