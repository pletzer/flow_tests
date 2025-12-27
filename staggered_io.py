import numpy as np

def save_staggered_fields(fname, x, y, u, v, Ax, Ay):
    """
    Save staggered velocity fields and face fractions to a .npz file.
    """
    np.savez(fname, x=x, y=y, u=u, v=v, Ax=Ax, Ay=Ay)
    print(f"Saved staggered fields to {fname}")

def load_staggered_fields(fname):
    """
    Load staggered velocity fields and face fractions from a .npz file.
    Returns x, y, u, v, Ax, Ay
    """
    data = np.load(fname)
    return data['x'], data['y'], data['u'], data['v'], data['Ax'], data['Ay']
