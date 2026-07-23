import numpy as np
from quadtree import *
from advance import *
from viz import *

def init_from(mesh, W_init):
    for c in mesh.get_active_cells():
        c.U = prim2con(W_init(c.x, c.y))

def run_blast(Nx=32, max_level=2, t_end=0.15, cfl=0.4,
              eta_refine=None, eta_coarsen=None, regrid_every=2,
              mode='in_step', indicator='detail',
              plot_file=None, verbose=True):
    """
    Cylindrical Sod-type blast on [0,1]^2, outflow boundaries.

    mode='in_step'  : new flagging
    mode='between'  : original flagging
    """
    if eta_refine is None:
        eta_refine = {'jump': 0.12, 'detail': 0.45}[indicator]
    if eta_coarsen is None:
        eta_coarsen = {'jump': 0.03, 'detail': 0.15}[indicator]

    def W_init(x, y):
        r = np.hypot(x - 0.5, y - 0.5)
        if r < 0.13:
            return np.array([1.0, 0.0, 0.0, 1.0])
        return np.array([0.125, 0.0, 0.0, 0.1])

    mesh = Mesh(Nx, Nx, 1.0, 1.0, bc='outflow')
    mesh.indicator = indicator

    init_from(mesh, W_init)
    
    t, step = 0.0, 0
    while t < t_end - 1e-12:
        if mode == 'in_step':
            dt = advance_adaptive(mesh, cfl, eta_refine, max_level,
                                  dt_max=t_end - t)
            t += dt
            step += 1
            if step % regrid_every == 0:
                coarsen_only(mesh, eta_coarsen)
        else:
            dt = advance(mesh, cfl, dt_max=t_end - t)
            t += dt
            step += 1
            if step % regrid_every == 0:
                regrid(mesh, eta_refine, eta_coarsen, max_level)
        if verbose and step % 20 == 0:
            print(f"step {step:4d}  t={t:.4f}  dt={dt:.2e}  "
                  f"get_active_cells={len(mesh.get_active_cells()):5d}")

    if verbose:
        print(f"done: {step} steps, t={t:.4f}, ")
    if plot_file:
        plot(mesh, plot_file,
             title=f'2D AMR blast, base {Nx}x{Nx}, max level {max_level}, '
                   f't={t:.3f}')
    return mesh


if __name__ == '__main__':
    run_blast(Nx=32, max_level=2, t_end=0.15,
              plot_file='amr2d_blast.png')