import numpy as np
from quadtree import *
from advance import *
from mesh_io import *

def init_from(mesh, W_init):
    for c in mesh.get_active_cells():
        c.U = prim2con(W_init(c.x, c.y))

def run_sim(mesh, max_level, t_end, cfl, eta_refine, eta_coarsen,
               regrid_every, mode, verbose, label):
    """
    Shared driver for the problem setups below (run_blast keeps its own
    copy of this logic untouched, for backward compatibility).

    mode='in_step'  : new flagging, RK2 time integration
    mode='between'  : original flagging, RK2 time integration
    mode='uniform'  : no refinement, RK2 time integration
    mode='ctu'      : corner-transport-upwind (single-step unsplit) time
                       integration, regridded every `regrid_every` steps
    """
    try:
        t, step = 0.0, 0
        while t < t_end - 1e-12:
            if mode == 'in_step':
                dt = advance_adaptive(mesh, cfl, eta_refine, max_level,
                                      dt_max=t_end - t)
                t += dt
                step += 1
                if step % regrid_every == 0:
                    coarsen_only(mesh, eta_coarsen)
            elif mode == 'between' or mode == 'uniform':
                dt = advance(mesh, cfl, dt_max=t_end - t)
                t += dt
                step += 1
                if mode == 'between':
                    if step % regrid_every == 0:
                        regrid(mesh, eta_refine, eta_coarsen, max_level)
                        grad_init(mesh)
            else: 
                raise ValueError("Unrecognised mode. STOP")

            if verbose and step % 20 == 0:
                print(f"[{label}] step {step:4d}  t={t:.4f}  dt={dt:.2e}  "
                      f"cells={len(mesh.get_active_cells()):5d}")

        if verbose:
            print(f"[{label}] done: {step} steps, t={t:.4f}")
        return mesh
    except KeyboardInterrupt:
        print("Keyboard Interrupt, returning mesh...")
        return mesh


def run_blast(Nx=32, max_level=2, t_end=0.15, cfl=0.4,
              eta_refine=None, eta_coarsen=None, regrid_every=2,
              mode='in_step', indicator='detail',
              plot_file=None, verbose=True, init=True, mesh=None):
    """
    Cylindrical Sod-type blast on [0,1]^2, outflow boundaries.

    mode='in_step'  : new flagging, RK2 time integration
    mode='between'  : original flagging, RK2 time integration
    mode='uniform'  : no refinement, RK2 time integration
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
    if init:
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
            if mode == 'between':
                if step % regrid_every == 0:
                    regrid(mesh, eta_refine, eta_coarsen, max_level)
                    grad_init(mesh)

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

def run_sod_1d(Nx=200, Ny=6, max_level=3, t_end=0.20, cfl=0.4,
               eta_refine=None, eta_coarsen=None, regrid_every=2,
               mode='between', indicator='detail',
               plot_file=None, verbose=True):
    """
    Sod's shock tube (Sod 1978): domain [0,1], discontinuity at x=0.5.
        left:  rho=1.0,   u=0, p=1.0
        right: rho=0.125, u=0, p=0.1
    Standard reference time t_end=0.2.
    """
    if eta_refine is None:
        eta_refine = {'jump': 0.12, 'detail': 0.45}[indicator]
    if eta_coarsen is None:
        eta_coarsen = {'jump': 0.03, 'detail': 0.15}[indicator]

    def W_init(x, y):
        return np.array([1.0, 0.0, 0.0, 1.0]) if x < 0.5 \
            else np.array([0.125, 0.0, 0.0, 0.1])

    h0 = 1.0 / Nx
    mesh = Mesh(Nx, Ny, 1.0, Ny * h0, bc='outflow', max_level=max_level)
    mesh.indicator = indicator
    init_from(mesh, W_init)

    run_sim(mesh, max_level, t_end, cfl, eta_refine, eta_coarsen,
               regrid_every, mode, verbose, label='sod_1d')

    if plot_file:
        plot(mesh, plot_file, title=f'Sod shock tube, t={t_end:.3f}')
    return mesh


def run_lax_1d(Nx=200, Ny=6, max_level=3, t_end=0.13, cfl=0.4,
               eta_refine=None, eta_coarsen=None, regrid_every=2,
               mode='between', indicator='detail',
               plot_file=None, verbose=True):

    if eta_refine is None:
        eta_refine = {'jump': 0.12, 'detail': 0.45}[indicator]
    if eta_coarsen is None:
        eta_coarsen = {'jump': 0.03, 'detail': 0.15}[indicator]

    def W_init(x, y):
        return np.array([0.445, 0.698, 0.0, 3.528]) if x < 0.5 \
            else np.array([0.5, 0.0, 0.0, 0.571])

    h0 = 1.0 / Nx
    mesh = Mesh(Nx, Ny, 1.0, Ny * h0, bc='outflow', max_level=max_level)
    mesh.indicator = indicator
    init_from(mesh, W_init)

    run_sim(mesh, max_level, t_end, cfl, eta_refine, eta_coarsen,
               regrid_every, mode, verbose, label='lax_1d')

    if plot_file:
        plot(mesh, plot_file, title=f'Lax shock tube, t={t_end:.3f}')
    return mesh

def run_implosion(Nx=64, max_level=3, t_end=2.5, cfl=0.4,
                  eta_refine=None, eta_coarsen=None, regrid_every=2,
                  mode='between', indicator='detail',
                  plot_file=None, verbose=True):
    
    if eta_refine is None:
        eta_refine = {'jump': 0.12, 'detail': 0.45}[indicator]
    if eta_coarsen is None:
        eta_coarsen = {'jump': 0.03, 'detail': 0.15}[indicator]

    def W_init(x, y):
        if x + y < 0.15:
            return np.array([0.125, 0.0, 0.0, 0.14])
        return np.array([1.0, 0.0, 0.0, 1.0])

    mesh = Mesh(Nx, Nx, 0.3, 0.3, bc='reflect', max_level=max_level)
    mesh.indicator = indicator
    init_from(mesh, W_init)

    run_sim(mesh, max_level, t_end, cfl, eta_refine, eta_coarsen,
               regrid_every, mode, verbose, label='implosion')

    if plot_file:
        plot(mesh, plot_file, title=f'Implosion, t={t_end:.3f}')
    return mesh


def run_riemann2d(Nx=64, max_level=3, t_end=0.3, cfl=0.4,
                  eta_refine=None, eta_coarsen=None, regrid_every=2,
                  mode='between', indicator='detail',
                  plot_file=None, verbose=True, init=True, mesh=None):
    """
    Quad 1 : rho=1.5,    u=0,     v=0,     p=1.5
    Quad 2 : rho=0.5323, u=1.206, v=0,     p=0.3
    Quad 3 : rho=0.138,  u=1.206, v=1.206, p=0.029
    Quad 4 : rho=0.5323, u=0,     v=1.206, p=0.3
    """
    if eta_refine is None:
        eta_refine = {'jump': 0.12, 'detail': 0.45}[indicator]
    if eta_coarsen is None:
        eta_coarsen = {'jump': 0.03, 'detail': 0.15}[indicator]

    def W_init(x, y):
        if x > 0.5 and y > 0.5:
            return np.array([1.5, 0.0, 0.0, 1.5])
        if x < 0.5 and y > 0.5:
            return np.array([0.5323, 1.206, 0.0, 0.3])
        if x < 0.5 and y < 0.5:
            return np.array([0.138, 1.206, 1.206, 0.029])
        return np.array([0.5323, 0.0, 1.206, 0.3])

    if init: 
        mesh = Mesh(Nx, Nx, 1.0, 1.0, bc='outflow', max_level=max_level)
        init_from(mesh, W_init)        
        mesh.indicator = indicator

    print('Start running')
    run_sim(mesh, max_level, t_end, cfl, eta_refine, eta_coarsen,
               regrid_every, mode, verbose, label='riemann2d')

    if plot_file:
        plot(mesh, plot_file,
             title=f'2D Riemann problem (config. 3), t={t_end:.3f}')
    return mesh


def run_kelvin_helmholtz(Nx=64, max_level=3, t_end=1.5, cfl=0.3,
                         eta_refine=None, eta_coarsen=None, regrid_every=2,
                         mode='between', indicator='detail',
                         plot_file=None, verbose=True, init=True, mesh=None):
    """
    Kelvin-Helmholtz instability
    BC: 'outflow'/'reflect'
    """
    if eta_refine is None:
        eta_refine = {'jump': 0.12, 'detail': 0.45}[indicator]
    if eta_coarsen is None:
        eta_coarsen = {'jump': 0.03, 'detail': 0.15}[indicator]

    delta = 0.02

    def W_init(x, y):
        step = np.tanh((y - 0.25) / delta) - np.tanh((y - 0.75) / delta)
        rho = 1.0 + 0.5 * step
        u = 0.5 * step - 0.5
        v = 0.01 * np.sin(4.0 * np.pi * x)
        p = 2.5
        return np.array([rho, u, v, p])

    if init:
        mesh = Mesh(Nx, Nx, 1.0, 1.0, bc='outflow', max_level=max_level)
        mesh.indicator = indicator
        init_from(mesh, W_init)

    run_sim(mesh, max_level, t_end, cfl, eta_refine, eta_coarsen,
               regrid_every, mode, verbose, label='kh')

    if plot_file:
        plot(mesh, plot_file, title=f'Kelvin-Helmholtz, t={t_end:.3f}')
    return mesh

    
def run_smooth_sine_1d(Nx=100, Ny=6, max_level=0, t_end=0.2, cfl=0.4,
                       eta_refine=None, eta_coarsen=None, regrid_every=2,
                       mode='uniform', indicator='detail',
                       plot_file=None, verbose=True,
                       rho0=1.0, amp=0.2, u0=1.0, p0=1.0, reconstruction=1):

    if eta_refine is None:
        eta_refine = {'jump': 0.12, 'detail': 0.45}[indicator]
    if eta_coarsen is None:
        eta_coarsen = {'jump': 0.03, 'detail': 0.15}[indicator]
 
    def W_init(x, y):
        rho = rho0 + amp * np.sin(2.0 * np.pi * x)
        return np.array([rho, u0, 0.0, p0])
 
    h0 = 1.0 / Nx
    mesh = Mesh(Nx, Ny, 1.0, Ny * h0, bc='periodic', max_level=max_level, reconstruction=reconstruction)
    mesh.indicator = indicator
    init_from(mesh, W_init)
 
    run_sim(mesh, max_level, t_end, cfl, eta_refine, eta_coarsen,
               regrid_every, mode, verbose, label='smooth_sine_1d')
 
    if plot_file:
        plot(mesh, plot_file, title=f'Smooth sine advection, t={t_end:.3f}')
    return mesh