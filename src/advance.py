from refinenent import *
from solver import *

def compute_dt(mesh, cfl):
    wmax = 0.0
    for c in mesh.get_active_cells():
        rho, u, v, p = con2prim(c.U)
        cs = np.sqrt(GAMMA * p / rho)
        wmax = max(wmax, ((abs(u) + cs) + (abs(v) + cs)) / c.h)
    return cfl / wmax


def advance(mesh, cfl, dt_max=np.inf):
    dt = min(compute_dt(mesh, cfl), dt_max)
    interior, boundary = build_faces(mesh)   # grid frozen for both stages

    compute_rhs(mesh, interior, boundary)    # stage 1
    for c in mesh.get_active_cells():
        c.U0 = c.U.copy()
        c.U = c.U0 + dt * c.dUdt

    compute_rhs(mesh, interior, boundary)    # stage 2 
    for c in mesh.get_active_cells():
        c.U = 0.5 * (c.U0 + c.U + dt * c.dUdt)
    return dt


def predictor_chi(mesh, dt):

    get_active_cells = mesh.get_active_cells()
    saved = [c.W for c in get_active_cells]
    for c in get_active_cells:
        c.W = con2prim(c.U + dt * c.dUdt)
    for c in get_active_cells:
        _, chi = grad_and_chi(mesh, c)
        c.chi = max(c.chi, chi)
    for c, w in zip(get_active_cells, saved):
        c.W = w


def advance_adaptive(mesh, cfl, eta_refine, max_level,
                     dt_max=np.inf, buffer=True):

    dt = min(compute_dt(mesh, cfl), dt_max)
    interior, boundary = build_faces(mesh)
    compute_rhs(mesh, interior, boundary)        
    get_active_cells = mesh.get_active_cells()
    for c in get_active_cells:
        c.L0 = c.dUdt.copy()
    predictor_chi(mesh, dt)                      
    flagged = flag_refinement(mesh, eta_refine, max_level, buffer)

    if flagged:
        grads = [get_grad(mesh, c) for c in flagged]   # snapshot before splitting
        for c, g in zip(flagged, grads):
            refine_cell(mesh, c, g)
        dt = min(compute_dt(mesh, cfl), dt_max)  
        interior, boundary = build_faces(mesh)
        compute_rhs(mesh, interior, boundary)    
        for c in mesh.get_active_cells():
            c.U0 = c.U.copy()
            c.U = c.U0 + dt * c.dUdt
    else:
        for c in get_active_cells:               
            c.U0 = c.U.copy()
            c.U = c.U0 + dt * c.L0

    compute_rhs(mesh, interior, boundary)        
    for c in mesh.get_active_cells():
        c.U = 0.5 * (c.U0 + c.U + dt * c.dUdt)
    return dt


def coarsen_only(mesh, eta_coarsen):
    """Deferred coarsening between steps (coarsening late costs only a
    little efficiency, never accuracy -- unlike refining late)."""
    refresh_chi(mesh)
    coarsen_pass(mesh, eta_coarsen)