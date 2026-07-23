import numpy as np
from constant import *

def prim2con(W):
    rho, u, v, p = W
    return np.array([rho, rho * u, rho * v,p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])


def con2prim(U):
    rho = max(U[0], EPSILON)
    u = U[1] / rho
    v = U[2] / rho
    p = (GAMMA - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v))
    return np.array([rho, u, v, max(p, EPSILON)])


def flux_x(W):
    rho, u, v, p = W
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.array([rho * u, rho * u * u + p, rho * u * v, u * (E + p)])


def hllc_x(WL, WR):
    rL, uL, vL, pL = WL
    rR, uR, vR, pR = WR
    cL = np.sqrt(GAMMA * pL / rL)
    cR = np.sqrt(GAMMA * pR / rR)
    SL = min(uL - cL, uR - cR)
    SR = max(uL + cL, uR + cR)

    UL = prim2con(WL)
    UR = prim2con(WR)
    FL = flux_x(WL)
    FR = flux_x(WR)

    if SL >= 0.0:
        return FL
    if SR <= 0.0:
        return FR

    SM = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) \
         / (rL * (SL - uL) - rR * (SR - uR))

    def U_star(W, U, S):
        r, u, v, _p = W
        fac = r * (S - u) / (S - SM)
        Es = U[3] / r + (SM - u) * (SM + _p / (r * (S - u)))
        return fac * np.array([1.0, SM, v, Es])

    if SM >= 0.0:
        return FL + SL * (U_star(WL, UL, SL) - UL)
    return FR + SR * (U_star(WR, UR, SR) - UR)


def riemann(WL, WR, axis):
    if axis == 0: # x-axis
        return hllc_x(WL, WR)
    
    # y-axis
    WLs = np.array([WL[0], WL[2], WL[1], WL[3]])
    WRs = np.array([WR[0], WR[2], WR[1], WR[3]])

    F = hllc_x(WLs, WRs)
    return np.array([F[0], F[2], F[1], F[3]])


def minmod(a, b):
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def grad_and_chi(mesh, c):
    """
    Compute the minmod-limited gradient of W (primitives) and the
    refinement indicator chi.

    mesh.indicator == 'jump'  : max relative undivided jump of rho and p.
    mesh.indicator == 'detail': Harten/Loehner multiresolution detail
        coefficient -- the normalised second difference against the two
        neighbour averages, with a small noise floor.
    """
    W = c.W
    g = np.zeros((4, 2))
    chi = 0.0
    detail = (mesh.indicator == 'detail')

    for axis, (dp, dm) in enumerate((('+x', '-x'), ('+y', '-y'))):
        positive_cells = mesh.neighbors(c, dp) # all cells in positive x/y direction
        negative_cells = mesh.neighbors(c, dm) # all cells of negative x/y direction

        avg_p = avg_M = grad_p = grad_m = None

        if positive_cells:
            avg_p = np.mean([n.W for n in positive_cells], axis=0) # average all cell on same side
            grad_p = (avg_p - W) / (0.5 * (c.h + positive_cells[0].h))
        if negative_cells: # same but -ve side
            avg_M = np.mean([n.W for n in negative_cells], axis=0)
            grad_m = (W - avg_M) / (0.5 * (c.h + negative_cells[0].h))

        if detail:
            if avg_p is not None and avg_M is not None:
                for k in (0, 3):
                    num = abs(avg_p[k] - 2.0 * W[k] + avg_M[k])
                    den = (abs(avg_p[k] - W[k]) + abs(W[k] - avg_M[k])
                           + 0.02 * (abs(avg_p[k]) + 2.0 * abs(W[k]) + abs(avg_M[k]))
                           + EPSILON)
                    chi = max(chi, num / den)
        else:
            for q_n in (avg_p, avg_M):
                if q_n is not None:
                    for k in (0, 3):
                        chi = max(chi, abs(q_n[k] - W[k]) / (abs(W[k]) + EPSILON)) # relative gradient

        if grad_p is not None and grad_m is not None:
            g[:, axis] = minmod(grad_p, grad_m)

    return g, chi

def grad_init(mesh):
    for c in mesh.get_active_cells():
        grad, _ = grad_and_chi(mesh, c)
        c.prim_grad = grad

def get_grad(mesh, c, var='U'):
    value = c.get_cell_value(var)

    g = np.zeros((4, 2))

    for axis, (dp, dm) in enumerate((('+x', '-x'), ('+y', '-y'))):
        positive_cells = mesh.neighbors(c, dp)
        negative_cells = mesh.neighbors(c, dm)

        if positive_cells and negative_cells:

            avg_p = np.mean([n.get_cell_value(var) for n in positive_cells], axis=0)
            avg_M = np.mean([n.get_cell_value(var) for n in negative_cells], axis=0)
            
            grad_p = (avg_p - value) / (0.5 * (c.h + positive_cells[0].h))
            grad_m = (value - avg_M) / (0.5 * (c.h + negative_cells[0].h))
        
            g[:, axis] = minmod(grad_p, grad_m)
    return g


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def L1_error(mesh, mesh_ref, var='W', plot=True):
    import matplotlib.colors as mcolors

    cell_list = mesh.get_active_cells()

    errors = []       # per-cell (value - ref_value), signed, vector len 4
    volumes = []       # per-cell volume, for weighting

    for c in cell_list:
        try:
            val = mesh.get_value(x=c.x, y=c.y, var=var)
        except:
            print(c)
            raise ValueError
        ref_val = mesh_ref.get_value(x=c.x, y=c.y, var=var)

        err = val - ref_val
        errors.append(err)
        volumes.append(c.volume)

    errors = np.array(errors)     # shape (Ncells, 4)
    volumes = np.array(volumes)   # shape (Ncells,)

    # volume-weighted L1 norm (always uses magnitude, regardless of plot coloring)
    L1 = np.sum(np.abs(errors) * volumes[:, None], axis=0) / np.sum(volumes)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        err_mag = np.abs(errors[:, 0])                 # magnitude: 0 at no error
        vmax = err_mag.max() if err_mag.max() > 0 else 1.0

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'white_to_red', ['white', 'firebrick'])
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

        for c, e in zip(cell_list, err_mag):
            color = cmap(norm(e))
            rect = patches.Rectangle(
                (c.x - c.h / 2, c.y - c.h / 2), c.h, c.h,
                facecolor=color, edgecolor='none'
            )
            ax.add_patch(rect)

        ax.set_xlim(0, mesh.Lx)
        ax.set_ylim(0, mesh.Ly)
        ax.set_aspect('equal')
        ax.set_title(f'{var}[0] |error| (density), L1 = {L1[0]:.4e}')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label='|error|')
        plt.show()

    return L1