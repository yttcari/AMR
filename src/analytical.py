import copy
import numpy as np
from constant import GAMMA
from utils import prim2con
from quadtree import Mesh
from solver import compute_rhs, build_faces


def _solve_pstar(rhoL, uL, pL, cL, rhoR, uR, pR, cR, gamma, tol, max_iter):
    AL = 2.0 / ((gamma + 1.0) * rhoL)
    BL = (gamma - 1.0) / (gamma + 1.0) * pL
    AR = 2.0 / ((gamma + 1.0) * rhoR)
    BR = (gamma - 1.0) / (gamma + 1.0) * pR

    def f_and_fprime(p, rho_K, p_K, c_K, A_K, B_K):
        if p > p_K:                                   # shock branch
            sq = np.sqrt(A_K / (p + B_K))
            f = (p - p_K) * sq
            fp = sq * (1.0 - (p - p_K) / (2.0 * (p + B_K)))
        else:                                          # rarefaction branch
            pr = p / p_K
            f = (2.0 * c_K / (gamma - 1.0)) * (pr ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
            fp = (1.0 / (rho_K * c_K)) * pr ** (-(gamma + 1.0) / (2.0 * gamma))
        return f, fp


    p0 = ((cL + cR - 0.5 * (gamma - 1.0) * (uR - uL))
          / (cL / pL ** ((gamma - 1.0) / (2.0 * gamma))
             + cR / pR ** ((gamma - 1.0) / (2.0 * gamma)))) ** (2.0 * gamma / (gamma - 1.0))
    p_old = max(tol, p0)

    for _ in range(max_iter):
        fL, fLp = f_and_fprime(p_old, rhoL, pL, cL, AL, BL)
        fR, fRp = f_and_fprime(p_old, rhoR, pR, cR, AR, BR)
        f = fL + fR + (uR - uL)
        fp = fLp + fRp
        p_new = max(tol, p_old - f / fp)
        if abs(p_new - p_old) / (0.5 * (p_new + p_old)) < tol:
            p_old = p_new
            break
        p_old = p_new

    p_star = p_old
    fL, _ = f_and_fprime(p_star, rhoL, pL, cL, AL, BL)
    fR, _ = f_and_fprime(p_star, rhoR, pR, cR, AR, BR)
    u_star = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    return p_star, u_star


def exact_riemann(WL, WR, x0, t, x, gamma=GAMMA, tol=1e-10, max_iter=100):
    # Exact solution of the 1D Riemann problem
    rhoL, uL, vL, pL = (float(w) for w in WL)
    rhoR, uR, vR, pR = (float(w) for w in WR)
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    if (2.0 * cL / (gamma - 1.0) + 2.0 * cR / (gamma - 1.0)) <= (uR - uL):
        raise ValueError("left/right states produce vacuum: not supported")

    p_star, u_star = _solve_pstar(rhoL, uL, pL, cL, rhoR, uR, pR, cR,
                                   gamma, tol, max_iter)

    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    if t > 0.0:
        S = (x_arr - x0) / t
    else:
        S = np.where(x_arr >= x0, np.inf, -np.inf)

    rho = np.empty_like(S)
    u = np.empty_like(S)
    p = np.empty_like(S)
    v = np.empty_like(S)

    left_mask = S < u_star
    if np.any(left_mask):
        Sl = S[left_mask]
        if p_star > pL:                                 # left shock
            S_L = uL - cL * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / pL
                                     + (gamma - 1.0) / (2.0 * gamma))
            behind = Sl > S_L
            rho_starL = rhoL * ((p_star / pL + (gamma - 1.0) / (gamma + 1.0))
                                 / ((gamma - 1.0) / (gamma + 1.0) * p_star / pL + 1.0))
            rho_l = np.where(behind, rho_starL, rhoL)
            u_l = np.where(behind, u_star, uL)
            p_l = np.where(behind, p_star, pL)
        else:                                            # left rarefaction
            c_starL = cL * (p_star / pL) ** ((gamma - 1.0) / (2.0 * gamma))
            S_HL = uL - cL
            S_TL = u_star - c_starL
            inside = (Sl > S_HL) & (Sl < S_TL)
            behind = Sl >= S_TL

            rho_fan = rhoL * (2.0 / (gamma + 1.0)
                               + (gamma - 1.0) / ((gamma + 1.0) * cL) * (uL - Sl)) ** (2.0 / (gamma - 1.0))
            u_fan = 2.0 / (gamma + 1.0) * (cL + (gamma - 1.0) / 2.0 * uL + Sl)
            p_fan = pL * (2.0 / (gamma + 1.0)
                          + (gamma - 1.0) / ((gamma + 1.0) * cL) * (uL - Sl)) ** (2.0 * gamma / (gamma - 1.0))
            rho_starL = rhoL * (p_star / pL) ** (1.0 / gamma)

            rho_l = np.where(behind, rho_starL, np.where(inside, rho_fan, rhoL))
            u_l = np.where(behind, u_star, np.where(inside, u_fan, uL))
            p_l = np.where(behind, p_star, np.where(inside, p_fan, pL))

        rho[left_mask], u[left_mask], p[left_mask] = rho_l, u_l, p_l
        v[left_mask] = vL

    right_mask = ~left_mask
    if np.any(right_mask):
        Sr = S[right_mask]
        if p_star > pR:                                 # right shock
            S_R = uR + cR * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / pR
                                     + (gamma - 1.0) / (2.0 * gamma))
            behind = Sr < S_R
            rho_starR = rhoR * ((p_star / pR + (gamma - 1.0) / (gamma + 1.0))
                                 / ((gamma - 1.0) / (gamma + 1.0) * p_star / pR + 1.0))
            rho_r = np.where(behind, rho_starR, rhoR)
            u_r = np.where(behind, u_star, uR)
            p_r = np.where(behind, p_star, pR)
        else:                                            # right rarefaction
            c_starR = cR * (p_star / pR) ** ((gamma - 1.0) / (2.0 * gamma))
            S_HR = uR + cR
            S_TR = u_star + c_starR
            inside = (Sr < S_HR) & (Sr > S_TR)
            behind = Sr <= S_TR

            rho_fan = rhoR * (2.0 / (gamma + 1.0)
                               - (gamma - 1.0) / ((gamma + 1.0) * cR) * (uR - Sr)) ** (2.0 / (gamma - 1.0))
            u_fan = 2.0 / (gamma + 1.0) * (-cR + (gamma - 1.0) / 2.0 * uR + Sr)
            p_fan = pR * (2.0 / (gamma + 1.0)
                          - (gamma - 1.0) / ((gamma + 1.0) * cR) * (uR - Sr)) ** (2.0 * gamma / (gamma - 1.0))
            rho_starR = rhoR * (p_star / pR) ** (1.0 / gamma)

            rho_r = np.where(behind, rho_starR, np.where(inside, rho_fan, rhoR))
            u_r = np.where(behind, u_star, np.where(inside, u_fan, uR))
            p_r = np.where(behind, p_star, np.where(inside, p_fan, pR))

        rho[right_mask], u[right_mask], p[right_mask] = rho_r, u_r, p_r
        v[right_mask] = vR

    out = np.stack([rho, u, v, p], axis=-1)
    return out if np.ndim(x) > 0 else out[0]


def sod_exact(x, t, x0=0.5):
    WL = np.array([1.0, 0.0, 0.0, 1.0])
    WR = np.array([0.125, 0.0, 0.0, 0.1])
    return exact_riemann(WL, WR, x0, t, x)


def lax_exact(x, t, x0=0.5):
    WL = np.array([0.445, 0.698, 0.0, 3.528])
    WR = np.array([0.5, 0.0, 0.0, 0.571])
    return exact_riemann(WL, WR, x0, t, x)


def exact_mesh_like(mesh, t, WL=np.array([1, 0, 0, 1]), WR=np.array([0.125, 0.0, 0.0, 0.1]), 
                    x0=0.5, gamma=GAMMA):
    # Build a reference Mesh for use as mesh_ref in L1_error() 
    mesh_ref = copy.deepcopy(mesh)
    cells = mesh_ref.get_active_cells()
    xs = np.array([c.x for c in cells])
    Ws = exact_riemann(WL, WR, x0, t, xs, gamma=gamma)
    for c, W in zip(cells, Ws):
        c.U = prim2con(W)

    compute_rhs(mesh_ref, *build_faces(mesh_ref))   
    return mesh_ref


def exact_mesh_1d(Nx, Ny, WL, WR, x0, t, Lx=1.0, gamma=GAMMA,
                  bc='outflow', max_level=0):

    h0 = Lx / Nx
    Ly = Ny * h0
    mesh = Mesh(Nx, Ny, Lx, Ly, bc=bc, max_level=max_level)
    mesh.indicator = 'detail'

    cells = mesh.get_active_cells()
    xs = np.array([c.x for c in cells])
    Ws = exact_riemann(WL, WR, x0, t, xs, gamma=gamma)
    for c, W in zip(cells, Ws):
        c.U = prim2con(W)

    compute_rhs(mesh, *build_faces(mesh)) 
    return mesh