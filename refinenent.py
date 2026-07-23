from constant import *
from utils import *
from quadtree import *


def refine_cell(mesh, c):
    """
    Refine cell
    # TODO: currently it is direct copy of parent cell value, 
    include reconstruction if possible
    """

    grad = get_grad(mesh, c)
    half_width = c.h / 2.0

    kids = []
    
    for j in (0, 1):
        for i in (0, 1):

            dx = (i - 0.5) * half_width
            dy = (j - 0.5) * half_width

            child = Cell(c.x + dx, c.y + dy, half_width, c.level + 1, parent=c)

            child_U = c.U + grad[:, 0] * dx + grad[:, 1] * dy
            child_W = con2prim(child_U)

            if child_W[0] <= EPSILON or child_W[3] <= EPSILON:   # positivity fallback
                child_U = c.U.copy()

            child.U = child_U

            child.W = con2prim(child_U)
            kids.append(child)

    c.children = kids
    c.flag = False


def coarsen_cell(parent):
    """
    coarsen the given parent cell
    """
    parent.U = sum(child.U for child in parent.children) / 4.0   # conservative
    parent.W = con2prim(parent.U)
    parent.chi = 0.0
    parent.flag = False
    parent.children = None


def coarsen_pass(mesh, eta_coarsen):
    for p in mesh.get_parents():
        kids = p.children
        if any(k.chi > eta_coarsen for k in kids):
            continue
        ok = True
        for k in kids:                      # keep 2:1 balance after merge
            for d in DIRS:
                for n in mesh.neighbors(k, d):
                    if n.h < k.h * (1 - 1e-12):
                        ok = False
        if ok:
            coarsen_cell(p)


def flag_refinement(mesh, eta_refine, max_level, buffer=True):

    get_active_cells = mesh.get_active_cells()
    for c in get_active_cells:
        c.flag = (c.chi > eta_refine and c.level < max_level)

    if buffer:                              
        extra = []
        for c in get_active_cells:
            if c.flag:
                for d in DIRS:
                    for n in mesh.neighbors(c, d):
                        if (not n.flag) and n.level < max_level \
                                and n.level <= c.level:
                            extra.append(n)
        for n in extra:
            n.flag = True

    # Ensure no neighbour level differ by more than 1
    changed = True
    while changed:
        changed = False
        for c in mesh.get_active_cells():
            if c.flag:
                for d in DIRS:
                    for n in mesh.neighbors(c, d):
                        if n.level < c.level and not n.flag:
                            n.flag = True
                            changed = True

    return [c for c in mesh.get_active_cells() if c.flag]


def refresh_chi(mesh):
    get_active_cells = mesh.get_active_cells()

    for c in get_active_cells:
        c.W = con2prim(c.U)

    for c in get_active_cells:
        c.prim_grad, c.chi = grad_and_chi(mesh, c)


def regrid(mesh, eta_refine, eta_coarsen, max_level, buffer=True):
    refresh_chi(mesh)
    coarsen_pass(mesh, eta_coarsen)
    for c in flag_refinement(mesh, eta_refine, max_level, buffer):
        refine_cell(mesh, c)