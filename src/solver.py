from constant import *
from utils import *

def build_faces(mesh):
    
    interior, boundary_cells = [], []
    for c in mesh.get_active_cells():
        for d in DIRS:
            ns = mesh.neighbors(c, d)
            
            if not ns: # no neighbour = mesh boundary
                boundary_cells.append((c, d))
                continue

            if len(ns) == 2 or ns[0].h < c.h:
                #  finer side owns this face
                continue                     

            n = ns[0]
            if not (n.h > c.h or d in ('+x', '+y')):
                # ignore negative dir to avoid double count at equal level
                continue
            
            face_length = c.h # face length = finer cell's edge
            
            if d == '+x':
                interior.append((c, n, 0, c.x + c.h / 2, c.y, face_length))
            elif d == '-x':
                interior.append((n, c, 0, c.x - c.h / 2, c.y, face_length))
            elif d == '+y':
                interior.append((c, n, 1, c.x, c.y + c.h / 2, face_length))
            else:
                interior.append((n, c, 1, c.x, c.y - c.h / 2, face_length))

    return interior, boundary_cells


def reconstruct(c, xface, yface):
    # MUSCL reconstruction
    # TODO: add selection for multipler order
    W = c.W + c.prim_grad[:, 0] * (xface - c.x) + c.prim_grad[:, 1] * (yface - c.y)
    if W[0] <= EPSILON or W[3] <= EPSILON:
        return c.W.copy()
    return W


def compute_rhs(mesh, interior, boundary_cells):

    get_active_cells = mesh.get_active_cells()

    for c in get_active_cells:
        c.W = con2prim(c.U)
        c.dUdt = np.zeros(4)
    
    for c in get_active_cells:
        c.prim_grad, c.chi = grad_and_chi(mesh, c)

    for (L, R, axis, xface, yface, face_length) in interior:
        primL = reconstruct(L, xface, yface)
        primR = reconstruct(R, xface, yface)
    
        flux = riemann(primL, primR, axis)
    
        L.dUdt -= flux * face_length / L.volume
        R.dUdt += flux * face_length / R.volume

    for (c, d) in boundary_cells:
        axis = 0 if d in ('+x', '-x') else 1
        sgn = 1.0 if d in ('+x', '+y') else -1.0

        if axis == 0:
            xface, yface = c.x + sgn * c.h / 2, c.y
        else:
            xface, yface = c.x, c.y + sgn * c.h / 2

        Wi = reconstruct(c, xface, yface)
        Wg = Wi.copy()

        if mesh.bc == 'reflect':
            Wg[1 + axis] = -Wg[1 + axis]
        # 'outflow': ghost = interior (zero-gradient)
        if sgn > 0:
            flux = riemann(Wi, Wg, axis)
            c.dUdt -= flux * c.h / c.volume
        else:
            flux = riemann(Wg, Wi, axis)
            c.dUdt += flux * c.h / c.volume