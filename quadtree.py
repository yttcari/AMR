from utils import *
from constant import *

class Cell:
    __slots__ = ('x', 'y', 'h', 'level', 'parent', 'children',
                 'U', 'U0', 'W', 'prim_grad', 'dUdt', 'L0', 'chi', 'flag', 'volume')

    def __init__(self, x, y, h, level, parent=None):
        self.x, self.y, self.h, self.level = x, y, h, level
        self.parent = parent
        self.children = None          # None (leaf) or list of 4 Cells
        self.U = None                 # conserved state
        self.U0 = None                # RK stage buffer
        self.W = None                 # primitive state
        self.prim_grad = None                # (4,2) limited primitive gradients
        self.dUdt = None
        self.chi = 0.0                # refinement indicator
        self.flag = False             # flagged for refinement
        self.volume = self.h ** 2

    @property
    def vol(self):
        return self.volume

    def get_cell_value(self, var):
        if var == 'U':
            return self.U
        elif var == 'W':
            return self.W
        else:
            raise ValueError("Does not recognise the input variable")

    def get_value(self, x, y, var, reconstruction):
        # make sure the input coord is within the cell
        assert x <= self.x + self.h/2 + EPSILON and x >= self.x - self.h/2 - EPSILON
        assert y <= self.y + self.h/2 + EPSILON and y >= self.y - self.h/2 - EPSILON

        if reconstruction == 0:
            return self.W if var == 'W' else self.U
        elif reconstruction == 1:
            assert self.prim_grad is not None 

            MUSCL_W = self.W + self.prim_grad[:, 0] * (x - self.x) + self.prim_grad[:, 1] * (y - self.y)
            return MUSCL_W if var == 'W' else prim2con(MUSCL_W)


class Mesh:

    def __init__(self, Nx, Ny, Lx, Ly, bc='outflow', max_level=2, reconstruction=1):
        self.Nx, self.Ny, self.Lx, self.Ly = Nx, Ny, Lx, Ly
        self.h0 = Lx / Nx # width
        self.bc = bc # boundary condition
        self.max_level = max_level
        self.reconstruction = reconstruction # 0: Godunov, 1: MUSCL

        # initialize mesh
        self.base = [[Cell((i + 0.5) * self.h0, (j + 0.5) * self.h0, self.h0, 0)
                      for j in range(Ny)] for i in range(Nx)]

    def get_active_cells(self):
        """
        find all active cell
        """
        out = []
        stack = [c for col in self.base for c in col]
        while stack:
            c = stack.pop()
            if c.children:
                stack.extend(c.children)
            else:
                out.append(c)
        return out

    def get_parents(self):
        """
        find all parent of active cell
        """
        out = []
        stack = [c for col in self.base for c in col]
        while stack:
            c = stack.pop()
            if c.children:
                if all(k.children is None for k in c.children):
                    out.append(c)
                else:
                    stack.extend(c.children)
        return out

    def get_value(self, x, y, var):
        c = self.find_cell(x, y)
        return c.get_value(x, y, var, self.reconstruction)


    def find_cell(self, x, y):
        """
        find a cell fora given coordinate
        """
        if not (0.0 < x < self.Lx and 0.0 < y < self.Ly):
            return None

        i = min(int(x / self.h0), self.Nx - 1)
        j = min(int(y / self.h0), self.Ny - 1)

        node = self.base[i][j]

        while node.children:
            idx = (1 if x > node.x else 0) + (2 if y > node.y else 0)
            node = node.children[idx]
        return node

    def get_cell_value(self, x, y, var):
        c = self.find_cell(x, y)
        if self.reconstruction == 0:
            return c.get_value(x, y, var, self.reconstruction)

    def neighbors(self, c, d):
        """
        Find all neightbour for a given cell

        @input
        c: cell
        d: direction

        @output
        out: list of neighbours
        """
        h = c.h
        eps = h / 8.0 # nudging it slightly outside the cell boundary

        if d == '+x':
            pts = ((c.x + h / 2 + eps, c.y - h / 4), (c.x + h / 2 + eps, c.y + h / 4))
        elif d == '-x':
            pts = ((c.x - h / 2 - eps, c.y - h / 4), (c.x - h / 2 - eps, c.y + h / 4))
        elif d == '+y':
            pts = ((c.x - h / 4, c.y + h / 2 + eps), (c.x + h / 4, c.y + h / 2 + eps))
        else:
            pts = ((c.x - h / 4, c.y - h / 2 - eps), (c.x + h / 4, c.y - h / 2 - eps))

        if self.bc == 'periodic':
            pts = tuple((px % self.Lx, py % self.Ly) for px, py in pts)

        out = []
        
        for p in pts:
            n = self.find_cell(*p)
            if n is not None and (not out or n is not out[0]):
                out.append(n)

        return out