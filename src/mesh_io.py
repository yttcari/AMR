import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
from utils import *
from quadtree import *


def plot(mesh, fname=None, var='rho', title=''):
    get_active_cells = mesh.get_active_cells()
 
    if mesh.Nx == 1 or mesh.Ny == 1:
        # degenerate direction: plot values along the surviving axis
        # instead of the 2D patch grid.
        along_x = mesh.Ny == 1
        coord = np.array([c.x if along_x else c.y for c in get_active_cells])
        if var == 'rho':
            rho = np.array([con2prim(c.U)[0] for c in get_active_cells])
        elif var == 'vx':
            rho = np.array([con2prim(c.W)[1] for c in get_active_cells])
        elif var == 'vy':
            rho = np.array([con2prim(c.W)[2] for c in get_active_cells])
        elif var == 'p':
            rho = np.array([con2prim(c.W)[3] for c in get_active_cells])
        print(f'Plotting {var}')
        lvl = np.array([c.level for c in get_active_cells], dtype=float)
 
        order = np.argsort(coord)
        coord, rho, lvl = coord[order], rho[order], lvl[order]
 
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
 
        axes[0].plot(coord, rho)
        axes[0].set_xlabel('x' if along_x else 'y')
        axes[0].set_ylabel(r'density $\rho$')
        axes[0].set_title(f'{var}')
        axes[0].grid(alpha=0.3)
 
        axes[1].plot(coord, lvl)
        axes[1].set_xlabel('x' if along_x else 'y')
        axes[1].set_ylabel('refinement level')
        axes[1].set_title('refinement level')
        axes[1].set_ylim(-0.5, max(lvl.max(), mesh.max_level) + 0.5)
        axes[1].grid(alpha=0.3)
 
        fig.suptitle(title)
        fig.tight_layout()
        if fname:
            fig.savefig(fname, dpi=150)
        else:
            plt.show()
        plt.close(fig)
        return
 
    if var == 'rho':
        rho = np.array([con2prim(c.U)[0] for c in get_active_cells])
    elif var == 'vx':
        rho = np.array([con2prim(c.W)[1] for c in get_active_cells])
    elif var == 'vy':
        rho = np.array([con2prim(c.W)[2] for c in get_active_cells])
    elif var == 'p':
        rho = np.array([con2prim(c.W)[3] for c in get_active_cells])

    lvl = np.array([c.level for c in get_active_cells], dtype=float)
    rects = [Rectangle((c.x - c.h / 2, c.y - c.h / 2), c.h, c.h)
             for c in get_active_cells]
 
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, vals, cmap, lab in ((axes[0], rho, 'viridis', var),
                                (axes[1], lvl, 'plasma', 'refinement level')):
        pc = PatchCollection([Rectangle(r.get_xy(), r.get_width(), r.get_height())
                              for r in rects],
                             edgecolor='k', linewidth=0.12)
        pc.set_array(vals)
        pc.set_cmap(cmap)
        ax.add_collection(pc)
        ax.set_xlim(0, mesh.Lx)
        ax.set_ylim(0, mesh.Ly)
        ax.set_aspect('equal')
        ax.set_title(lab)
        fig.colorbar(pc, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    else:
        plt.show()
    plt.close(fig)

def write_mesh(mesh, fname):
    with open(fname, "w") as f:
        f.write(f"{mesh.Nx} {mesh.Ny} {mesh.Lx} {mesh.Ly} {mesh.bc} {mesh.max_level}\n")

        def write_node(c):
            if c.children:
                f.write("N\n")
                for child in c.children:
                    write_node(child)
            else:
                rho, u, v, p = c.W
                f.write(f"L {c.x:.8e} {c.y:.8e} {c.h:.8e} {c.level:d} {rho:.8e} {u:.8e} {v:.8e} {p:.8e} {c.chi:.8e}\n")

        for col in mesh.base:
            for root_cell in col:
                write_node(root_cell)

def read_mesh(fname):
    with open(fname, "r") as f:
        Nx, Ny, Lx, Ly, bc, max_level = f.readline().split()
        mesh = Mesh(int(Nx), int(Ny), float(Lx), float(Ly), bc, int(max_level))

        def read_node(c):
            line = f.readline().split()
            tag = line[0]
            if tag == "N":
                c.children = [
                    Cell(c.x - c.h/4, c.y - c.h/4, c.h/2, c.level+1, c),
                    Cell(c.x + c.h/4, c.y - c.h/4, c.h/2, c.level+1, c),
                    Cell(c.x - c.h/4, c.y + c.h/4, c.h/2, c.level+1, c),
                    Cell(c.x + c.h/4, c.y + c.h/4, c.h/2, c.level+1, c),
                ]
                for child in c.children:
                    read_node(child)
            else:  # "L"
                _, x, y, h, level, rho, u, v, p, chi = line
                c.W = np.array([float(rho), float(u), float(v), float(p)])
                c.U = prim2con(c.W)
                c.chi = float(chi)
                c.dUdt = np.zeros(4)
                c.prim_grad = np.zeros((4, 2))

        for col in mesh.base:
            for root_cell in col:
                read_node(root_cell)

        return mesh