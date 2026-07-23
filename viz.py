import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
from utils import *

def plot(mesh, fname, title=''):
    get_active_cells = mesh.get_active_cells()
    rho = np.array([con2prim(c.U)[0] for c in get_active_cells])
    lvl = np.array([c.level for c in get_active_cells], dtype=float)
    rects = [Rectangle((c.x - c.h / 2, c.y - c.h / 2), c.h, c.h)
             for c in get_active_cells]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, vals, cmap, lab in ((axes[0], rho, 'viridis', r'density $\rho$'),
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
    fig.savefig(fname, dpi=150)
    plt.close(fig)

