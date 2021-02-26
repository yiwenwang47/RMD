import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .molecule import simple_mol
from .elements import elementdict
import numpy as np

colors = {
    'Ir': (85, 91, 94),
    'C': (7, 7, 7),
    'N': (61, 159, 218),
    'O': (46, 200, 104),
    'S': (249, 238, 65),
    'H': (160, 222, 252),
    'F': (197, 238, 167 ),
    'Br': (209, 1, 1),
    'Cl': (180, 247, 98),
    'I': (107, 106, 223)
}

def _color(RGB):
    color = (RGB[0]/256, RGB[1]/256, RGB[2]/256)
    return color

def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    mean = lambda pair: (pair[0]+pair[1])/2

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = mean(z_limits)

    plot_radius = 0.4*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - 0.8*plot_radius, z_middle + 0.8*plot_radius])

def plot_3d(mol: simple_mol):

    """
    A very simple way to plot a molecule in a 3d fashion. For the sake of visualizing bonds.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = mol.natoms
    # ax.set_aspect('equal')
    for i in range(n):
        a = mol.atoms[i]
        ax.scatter(xs=mol.coords_all[i, 0],ys=mol.coords_all[i, 1],zs=mol.coords_all[i, 2],
                   s=elementdict[a][2]*150,lw=0,c=[_color(colors[a])])
        for j in range(i+1, n):
            if mol.graph[i][j] == 1:
                c1, c2 = mol.coords_all[i], mol.coords_all[j]
                ax.plot(xs=[c1[0], c2[0]], ys=[c1[1], c2[1]], zs=[c1[2], c2[2]],c=_color((195, 193, 192)), linewidth=4)
    set_axes_equal(ax)
    ax.grid(False)
    return ax

def plot_partial_3d(mol: simple_mol, scope: set):

    """
    A very simple way to plot a molecule in a 3d fashion with a specified scope.
    """

    scope = list(scope)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    coords = mol.coords_all[scope]
    for i in scope:
        a = mol.atoms[i]
        ax.scatter(xs=mol.coords_all[i, 0],ys=mol.coords_all[i, 1],zs=mol.coords_all[i, 2],
                   s=elementdict[a][2]*150,lw=0,c=[_color(colors[a])])
    graph = mol.graph[scope][:, scope]
    n = len(scope)
    for i in range(n):
        for j in range(i+1, n):
            if graph[i][j] == 1:
                c1, c2 = coords[i], coords[j]
                ax.plot(xs=[c1[0], c2[0]], ys=[c1[1], c2[1]], zs=[c1[2], c2[2]],c=_color((195, 193, 192)), linewidth=4)
    set_axes_equal(ax)
    ax.grid(False)
    return ax