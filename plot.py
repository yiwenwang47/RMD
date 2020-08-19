import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .molecule import simple_mol
import numpy as np

def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_3d(mol: simple_mol):

    """
    A very simple way to plot a molecule in a 3d fashion. For the sake of visualizing bonds.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    ax.scatter(xs=mol.coords_all[:, 0],ys=mol.coords_all[:, 1],zs=mol.coords_all[:, 2],s=50)

    n = len(mol.atoms)
    for i in range(n):
        for j in range(i+1, n):
            if mol.graph[i][j] == 1:
                c1, c2 = mol.coords_all[i], mol.coords_all[j]
                ax.plot(xs=[c1[0], c2[0]], ys=[c1[1], c2[1]], zs=[c1[2], c2[2]])
    set_axes_equal(ax)
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
    graph = mol.graph[scope][:, scope]
    ax.scatter(xs=coords[:, 0],ys=coords[:, 1],zs=coords[:, 2],s=50)
    n = len(scope)
    for i in range(n):
        for j in range(i+1, n):
            if graph[i][j] == 1:
                c1, c2 = coords[i], coords[j]
                ax.plot(xs=[c1[0], c2[0]], ys=[c1[1], c2[1]], zs=[c1[2], c2[2]])
    set_axes_equal(ax)
    return ax

def plot_ligand_3d(mol: simple_mol, ligand_type: str, plot_mc=True):

    """
    A very simple way to plot a molecule in a 3d fashion with only the specified type of ligand.
    This is to verify that the ligands are categorized correctly.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    ligands = mol.get_specific_ligand(ligand_type)
    if plot_mc:
        mc_coords = mol.coords_all[mol.mcs]
        ax.scatter(xs=mc_coords[:, 0],ys=mc_coords[:, 1],zs=mc_coords[:, 2],s=50)
    for l in ligands:
        ligand = mol.ligand_ind[l]
        ax.scatter(xs=mol.coords_all[ligand, 0],ys=mol.coords_all[ligand, 1],zs=mol.coords_all[ligand, 2],s=50)
        if plot_mc:
            for mc in mol.mcs:
                for lc in mol.lcs[l]:
                    if mol.graph[mc][lc] == 1:
                        c1, c2 = mol.coords_all[mc], mol.coords_all[lc]
                        ax.plot(xs=[c1[0], c2[0]], ys=[c1[1], c2[1]], zs=[c1[2], c2[2]])
        _length = len(ligand)
        for ind_i in range(_length):
            for ind_j in range(ind_i+1, _length):
                i, j = ligand[ind_i], ligand[ind_j]
                if mol.graph[i][j] == 1:
                    c1, c2 = mol.coords_all[i], mol.coords_all[j]
                    ax.plot(xs=[c1[0], c2[0]], ys=[c1[1], c2[1]], zs=[c1[2], c2[2]])
    set_axes_equal(ax)
    return ax