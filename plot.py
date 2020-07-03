import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(mol):
    """
    A very simple way to plot a molecule in a 3d fashion
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=mol.coords_all[:, 0],ys=mol.coords_all[:, 1],zs=mol.coords_all[:, 2],s=50)

    n = len(mol.atoms)
    for i in range(n):
        for j in range(i+1, n):
            if mol.graph[i][j] == 1:
                c1, c2 = mol.coords_all[i], mol.coords_all[j]
                ax.plot(xs=[c1[0], c2[0]], ys=[c1[1], c2[1]], zs=[c1[2], c2[2]])
    return ax