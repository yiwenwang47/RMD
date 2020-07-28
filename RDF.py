import numpy as np
from .elements import *

def radial_distribution_function(array_1: np.ndarray, d_matrix: np.ndarray, array_2: np.ndarray, \
    beta: float, R: float, with_origin=False) -> np.float:

    """
    A very simple radial distribution function. If starts from one atom, set with_origin=True.
    """

    matrix = d_matrix - R
    matrix = np.exp(matrix * matrix *(-beta))
    matrix[np.where(d_matrix==0)] = 0
    f_inv = matrix.sum() #inverse of scaling factor
    if f_inv == 0:
        return 0
    else:
        f = 1/f_inv
    if with_origin:
        assert len(matrix.shape) == 1
        return (array_1 * matrix).dot(array_2)  * f
    else:
        assert len(matrix.shape) == 2
        assert ((array_1-array_2)**2).sum() < 1e-5
        return matrix.dot(array_1).dot(array_2) * f

# def RDF