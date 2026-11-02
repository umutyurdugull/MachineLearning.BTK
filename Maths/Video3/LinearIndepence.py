import numpy as np
from scipy.linalg import null_space


v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [0, 0, 1]

def isLinearIndependent(matrix):
    
    matrix = np.array(matrix)
    rank = np.linalg.matrix_rank(matrix)
    det = np.linalg.det(matrix)

    vector = matrix.shape[1]
    if (rank == vector):
        print("Independent")
    else:
        ns = null_space(matrix) ## Finds nonzero coefficient vector(s) c such that matrix @ c = 0.
        print(np.round(ns[:, 0] / np.min(np.abs(ns[np.abs(ns) > 1e-5])), 2))    



isLinearIndependent([v1,v2,v3])
