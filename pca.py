from PIL import Image
import numpy as np

def pca(X):
    num_data, dim = X.shape
    mean_X = X.mean(axis=0)
    X -= mean_X

    if dim > num_data:
        M = np.dot(X, X.T)
        e, EV = np.linalg.eigh(M)
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        S = np.sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]

    return V, S, mean_X
