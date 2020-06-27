import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster.k_means_ import KMeans

def k_way_spectral_clustering():
    x = np.load('q2data.npy')
    A = np.load('AMatrix.npy')
    WeightMatrix = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            if A[i][j] == 1:
                WeightMatrix[i][j] = np.exp(-1 * ((np.linalg.norm(x[i] - x[j]) ** 2)))
            else:
                WeightMatrix[i][j] = 0

    DegreeMatrix = np.sum(WeightMatrix, axis=1)
    L = DegreeMatrix - WeightMatrix
    DSquareRoot = np.diag(1.0 / (DegreeMatrix ** (0.5)))
    Lnorm = np.dot(np.dot(DSquareRoot, L), DSquareRoot)

    eigvals, eigvecs = np.linalg.eig(Lnorm)
    eigvecs = np.array(eigvecs, dtype=np.float64)
    sortedinds = eigvals.argsort()
    eigvec1, eigvec2, eigvec3, eigvec4 = eigvecs[:, 10], eigvecs[:, 11], eigvecs[:, 13], eigvecs[:, 14]

    kmeans = KMeans(n_clusters=3, init='random')
    kmeans.fit(eigvecs)
    components = kmeans.labels_
    return components

k_way_spectral_clustering()