import numpy as np


def get_row(i, j):
    return np.array([1, i, j, i * j])

def get_matrices(n=128):
    X, Y = np.meshgrid(range(n), range(n))

    vec_get_row = np.vectorize(get_row, signature='(),()->(n)')

    n_4 = vec_get_row(X, Y).reshape(n * n, 4)

    return np.linalg.inv(
        np.dot(
            n_4.T,
            n_4
        )
    )