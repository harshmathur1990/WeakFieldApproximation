import numpy as np


def least_sq_dist(source, dest):
    return np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    source,
                    dest
                )
            )
        )
    )


def find_k(norm_line, norm_atlas, k_start, k_end, k_iter):

    result_k = -1
    coeff = None
    score = np.Inf

    for k in np.arange(k_start, k_end, k_iter):
        this_coeff = 1 + k * (
            (norm_atlas - norm_line) / (norm_line + norm_atlas)
        )
        this_score = least_sq_dist(norm_atlas, norm_line * this_coeff)

        if this_score < score:
            score = this_score
            coeff = this_coeff
            result_k = k

    return coeff, score, result_k
