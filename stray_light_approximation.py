import numpy as np
import scipy.ndimage


def prepare_get_indice(arr):
    def get_indice(wave):
        return np.argmin(np.abs(arr - wave))
    return get_indice


def normalise_profiles(
    line_profile,
    line_wave,
    atlas_profile,
    atlas_wave,
    cont_wave
):
    indice_line = np.argmin(np.abs(line_wave - cont_wave))
    indice_atlas = np.argmin(np.abs(atlas_wave - cont_wave))

    get_indice = prepare_get_indice(atlas_wave)

    vec_get_indice = np.vectorize(get_indice)

    atlas_indice = vec_get_indice(line_wave)

    norm_atlas = atlas_profile / atlas_profile[indice_atlas]

    return line_profile / line_profile[indice_line], \
        norm_atlas[atlas_indice], atlas_wave[atlas_indice]


def mean_squarred_error(line_profile, atlas_profile):
    return np.sqrt(
        np.sum(
            np.power(
                np.subtract(
                    line_profile,
                    atlas_profile
                ),
                2
            )
        )
    )


def approximate_stray_light_and_sigma(
    line_profile,
    atlas_profile,
    continuum=1.0,
    indices=None
):
    fwhm = np.linspace(2, 30, 50)

    sigma = fwhm / 2.355

    k_values = np.linspace(0, 1, 100)

    result = np.zeros(shape=(sigma.size, k_values.size))

    result_atlas = np.zeros(
        shape=(
            sigma.size,
            k_values.size,
            atlas_profile.shape[0]
        )
    )

    for i, _sig in enumerate(sigma):
        for j, k_value in enumerate(k_values):
            # degraded_atlas = (
            #     atlas_profile + (k_value * continuum)
            # ) / (1 + k_value)
            # degraded_atlas = scipy.ndimage.gaussian_filter(
            #     degraded_atlas,
            #     _sig
            # )
            degraded_atlas = scipy.ndimage.gaussian_filter(
                (1 - k_value) * atlas_profile,
                _sig
            ) + (k_value * continuum)
            result_atlas[i][j] = degraded_atlas
            result[i][j] = mean_squarred_error(
                degraded_atlas[indices] / degraded_atlas[indices][0],
                line_profile / line_profile[0]
            )
    return result, result_atlas
