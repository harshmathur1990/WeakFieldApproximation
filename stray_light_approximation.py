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
    if indices == None:
        indices = np.arange(line_profile.size)

    fwhm = np.linspace(2, 30, 50)

    sigma = fwhm / 2.355

    k_values = np.arange(0, 1, 0.01)

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
    
    return result, result_atlas, fwhm, sigma, k_values


def prepare_give_output_value(result_list):
    def give_output_value(*args):

        sigma_index = args[0]

        total_error = 0.0

        for indd, arg in enumerate(args[1:]):
            sigma_index = int(sigma_index)
            arg = int(sigma_index)
            total_error += result_list[indd-1][sigma_index, arg]

        return total_error

    return give_output_value


def prepare_give_output_atlas(result_atlas_list):
    def give_output_atlas(*args):

        sigma_index = args[0]

        atlas = None

        for indd, arg in enumerate(args[1:]):
            sigma_index = int(sigma_index)
            arg = int(sigma_index)
            if atlas is None:
                atlas = result_atlas_list[indd-1][sigma_index, arg]
            else:
                atlas = np.concatenate((atlas, result_atlas_list[indd-1][sigma_index, arg]), 2)

        return atlas

    return give_output_atlas


def approximate_stray_light_and_sigma_multiple_lines(
        line_profile_list,
        atlas_profile_list,
):
    fwhm = np.linspace(2, 30, 50)

    sigma = fwhm / 2.355

    k_values = np.arange(0, 1, 0.01)

    dimensions = list()

    dimensions.append(sigma.size)

    for _ in len(line_profile_list):
        dimensions.append(k_values.size)

    merged_result = np.zeros(
        tuple(dimensions),
        dtype=np.float64
    )

    result_list = list()

    result_atlas_list = list()

    for index, line_profile, atlas_profile in enumerate(zip(line_profile_list, atlas_profile_list)):
        result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(line_profile, atlas_profile)

        result /= line_profile.size()

        result_list.append(result)

        result_atlas_list.append(result_atlas)

    give_output_value = prepare_give_output_value(result_list)

    vec_give_output_value = np.vectorize(give_output_value)

    merged_result = np.fromfunction(vec_give_output_value)

    give_output_atlas = prepare_give_output_atlas(result_atlas_list)

    vec_give_output_atlas = np.vectorize(give_output_atlas)

    merged_atlas_profiles = np.fromfunction(vec_give_output_atlas)

    return merged_result, merged_atlas_profiles, fwhm, sigma, k_values
