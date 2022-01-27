import numpy as np


def calculate_b_los(
    stokes_I,
    stokes_V,
    wavelength_arr,
    lambda0,
    lambda_range_min,
    lambda_range_max,
    g_eff
):
    '''
    stokes_I: Array of Intensties
    stokes_V: Array of Stokes V
    wavelength_arr: Units is nm
    lambda0: Units is nm
    lambda_range_min: Units is nm
    lambda_range_max: Units is nm
    g_eff: Effective lande g factor
    '''

    indices = np.where(
        (
            np.array(wavelength_arr) >= (lambda_range_min)
        ) & (
            np.array(wavelength_arr) <= (lambda_range_max)
        )
    )[0]

    wavelength = np.array(wavelength_arr)[indices]

    intensity = np.array(stokes_I)[indices]

    stokes_V_cropped = np.array(stokes_V)[indices]

    derivative = np.gradient(intensity, wavelength * 10)

    constant = 4.66e-13 * g_eff * (lambda0 * 10)**2

    numerator = np.sum(derivative * stokes_V_cropped)

    denominator = np.sum(np.square(derivative))

    return -numerator / (constant * denominator)


def prepare_calculate_blos(
    obs,
    wavelength_arr,
    lambda0,
    lambda_range_min,
    lambda_range_max,
    g_eff,
    ind
):
    def actual_calculate_blos(i, j):

        i = int(i)
        j = int(j)
        stokes_I, stokes_V = obs[0, i, j, ind, 0], obs[0, i, j, ind, 3]
        return calculate_b_los(
            stokes_I,
            stokes_V,
            wavelength_arr,
            lambda0,
            lambda_range_min,
            lambda_range_max,
            g_eff
        )
    return actual_calculate_blos

lambda_range = [6560, 6562]
lambda_range = [6562, 6562.8]

def calculate_b_transverse_wing(
    stokes_I,
    stokes_Q,
    stokes_U,
    wavelength_arr,
    lambda0,
    lambda_range_min,
    lambda_range_max,
    g_eff
):

    norm_Q = np.array(stokes_Q) / np.array(stokes_I)

    norm_U = np.array(stokes_U) / np.array(stokes_I)

    total_linear_polarization = np.sqrt(
        np.add(
            np.square(
                norm_Q
            ),
            np.square(
                norm_U
            )
        )
    )

    indices = np.where(
        (
            np.array(wavelength_arr) > (lambda0 + lambda_range_min)
        ) & (
            np.array(wavelength_arr) < (lambda0 + lambda_range_max)
        )
    )[0]

    wavelength = np.array(wavelength_arr)[indices]

    intensity = np.array(stokes_I)[indices]

    # norm_Q_cropped = norm_Q[indices]

    # norm_U_cropped = norm_U[indices]

    total_linear_polarization_cropped = total_linear_polarization[indices]

    derivative = np.abs(np.gradient(intensity, wavelength * 10))

    constant = (4.6686e-10 * (lambda0 * 10)**2)**2 * g_eff

    diff_lambda = 1 / (np.abs(wavelength - lambda0) * 10)

    numerator = 4 * np.sum(
        total_linear_polarization_cropped * diff_lambda * derivative
    ) / (
        3 * constant
    )

    denominator = np.sum(np.square(diff_lambda) * np.square(derivative))

    return np.sqrt(numerator / denominator)
