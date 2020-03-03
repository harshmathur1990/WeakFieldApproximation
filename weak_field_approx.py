import numpy as np


def calculate_b_los(
    stokes_I,
    stokes_V,
    wavelength_arr,
    lambda0,
    lambda_range,
    g_eff
):
    '''
    stokes_I: Array of Intensties
    stokes_V: Array of Stokes V
    wavelength_arr: Units is nm
    lambda0: Units is nm
    lambda_range: Units is nm
    g_eff: Effective lande g factor
    '''

    indices = np.where(
        (
            np.array(wavelength_arr) > (lambda0 - lambda_range)
        ) & (
            np.array(wavelength_arr) < (lambda0 + lambda_range)
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
