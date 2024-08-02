from scipy.integrate.quadrature import cumtrapz
import numpy as np
import h5py


def gaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def instrument_profile(no_wavelength_points, sigma):
    if no_wavelength_points % 2 == 0:
        return ValueError('no_wavelength_points must be odd')

    return gaussian_kernel1d(sigma, no_wavelength_points / 2)


def save_instrument_profile(fwhm, no_wavelength_points):
    np.savetxt(
        'Instrumental_profile.dat',
        instrument_profile(no_wavelength_points, fwhm / 2.355)
    )


def get_tau(f_rhout, f_atmos):
    height = f_atmos['z'][()][0]

    index500 = np.argmin(
        np.abs(
            f_rhout['wavelength_selected'][()] - 500
        )
    )

    tau500 = cumtrapz(
        f_rhout['chi'][()][0, 0, :, index500],
        x=-height
    )

    tau500 = np.concatenate([[1e-20], tau500])

    return tau500


def generate_atmosphere_file_from_rh(
    filename,
    rh_out='.',
    rh_atmos='.',
    straylight_fraction=0,
    velocity_macroturbulent=0
):

    remove_snapshot_keys = [
        'z', 'B_x', 'B_y', 'B_z', 'electron_density',
        'temperature', 'velocity_z', 'velocity_turbulent',
        'density'
    ]

    as_is_keys = [
        'x', 'y'
    ]

    conversion_factor = {
        'z': 1e-3,
        'electron_density': 1e-6,
        'velocity_z': 1e2,
        'velocity_turbulent': 1e2,
        'density': 1e-3,
        'hydrogen_populations': 1e-6
    }

    f_rhout = h5py.File(rh_out, 'r')

    f_atmos = h5py.File(rh_atmos, 'r')

    f_out = h5py.File(filename, 'w')

    for key in remove_snapshot_keys:
        conv_factor = conversion_factor.get(key, 1)
        try:
            f_out[key] = f_atmos[key][()][0] * conv_factor
        except Exception:
            pass

    for key in as_is_keys:
        f_out[key] = f_atmos[key][()]

    hp_conv_factor = conversion_factor.get(
        'hydrogen_populations',
        1
    )

    f_out['hydrogen_populations'] = np.sum(
        f_atmos['hydrogen_populations'][()][0] * hp_conv_factor,
        axis=0
    )

    f_out['tau'] = np.log(get_tau(f_rhout, f_atmos))

    f_out['straylight_fraction'] = straylight_fraction

    f_out['velocity_macroturbulent'] = velocity_macroturbulent

    f_out.close()

    f_atmos.close()

    f_rhout.close()
