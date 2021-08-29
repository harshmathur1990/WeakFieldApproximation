from helita.sim import rh15d
from pathlib import Path
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from helita.utils.utilsmath import planck
from astropy import units as u


base_path = Path(
    '/home/harsh/CourseworkRepo/rh/rh/rh15d/run'
)


def get_doppler_velocity(wavelength, center_wavelength):
    return (wavelength - center_wavelength) * 2.99792458e5 / center_wavelength

@np.vectorize
def get_doppler_velocity_halpha(wavelength):
    return get_doppler_velocity(wavelength, 656.2819)


wavelength_selected_line_core_indice = 542


def plot_contribution_plot():
    out = rh15d.Rh15dout(fdir='output')

    height = out.atmos.height_scale[0, 0]

    tau = np.zeros((height.size-1, out.ray.wavelength_indices.data.size))

    height_at_tau_1 = np.zeros(out.ray.wavelength_indices.data.size)

    for index, wave_indice in enumerate(out.ray.wavelength_indices.data):
        tau[:, index] = cumtrapz(
            out.ray.chi[0, 0, :, index],
            x=-height
        )

        height_at_tau_1[index] = height[1:][
            np.argmin(
                np.abs(
                    tau[:, index] - 1
                )
            )
        ] / 1000

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(
        figsize=(
            6,
            6
        )
    )

    gs = gridspec.GridSpec(2, 2)

    gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
    
    # ------------------- #
    axs = plt.subplot(gs[0])

    doppler_wave = get_doppler_velocity_halpha(out.ray.wavelength_selected.data[:-1])

    X, Y = np.meshgrid(
        doppler_wave,
        height[1:] / 1000
    )

    axs.pcolormesh(
        X,
        Y,
        np.log(
            out.ray.chi.data[0, 0, 1:, :-1] / tau[:, :-1]
        ),
        shading='nearest',
        cmap='gray'
    )

    axs2 = axs.twinx()

    axs2.plot(doppler_wave, height_at_tau_1[:-1], color='white')

    # ------------------- #

    axs = plt.subplot(gs[1])

    axs.pcolormesh(X, Y, out.ray.source_function.data[0, 0, 1:, :-1], shading='nearest', cmap='gray')

    axs2 = axs.twinx()

    axs2.plot(doppler_wave, height_at_tau_1[:-1], color='white')

    pf = planck(
        out.ray.wavelength_selected[wavelength_selected_line_core_indice].values * u.nm,
        out.atmos.temperature[0, 0, 1:].values * u.K,
        dist='wavelength'
    ).value

    axs3 = axs.twiny()

    axs3.plot(
        out.ray.source_function.data[0, 0, 1:, wavelength_selected_line_core_indice],
        height[1:] / 1000,
        '-.',
        color='white'
    )

    axs3.plot(pf, height[1:] / 1000, '.', color='white')

    # ------------------- #

    axs = plt.subplot(gs[2])

    axs.pcolormesh(X, Y, tau[:, :-1] * np.exp(-tau[:, :-1]), shading='nearest', cmap='gray')

    axs2 = axs.twinx()

    axs2.plot(doppler_wave, height_at_tau_1[:-1], color='white')

    # ------------------- #

    axs = plt.subplot(gs[3])

    axs.pcolormesh(
        X,
        Y,
        np.exp(-tau[:, :-1]) * out.ray.source_function.data[0, 0, 1:, :-1] * out.ray.chi.data[0, 0, 1:, :-1],
        shading='nearest',
        cmap='gray'
    )

    axs2 = axs.twinx()

    axs2.plot(doppler_wave, height_at_tau_1[:-1], color='white')

    axs3 = axs.twinx()

    axs3.plot(doppler_wave, out.ray.intensity.data[0, 0, out.ray.wavelength_indices.data[:-1]] , color='white')

    # ------------------- #

    plt.show()



