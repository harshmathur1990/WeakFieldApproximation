import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict
import queue


processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

ca_ha_data_file = processed_inputs / 'aligned_Ca_Ha_stic_profiles.nc'

me_data_file = processed_inputs / 'me_results_6569.nc'

wfa_8542_data_file = processed_inputs / 'wfa_8542.nc'


def get_linear_polarisation(f, ind):
    data = f['profiles'][()]

    linpol = np.mean(
        np.sqrt(
            (data[0, :, :, ind, 1]**2 + data[0, :, :, ind, 2]**2) / data[0, :, :, ind, 0]**2
        ),
        0
    )

    return linpol


def get_net_circular_polarisation(f, ind):
    data = f['profiles'][()]

    circpol = np.sum(
        data[0, :, :, ind, 3] / data[0, :, :, ind, 0],
        0
    )

    return circpol


def get_neighbor(pixel, shape_1, shape_2, type=4):

    pixel_list = list()

    four_connected = [
        (pixel[0] - 1, pixel[1]),
        (pixel[0], pixel[1] - 1),
        (pixel[0] + 1, pixel[1]),
        (pixel[0], pixel[1] + 1)
    ]

    eight_connected = [
        (pixel[0] - 1, pixel[1] - 1),
        (pixel[0] - 1, pixel[1] + 1),
        (pixel[0] + 1, pixel[1] - 1),
        (pixel[0] + 1, pixel[1] + 1)
    ]

    for _pixel in four_connected:
        if 0 <= _pixel[0] < shape_1 and 0 <= _pixel[1] < shape_2:
            pixel_list.append(_pixel)

    if type == 8:
        for _pixel in eight_connected:
            if 0 <= _pixel[0] < shape_1 and 0 <= _pixel[1] < 2:
                pixel_list.append(_pixel)

    return pixel_list


def relaxed_pore_image_in_c(image, threshold):

    points_with_min_intensity = np.unravel_index(
        np.argmin(image), image.shape
    )

    intensity_set = set()

    checking_dict = defaultdict(dict)

    intensity_set.add(np.nanmin(image))

    a = points_with_min_intensity

    seed_pixel = a[0], a[1]

    segment = np.zeros_like(image, dtype=np.int64)

    visited = np.zeros_like(image)

    visiting_queue = queue.Queue()

    visiting_queue.put(seed_pixel)

    checking_dict[seed_pixel[0]][seed_pixel[1]] = 1.0

    while not visiting_queue.empty():

        element = visiting_queue.get()

        visited[element] = 1.0

        if image[element] < threshold:
            segment[element] = 1.0

            neighbors = get_neighbor(
                element,
                image.shape[0],
                image.shape[1]
            )

            for neighbor in neighbors:
                if visited[neighbor] == 0.0:
                    if neighbor[1] not in checking_dict[neighbor[0]]:
                        visiting_queue.put(neighbor)
                        checking_dict[neighbor[0]][neighbor[1]] = 1.0

    return segment


def get_fov_data():
    data = np.zeros((4, 2, 17, 60), dtype=np.float64)
    mask = np.zeros((4, 2, 17, 60), dtype=np.int64)

    fcaha = h5py.File(ca_ha_data_file, 'r')

    fme = h5py.File(me_data_file, 'r')

    fwfa = h5py.File(wfa_8542_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

    ind_8542_core = ind[np.argmin(np.abs(fcaha['wav'][ind] - 8542.09))]

    ind_8542_wing = ind[np.argmin(np.abs(fcaha['wav'][ind] - 8534.288))]

    ind_6563_core = ind[np.argmin(np.abs(fcaha['wav'][ind] - 6562.8))]

    ind_6563_wing = ind[np.argmin(np.abs(fcaha['wav'][ind] - 6569.826))]

    ind_6569_wav = ind[np.where((fcaha['wav'][ind] >= (6569.2151 - 0.227)) & (fcaha['wav'][ind] <= (6569.2151 + 0.227)))[0]]

    ind_6562_wav = ind[np.where((fcaha['wav'][ind] >= (6562.8 - 0.7)) & (fcaha['wav'][ind] <= (6562.8 + 0.7)))[0]]

    ind_8542_wav = ind[np.where((fcaha['wav'][ind] >= (8542.09 - 0.40)) & (fcaha['wav'][ind] <= (8542.09 + 0.40)))[0]]

    data[0][0] = fcaha['profiles'][0, :, :, ind_8542_wing, 0]
    data[0][1] = fcaha['profiles'][0, :, :, ind_8542_core, 0]
    data[1][0] = fcaha['profiles'][0, :, :, ind_6563_wing, 0]
    data[1][1] = fcaha['profiles'][0, :, :, ind_6563_core, 0]
    data[2][0] = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])
    data[2][1] = fwfa['blos_gauss'][()]
    data[3][0] = get_linear_polarisation(fcaha, ind_6569_wav)
    data[3][1] = get_linear_polarisation(fcaha, ind_8542_wav)

    mask_arr = relaxed_pore_image_in_c(data[1][0], data[1, 0].mean() + (-1 * data[1, 0].std()))
    mask[:, :] = mask_arr[np.newaxis, np.newaxis, :, :]

    aa, bb, cc, dd = fcaha['wav'][ind_8542_wing], fcaha['wav'][ind_8542_core], fcaha['wav'][ind_6563_wing], fcaha['wav'][ind_6563_core]

    fwfa.close()

    fme.close()

    fcaha.close()

    return data, aa, bb, cc, dd, mask


def make_fov_plots():
    data, wing_ca, core_ca, wing_ha, core_ha, mask = get_fov_data()

    fig, axs = plt.subplots(4, 2, figsize=(7, 4.4))

    extent = [0, 22.8, 0, 6.46]

    fontsize = 8

    for i in range(4):
        for j in range(2):
            if i == 2:
                if j == 0:
                    vmin = -700
                    vmax = 700
                else:
                    vmin = -600
                    vmax = 600

                gh = axs[i][j].imshow(data[i][j], cmap='RdGy', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
                if j == 0:
                    im20 = gh
                else:
                    im21 = gh
            elif i == 3:
                vmin = 0
                vmax = np.round(data[i][j].max() * 100, 2)
                gh = axs[i][j].imshow(data[i][j] * 100, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                if j == 0:
                    im30 = gh
                else:
                    im31 = gh
            else:
                axs[i][j].imshow(data[i][j], cmap='gray', origin='lower')
            axs[i][j].contour(mask[i][j], levels=0, origin='lower', colors='blue', linewidths=0.5)
            axs[i][j].plot(np.ones(60) * 12, linestyle='--', color='brown', linewidth=0.5)

    axs[0][0].text(
        0.02, 0.04,
        r'(a) Ca II 8542 $\mathrm{{\AA}}$ {} $\mathrm{{\AA}}$'.format(
            np.round(wing_ca - 8542.09), 2
        ),
        transform=axs[0][0].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[0][1].text(
        0.02, 0.04,
        r'(b) Ca II 8542 $\mathrm{{\AA}}$ core',
        transform=axs[0][1].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[1][0].text(
        0.02, 0.04,
        r'(c) H$\alpha$ +{} $\mathrm{{\AA}}$'.format(
            np.round(wing_ha - 6562.8), 2
        ),
        transform=axs[1][0].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[1][1].text(
        0.02, 0.04,
        r'(d) H$\alpha$ core',
        transform=axs[1][1].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[2][0].text(
        0.02, 0.04,
        r'(e) $B_{{\mathrm{LOS}}}$ (Fe I 6569 $\AA$) [G]',
        transform=axs[2][0].transAxes,
        color='black',
        fontsize=fontsize
    )

    axs[2][1].text(
        0.02, 0.04,
        r'(f) $B_{{\mathrm{LOS}}}$ (Ca II 8542 $\AA$) [G]',
        transform=axs[2][1].transAxes,
        color='black',
        fontsize=fontsize
    )

    axs[3][0].text(
        0.02, 0.04,
        r'(g) Linear Polarisation (Fe I 6569 $\AA$) [%]',
        transform=axs[3][0].transAxes,
        color='white',
        fontsize=fontsize
    )

    axs[3][1].text(
        0.02, 0.04,
        r'(h) Linear Polarisation (Ca II 8542 $\AA$) [%]',
        transform=axs[3][1].transAxes,
        color='white',
        fontsize=fontsize
    )

    cbaxes = inset_axes(
        axs[2][0],
        width="2%",
        height="50%",
        loc=4,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im20,
        cax=cbaxes,
        ticks=[-600, 0, 600],
        orientation='vertical'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='black')

    cbar.ax.yaxis.set_ticks_position('left')

    cbaxes = inset_axes(
        axs[2][1],
        width="2%",
        height="50%",
        loc=4,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im21,
        cax=cbaxes,
        ticks=[-400, 0, 400],
        orientation='vertical'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbar.ax.yaxis.set_ticks_position('left')

    cbaxes = inset_axes(
        axs[3][0],
        width="2%",
        height="50%",
        loc=4,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im30,
        cax=cbaxes,
        ticks=[0, 0.5, 1],
        orientation='vertical'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='white')

    cbar.ax.yaxis.set_ticks_position('left')

    cbaxes = inset_axes(
        axs[3][1],
        width="2%",
        height="50%",
        loc=4,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im31,
        cax=cbaxes,
        ticks=[0, 0.5, 1],
        orientation='vertical'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='white')

    cbar.ax.yaxis.set_ticks_position('left')

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    # axs[0][1].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    # axs[1][1].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    axs[2][0].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    # axs[2][1].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    axs[3][0].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    # axs[3][1].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    #
    # axs[0][0].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    # axs[0][1].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    # axs[1][0].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    # axs[1][1].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    # axs[2][0].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    # axs[2][1].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    axs[3][0].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    axs[3][1].xaxis.set_minor_locator(MultipleLocator(60/22.8))

    axs[0][0].tick_params(direction='out', which='both', color='black')
    # axs[0][1].tick_params(direction='out', which='both', color='black')
    axs[1][0].tick_params(direction='out', which='both', color='black')
    # axs[1][1].tick_params(direction='out', which='both', color='black')
    axs[2][0].tick_params(direction='out', which='both', color='black')
    # axs[2][1].tick_params(direction='out', which='both', color='black')
    axs[3][0].tick_params(direction='out', which='both', color='black')
    axs[3][1].tick_params(direction='out', which='both', color='black')

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[1][0].set_xticklabels([])
    axs[1][1].set_xticklabels([])
    axs[2][0].set_xticklabels([])
    axs[2][1].set_xticklabels([])
    axs[3][0].set_xticks([0, 10 * 60/22.8, 20 * 60/22.8])
    axs[3][0].set_xticklabels([0, 10, 20])
    axs[3][1].set_xticks([0, 10 * 60/22.8, 20 * 60/22.8])
    axs[3][1].set_xticklabels([0, 10, 20])
    axs[0][0].set_yticks([0, 2 * 17/6.46, 4 * 17/6.46])
    axs[0][0].set_yticklabels([0, 2, 4])
    axs[1][0].set_yticks([0, 2 * 17/6.46, 4 * 17/6.46])
    axs[1][0].set_yticklabels([0, 2, 4])
    axs[2][0].set_yticks([0, 2 * 17/6.46, 4 * 17/6.46])
    axs[2][0].set_yticklabels([0, 2, 4])
    axs[3][0].set_yticks([0, 2 * 17/6.46, 4 * 17/6.46])
    axs[3][0].set_yticklabels([0, 2, 4])
    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticklabels([])
    axs[2][1].set_yticklabels([])
    axs[3][1].set_yticklabels([])

    axs[0][0].set_ylabel('y [arcsec]')
    axs[1][0].set_ylabel('y [arcsec]')
    axs[2][0].set_ylabel('y [arcsec]')
    axs[3][0].set_ylabel('y [arcsec]')

    axs[3][0].set_xlabel('x [arcsec]')
    axs[3][1].set_xlabel('x [arcsec]')

    plt.subplots_adjust(left=0.08, bottom=0.16, right=1, top=1, hspace=0.0, wspace=0.0)

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'FOV.pdf', format='pdf', dpi=300)

    plt.show()


if __name__ == '__main__':
    make_fov_plots()
