import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example/')
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict
import queue
from matplotlib.colors import LinearSegmentedColormap
from weak_field_approx import prepare_calculate_blos
from prepare_data import *


processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

ca_ha_data_file = processed_inputs / 'aligned_Ca_Ha_stic_profiles.nc'

ca_stray_files = [
    processed_inputs / 'alignedspectra_scan1_map01_Ca.fits_stray_corrected_SiI_8536.h5',
    processed_inputs / 'alignedspectra_scan1_map01_Ca.fits_stray_corrected_FeI_8538.h5',
    processed_inputs / 'alignedspectra_scan1_map01_Ca.fits_stray_corrected_CaII_8542.h5'
]

ha_stray_file = processed_inputs / 'alignedspectra_scan1_map01_Ha.fits_stray_corrected.h5'

me_data_file = processed_inputs / 'me_results_6569.nc'

wfa_8542_data_file = processed_inputs / 'wfa_8542.nc'



ltau500 = np.array(
    [
        -8.        , -7.78133011, -7.77447987, -7.76711988, -7.76003981,
        -7.75249004, -7.74428988, -7.73559999, -7.72637987, -7.71590996,
        -7.7047801 , -7.69357014, -7.6876502 , -7.68174982, -7.67588997,
        -7.66997004, -7.66374016, -7.65712023, -7.64966011, -7.64093018,
        -7.63092995, -7.61920023, -7.60529995, -7.58876991, -7.56925011,
        -7.54674006, -7.52177   , -7.49316978, -7.45849991, -7.41659021,
        -7.36724997, -7.3108902 , -7.24834013, -7.18071985, -7.11129999,
        -7.04137993, -6.97006989, -6.89697981, -6.82298994, -6.74880981,
        -6.6747098 , -6.60046005, -6.52598   , -6.45187998, -6.37933016,
        -6.30926991, -6.24280977, -6.1792798 , -6.11685991, -6.05597019,
        -5.9974699 , -5.94147015, -5.88801003, -5.8468399 , -5.81285   ,
        -5.78013992, -5.74853992, -5.71774006, -5.68761015, -5.65824986,
        -5.62930012, -5.60065985, -5.57245016, -5.54456997, -5.51687002,
        -5.4893198 , -5.46182013, -5.43416977, -5.40622997, -5.3780098 ,
        -5.34959984, -5.32110977, -5.29247999, -5.26357985, -5.23412991,
        -5.20391989, -5.1728301 , -5.1407299 , -5.10780001, -5.07426023,
        -5.03998995, -5.00492001, -4.96953011, -4.9340601 , -4.89821005,
        -4.86195993, -4.82533979, -4.78824997, -4.75065994, -4.71243   ,
        -4.67438984, -4.63696003, -4.59945011, -4.56069994, -4.52212   ,
        -4.48434019, -4.44652987, -4.40795994, -4.36862993, -4.32842016,
        -4.28650999, -4.24205017, -4.19485998, -4.14490986, -4.09186983,
        -4.03446007, -3.97196007, -3.90451002, -3.83087993, -3.74959993,
        -3.66000009, -3.56112003, -3.45190001, -3.33172989, -3.20393991,
        -3.07448006, -2.94443989, -2.81389999, -2.68294001, -2.55164003,
        -2.4200201 , -2.28814006, -2.15604997, -2.02377009, -1.89135003,
        -1.75880003, -1.62612998, -1.49337006, -1.36126995, -1.23139   ,
        -1.10698998, -0.99208999, -0.884893  , -0.78278702, -0.68348801,
        -0.58499599, -0.48555899, -0.38308501, -0.27345601, -0.15217701,
        -0.0221309 ,  0.110786  ,  0.244405  ,  0.378378  ,  0.51182002,
        0.64473999,  0.777188  ,  0.90906298,  1.04043996,  1.17110002
    ]
)
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
    data[0][1] = fcaha['profiles'][0, :, :, ind_6563_wing, 0]
    data[1][0] = fcaha['profiles'][0, :, :, ind_8542_core, 0]
    data[1][1] = fcaha['profiles'][0, :, :, ind_6563_core, 0]
    data[2][0] = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])
    data[2][1] = fwfa['blos_gauss'][()]
    data[3][0] = get_linear_polarisation(fcaha, ind_6569_wav)
    data[3][1] = get_linear_polarisation(fcaha, ind_8542_wav)

    mask_arr = relaxed_pore_image_in_c(data[0][1], data[0, 1].mean() + (-1 * data[0, 1].std()))
    mask[:, :] = mask_arr[np.newaxis, np.newaxis, :, :]

    aa, bb, cc, dd = fcaha['wav'][ind_8542_wing], fcaha['wav'][ind_8542_core], fcaha['wav'][ind_6563_wing], fcaha['wav'][ind_6563_core]

    fwfa.close()

    fme.close()

    fcaha.close()

    return data, aa, bb, cc, dd, mask


def make_fov_plots(points, colors_scatter):
    data, wing_ca, core_ca, wing_ha, core_ha, mask = get_fov_data()

    fig, axs = plt.subplots(2, 4, figsize=(4.15, 7))

    extent = [0, 6.46, 0, 22.8]

    fontsize = 10

    colors = ["darkred", "darkgoldenrod", "white", "green", "blue"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    for i in range(2):
        for j in range(4):
            if j == 2:
                maxval = np.abs(data[j][i]).max()
                limval = (maxval // 100) * 100 + 100
                vmin = -limval
                vmax = limval

                print(limval)

                gh = axs[i][j].imshow(data[j][i].T, cmap=cmap1, origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
                if i == 0:
                    im02 = gh
                else:
                    im12 = gh
            elif j == 3:
                vmin = 0
                vmax = np.round(data[j][i].max() * 100, 2)
                gh = axs[i][j].imshow(data[j][i].T * 100, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                if i == 0:
                    im03 = gh
                else:
                    im13 = gh
            else:
                axs[i][j].imshow(data[j][i].T, cmap='gray', origin='lower')
            axs[i][j].contour(mask[j][i].T, levels=0, origin='lower', colors='blue', linewidths=0.5)
            axs[i][j].axvline(12, linestyle='--', color='brown', linewidth=0.5)
            axs[i][j].axvline(8, linestyle='--', color='darkgreen', linewidth=0.5)
            for point, color in zip(points, colors_scatter):
                axs[i][j].scatter(point[0], point[1], color=color, marker='x')

    axs[0][0].text(
        0.02, 0.04,
        # r'(a) Ca II 8542 $\mathrm{{\AA}}$ {} $\mathrm{{\AA}}$'.format(
        #     np.round(wing_ca - 8542.09), 2
        # ),
        r'(a)',
        transform=axs[0][0].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[1][0].text(
        0.02, 0.04,
        # r'(b) H$\alpha$ +{} $\mathrm{{\AA}}$'.format(
        #     np.round(wing_ha - 6562.8), 2
        # ),
        r'(b)',
        transform=axs[1][0].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[0][1].text(
        0.02, 0.04,
        # r'(c) Ca II 8542 $\mathrm{{\AA}}$ core',
        r'(c)',
        transform=axs[0][1].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[1][1].text(
        0.02, 0.04,
        # r'(d) H$\alpha$ core',
        r'(d)',
        transform=axs[1][1].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[0][2].text(
        0.02, 0.04,
        # r'(e) $B_{{\mathrm{LOS}}}$ (Fe I 6569 $\AA$) [G]',
        r'(e)',
        transform=axs[0][2].transAxes,
        color='black',
        fontsize=fontsize
    )

    axs[1][2].text(
        0.02, 0.04,
        # r'(f) $B_{{\mathrm{LOS}}}$ (Ca II 8542 $\AA$) [G]',
        r'(f)',
        transform=axs[1][2].transAxes,
        color='black',
        fontsize=fontsize
    )

    axs[0][3].text(
        0.02, 0.04,
        # r'(g) Linear Polarisation (Fe I 6569 $\AA$) [%]',
        r'(g)',
        transform=axs[0][3].transAxes,
        color='white',
        fontsize=fontsize
    )

    axs[1][3].text(
        0.02, 0.04,
        # r'(h) Linear Polarisation (Ca II 8542 $\AA$) [%]',
        r'(h)',
        transform=axs[1][3].transAxes,
        color='white',
        fontsize=fontsize
    )

    cbaxes = inset_axes(
        axs[0][2],
        width="70%",
        height="3%",
        loc=1,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im02,
        cax=cbaxes,
        ticks=[-700, 0, 700],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize - 2, colors='black')

    cbar.ax.yaxis.set_ticks_position('left')

    cbaxes = inset_axes(
        axs[1][2],
        width="70%",
        height="3%",
        loc=1,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im12,
        cax=cbaxes,
        ticks=[-500, 0, 500],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize - 2, colors='black')

    cbar.ax.yaxis.set_ticks_position('left')

    cbaxes = inset_axes(
        axs[0][3],
        width="70%",
        height="3%",
        loc=1,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im03,
        cax=cbaxes,
        ticks=[0, 0.5, 1],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize - 2, colors='white')

    cbar.ax.yaxis.set_ticks_position('left')

    cbaxes = inset_axes(
        axs[1][3],
        width="70%",
        height="3%",
        loc=1,
        borderpad=0.5
    )
    cbar = fig.colorbar(
        im13,
        cax=cbaxes,
        ticks=[0, 0.4, 0.8],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize - 2, colors='white')

    cbar.ax.yaxis.set_ticks_position('left')

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(17/6.46))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(17/6.46))
    axs[0][2].xaxis.set_minor_locator(MultipleLocator(17 / 6.46))
    axs[0][3].xaxis.set_minor_locator(MultipleLocator(17 / 6.46))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(60/22.8))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(60/22.8))
    axs[0][2].yaxis.set_minor_locator(MultipleLocator(60/22.8))
    axs[0][3].yaxis.set_minor_locator(MultipleLocator(60/22.8))

    axs[1][0].xaxis.set_minor_locator(MultipleLocator(17 / 6.46))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(17 / 6.46))
    axs[1][2].xaxis.set_minor_locator(MultipleLocator(17 / 6.46))
    axs[1][3].xaxis.set_minor_locator(MultipleLocator(17 / 6.46))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(60 / 22.8))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(60 / 22.8))
    axs[1][2].yaxis.set_minor_locator(MultipleLocator(60 / 22.8))
    axs[1][3].yaxis.set_minor_locator(MultipleLocator(60 / 22.8))

    axs[0][0].tick_params(direction='out', which='both', color='black', labelsize=fontsize)
    axs[0][1].tick_params(direction='out', which='both', color='black', labelsize=fontsize)
    axs[0][2].tick_params(direction='out', which='both', color='black', labelsize=fontsize)
    axs[0][3].tick_params(direction='out', which='both', color='black', labelsize=fontsize)

    axs[1][0].tick_params(direction='out', which='both', color='black', labelsize=fontsize)
    axs[1][1].tick_params(direction='out', which='both', color='black', labelsize=fontsize)
    axs[1][2].tick_params(direction='out', which='both', color='black', labelsize=fontsize)
    axs[1][3].tick_params(direction='out', which='both', color='black', labelsize=fontsize)

    axs[0][0].set_xticks([0, 2 * 17 / 6.46, 4 * 17 / 6.46])
    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticks([0, 2 * 17 / 6.46, 4 * 17 / 6.46])
    axs[0][1].set_xticklabels([])
    axs[0][2].set_xticks([0, 2 * 17 / 6.46, 4 * 17 / 6.46])
    axs[0][2].set_xticklabels([])
    axs[0][3].set_xticks([0, 2 * 17 / 6.46, 4 * 17 / 6.46])
    axs[0][3].set_xticklabels([])
    axs[1][0].set_xticks([0, 2 * 17/6.46, 4 * 17/6.46])
    axs[1][0].set_xticklabels([0, 2, 4])
    axs[1][1].set_xticks([0, 2 * 17 / 6.46, 4 * 17 / 6.46])
    axs[1][1].set_xticklabels([0, 2, 4])
    axs[1][2].set_xticks([0, 2 * 17 / 6.46, 4 * 17 / 6.46])
    axs[1][2].set_xticklabels([0, 2, 4])
    axs[1][3].set_xticks([0, 2 * 17 / 6.46, 4 * 17 / 6.46])
    axs[1][3].set_xticklabels([0, 2, 4])

    axs[0][0].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[0][0].set_yticklabels([0, 10, 20])
    axs[0][1].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[0][1].set_yticklabels([])
    axs[0][2].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[0][2].set_yticklabels([])
    axs[0][3].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[0][3].set_yticklabels([])
    axs[1][0].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[1][0].set_yticklabels([0, 10, 20])
    axs[1][1].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[1][1].set_yticklabels([])
    axs[1][2].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[1][2].set_yticklabels([])
    axs[1][3].set_yticks([0, 10 * 60 / 22.8, 20 * 60 / 22.8])
    axs[1][3].set_yticklabels([])

    axs[1][2].text(
        -0.9, -0.13,
        'scan direction [arcsec]',
        transform=axs[1][2].transAxes,
        color='black',
        fontsize=fontsize
    )

    axs[0][0].set_ylabel('slit position [arcsec]', fontsize=fontsize)

    axs[1][0].set_ylabel('slit position [arcsec]', fontsize=fontsize)

    plt.subplots_adjust(left=0.12, bottom=0.08, right=1, top=1, hspace=0.0, wspace=0.0)

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'FOV.pdf', format='pdf', dpi=300)

    plt.show()


def plot_stokes_parameters(cut_indice, points, colors_p):

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    colors = ["red", "yellow", "white", "blue", "green"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    ind_ca_1 = np.where((fcaha['wav'][ind[0:306]] >= 8535.75) & (fcaha['wav'][ind[0:306]] <= 8536.75))[0]
    ind_ca_2 = np.where((fcaha['wav'][ind[0:306]] >= 8537.5) & (fcaha['wav'][ind[0:306]] <= 8538.5))[0]
    ind_ca_3 = np.where((fcaha['wav'][ind[0:306]] >= 8541) & (fcaha['wav'][ind[0:306]] <= 8543))[0]

    ind_ha_1 = np.where((fcaha['wav'][ind[306:]] >= 6561.5) & (fcaha['wav'][ind[306:]] <= 6564.5))[0]
    ind_ha_2 = np.where((fcaha['wav'][ind[306:]] >= 6568.5) & (fcaha['wav'][ind[306:]] <= 6570))[0]

    fontsize = 8

    fig = plt.figure(figsize=(7, 4.5))

    gs1 = gridspec.GridSpec(2, 3, width_ratios=[0.2, 0.2, 0.6])

    gs1.update(left=0.07, bottom=0.09, right=0.48, top=0.95, wspace=0.03, hspace=0.15)

    gs3 = gridspec.GridSpec(2, 2, width_ratios=[0.7, 0.3])

    gs3.update(left=0.57, bottom=0.09, right=0.93, top=0.95, wspace=0.03, hspace=0.15)

    axs1 = [[], []]
    axs2 = [[], []]

    k = 0
    for i in range(2):
        for j in range(3):
            axs1[i].append(fig.add_subplot(gs1[k]))
            k += 1

    k = 0
    for i in range(2):
        for j in range(2):
            axs2[i].append(fig.add_subplot(gs3[k]))
            k += 1

    axs1[0][0].spines['right'].set_visible(False)
    axs1[0][1].spines['left'].set_visible(False)
    axs1[0][0].yaxis.tick_left()
    axs1[0][0].tick_params(labelright='off')
    axs1[0][1].tick_params(labelleft='off')
    axs1[0][0].tick_params(labelright=False)
    axs1[0][1].yaxis.tick_right()
    axs1[0][0].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_1[0]], fcaha['wav'][ind[0:306]][ind_ca_1[-1]])
    axs1[0][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])

    axs1[0][1].spines['right'].set_visible(False)
    axs1[0][2].spines['left'].set_visible(False)
    axs1[0][1].yaxis.tick_left()
    axs1[0][1].tick_params(labelright='off')
    axs1[0][2].tick_params(labelleft='off')
    axs1[0][1].tick_params(labelright=False)
    axs1[0][2].yaxis.tick_right()
    axs1[0][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])
    axs1[0][2].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_3[0]], fcaha['wav'][ind[0:306]][ind_ca_3[-1]])

    axs1[1][0].spines['right'].set_visible(False)
    axs1[1][1].spines['left'].set_visible(False)
    axs1[1][0].yaxis.tick_left()
    axs1[1][0].tick_params(labelright='off')
    axs1[1][1].tick_params(labelleft='off')
    axs1[1][0].tick_params(labelright=False)
    axs1[1][1].yaxis.tick_right()
    axs1[1][0].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_1[0]], fcaha['wav'][ind[0:306]][ind_ca_1[-1]])
    axs1[1][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])

    axs1[1][1].spines['right'].set_visible(False)
    axs1[1][2].spines['left'].set_visible(False)
    axs1[1][1].yaxis.tick_left()
    axs1[1][1].tick_params(labelright='off')
    axs1[1][2].tick_params(labelleft='off')
    axs1[1][1].tick_params(labelright=False)
    axs1[1][2].yaxis.tick_right()
    axs1[1][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])
    axs1[1][2].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_3[0]], fcaha['wav'][ind[0:306]][ind_ca_3[-1]])

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1[0][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[0][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[0][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[0][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[0][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[0][1].plot((-d, +d), (-d, +d), **kwargs)

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1[0][1].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[0][1].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[0][1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[0][2].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[0][2].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[0][2].plot((-d, +d), (-d, +d), **kwargs)

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1[1][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[1][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[1][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[1][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[1][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[1][1].plot((-d, +d), (-d, +d), **kwargs)

    kwargs = dict(transform=axs1[1][1].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[1][1].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[1][1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[1][2].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[1][2].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[1][2].plot((-d, +d), (-d, +d), **kwargs)

    axs2[0][0].spines['right'].set_visible(False)
    axs2[0][1].spines['left'].set_visible(False)
    axs2[0][0].yaxis.tick_left()
    axs2[0][0].tick_params(labelright='off')
    axs2[0][1].tick_params(labelleft='off')
    axs2[0][0].tick_params(labelright=False)
    axs2[0][1].yaxis.tick_right()
    axs2[0][0].set_xlim(fcaha['wav'][ind[306:]][ind_ha_1[0]], fcaha['wav'][ind[306:]][ind_ha_1[-1]])
    axs2[0][1].set_xlim(fcaha['wav'][ind[306:]][ind_ha_2[0]], fcaha['wav'][ind[306:]][ind_ha_2[-1]])

    axs2[1][0].spines['right'].set_visible(False)
    axs2[1][1].spines['left'].set_visible(False)
    axs2[1][0].yaxis.tick_left()
    axs2[1][0].tick_params(labelright='off')
    axs2[1][1].tick_params(labelleft='off')
    axs2[1][0].tick_params(labelright=False)
    axs2[1][1].yaxis.tick_right()
    axs2[1][0].set_xlim(fcaha['wav'][ind[306:]][ind_ha_1[0]], fcaha['wav'][ind[306:]][ind_ha_1[-1]])
    axs2[1][1].set_xlim(fcaha['wav'][ind[306:]][ind_ha_2[0]], fcaha['wav'][ind[306:]][ind_ha_2[-1]])

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs2[0][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs2[0][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs2[0][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs2[0][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs2[0][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs2[0][1].plot((-d, +d), (-d, +d), **kwargs)

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs2[1][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs2[1][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs2[1][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs2[1][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs2[1][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs2[1][1].plot((-d, +d), (-d, +d), **kwargs)

    factor = 1
    X1, Y1 = np.meshgrid(fcaha['wav'][ind[0:306]][ind_ca_1], np.arange(0, 60 * 0.38, 0.38))
    X2, Y2 = np.meshgrid(fcaha['wav'][ind[0:306]][ind_ca_2], np.arange(0, 60 * 0.38, 0.38))
    X3, Y3 = np.meshgrid(fcaha['wav'][ind[0:306]][ind_ca_3], np.arange(0, 60 * 0.38, 0.38))
    im00 = axs1[0][0].pcolormesh(X1, Y1, fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_1], 0], cmap='gray', shading='nearest', linewidth=0, rasterized=True, vmin=0.1, vmax=1)
    im01 = axs1[0][1].pcolormesh(X2, Y2, fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_2], 0], cmap='gray', shading='nearest', linewidth=0, rasterized=True, vmin=0.1, vmax=1)
    im02 = axs1[0][2].pcolormesh(X3, Y3, fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_3], 0], cmap='gray', shading='nearest', linewidth=0, rasterized=True, vmin=0.1, vmax=1)
    im10 = axs1[1][0].pcolormesh(X1, Y1, fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_1], 3] * factor * 100 / fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_1], 0], cmap=cmap1, shading='nearest', linewidth=0, rasterized=True, vmin=-10, vmax=10)
    im11 = axs1[1][1].pcolormesh(X2, Y2, fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_2], 3] * 100 / fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_2], 0], cmap=cmap1, shading='nearest', linewidth=0, rasterized=True, vmin=-10, vmax=10)
    im12 = axs1[1][2].pcolormesh(X3, Y3, fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_3], 3] * 100 / fcaha['profiles'][0, cut_indice, :, ind[0:306][ind_ca_3], 0], cmap=cmap1, shading='nearest', linewidth=0, rasterized=True, vmin=-10, vmax=10)

    im00.set_edgecolor('face')
    im01.set_edgecolor('face')
    im02.set_edgecolor('face')
    im10.set_edgecolor('face')
    im11.set_edgecolor('face')
    im12.set_edgecolor('face')

    cbar02 = fig.colorbar(im02, ax=axs1[0][2])
    cbar12 = fig.colorbar(im12, ax=axs1[1][2], ticks=[9, 6, 3, 0, -3, -6, -9])
    cbar02.ax.tick_params(labelsize=fontsize)
    cbar12.ax.tick_params(labelsize=fontsize)

    factor = 1
    X1, Y1 = np.meshgrid(fcaha['wav'][ind[306:]][ind_ha_1], np.arange(0, 60 * 0.38, 0.38))
    X2, Y2 = np.meshgrid(fcaha['wav'][ind[306:]][ind_ha_2], np.arange(0, 60 * 0.38, 0.38))
    im00 = axs2[0][0].pcolormesh(X1, Y1, fcaha['profiles'][0, cut_indice, :, ind[306:][ind_ha_1], 0], cmap='gray', shading='nearest', linewidth=0, rasterized=True, vmin=0.1, vmax=1)
    im01 = axs2[0][1].pcolormesh(X2, Y2, fcaha['profiles'][0, cut_indice, :, ind[306:][ind_ha_2], 0], cmap='gray', shading='nearest', linewidth=0, rasterized=True, vmin=0.1, vmax=1)
    im10 = axs2[1][0].pcolormesh(X1, Y1, fcaha['profiles'][0, cut_indice, :, ind[306:][ind_ha_1], 3] * factor * 100 / fcaha['profiles'][0, cut_indice, :, ind[306:][ind_ha_1], 0], cmap=cmap1, shading='nearest', linewidth=0, rasterized=True, vmin=-10, vmax=10)
    im11 = axs2[1][1].pcolormesh(X2, Y2, fcaha['profiles'][0, cut_indice, :, ind[306:][ind_ha_2], 3] * 100 / fcaha['profiles'][0, cut_indice, :, ind[306:][ind_ha_2], 0], cmap=cmap1, shading='nearest', linewidth=0, rasterized=True, vmin=-10, vmax=10)

    im00.set_edgecolor('face')
    im01.set_edgecolor('face')
    im10.set_edgecolor('face')
    im11.set_edgecolor('face')

    cbar01 = fig.colorbar(im01, ax=axs2[0][1])
    cbar11 = fig.colorbar(im11, ax=axs2[1][1], ticks=[9, 6, 3, 0, -3, -6, -9])
    cbar01.ax.tick_params(labelsize=fontsize)
    cbar11.ax.tick_params(labelsize=fontsize)

    colors = colors_p
    for point, color in zip(points, colors):
        i = 0
        for j in range(2):
            axs1[j][i].plot(fcaha['wav'][ind[0:306]][ind_ca_1], np.ones_like(fcaha['wav'][ind[0:306]][ind_ca_1]) * point * 0.38, color=color, linestyle='--', linewidth=0.5)

    for point, color in zip(points, colors):
        i = 1
        for j in range(2):
            axs1[j][i].plot(fcaha['wav'][ind[0:306]][ind_ca_2], np.ones_like(fcaha['wav'][ind[0:306]][ind_ca_2]) * point * 0.38, color=color, linestyle='--', linewidth=0.5)

    for point, color in zip(points, colors):
        i = 2
        for j in range(2):
            axs1[j][i].plot(fcaha['wav'][ind[0:306]][ind_ca_3], np.ones_like(fcaha['wav'][ind[0:306]][ind_ca_3]) * point * 0.38, color=color, linestyle='--', linewidth=0.5)

    for point, color in zip(points, colors):
        i = 0
        for j in range(2):
            axs2[j][i].plot(fcaha['wav'][ind[306:]][ind_ha_1], np.ones_like(fcaha['wav'][ind[306:]][ind_ha_1]) * point * 0.38, color=color, linestyle='--', linewidth=0.5)

    for point, color in zip(points, colors):
        i = 1
        for j in range(2):
            axs2[j][i].plot(fcaha['wav'][ind[306:]][ind_ha_2], np.ones_like(fcaha['wav'][ind[306:]][ind_ha_2]) * point * 0.38, color=color, linestyle='--', linewidth=0.5)


    axs1[0][0].set_ylabel('slit position [arcsec]', fontsize=fontsize)
    axs1[1][0].set_ylabel('slit position [arcsec]', fontsize=fontsize)

    axs1[0][0].set_yticks([0, 5, 10, 15, 20])
    axs1[0][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs1[0][1].set_yticks([])
    axs1[0][1].set_yticklabels([])
    axs1[0][2].set_yticks([])
    axs1[0][2].set_yticklabels([])
    axs1[1][0].set_yticks([0, 5, 10, 15, 20])
    axs1[1][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs1[1][1].set_yticks([])
    axs1[1][1].set_yticklabels([])
    axs1[1][2].set_yticks([])
    axs1[1][2].set_yticklabels([])

    axs1[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs1[1][0].yaxis.set_minor_locator(MultipleLocator(1))

    axs1[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[0][2].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[1][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[1][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[1][2].xaxis.set_minor_locator(MultipleLocator(0.25))

    axs2[0][0].set_yticks([0, 5, 10, 15, 20])
    axs2[0][0].set_yticklabels([], fontsize=fontsize)
    axs2[0][1].set_yticks([])
    axs2[0][1].set_yticklabels([])
    axs2[1][0].set_yticks([0, 5, 10, 15, 20])
    axs2[1][0].set_yticklabels([], fontsize=fontsize)
    axs2[1][1].set_yticks([])
    axs2[1][1].set_yticklabels([])

    axs2[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs2[1][0].yaxis.set_minor_locator(MultipleLocator(1))

    axs2[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs2[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs2[1][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs2[1][1].xaxis.set_minor_locator(MultipleLocator(0.25))

    axs2[0][0].tick_params(labelsize=fontsize)
    axs2[0][1].tick_params(labelsize=fontsize)
    axs2[1][0].tick_params(labelsize=fontsize)
    axs2[1][1].tick_params(labelsize=fontsize)

    axs1[0][0].tick_params(labelsize=fontsize)
    axs1[0][1].tick_params(labelsize=fontsize)
    axs1[0][2].tick_params(labelsize=fontsize)
    axs2[0][0].tick_params(labelsize=fontsize)
    axs2[0][1].tick_params(labelsize=fontsize)
    axs1[1][0].tick_params(labelsize=fontsize)
    axs1[1][1].tick_params(labelsize=fontsize)
    axs1[1][2].tick_params(labelsize=fontsize)
    axs2[1][0].tick_params(labelsize=fontsize)
    axs2[1][1].tick_params(labelsize=fontsize)

    axs1[0][0].set_xticks([8536])
    axs1[0][0].set_xticklabels([], fontsize=fontsize)

    axs1[1][0].set_xticks([8536])
    axs1[1][0].set_xticklabels([], fontsize=fontsize)

    axs1[0][1].set_xticks([8538])
    axs1[0][1].set_xticklabels([], fontsize=fontsize)

    axs1[1][1].set_xticks([8538])
    axs1[1][1].set_xticklabels([], fontsize=fontsize)

    axs1[0][2].set_xticks([8542])
    axs1[0][2].set_xticklabels([], fontsize=fontsize)

    axs1[1][2].set_xticks([8542])
    axs1[1][2].set_xticklabels([], fontsize=fontsize)

    axs2[0][0].set_xticks([6562, 6563, 6564])
    axs2[0][0].set_xticklabels([], fontsize=fontsize)

    axs2[1][0].set_xticks([6562, 6563, 6564])
    axs2[1][0].set_xticklabels([], fontsize=fontsize)

    axs2[0][1].set_xticks([6569])
    axs2[0][1].set_xticklabels([], fontsize=fontsize)

    axs2[1][1].set_xticks([6569])
    axs2[1][1].set_xticklabels([], fontsize=fontsize)

    axs1[0][2].text(
        0.93, 1.07,
        'Intensity [Normalised]',
        transform=axs1[0][2].transAxes,
        fontsize=fontsize
    )
    axs1[1][2].text(
        1.3, 1.07,
        r'$V/I$ [%]',
        transform=axs1[1][2].transAxes,
        fontsize=fontsize
    )

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'CaII_Stokes_{}.pdf'.format(cut_indice), format='pdf', dpi=300)

    fcaha.close()


def plot_profiles():
    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    plt.pcolormesh(fcaha['profiles'][0, 12, :, ind[0:306], 0], cmap='gray', shading='gouraud')

    plt.show()

    fcaha.close()


def plot_spatial_variation_of_profiles(cut_indice, points, colors, factor_ca_list, factor_ha_list):

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    medprofca = list()
    for ca_stray_file in ca_stray_files:
        fstrayca = h5py.File(ca_stray_file, 'r')
        medprofca += list(fstrayca['stray_corrected_median'][()] / fstrayca['stic_cgs_calib_factor'][()])
        fstrayca.close()
    medprofca = np.array(medprofca)

    fstrayha = h5py.File(ha_stray_file, 'r')
    medprofha = fstrayha['stray_corrected_median'][()] / fstrayha['stic_cgs_calib_factor'][()]
    fstrayha.close()

    max_ca = list()
    norm_profiles_ca = list()
    norm_stokes_I_ca_list = list()
    medprofca_list = list()
    max_ha = list()
    norm_profiles_ha = list()
    norm_stokes_I_ha_list = list()
    medprofha_list = list()

    center_list = list()
    ylim = list()

    ind_ca_1 = np.where((fcaha['wav'][ind[0:306]] >= 8535.75) & (fcaha['wav'][ind[0:306]] <= 8536.75))[0]
    ind_ca_2 = np.where((fcaha['wav'][ind[0:306]] >= 8537.5) & (fcaha['wav'][ind[0:306]] <= 8538.5))[0]
    ind_ca_3 = np.where((fcaha['wav'][ind[0:306]] >= 8541) & (fcaha['wav'][ind[0:306]] <= 8543))[0]

    ind_ha_1 = np.where((fcaha['wav'][ind[306:]] >= 6561.5) & (fcaha['wav'][ind[306:]] <= 6564.5))[0]
    ind_ha_2 = np.where((fcaha['wav'][ind[306:]] >= 6568.5) & (fcaha['wav'][ind[306:]] <= 6570))[0]

    for index, (point, factor_ca, factor_ha) in enumerate(zip(points, factor_ca_list, factor_ha_list)):
        factor_ca_arr = np.ones_like(fcaha['wav'][ind[0:306]], dtype=np.float64)
        factor_ha_arr = np.ones_like(fcaha['wav'][ind[306:]], dtype=np.float64)

        factor_ca_arr[ind_ca_1] = factor_ca[0]
        factor_ca_arr[ind_ca_2] = factor_ca[1]
        factor_ca_arr[ind_ca_3] = factor_ca[2]

        factor_ha_arr[ind_ha_1] = factor_ha[0]
        factor_ha_arr[ind_ha_2] = factor_ha[1]

        center = 2 * index - ((len(points) // 2) * 2)
        center *= -1
        center_list.append(center)
        print(center)
        if index == 0:
            ylim.append(center + 1)
        if index == len(points) - 1:
            ylim.append(center - 1)

        amax = np.round(
            np.abs(fcaha['profiles'][0, cut_indice, point, ind[0:306], 3] * 100 / fcaha['profiles'][0, cut_indice, point, ind[0:306], 0]).max(),
            1
        )
        norm_profile_ca = (fcaha['profiles'][0, cut_indice, point, ind[0:306], 3] * 100 / fcaha['profiles'][0, cut_indice, point, ind[0:306], 0]) / amax
        norm_profile_ca *= factor_ca_arr
        max_ca.append(amax)
        norm_profile_ca += center
        norm_profiles_ca.append(norm_profile_ca)
        norm_stokes_I_ca = fcaha['profiles'][0, cut_indice, point, ind[0:306], 0]
        norm_stokes_I_ca_list.append(norm_stokes_I_ca)
        medprofca_list.append(medprofca)

        amax = np.round(
            np.abs(fcaha['profiles'][0, cut_indice, point, ind[306:], 3] * 100 / fcaha['profiles'][0, cut_indice, point, ind[306:], 0]).max(),
            1
        )
        norm_profile_ha = (fcaha['profiles'][0, cut_indice, point, ind[306:], 3] * 100 / fcaha['profiles'][0, cut_indice, point, ind[306:], 0]) / amax
        norm_profile_ha *= factor_ha_arr
        max_ha.append(amax)
        norm_profile_ha += center
        norm_profiles_ha.append(norm_profile_ha)
        norm_stokes_I_ha = fcaha['profiles'][0, cut_indice, point, ind[306:], 0]
        norm_stokes_I_ha_list.append(norm_stokes_I_ha)
        medprofha_list.append(medprofha)

    linewidth = 0.5

    fontsize = 8

    fig = plt.figure(figsize=(7, 4.5))

    gs1 = gridspec.GridSpec(2, 3, width_ratios=[0.23, 0.23, 0.54])

    gs1.update(left=0.07, bottom=0.09, right=0.43, top=0.95, wspace=0.03, hspace=0.15)

    gs3 = gridspec.GridSpec(2, 2, width_ratios=[0.74, 0.26])

    gs3.update(left=0.57, bottom=0.09, right=0.9105, top=0.95, wspace=0.03, hspace=0.15)

    axs1 = [[], []]
    axs2 = [[], []]

    k = 0
    for i in range(2):
        for j in range(3):
            axs1[i].append(fig.add_subplot(gs1[k]))
            k += 1

    k = 0
    for i in range(2):
        for j in range(2):
            axs2[i].append(fig.add_subplot(gs3[k]))
            k += 1

    axs1[0][0].spines['right'].set_visible(False)
    axs1[0][1].spines['left'].set_visible(False)
    axs1[0][0].yaxis.tick_left()
    axs1[0][0].tick_params(labelright='off')
    axs1[0][1].tick_params(labelleft='off')
    axs1[0][0].tick_params(labelright=False)
    axs1[0][1].yaxis.tick_right()
    axs1[0][0].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_1[0]], fcaha['wav'][ind[0:306]][ind_ca_1[-1]])
    axs1[0][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])

    axs1[0][1].spines['right'].set_visible(False)
    axs1[0][2].spines['left'].set_visible(False)
    axs1[0][1].yaxis.tick_left()
    axs1[0][1].tick_params(labelright='off')
    axs1[0][2].tick_params(labelleft='off')
    axs1[0][1].tick_params(labelright=False)
    axs1[0][2].yaxis.tick_right()
    axs1[0][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])
    axs1[0][2].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_3[0]], fcaha['wav'][ind[0:306]][ind_ca_3[-1]])

    axs1[1][0].spines['right'].set_visible(False)
    axs1[1][1].spines['left'].set_visible(False)
    axs1[1][0].yaxis.tick_left()
    axs1[1][0].tick_params(labelright='off')
    axs1[1][1].tick_params(labelleft='off')
    axs1[1][0].tick_params(labelright=False)
    axs1[1][1].yaxis.tick_right()
    axs1[1][0].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_1[0]], fcaha['wav'][ind[0:306]][ind_ca_1[-1]])
    axs1[1][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])

    axs1[1][1].spines['right'].set_visible(False)
    axs1[1][2].spines['left'].set_visible(False)
    axs1[1][1].yaxis.tick_left()
    axs1[1][1].tick_params(labelright='off')
    axs1[1][2].tick_params(labelleft='off')
    axs1[1][1].tick_params(labelright=False)
    axs1[1][2].yaxis.tick_right()
    axs1[1][1].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_2[0]], fcaha['wav'][ind[0:306]][ind_ca_2[-1]])
    axs1[1][2].set_xlim(fcaha['wav'][ind[0:306]][ind_ca_3[0]], fcaha['wav'][ind[0:306]][ind_ca_3[-1]])

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1[0][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[0][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[0][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[0][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[0][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[0][1].plot((-d, +d), (-d, +d), **kwargs)

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1[0][1].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[0][1].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[0][1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[0][2].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[0][2].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[0][2].plot((-d, +d), (-d, +d), **kwargs)

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1[1][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[1][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[1][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[1][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[1][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[1][1].plot((-d, +d), (-d, +d), **kwargs)

    kwargs = dict(transform=axs1[1][1].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs1[1][1].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs1[1][1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs1[1][2].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs1[1][2].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs1[1][2].plot((-d, +d), (-d, +d), **kwargs)

    axs2[0][0].spines['right'].set_visible(False)
    axs2[0][1].spines['left'].set_visible(False)
    axs2[0][0].yaxis.tick_left()
    axs2[0][0].tick_params(labelright='off')
    axs2[0][1].tick_params(labelleft='off')
    axs2[0][0].tick_params(labelright=False)
    axs2[0][1].yaxis.tick_right()
    axs2[0][0].set_xlim(fcaha['wav'][ind[306:]][ind_ha_1[0]], fcaha['wav'][ind[306:]][ind_ha_1[-1]])
    axs2[0][1].set_xlim(fcaha['wav'][ind[306:]][ind_ha_2[0]], fcaha['wav'][ind[306:]][ind_ha_2[-1]])

    axs2[1][0].spines['right'].set_visible(False)
    axs2[1][1].spines['left'].set_visible(False)
    axs2[1][0].yaxis.tick_left()
    axs2[1][0].tick_params(labelright='off')
    axs2[1][1].tick_params(labelleft='off')
    axs2[1][0].tick_params(labelright=False)
    axs2[1][1].yaxis.tick_right()
    axs2[1][0].set_xlim(fcaha['wav'][ind[306:]][ind_ha_1[0]], fcaha['wav'][ind[306:]][ind_ha_1[-1]])
    axs2[1][1].set_xlim(fcaha['wav'][ind[306:]][ind_ha_2[0]], fcaha['wav'][ind[306:]][ind_ha_2[-1]])

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs2[0][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs2[0][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs2[0][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs2[0][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs2[0][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs2[0][1].plot((-d, +d), (-d, +d), **kwargs)

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs2[1][0].transAxes, color='k', clip_on=False, linewidth=0.5)
    axs2[1][0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axs2[1][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=axs2[1][1].transAxes, linewidth=0.5)  # switch to the bottom axes
    axs2[1][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs2[1][1].plot((-d, +d), (-d, +d), **kwargs)

    for point, color, norm_profile_ca, norm_stokes_I_ca, amedprofca in zip(points, colors, norm_profiles_ca, norm_stokes_I_ca_list, medprofca_list):
        print ('{} - {}'.format(point, color))
        axs1[0][0].plot(fcaha['wav'][ind[0:306]][ind_ca_1], norm_stokes_I_ca[ind_ca_1], color=color, linewidth=linewidth, linestyle='-')
        axs1[0][1].plot(fcaha['wav'][ind[0:306]][ind_ca_2], norm_stokes_I_ca[ind_ca_2], color=color, linewidth=linewidth, linestyle='-')
        axs1[0][2].plot(fcaha['wav'][ind[0:306]][ind_ca_3], norm_stokes_I_ca[ind_ca_3], color=color, linewidth=linewidth, linestyle='-')
        axs1[1][0].plot(fcaha['wav'][ind[0:306]][ind_ca_1], norm_profile_ca[ind_ca_1], color=color, linewidth=linewidth, linestyle='-')
        axs1[1][1].plot(fcaha['wav'][ind[0:306]][ind_ca_2], norm_profile_ca[ind_ca_2], color=color, linewidth=linewidth, linestyle='-')
        axs1[1][2].plot(fcaha['wav'][ind[0:306]][ind_ca_3], norm_profile_ca[ind_ca_3], color=color, linewidth=linewidth, linestyle='-')
    axs1[0][0].plot(fcaha['wav'][ind[0:306]][ind_ca_1], amedprofca[ind_ca_1], color='grey', linewidth=linewidth, linestyle='--')
    axs1[0][1].plot(fcaha['wav'][ind[0:306]][ind_ca_2], amedprofca[ind_ca_2], color='grey', linewidth=linewidth, linestyle='--')
    axs1[0][2].plot(fcaha['wav'][ind[0:306]][ind_ca_3], amedprofca[ind_ca_3], color='grey', linewidth=linewidth, linestyle='--')

    for point, color, norm_profile_ha, norm_stokes_I_ha, amedprofha in zip(points, colors, norm_profiles_ha, norm_stokes_I_ha_list, medprofha_list):
        axs2[0][0].plot(fcaha['wav'][ind[306:]][ind_ha_1], norm_stokes_I_ha[ind_ha_1], color=color, linewidth=linewidth, linestyle='-')
        axs2[0][1].plot(fcaha['wav'][ind[306:]][ind_ha_2], norm_stokes_I_ha[ind_ha_2], color=color, linewidth=linewidth, linestyle='-')
        axs2[1][0].plot(fcaha['wav'][ind[306:]][ind_ha_1], norm_profile_ha[ind_ha_1], color=color, linewidth=linewidth, linestyle='-')
        axs2[1][1].plot(fcaha['wav'][ind[306:]][ind_ha_2], norm_profile_ha[ind_ha_2], color=color, linewidth=linewidth, linestyle='-')
    axs2[0][0].plot(fcaha['wav'][ind[306:]][ind_ha_1], amedprofha[ind_ha_1], color='grey', linewidth=linewidth, linestyle='--')
    axs2[0][1].plot(fcaha['wav'][ind[306:]][ind_ha_2], amedprofha[ind_ha_2], color='grey', linewidth=linewidth, linestyle='--')

    for center in center_list:
        axs1[1][0].plot(fcaha['wav'][ind[0:306]][ind_ca_1], np.ones_like(ind_ca_1) * (center - 1), linewidth=0.5, linestyle='--', color='gray')
        axs1[1][1].plot(fcaha['wav'][ind[0:306]][ind_ca_2], np.ones_like(ind_ca_2) * (center - 1), linewidth=0.5, linestyle='--', color='gray')
        axs1[1][2].plot(fcaha['wav'][ind[0:306]][ind_ca_3], np.ones_like(ind_ca_3) * (center - 1), linewidth=0.5, linestyle='--', color='gray')
        axs2[1][0].plot(fcaha['wav'][ind[306:]][ind_ha_1], np.ones_like(ind_ha_1) * (center - 1), linewidth=0.5, linestyle='--', color='gray')
        axs2[1][1].plot(fcaha['wav'][ind[306:]][ind_ha_2], np.ones_like(ind_ha_2) * (center - 1), linewidth=0.5, linestyle='--', color='gray')

    yticks = list()
    yticklabels1 = list()
    yticklabels2 = list()
    yticklabels01 = list()

    for a_max_ha, a_max_ca, center in zip(max_ha, max_ca, center_list):
        yticks.append(center - 0.5)
        yticks.append(center + 0.5)
        yticklabels1.append(np.round(-0.5 * a_max_ca, 1))
        yticklabels1.append(np.round(0.5 * a_max_ca, 1))
        yticklabels2.append(np.round(-0.5 * a_max_ha, 1))
        yticklabels2.append(np.round(0.5 * a_max_ha, 1))
        yticklabels01.append(0.25)
        yticklabels01.append(0.5)
        yticklabels01.append(0.75)

    axs1[1][1].set_yticks([])
    axs1[1][2].set_yticks([])
    axs2[1][1].set_yticks([])
    axs1[1][0].set_yticks(yticks)
    axs2[1][0].set_yticks(yticks)
    axs1[1][0].set_yticklabels(yticklabels1, fontsize=fontsize)
    axs2[1][0].set_yticklabels(yticklabels2, fontsize=fontsize)

    axs1[0][0].set_ylabel(r'$I/I_{c}$', fontsize=fontsize)
    axs1[1][0].set_ylabel(r'$V/I$ [%]', fontsize=fontsize)

    axs1[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[0][2].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[1][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[1][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs1[1][2].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs2[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs2[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs2[1][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs2[1][1].xaxis.set_minor_locator(MultipleLocator(0.25))

    axs1[1][0].yaxis.set_minor_locator(MultipleLocator(0.5))
    axs2[1][0].yaxis.set_minor_locator(MultipleLocator(0.5))

    axs1[0][0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs2[0][0].yaxis.set_minor_locator(MultipleLocator(0.1))

    ylim.reverse()
    axs1[1][0].set_ylim(*ylim)
    axs1[1][1].set_ylim(*ylim)
    axs1[1][2].set_ylim(*ylim)

    axs2[1][0].set_ylim(*ylim)
    axs2[1][1].set_ylim(*ylim)

    axs1[0][0].set_ylim(0.15, 1)
    axs1[0][1].set_ylim(0.15, 1)
    axs1[0][2].set_ylim(0.15, 1)
    axs2[0][0].set_ylim(0.15, 1)
    axs2[0][1].set_ylim(0.15, 1)

    axs1[0][1].set_yticks([])
    axs1[0][2].set_yticks([])
    axs2[0][0].set_yticklabels([])
    axs2[0][1].set_yticks([])

    axs1[0][0].tick_params(labelsize=fontsize)
    axs1[0][1].tick_params(labelsize=fontsize)
    axs1[0][2].tick_params(labelsize=fontsize)
    axs2[0][0].tick_params(labelsize=fontsize)
    axs2[0][1].tick_params(labelsize=fontsize)
    axs1[1][0].tick_params(labelsize=fontsize)
    axs1[1][1].tick_params(labelsize=fontsize)
    axs1[1][2].tick_params(labelsize=fontsize)
    axs2[1][0].tick_params(labelsize=fontsize)
    axs2[1][1].tick_params(labelsize=fontsize)

    axs1[0][0].set_xticks([8536])
    axs1[0][0].set_xticklabels([], fontsize=fontsize)

    axs1[1][0].set_xticks([8536])
    axs1[1][0].set_xticklabels([8536], fontsize=fontsize)

    axs1[0][1].set_xticks([8538])
    axs1[0][1].set_xticklabels([], fontsize=fontsize)

    axs1[1][1].set_xticks([8538])
    axs1[1][1].set_xticklabels([8538], fontsize=fontsize)

    axs1[0][2].set_xticks([8542])
    axs1[0][2].set_xticklabels([], fontsize=fontsize)

    axs1[1][2].set_xticks([8542])
    axs1[1][2].set_xticklabels([8542], fontsize=fontsize)

    axs2[0][0].set_xticks([6562, 6563, 6564])
    axs2[0][0].set_xticklabels([], fontsize=fontsize)

    axs2[1][0].set_xticks([6562, 6563, 6564])
    axs2[1][0].set_xticklabels([6562, 6563, 6564], fontsize=fontsize)

    axs2[0][1].set_xticks([6569])
    axs2[0][1].set_xticklabels([], fontsize=fontsize)

    axs2[1][1].set_xticks([6569])
    axs2[1][1].set_xticklabels([6569], fontsize=fontsize)

    axs1[0][2].text(
        0.93, 1.07,
        'Intensity [Normalised]',
        transform=axs1[0][2].transAxes,
        fontsize=fontsize
    )
    axs1[1][2].text(
        1.3, 1.07,
        r'$V/I$ [%]',
        transform=axs1[1][2].transAxes,
        fontsize=fontsize
    )

    axs1[1][0].text(
        1.55, -0.2,
        r'Wavelength [$\mathrm{\AA}$]',
        transform=axs1[1][0].transAxes,
        fontsize=fontsize
    )
    axs2[1][0].text(
        0.55, -0.2,
        r'Wavelength [$\mathrm{\AA}$]',
        transform=axs2[1][0].transAxes,
        fontsize=fontsize
    )

    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'SpatialVariationProfiles_{}_{}.pdf'.format(cut_indice, '_'.join(colors)), format='pdf', dpi=300)

    fig.savefig(write_path / 'SpatialVariationProfiles_{}_{}.png'.format(cut_indice, '_'.join(colors)), format='png', dpi=300)

    fcaha.close()

    plt.close('all')

    plt.clf()

    plt.cla()


def make_output_param_plots(points, colors_scatter):

    fontsize = 8

    size = plt.rcParams['lines.markersize']

    _, _, _, _, _, mask = get_fov_data()

    interesting_ltaus = [-4.5, -3, -1]

    ltau_indice = list()

    for interesting_ltau in interesting_ltaus:
        ltau_indice.append(np.argmin(np.abs(ltau500 - interesting_ltau)))

    ltau_indice = np.array(ltau_indice)

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output.nc', 'r')

    fig, axs = plt.subplots(3, 3, figsize=(3.5, 7))

    ind = np.where((ltau500 >= -1) & (ltau500 <= 0))[0]

    a, b = np.where(mask[0, 0] == 1)

    print(a)
    print(b)
    calib_vlos = np.mean(f['vlos'][()][0, a, b][:, ind])

    print(calib_vlos / 1e5)

    X, Y = np.meshgrid(np.arange(0, 17 * 0.38, 0.38), np.arange(0, 60 * 0.38, 0.38))

    im00 = axs[0][0].pcolormesh(X, Y, f['temp'][0, 0:17, :, ltau_indice[0]].T / 1e3, cmap='hot', shading='nearest', linewidth=0, rasterized=True)
    im01 = axs[0][1].pcolormesh(X, Y, f['temp'][0, 0:17, :, ltau_indice[1]].T / 1e3, cmap='hot', shading='nearest', linewidth=0, rasterized=True)
    im02 = axs[0][2].pcolormesh(X, Y, f['temp'][0, 0:17, :, ltau_indice[2]].T / 1e3, cmap='hot', shading='nearest', linewidth=0, rasterized=True)

    im10 = axs[1][0].pcolormesh(X, Y, (f['vlos'][0, 0:17, :, ltau_indice[0]].T - calib_vlos) / 1e5, cmap='bwr', vmin=-5, vmax=5, shading='nearest', linewidth=0, rasterized=True)
    im11 = axs[1][1].pcolormesh(X, Y, (f['vlos'][0, 0:17, :, ltau_indice[1]].T - calib_vlos) / 1e5, cmap='bwr', vmin=-5, vmax=5, shading='nearest', linewidth=0, rasterized=True)
    im12 = axs[1][2].pcolormesh(X, Y, (f['vlos'][0, 0:17, :, ltau_indice[2]].T - calib_vlos) / 1e5, cmap='bwr', vmin=-5, vmax=5, shading='nearest', linewidth=0, rasterized=True)

    im20 = axs[2][0].pcolormesh(X, Y, f['vturb'][0, 0:17, :, ltau_indice[0]].T / 1e5, cmap='copper', vmin=0, vmax=5, shading='nearest', linewidth=0, rasterized=True)
    im21 = axs[2][1].pcolormesh(X, Y, f['vturb'][0, 0:17, :, ltau_indice[1]].T / 1e5, cmap='copper', vmin=0, vmax=5, shading='nearest', linewidth=0, rasterized=True)
    im22 = axs[2][2].pcolormesh(X, Y, f['vturb'][0, 0:17, :, ltau_indice[2]].T / 1e5, cmap='copper', vmin=0, vmax=5, shading='nearest', linewidth=0, rasterized=True)

    im00.set_edgecolor('face')
    im01.set_edgecolor('face')
    im02.set_edgecolor('face')

    im10.set_edgecolor('face')
    im11.set_edgecolor('face')
    im12.set_edgecolor('face')

    im20.set_edgecolor('face')
    im21.set_edgecolor('face')
    im22.set_edgecolor('face')

    cbaxes = inset_axes(
        axs[0][0],
        width="80%",
        height="4%",
        loc='upper center',
        borderpad=-1.2
    )
    cbar = fig.colorbar(
        im00,
        cax=cbaxes,
        # ticks=[all_vmin[j], (all_vmin[j] + all_vmax[j]) // 2, all_vmax[j]],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][1],
        width="80%",
        height="4%",
        loc='upper center',
        borderpad=-1.2
    )
    cbar = fig.colorbar(
        im01,
        cax=cbaxes,
        # ticks=[all_vmin[j], (all_vmin[j] + all_vmax[j]) // 2, all_vmax[j]],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][2],
        width="80%",
        height="4%",
        loc='upper center',
        borderpad=-1.2
    )
    cbar = fig.colorbar(
        im02,
        cax=cbaxes,
        ticks=[5, 5.4],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][2],
        width="7%",
        height="80%",
        loc='right',
        borderpad=-1.2
    )
    cbar = fig.colorbar(
        im12,
        cax=cbaxes,
        ticks=[-5, 0, 5],
        # orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[2][2],
        width="7%",
        height="80%",
        loc='right',
        borderpad=-1.2
    )
    cbar = fig.colorbar(
        im22,
        cax=cbaxes,
        ticks=[0, 3, 5],
        # orientation='vertical'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    for i in range(3):
        for j in range(3):
            color = 'black'
            if i == 2:
                color = 'white'
            axs[i][j].contour(X, Y, mask[0, 0].T, levels=0, colors=color, linewidths=0.5)
            axs[i][j].set_yticks([0, 5, 10, 15, 20])
            axs[i][j].set_xticks([0, 5])
            axs[i][j].set_yticklabels([])
            axs[i][j].set_xticklabels([])
            axs[i][j].axvline(12 * 0.38, linestyle='--', color='brown', linewidth=0.5)
            axs[i][j].axvline(8 * 0.38, linestyle='--', color='darkgreen', linewidth=0.5)
            for point, color in zip(points, colors_scatter):
                axs[i][j].scatter(point[0] * 0.38, point[1] * 0.38, color=color, marker='x', s=size**2, linewidths=1)

    for i in range(3):
        axs[2][i].set_xticklabels([0, 5], fontsize=fontsize)
        axs[i][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].yaxis.set_minor_locator(MultipleLocator(1))

    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].yaxis.set_minor_locator(MultipleLocator(1))

    axs[2][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[2][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[2][2].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_ylabel('slit position [arcsec]', fontsize=fontsize)
    axs[1][0].set_ylabel('slit position [arcsec]', fontsize=fontsize)
    axs[2][0].set_ylabel('slit position [arcsec]', fontsize=fontsize)

    axs[2][1].set_xlabel('scan direction [arcsec]', fontsize=fontsize)

    axs[0][0].text(
        -0.05, 1.25,
        r'$\log \tau_{\mathrm{500}}=-4.5$',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.1, 1.25,
        r'$\log \tau_{\mathrm{500}}=-3$',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[0][2].text(
        0.1, 1.25,
        r'$\log \tau_{\mathrm{500}}=-1$',
        transform=axs[0][2].transAxes,
        fontsize=fontsize
    )

    plt.subplots_adjust(left=0.12, bottom=0.06, right=0.85, top=0.9, wspace=0.0, hspace=0.0)

    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'InversionResults.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()

    f.close()


def get_wfa():

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    ha_center_wave = 6562.8 / 10
    wave_range = 0.9 / 10

    transition_skip_list = np.array(
        [
            [6560.57, 0.25],
            [6561.09, 0.1],
            [6562.44, 0.05],
            [6563.51, 0.15],
            [6564.15, 0.35]
        ]
    ) / 10


    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave - wave_range,
        ha_center_wave + wave_range,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    fcaha.close()

    return magha


def get_wfa_new():

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    ha_center_wave = 6562.8 / 10

    transition_skip_list = np.array(
        [
            [6560.57, 0.25],
            [6561.09, 0.1],
            [6562.44, 0.05],
            [6563.51, 0.15],
            [6564.15, 0.35]
        ]
    ) / 10

    wave_range = 0.3 / 10

    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave - wave_range,
        ha_center_wave,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha_left = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave,
        ha_center_wave + wave_range,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha_right = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    magha = np.add(magha_left, magha_right) / 2

    wave_range_1 = 0.9 / 10

    wave_range_2 = 0.5 / 10

    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave - wave_range_1,
        ha_center_wave - wave_range_2,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha_left = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave + wave_range_2,
        ha_center_wave + wave_range_1,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha_right = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    magha_p = np.add(magha_left, magha_right) / 2

    fcaha.close()


def get_wfanew_alternate():

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    ha_center_wave = 6562.8 / 10
    wave_range = 0.35 / 10

    transition_skip_list = np.array(
        [
            [6560.57, 0.25],
            [6561.09, 0.1],
            [6562.44, 0.05],
            [6563.51, 0.15],
            [6564.15, 0.35]
        ]
    ) / 10


    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave - wave_range,
        ha_center_wave + wave_range,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    wave_range = 1.5 / 10

    transition_skip_list = np.array(
        [
            [6560.57, 0.25],
            [6561.09, 0.1],
            [6562.44, 0.05],
            [6563.51, 0.15],
            [6564.15, 0.35],
            [6562.8, 0.6]
        ]
    ) / 10


    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave - wave_range,
        ha_center_wave + wave_range,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha_p = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    transition_skip_list = np.array(
        [
            [6560.57, 0.25],
            [6561.09, 0.1],
            [6562.44, 0.05],
            [6563.51, 0.15],
            [6564.15, 0.35]
        ]
    ) / 10

    actual_calculate_blos = prepare_calculate_blos(
        fcaha['profiles'][:, :, :, ind[306:]],
        fcaha['wav'][ind[306:]] / 10,
        ha_center_wave,
        ha_center_wave - wave_range,
        ha_center_wave + wave_range,
        1.048,
        transition_skip_list=transition_skip_list
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha_full_line = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    fcaha.close()

    # print (magha.min(), magha.max())
    # print (magha_p.min(), magha_p.max())
    # print (magha_full_line.min(), magha_full_line.max())

    return magha, magha_p, magha_full_line


def plot_mag_field_compare():

    fontsize = 8

    magha = get_wfa().T

    interesting_ltaus = [0, -2, -3, -5]

    ltau_indice = list()

    for interesting_ltau in interesting_ltaus:
        ltau_indice.append(np.argmin(np.abs(ltau500 - interesting_ltau)))

    ltau_indice = np.array(ltau_indice)

    _, _, _, _, _, mask = get_fov_data()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output.nc', 'r')

    fig, axs = plt.subplots(2, 4, figsize=(7, 7))

    a0 = f['blong'][0, 0:17, :, ltau_indice[0]].T
    a1 = f['blong'][0, 0:17, :, ltau_indice[1]].T
    a2 = f['blong'][0, 0:17, :, ltau_indice[2]].T
    a3 = f['blong'][0, 0:17, :, ltau_indice[3]].T

    X, Y = np.meshgrid(np.arange(0, 17 * 0.38, 0.38), np.arange(0, 60 * 0.38, 0.38))

    im00 = axs[0][0].pcolormesh(X, Y, a1, cmap='RdGy', shading='gouraud', vmin=-1000, vmax=1000)

    im01 = axs[0][1].pcolormesh(X, Y, a2, cmap='RdGy', shading='gouraud', vmin=-900, vmax=900)

    im02 = axs[0][2].pcolormesh(X, Y, a3, cmap='RdGy', shading='gouraud', vmin=-600, vmax=600)

    im03 = axs[0][3].pcolormesh(X, Y, magha, cmap='RdGy', shading='gouraud', vmin=-500, vmax=500)

    im10 = axs[1][0].pcolormesh(X, Y, np.abs(a1) - np.abs(a0), cmap='bwr', shading='gouraud', vmin=-300, vmax=300)

    im11 = axs[1][1].pcolormesh(X, Y, np.abs(a2) - np.abs(a1), cmap='bwr', shading='gouraud', vmin=-400, vmax=400)

    im12 = axs[1][2].pcolormesh(X, Y, np.abs(a3) - np.abs(a2), cmap='bwr', shading='gouraud', vmin=-500, vmax=500)

    im13 = axs[1][3].pcolormesh(X, Y, np.abs(magha) - np.abs(a3), cmap='bwr', shading='gouraud', vmin=-300, vmax=300)

    for i in range(2):
        for j in range(4):
            color = 'black'
            axs[i][j].contour(X, Y, mask[0, 0].T, levels=0, colors=color, linewidths=0.5)

    cbaxes = inset_axes(
        axs[0][0],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im00,
        cax=cbaxes,
        ticks=[-900, 0, 900],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][1],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im01,
        cax=cbaxes,
        ticks=[-800, 0, 800],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][2],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im02,
        cax=cbaxes,
        ticks=[-500, 0, 500],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][3],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im03,
        cax=cbaxes,
        ticks=[-400, 0, 400],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][0],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im10,
        cax=cbaxes,
        ticks=[-200, 0, 200],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][1],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im11,
        cax=cbaxes,
        ticks=[-300, 0, 300],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][2],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im12,
        cax=cbaxes,
        ticks=[-400, 0, 400],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][3],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im13,
        cax=cbaxes,
        ticks=[-200, 0, 200],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[0][2].set_xticklabels([])
    axs[0][2].set_yticklabels([])
    axs[0][3].set_xticklabels([])
    axs[0][3].set_yticklabels([])

    axs[1][1].set_yticklabels([])
    axs[1][2].set_yticklabels([])
    axs[1][3].set_yticklabels([])

    axs[0][0].set_yticks([0, 5, 10, 15, 20])
    axs[0][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs[0][1].set_yticks([0, 5, 10, 15, 20])
    axs[0][2].set_yticks([0, 5, 10, 15, 20])
    axs[0][3].set_yticks([0, 5, 10, 15, 20])
    axs[1][0].set_yticks([0, 5, 10, 15, 20])
    axs[1][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs[1][1].set_yticks([0, 5, 10, 15, 20])
    axs[1][2].set_yticks([0, 5, 10, 15, 20])
    axs[1][3].set_yticks([0, 5, 10, 15, 20])

    axs[0][0].set_xticks([0, 5])
    axs[0][1].set_xticks([0, 5])
    axs[0][2].set_xticks([0, 5])
    axs[0][3].set_xticks([0, 5])

    axs[1][0].set_xticks([0, 5])
    axs[1][0].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][1].set_xticks([0, 5])
    axs[1][1].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][2].set_xticks([0, 5])
    axs[1][2].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][3].set_xticks([0, 5])
    axs[1][3].set_xticklabels([0, 5], fontsize=fontsize)

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][3].yaxis.set_minor_locator(MultipleLocator(1))

    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][3].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_ylabel('x [arcsec]', fontsize=fontsize)
    axs[1][0].set_ylabel('x [arcsec]', fontsize=fontsize)

    axs[1][0].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][1].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][2].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][3].set_xlabel('y [arcsec]', fontsize=fontsize)

    axs[0][0].text(
        0.05, 1.2,
        r'$\log \tau_{\mathrm{500}}=-2$',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.05, 1.2,
        r'$\log \tau_{\mathrm{500}}=-3$',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[0][2].text(
        0.05, 1.2,
        r'$\log \tau_{\mathrm{500}}=-5$',
        transform=axs[0][2].transAxes,
        fontsize=fontsize
    )

    axs[0][3].text(
        0.3, 1.2,
        r'WFA (H$\alpha$)',
        transform=axs[0][3].transAxes,
        fontsize=fontsize
    )

    plt.subplots_adjust(left=0.1, bottom=0.07, right=0.96, top=0.9, wspace=0.2, hspace=0.2)

    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'MagneticField.pdf', format='pdf', dpi=300)

    f.close()


def plot_mag_field_compare_new(points, colors_scatter):

    fontsize = 8

    a, b, c = get_wfanew_alternate()

    magha, magha_p, magha_full_line = a.T, b.T, c.T

    interesting_ltaus = [0, -2, -3, -4.5]

    ltau_indice = list()

    for interesting_ltau in interesting_ltaus:
        ltau_indice.append(np.argmin(np.abs(ltau500 - interesting_ltau)))

    ltau_indice = np.array(ltau_indice)

    _, _, _, _, _, mask = get_fov_data()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output.nc', 'r')

    fig, axs = plt.subplots(2, 6, figsize=(7, 7))

    a0 = f['blong'][0, 0:17, :, ltau_indice[0]].T
    a1 = f['blong'][0, 0:17, :, ltau_indice[1]].T
    a2 = f['blong'][0, 0:17, :, ltau_indice[2]].T
    a3 = f['blong'][0, 0:17, :, ltau_indice[3]].T

    colors = ["darkred", "darkgoldenrod", "white", "green", "blue"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    cmap = cmap1  #RdGy

    X, Y = np.meshgrid(np.arange(0, 17 * 0.38, 0.38), np.arange(0, 60 * 0.38, 0.38))

    vlim = np.zeros((2, 6, 2), dtype=np.int64)
    temp = np.zeros((12, 60, 17), dtype=np.float64)

    temp[0] = a1
    temp[1] = a2
    temp[2] = a3
    temp[3] = magha
    temp[4] = magha_p
    temp[5] = magha_full_line
    temp[6] = np.abs(a1) - np.abs(a0)
    temp[7] = np.abs(a2) - np.abs(a1)
    temp[8] = np.abs(a3) - np.abs(a2)
    temp[9] = np.abs(magha) - np.abs(a3)
    temp[10] = np.abs(magha_p) - np.abs(a1)
    temp[11] = np.abs(magha_full_line) - np.abs(a1)

    k = 0
    for ii in range(2):
        for jj in range(6):
            maxval = np.abs(temp[k]).max()
            limval = (maxval // 100) * 100 + 100
            vlim[ii][jj][0] = -limval
            vlim[ii][jj][1] = limval
            print(vlim[ii][jj])
            k += 1
    
    im00 = axs[0][0].pcolormesh(X, Y, a1, cmap=cmap, shading='nearest', linewidth=0, rasterized=True, vmin=vlim[0][0][0], vmax=vlim[0][0][1])

    im01 = axs[0][1].pcolormesh(X, Y, a2, cmap=cmap, shading='nearest', linewidth=0, rasterized=True, vmin=vlim[0][1][0], vmax=vlim[0][1][1])

    im02 = axs[0][2].pcolormesh(X, Y, a3, cmap=cmap, shading='nearest', linewidth=0, rasterized=True, vmin=vlim[0][2][0], vmax=vlim[0][2][1])

    im03 = axs[0][3].pcolormesh(X, Y, magha, cmap=cmap, shading='nearest', linewidth=0, rasterized=True, vmin=vlim[0][3][0], vmax=vlim[0][3][1])

    im04 = axs[0][4].pcolormesh(X, Y, magha_p, cmap=cmap, shading='nearest', linewidth=0, rasterized=True, vmin=vlim[0][4][0], vmax=vlim[0][4][1])

    im05 = axs[0][5].pcolormesh(X, Y, magha_full_line, cmap=cmap, shading='nearest', linewidth=0, rasterized=True, vmin=vlim[0][5][0], vmax=vlim[0][5][1])

    im10 = axs[1][0].pcolormesh(X, Y, np.abs(a1) - np.abs(a0), cmap='bwr', shading='nearest', linewidth=0, rasterized=True, vmin=vlim[1][0][0], vmax=vlim[1][0][1])

    im11 = axs[1][1].pcolormesh(X, Y, np.abs(a2) - np.abs(a1), cmap='bwr', shading='nearest', linewidth=0, rasterized=True, vmin=vlim[1][1][0], vmax=vlim[1][1][1])

    im12 = axs[1][2].pcolormesh(X, Y, np.abs(a3) - np.abs(a2), cmap='bwr', shading='nearest', linewidth=0, rasterized=True, vmin=vlim[1][2][0], vmax=vlim[1][2][1])

    im13 = axs[1][3].pcolormesh(X, Y, np.abs(magha) - np.abs(a3), cmap='bwr', shading='nearest', linewidth=0, rasterized=True, vmin=vlim[1][3][0], vmax=vlim[1][3][1])

    im14 = axs[1][4].pcolormesh(X, Y, np.abs(magha_p) - np.abs(a1), cmap='bwr', shading='nearest', linewidth=0, rasterized=True, vmin=vlim[1][4][0], vmax=vlim[1][4][1])

    im15 = axs[1][5].pcolormesh(X, Y, np.abs(magha_full_line) - np.abs(a1), cmap='bwr', shading='nearest', linewidth=0, rasterized=True, vmin=vlim[1][5][0], vmax=vlim[1][5][1])

    im00.set_edgecolor('face')
    im01.set_edgecolor('face')
    im02.set_edgecolor('face')
    im03.set_edgecolor('face')
    im04.set_edgecolor('face')
    im05.set_edgecolor('face')

    im10.set_edgecolor('face')
    im11.set_edgecolor('face')
    im12.set_edgecolor('face')
    im13.set_edgecolor('face')
    im14.set_edgecolor('face')
    im15.set_edgecolor('face')

    for i in range(2):
        for j in range(6):
            color = 'black'
            axs[i][j].contour(X, Y, mask[0, 0].T, levels=0, colors=color, linewidths=0.5)
            axs[i][j].axvline(12 * 0.38, linestyle='--', color='brown', linewidth=0.5)
            axs[i][j].axvline(8 * 0.38, linestyle='--', color='darkgreen', linewidth=0.5)
            for point, color in zip(points, colors_scatter):
                axs[i][j].scatter(point[0] * 0.38, point[1] * 0.38, color=color, marker='x')

    cbaxes = inset_axes(
        axs[0][0],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im00,
        cax=cbaxes,
        ticks=[vlim[0][0][0]+100, 0, vlim[0][0][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][1],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im01,
        cax=cbaxes,
        ticks=[vlim[0][1][0]+100, 0, vlim[0][1][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][2],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im02,
        cax=cbaxes,
        ticks=[vlim[0][2][0]+100, 0, vlim[0][2][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][3],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im03,
        cax=cbaxes,
        ticks=[vlim[0][3][0]+100, 0, vlim[0][3][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][4],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im04,
        cax=cbaxes,
        ticks=[vlim[0][4][0]+100, 0, vlim[0][4][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[0][5],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im05,
        cax=cbaxes,
        ticks=[vlim[0][5][0]+100, 0, vlim[0][5][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][0],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im10,
        cax=cbaxes,
        ticks=[vlim[1][0][0]+100, 0, vlim[1][0][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][1],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im11,
        cax=cbaxes,
        ticks=[vlim[1][1][0]+100, 0, vlim[1][1][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][2],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im12,
        cax=cbaxes,
        ticks=[vlim[1][2][0]+100, 0, vlim[1][2][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][3],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im13,
        cax=cbaxes,
        ticks=[vlim[1][3][0]+100, 0, vlim[1][3][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][4],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im14,
        cax=cbaxes,
        ticks=[vlim[1][4][0]+100, 0, vlim[1][4][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    cbaxes = inset_axes(
        axs[1][5],
        width="100%",
        height="5%",
        loc='upper center',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im15,
        cax=cbaxes,
        ticks=[vlim[1][5][0]+100, 0, vlim[1][5][1]-100],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[0][2].set_xticklabels([])
    axs[0][2].set_yticklabels([])
    axs[0][3].set_xticklabels([])
    axs[0][3].set_yticklabels([])
    axs[0][4].set_xticklabels([])
    axs[0][4].set_yticklabels([])
    axs[0][5].set_xticklabels([])
    axs[0][5].set_yticklabels([])

    axs[1][1].set_yticklabels([])
    axs[1][2].set_yticklabels([])
    axs[1][3].set_yticklabels([])
    axs[1][4].set_yticklabels([])
    axs[1][5].set_yticklabels([])

    axs[0][0].set_yticks([0, 5, 10, 15, 20])
    axs[0][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs[0][1].set_yticks([0, 5, 10, 15, 20])
    axs[0][2].set_yticks([0, 5, 10, 15, 20])
    axs[0][3].set_yticks([0, 5, 10, 15, 20])
    axs[0][4].set_yticks([0, 5, 10, 15, 20])
    axs[0][5].set_yticks([0, 5, 10, 15, 20])
    axs[1][0].set_yticks([0, 5, 10, 15, 20])
    axs[1][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs[1][1].set_yticks([0, 5, 10, 15, 20])
    axs[1][2].set_yticks([0, 5, 10, 15, 20])
    axs[1][3].set_yticks([0, 5, 10, 15, 20])
    axs[1][4].set_yticks([0, 5, 10, 15, 20])
    axs[1][5].set_yticks([0, 5, 10, 15, 20])

    axs[0][0].set_xticks([0, 5])
    axs[0][1].set_xticks([0, 5])
    axs[0][2].set_xticks([0, 5])
    axs[0][3].set_xticks([0, 5])
    axs[0][4].set_xticks([0, 5])
    axs[0][5].set_xticks([0, 5])

    axs[1][0].set_xticks([0, 5])
    axs[1][0].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][1].set_xticks([0, 5])
    axs[1][1].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][2].set_xticks([0, 5])
    axs[1][2].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][3].set_xticks([0, 5])
    axs[1][3].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][4].set_xticks([0, 5])
    axs[1][4].set_xticklabels([0, 5], fontsize=fontsize)
    axs[1][5].set_xticks([0, 5])
    axs[1][5].set_xticklabels([0, 5], fontsize=fontsize)

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][4].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][5].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][3].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][4].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][5].yaxis.set_minor_locator(MultipleLocator(1))

    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][4].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][5].xaxis.set_minor_locator(MultipleLocator(1))

    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][3].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][4].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][5].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_ylabel('x [arcsec]', fontsize=fontsize)
    axs[1][0].set_ylabel('x [arcsec]', fontsize=fontsize)

    axs[1][0].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][1].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][2].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][3].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][4].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][5].set_xlabel('y [arcsec]', fontsize=fontsize)


    axs[0][0].text(
        0.05, 1.2,
        r'$\log \tau_{\mathrm{500}}=-2$',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][0].text(
        0.05, 0.95,
        r'(a)',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.05, 1.2,
        r'$\log \tau_{\mathrm{500}}=-3$',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.05, 0.95,
        r'(b)',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[0][2].text(
        0.05, 1.2,
        r'$\log \tau_{\mathrm{500}}=-4.5$',
        transform=axs[0][2].transAxes,
        fontsize=fontsize
    )
    axs[0][2].text(
        0.05, 0.95,
        r'(c)',
        transform=axs[0][2].transAxes,
        fontsize=fontsize
    )
    axs[0][3].text(
        0.03, 1.2,
        r'WFA (H$\alpha$ core)',
        transform=axs[0][3].transAxes,
        fontsize=fontsize
    )
    axs[0][3].text(
        0.05, 0.95,
        r'(d)',
        transform=axs[0][3].transAxes,
        fontsize=fontsize
    )
    axs[0][4].text(
        0.03, 1.2,
        r'WFA (H$\alpha$ wing)',
        transform=axs[0][4].transAxes,
        fontsize=fontsize
    )
    axs[0][4].text(
        0.05, 0.95,
        r'(e)',
        transform=axs[0][4].transAxes,
        fontsize=fontsize
    )
    axs[0][5].text(
        0.03, 1.2,
        r'WFA (H$\alpha\pm1.5\AA$)',
        transform=axs[0][5].transAxes,
        fontsize=fontsize
    )
    axs[0][5].text(
        0.05, 0.95,
        r'(f)',
        transform=axs[0][5].transAxes,
        fontsize=fontsize
    )

    axs[1][0].text(
        0.03, 0.95,
        r'|(a)|$-$|$B_{\log\tau_{\mathrm{500} = 0}}$|',
        transform=axs[1][0].transAxes,
        fontsize=fontsize
    )
    axs[1][1].text(
        0.05, 0.95,
        r'|(b)|$-$|(a)|',
        transform=axs[1][1].transAxes,
        fontsize=fontsize
    )
    axs[1][2].text(
        0.05, 0.95,
        r'|(c)|$-$|(b)|',
        transform=axs[1][2].transAxes,
        fontsize=fontsize
    )
    axs[1][3].text(
        0.05, 0.95,
        r'|(d)|$-$|(c)|',
        transform=axs[1][3].transAxes,
        fontsize=fontsize
    )
    axs[1][4].text(
        0.05, 0.95,
        r'|(e)|$-$|(a)|',
        transform=axs[1][4].transAxes,
        fontsize=fontsize
    )
    axs[1][5].text(
        0.05, 0.95,
        r'|(f)|$-$|(a)|',
        transform=axs[1][5].transAxes,
        fontsize=fontsize
    )

    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.99, top=0.9, wspace=0.25, hspace=0.2)

    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'MagneticField.pdf', format='pdf', dpi=300)

    # plt.show()

    f.close()


def get_quality_of_fits_data(points):

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

    observed = np.zeros((len(points), 306, 2), dtype=np.float64)

    for index, point in enumerate(points):
        observed[index, :, 0] = fcaha['profiles'][0, point[0], point[1], ind[0:306], 0]
        observed[index, :, 1] = fcaha['profiles'][0, point[0], point[1], ind[0:306], 3] / fcaha['profiles'][0, point[0], point[1], ind[0:306], 0]

    fcaha.close()

    synthesized = np.zeros((len(points), 306, 2), dtype=np.float64)

    wave = None

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output_profs.nc', 'r')

    for index, point in enumerate(points):
        synthesized[index, :, 0] = f['profiles'][0, point[0], point[1], ind[0:306], 0]
        synthesized[index, :, 1] = f['profiles'][0, point[0], point[1], ind[0:306], 3] / f['profiles'][0, point[0], point[1], ind[0:306], 0]

    wave = f['wav'][ind[0:306]]

    f.close()

    return wave, observed, synthesized


def make_quality_of_fits(points, colors):

    wave, observed, synthesized = get_quality_of_fits_data(points)

    size = plt.rcParams['lines.markersize']

    fontsize = 6

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(4, 5, figsize=(7, 5))

    k = 0

    for i in [0, 2]:
        for j in range(5):
            axs[i][j].plot(wave, observed[k + j, :, 0], linestyle='dotted', linewidth=0.5, color=colors[k + j]) # s=size/8
            axs[i][j].plot(wave, synthesized[k + j, :, 0], linewidth=0.25, linestyle='-', color=colors[k + j])
            axs[i + 1][j].plot(wave, observed[k + j, :, 1] * 100, linestyle='dotted', linewidth=0.5, color=colors[k + j]) # s=size/8
            axs[i + 1][j].plot(wave, synthesized[k + j, :, 1] * 100, linewidth=0.25, linestyle='-', color=colors[k + j])
            axs[i][j].set_xticks([8536, 8538, 8540, 8542, 8544])
            axs[i + 1][j].set_xticks([8536, 8538, 8540, 8542, 8544])
            axs[i][j].set_xticklabels([])
            axs[i + 1][j].set_xticklabels([])
            axs[i][j].set_ylim(0.1, 1)
            axs[i][j].set_yticks([0.2, 0.4, 0.6, 0.8])
            axs[i][j].set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
            ymax = 1.1 * 100 * max(np.abs(observed[k + j, :, 1]).max(), np.abs(synthesized[k + j, :, 1]).max())
            axs[i + 1][j].set_ylim(-ymax, ymax)
            yticks = [-np.round(ymax * 2 / 3, 1), -np.round(ymax * 1 / 3, 1), 0, np.round(ymax * 1 / 3, 1), np.round(ymax * 2 / 3, 1)]
            axs[i + 1][j].set_yticks(yticks)
            axs[i + 1][j].set_yticklabels(yticks, fontsize=fontsize)
        k += 5

    axs[3][0].set_xticklabels(np.round(np.array([8536, 8538, 8540, 8542, 8544]) - 8530, 0), fontsize=fontsize)
    axs[3][1].set_xticklabels(np.round(np.array([8536, 8538, 8540, 8542, 8544]) - 8530, 0), fontsize=fontsize)
    axs[3][2].set_xticklabels(np.round(np.array([8536, 8538, 8540, 8542, 8544]) - 8530, 0), fontsize=fontsize)
    axs[3][3].set_xticklabels(np.round(np.array([8536, 8538, 8540, 8542, 8544]) - 8530, 0), fontsize=fontsize)
    axs[3][4].set_xticklabels(np.round(np.array([8536, 8538, 8540, 8542, 8544]) - 8530, 0), fontsize=fontsize)
    axs[3][0].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 8530) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][1].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 8530)[$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][2].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 8530) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][3].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 8530) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][4].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 8530) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[0][0].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
    axs[2][0].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
    axs[1][0].set_ylabel(r'$V/I$ [%]', fontsize=fontsize)
    axs[3][0].set_ylabel(r'$V/I$ [%]', fontsize=fontsize)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.995, top=0.995, wspace=0.3, hspace=0.2)
    # plt.show()
    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'QualityofFits_new.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


def make_atmos_for_response_functions(points):
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output.nc', 'r')

    m = sp.model(nx=len(points), ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    for index, point in enumerate(points):
        m.temp[0, 0, index] = f['temp'][0, point[0], point[1]]

        m.vlos[0, 0, index] = f['vlos'][0, point[0], point[1]]

        m.vturb[0, 0, index] = f['vturb'][0, point[0], point[1]]

        m.Bln[0, 0, index] = f['blong'][0, point[0], point[1]]

    m.write(base_path / 'output_atmosphere_selected_points.nc')

    f.close()


def get_data_for_make_forward_synthesis_plots(forward_file, points):
    data = np.zeros((2, 2, 17, 60), dtype=np.float64)
    observed = np.zeros((len(points), 444, 2), dtype=np.float64)
    synthesized = np.zeros((len(points), 444, 2), dtype=np.float64)

    fcaha = h5py.File(ca_ha_data_file, 'r')
    fforward = h5py.File(forward_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

    ind_6563_core = ind[np.argmin(np.abs(fcaha['wav'][ind] - 6562.8))]

    ind_6563_wing = ind[np.argmin(np.abs(fcaha['wav'][ind] - 6569.826))]

    data[0][0] = fcaha['profiles'][0, 0:17, :, ind_6563_wing, 0]

    data[0][1] = fforward['profiles'][0, 0:17, :, ind_6563_wing - 1236, 0]

    data[1][0] = fcaha['profiles'][0, 0:17, :, ind_6563_core, 0]

    data[1][1] = fforward['profiles'][0, 0:17, :, ind_6563_core - 1236, 0]

    a, b, c, d = np.where(data > 2)
    data[a, b, c, d] = 0

    for index, point in enumerate(points):
        observed[index, :, 0] = fcaha['profiles'][0, point[0], point[1], ind[306:], 0]
        observed[index, :, 1] = fcaha['profiles'][0, point[0], point[1], ind[306:], 3] / fcaha['profiles'][0, point[0], point[1], ind[306:], 0]

    for index, point in enumerate(points):
        synthesized[index, :, 0] = fforward['profiles'][0, point[0], point[1], ind[306:] - 1236, 0]
        synthesized[index, :, 1] = fforward['profiles'][0, point[0], point[1], ind[306:] - 1236, 3] / fforward['profiles'][0, point[0], point[1], ind[306:] - 1236, 0]

    wave = fforward['wav'][ind[306:] - 1236]

    a, b, c = np.where(synthesized > 2)
    synthesized[a, b, c] = 0

    fforward.close()
    fcaha.close()

    return data, observed, synthesized, wave


def make_forward_synthesis_plots(forward_file, points, colors_scatter, name):
    data, observed, synthesized, wave = get_data_for_make_forward_synthesis_plots(forward_file, points)

    _, _, _, wing_ha, core_ha, mask = get_fov_data()

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(2, 2, figsize=(7, 2.4))

    extent = [0, 22.8, 0, 6.46]

    fontsize = 8

    for i in range(2):
        for j in range(2):
            axs[i][j].imshow(data[i][j], cmap='gray', origin='lower')
            axs[i][j].contour(mask[i][j], levels=0, origin='lower', colors='blue', linewidths=0.5)
            axs[i][j].plot(np.ones(60) * 12, linestyle='--', color='brown', linewidth=0.5)
            axs[i][j].plot(np.ones(60) * 8, linestyle='--', color='darkgreen', linewidth=0.5)
            for point, color in zip(points, colors_scatter):
                axs[i][j].scatter(point[1], point[0], color=color, marker='x')

    axs[0][0].text(
        0.02, 0.04,
        r'(a) H$\alpha$ +{} $\mathrm{{\AA}}$ (Observed)'.format(
            np.round(wing_ha - 6562.8), 2
        ),
        transform=axs[0][0].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[0][1].text(
        0.02, 0.04,
        r'(b) H$\alpha$ +{} $\mathrm{{\AA}}$ (Synthesized)'.format(
            np.round(wing_ha - 6562.8), 2
        ),
        transform=axs[0][1].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[1][0].text(
        0.02, 0.04,
        r'(c) H$\alpha$ core (Observed)',
        transform=axs[1][0].transAxes,
        color='white',
        fontsize=fontsize
    )
    axs[1][1].text(
        0.02, 0.04,
        r'(d) H$\alpha$ core (Synthesized)',
        transform=axs[1][1].transAxes,
        color='white',
        fontsize=fontsize
    )

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(17/6.46))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(17/6.46))

    axs[1][0].xaxis.set_minor_locator(MultipleLocator(60/22.8))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(60/22.8))

    axs[0][0].tick_params(direction='out', which='both', color='black')
    axs[1][0].tick_params(direction='out', which='both', color='black')
    axs[1][1].tick_params(direction='out', which='both', color='black')


    axs[0][0].set_xticks([])
    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticks([])
    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticks([])
    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticks([])
    axs[1][1].set_yticklabels([])
    axs[1][0].set_xticks([0, 10 * 60/22.8, 20 * 60/22.8])
    axs[1][0].set_xticklabels([0, 10, 20])
    axs[1][1].set_xticks([0, 10 * 60/22.8, 20 * 60/22.8])
    axs[1][1].set_xticklabels([0, 10, 20])
    axs[0][0].set_yticks([0, 2 * 17/6.46, 4 * 17/6.46])
    axs[0][0].set_yticklabels([0, 2, 4])
    axs[1][0].set_yticks([0, 2 * 17/6.46, 4 * 17/6.46])
    axs[1][0].set_yticklabels([0, 2, 4])

    axs[0][0].set_ylabel('y [arcsec]')
    axs[1][0].set_ylabel('y [arcsec]')

    axs[1][0].set_xlabel('x [arcsec]')
    axs[1][1].set_xlabel('x [arcsec]')

    plt.subplots_adjust(left=0.07, bottom=0.23, right=1, top=1, hspace=0.0, wspace=0.0)

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'Ha_synth_{}.pdf'.format(name), format='pdf', dpi=300)

    # plt.show()

    size = plt.rcParams['lines.markersize']

    fontsize = 6

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(4, 5, figsize=(7, 5))

    k = 0

    for i in [0, 2]:
        for j in range(5):
            axs[i][j].plot(wave, observed[k + j, :, 0], linestyle='dotted', linewidth=0.5,
                           color=colors[k + j])  # s=size/8
            axs[i][j].plot(wave, synthesized[k + j, :, 0], linewidth=0.25, linestyle='-', color=colors[k + j])
            axs[i + 1][j].plot(wave, observed[k + j, :, 1] * 100, linestyle='dotted', linewidth=0.5,
                               color=colors[k + j])  # s=size/8
            axs[i + 1][j].plot(wave, synthesized[k + j, :, 1] * 100, linewidth=0.25, linestyle='-', color=colors[k + j])
            axs[i][j].set_xticks([6560, 6562, 6564, 6566, 6568, 6570])
            axs[i + 1][j].set_xticks([6560, 6562, 6564, 6566, 6568, 6570])
            axs[i][j].set_xticklabels([])
            axs[i + 1][j].set_xticklabels([])
            axs[i][j].set_ylim(0.0, 1)
            axs[i][j].set_yticks([0.2, 0.4, 0.6, 0.8])
            axs[i][j].set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
            ymax = 1.1 * 100 * max(np.abs(observed[k + j, :, 1]).max(), np.abs(synthesized[k + j, :, 1]).max())
            axs[i + 1][j].set_ylim(-ymax, ymax)
            yticks = [-np.round(ymax * 2 / 3, 1), -np.round(ymax * 1 / 3, 1), 0, np.round(ymax * 1 / 3, 1),
                      np.round(ymax * 2 / 3, 1)]
            axs[i + 1][j].set_yticks(yticks)
            axs[i + 1][j].set_yticklabels(yticks, fontsize=fontsize)
        k += 5

    axs[3][0].set_xticklabels(np.round(np.array([6560, 6562, 6564, 6566, 6568, 6570]) - 6560, 0), fontsize=fontsize)
    axs[3][1].set_xticklabels(np.round(np.array([6560, 6562, 6564, 6566, 6568, 6570]) - 6560, 0), fontsize=fontsize)
    axs[3][2].set_xticklabels(np.round(np.array([6560, 6562, 6564, 6566, 6568, 6570]) - 6560, 0), fontsize=fontsize)
    axs[3][3].set_xticklabels(np.round(np.array([6560, 6562, 6564, 6566, 6568, 6570]) - 6560, 0), fontsize=fontsize)
    axs[3][4].set_xticklabels(np.round(np.array([6560, 6562, 6564, 6566, 6568, 6570]) - 6560, 0), fontsize=fontsize)
    axs[3][0].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 6560) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][1].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 6560)[$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][2].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 6560) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][3].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 6560) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[3][4].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 6560) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[0][0].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
    axs[2][0].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
    axs[1][0].set_ylabel(r'$V/I$ [%]', fontsize=fontsize)
    axs[3][0].set_ylabel(r'$V/I$ [%]', fontsize=fontsize)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.995, top=0.995, wspace=0.3, hspace=0.2)
    # plt.show()
    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'SynthProf_{}.pdf'.format(name), format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


def get_data_for_mean_profile_inversion_plots():

    calib_velocity = 2.72363206e5

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    pore_observed_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_mean_10_34_1_total_1.nc'
    pore_synth_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_mean_10_34_1_total_1_cycle_1_t_6_vl_2_vt_4_blong_2_profs.nc'
    pore_atmos_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_mean_10_34_1_total_1_cycle_1_t_6_vl_2_vt_4_blong_2_atmos.nc'

    emission_observed_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_mean_12_40_1_total_1.nc'
    emission_synth_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_mean_12_40_1_total_1_cycle_1_t_6_vl_4_vt_4_blong_2_profs.nc'
    emission_atmos_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_mean_12_40_1_total_1_cycle_1_t_6_vl_4_vt_4_blong_2_atmos.nc'

    opposite_polarity_observed_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_mean_8_51_1_total_1.nc'
    opposite_polarity_synth_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_mean_8_51_1_total_1_cycle_1_t_6_vl_2_vt_4_blong_2_nw_profs.nc'
    opposite_polarity_atmos_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_mean_8_51_1_total_1_cycle_1_t_6_vl_2_vt_4_blong_2_nw_atmos.nc'

    quiet_atmos_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_mean_13_17_1_total_1_cycle_1_t_6_vl_2_vt_4_blong_2_atmos.nc'

    data_1 = [[], [], []]
    data_2 = [[], [], []]

    f = h5py.File(pore_observed_file, 'r')
    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]
    wave = f['wav'][ind]
    data_1[0].append(f['profiles'][0, 0, 0, ind, 0])
    data_1[0].append(f['profiles'][0, 0, 0, ind, 3] * 100 / f['profiles'][0, 0, 0, ind, 0])
    f.close()

    f = h5py.File(pore_atmos_file, 'r')
    ltau = f['ltau500'][0, 0, 0]
    data_1[0].append(f['temp'][0, 0, 0] / 1e3)
    data_1[0].append((f['vlos'][0, 0, 0] - calib_velocity) / 1e5)
    data_1[0].append(f['vturb'][0, 0, 0] / 1e5)
    data_1[0].append(f['blong'][0, 0, 0] / 1e2)
    f.close()

    f = h5py.File(emission_observed_file, 'r')
    data_1[1].append(f['profiles'][0, 0, 0, ind, 0])
    data_1[1].append(f['profiles'][0, 0, 0, ind, 3] * 100 / f['profiles'][0, 0, 0, ind, 0])
    f.close()

    f = h5py.File(emission_atmos_file, 'r')
    data_1[1].append(f['temp'][0, 0, 0] / 1e3)
    data_1[1].append((f['vlos'][0, 0, 0] - calib_velocity) / 1e5)
    data_1[1].append(f['vturb'][0, 0, 0] / 1e5)
    data_1[1].append(f['blong'][0, 0, 0] / 1e2)
    f.close()

    f = h5py.File(opposite_polarity_observed_file, 'r')
    data_1[2].append(f['profiles'][0, 0, 0, ind, 0])
    data_1[2].append(f['profiles'][0, 0, 0, ind, 3] * 100 / f['profiles'][0, 0, 0, ind, 0])
    f.close()

    f = h5py.File(opposite_polarity_atmos_file, 'r')
    data_1[2].append(f['temp'][0, 0, 0] / 1e3)
    data_1[2].append((f['vlos'][0, 0, 0] - calib_velocity) / 1e5)
    data_1[2].append(f['vturb'][0, 0, 0] / 1e5)
    data_1[2].append(f['blong'][0, 0, 0] / 1e2)
    f.close()

    f = h5py.File(pore_synth_file, 'r')
    data_2[0].append(f['profiles'][0, 0, 0, ind, 0])
    data_2[0].append(f['profiles'][0, 0, 0, ind, 3] * 100 / f['profiles'][0, 0, 0, ind, 0])
    f.close()

    f = h5py.File(quiet_atmos_file, 'r')
    data_2[0].append(f['temp'][0, 0, 0] / 1e3)
    data_2[0].append((f['vlos'][0, 0, 0] - calib_velocity) / 1e5)
    data_2[0].append(f['vturb'][0, 0, 0] / 1e5)
    data_2[0].append(f['blong'][0, 0, 0] / 1e2)
    f.close()

    f = h5py.File(emission_synth_file, 'r')
    data_2[1].append(f['profiles'][0, 0, 0, ind, 0])
    data_2[1].append(f['profiles'][0, 0, 0, ind, 3] * 100 / f['profiles'][0, 0, 0, ind, 0])
    f.close()

    f = h5py.File(quiet_atmos_file, 'r')
    data_2[1].append(f['temp'][0, 0, 0] / 1e3)
    data_2[1].append((f['vlos'][0, 0, 0] - calib_velocity) / 1e5)
    data_2[1].append(f['vturb'][0, 0, 0] / 1e5)
    data_2[1].append(f['blong'][0, 0, 0] / 1e2)
    f.close()

    f = h5py.File(opposite_polarity_synth_file, 'r')
    data_2[2].append(f['profiles'][0, 0, 0, ind, 0])
    data_2[2].append(f['profiles'][0, 0, 0, ind, 3] * 100 / f['profiles'][0, 0, 0, ind, 0])
    f.close()

    f = h5py.File(quiet_atmos_file, 'r')
    data_2[2].append(f['temp'][0, 0, 0] / 1e3)
    data_2[2].append((f['vlos'][0, 0, 0] - calib_velocity) / 1e5)
    data_2[2].append(f['vturb'][0, 0, 0] / 1e5)
    data_2[2].append(f['blong'][0, 0, 0] / 1e2)
    f.close()

    return wave, ltau, data_1, data_2


def mean_profile_inversion_plots():
    wave, ltau, data_1, data_2 = get_data_for_mean_profile_inversion_plots()

    fontsize = 6

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(3, 6, figsize=(7, 3.5))

    color = 'black'
    axs[0][0].plot(wave, data_1[0][0], linestyle='dotted', linewidth=0.5, color=color)
    axs[0][1].plot(wave, data_1[0][1], linestyle='dotted', linewidth=0.5, color=color)
    axs[0][2].plot(ltau, data_1[0][2], linestyle='-', linewidth=0.5, color=color)
    axs[0][3].plot(ltau, data_1[0][3], linestyle='-', linewidth=0.5, color=color)
    axs[0][4].plot(ltau, data_1[0][4], linestyle='-', linewidth=0.5, color=color)
    axs[0][5].plot(ltau, data_1[0][5], linestyle='-', linewidth=0.5, color=color)

    color = 'darkblue'
    axs[1][0].plot(wave, data_1[1][0], linestyle='dotted', linewidth=0.5, color=color)
    axs[1][1].plot(wave, data_1[1][1], linestyle='dotted', linewidth=0.5, color=color)
    axs[1][2].plot(ltau, data_1[1][2], linestyle='-', linewidth=0.5, color=color)
    axs[1][3].plot(ltau, data_1[1][3], linestyle='-', linewidth=0.5, color=color)
    axs[1][4].plot(ltau, data_1[1][4], linestyle='-', linewidth=0.5, color=color)
    axs[1][5].plot(ltau, data_1[1][5], linestyle='-', linewidth=0.5, color=color)

    color = 'mediumvioletred'
    axs[2][0].plot(wave, data_1[2][0], linestyle='dotted', linewidth=0.5, color=color)
    axs[2][1].plot(wave, data_1[2][1], linestyle='dotted', linewidth=0.5, color=color)
    axs[2][2].plot(ltau, data_1[2][2], linestyle='-', linewidth=0.5, color=color)
    axs[2][3].plot(ltau, data_1[2][3], linestyle='-', linewidth=0.5, color=color)
    axs[2][4].plot(ltau, data_1[2][4], linestyle='-', linewidth=0.5, color=color)
    axs[2][5].plot(ltau, data_1[2][5], linestyle='-', linewidth=0.5, color=color)

    color = 'black'
    axs[0][0].plot(wave, data_2[0][0], linestyle='-', linewidth=0.5, color=color)
    axs[0][1].plot(wave, data_2[0][1], linestyle='-', linewidth=0.5, color=color)
    color = 'gray'
    axs[0][2].plot(ltau, data_2[0][2], linestyle='-', linewidth=0.5, color=color)
    axs[0][3].plot(ltau, data_2[0][3], linestyle='-', linewidth=0.5, color=color)
    axs[0][4].plot(ltau, data_2[0][4], linestyle='-', linewidth=0.5, color=color)
    axs[0][5].plot(ltau, data_2[0][5], linestyle='-', linewidth=0.5, color=color)

    color = 'darkblue'
    axs[1][0].plot(wave, data_2[1][0], linestyle='-', linewidth=0.5, color=color)
    axs[1][1].plot(wave, data_2[1][1], linestyle='-', linewidth=0.5, color=color)
    color = 'gray'
    axs[1][2].plot(ltau, data_2[1][2], linestyle='-', linewidth=0.5, color=color)
    axs[1][3].plot(ltau, data_2[1][3], linestyle='-', linewidth=0.5, color=color)
    axs[1][4].plot(ltau, data_2[1][4], linestyle='-', linewidth=0.5, color=color)
    axs[1][5].plot(ltau, data_2[1][5], linestyle='-', linewidth=0.5, color=color)

    color = 'mediumvioletred'
    axs[2][0].plot(wave, data_2[2][0], linestyle='-', linewidth=0.5, color=color)
    axs[2][1].plot(wave, data_2[2][1], linestyle='-', linewidth=0.5, color=color)
    color = 'gray'
    axs[2][2].plot(ltau, data_2[2][2], linestyle='-', linewidth=0.5, color=color)
    axs[2][3].plot(ltau, data_2[2][3], linestyle='-', linewidth=0.5, color=color)
    axs[2][4].plot(ltau, data_2[2][4], linestyle='-', linewidth=0.5, color=color)
    axs[2][5].plot(ltau, data_2[2][5], linestyle='-', linewidth=0.5, color=color)

    axs[0][0].set_ylim(0.1, 1)
    axs[1][0].set_ylim(0.1, 1)
    axs[2][0].set_ylim(0.1, 1)

    axs[0][0].set_yticks([0.2, 0.4, 0.6, 0.8])
    axs[0][0].set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
    axs[1][0].set_yticks([0.2, 0.4, 0.6, 0.8])
    axs[1][0].set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
    axs[2][0].set_yticks([0.2, 0.4, 0.6, 0.8])
    axs[2][0].set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=fontsize)

    ymax = 1.1 * max(
        np.abs(data_1[0][1]).max(),
        np.abs(data_2[0][1]).max()
    )
    axs[0][1].set_ylim(-ymax, ymax)
    yticks = [-np.round(ymax * 2 / 3, 1), -np.round(ymax * 1 / 3, 1), 0, np.round(ymax * 1 / 3, 1), np.round(ymax * 2 / 3, 1)]
    axs[0][1].set_yticks(yticks)
    axs[0][1].set_yticklabels(yticks, fontsize=fontsize)

    ymax = 1.1 * max(
        np.abs(data_1[1][1]).max(),
        np.abs(data_2[1][1]).max()
    )
    axs[1][1].set_ylim(-ymax, ymax)
    yticks = [-np.round(ymax * 2 / 3, 1), -np.round(ymax * 1 / 3, 1), 0, np.round(ymax * 1 / 3, 1),
              np.round(ymax * 2 / 3, 1)]
    axs[1][1].set_yticks(yticks)
    axs[1][1].set_yticklabels(yticks, fontsize=fontsize)

    ymax = 1.1 * max(
        np.abs(data_1[2][1]).max(),
        np.abs(data_2[2][1]).max()
    )
    axs[2][1].set_ylim(-ymax, ymax)
    yticks = [-np.round(ymax * 2 / 3, 1), -np.round(ymax * 1 / 3, 1), 0, np.round(ymax * 1 / 3, 1),
              np.round(ymax * 2 / 3, 1)]
    axs[2][1].set_yticks(yticks)
    axs[2][1].set_yticklabels(yticks, fontsize=fontsize)

    yticks = [4, 6, 8, 10, 12]
    axs[0][2].set_ylim(3.5, 14)
    axs[1][2].set_ylim(3.5, 14)
    axs[2][2].set_ylim(3.5, 14)
    axs[0][2].set_yticks(yticks)
    axs[1][2].set_yticks(yticks)
    axs[2][2].set_yticks(yticks)
    axs[0][2].set_yticklabels(yticks, fontsize=fontsize)
    axs[1][2].set_yticklabels(yticks, fontsize=fontsize)
    axs[2][2].set_yticklabels(yticks, fontsize=fontsize)

    axs[0][3].tick_params(axis='y', labelsize=fontsize)
    axs[1][3].tick_params(axis='y', labelsize=fontsize)
    axs[2][3].tick_params(axis='y', labelsize=fontsize)

    axs[0][4].tick_params(axis='y', labelsize=fontsize)
    axs[1][4].tick_params(axis='y', labelsize=fontsize)
    axs[2][4].tick_params(axis='y', labelsize=fontsize)

    axs[0][5].tick_params(axis='y', labelsize=fontsize)
    axs[1][5].tick_params(axis='y', labelsize=fontsize)
    axs[2][5].tick_params(axis='y', labelsize=fontsize)

    xticks = np.array([8536, 8538, 8540, 8540, 8542, 8544])
    axs[0][0].set_xticks(xticks)
    axs[0][1].set_xticks(xticks)
    axs[1][0].set_xticks(xticks)
    axs[1][1].set_xticks(xticks)
    axs[2][0].set_xticks(xticks)
    axs[2][1].set_xticks(xticks)
    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[1][0].set_xticklabels([])
    axs[1][1].set_xticklabels([])
    axs[2][0].set_xticklabels(xticks - 8530, fontsize=fontsize)
    axs[2][1].set_xticklabels(xticks - 8530, fontsize=fontsize)

    xticks = np.array([-6, -4, -2, 0])
    axs[0][2].set_xticks(xticks)
    axs[0][3].set_xticks(xticks)
    axs[0][4].set_xticks(xticks)
    axs[0][5].set_xticks(xticks)
    axs[1][2].set_xticks(xticks)
    axs[1][3].set_xticks(xticks)
    axs[1][4].set_xticks(xticks)
    axs[1][5].set_xticks(xticks)
    axs[2][2].set_xticks(xticks)
    axs[2][3].set_xticks(xticks)
    axs[2][4].set_xticks(xticks)
    axs[2][5].set_xticks(xticks)
    axs[0][2].set_xticklabels([])
    axs[0][3].set_xticklabels([])
    axs[0][4].set_xticklabels([])
    axs[0][5].set_xticklabels([])
    axs[1][2].set_xticklabels([])
    axs[1][3].set_xticklabels([])
    axs[1][4].set_xticklabels([])
    axs[1][5].set_xticklabels([])
    axs[2][2].set_xticklabels(xticks, fontsize=fontsize)
    axs[2][3].set_xticklabels(xticks, fontsize=fontsize)
    axs[2][4].set_xticklabels(xticks, fontsize=fontsize)
    axs[2][5].set_xticklabels(xticks, fontsize=fontsize)

    axs[2][0].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 8530) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[2][1].set_xlabel(r'$\Delta\lambda$ ($\lambda$ - 8530) [$\mathrm{\AA}$]', fontsize=fontsize)
    axs[2][2].set_xlabel(r'$\log \tau_{500}$', fontsize=fontsize)
    axs[2][3].set_xlabel(r'$\log \tau_{500}$', fontsize=fontsize)
    axs[2][4].set_xlabel(r'$\log \tau_{500}$', fontsize=fontsize)
    axs[2][5].set_xlabel(r'$\log \tau_{500}$', fontsize=fontsize)

    axs[0][0].set_title(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
    # axs[1][0].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
    # axs[2][0].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)

    axs[0][1].set_title(r'$V/I$ [%]', fontsize=fontsize)
    # axs[1][1].set_ylabel(r'$V/I$ [%]', fontsize=fontsize)
    # axs[2][1].set_ylabel(r'$V/I$ [%]', fontsize=fontsize)

    axs[0][2].set_title('T [kK]', fontsize=fontsize)
    # axs[1][2].set_ylabel('T [kK]', fontsize=fontsize)
    # axs[2][2].set_ylabel('T [kK]', fontsize=fontsize)

    axs[0][3].set_title(r'$V_{\mathrm{LOS}}$ $\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)
    # axs[1][3].set_ylabel(r'$V_{\mathrm{LOS}}$ $\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)
    # axs[2][3].set_ylabel(r'$V_{\mathrm{LOS}}$ $\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)

    axs[0][4].set_title(r'$V_{\mathrm{turb}}$ $\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)
    # axs[1][4].set_ylabel(r'$V_{\mathrm{turb}}$ $\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)
    # axs[2][4].set_ylabel(r'$V_{\mathrm{turb}}$ $\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)

    axs[0][5].set_title(r'$B_{\mathrm{LOS}}$ [x 100 G]', fontsize=fontsize)
    # axs[1][5].set_ylabel(r'$B_{\mathrm{LOS}}$ [G]', fontsize=fontsize)
    # axs[2][5].set_ylabel(r'$B_{\mathrm{LOS}}$ [G]', fontsize=fontsize)

    plt.subplots_adjust(left=0.04, bottom=0.12, right=0.99, top=0.94, wspace=0.4, hspace=0.2)

    write_path = Path('/home/harsh/Spinor Paper/')

    plt.show()

    fig.savefig(write_path / 'mean_profile_inversions.pdf', format='pdf', dpi=300)

def make_legend(fontsize=6):
    color = ['black', 'darkblue', 'mediumvioletred', 'gray']
    label_list = ['pore', 'microflare', 'opposite polarity', 'median profile']

    handles = [Patch(color=c, label=l) for l, c in zip(label_list, color)]
    plt.close('all')
    plt.clf()
    plt.cla()

    fig = plt.figure(figsize=(7, 3.5))
    legend = plt.legend(
        handles,
        label_list,
        ncol=4,
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc='lower left',
        mode="expand",
        borderaxespad=0.,
        fontsize=fontsize
    )
    fig.canvas.draw()
    bbox = legend.get_window_extent().padded(2)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'legends_{}.pdf'.format(fontsize), dpi=300, transparent=True, bbox_inches=bbox)

    plt.close('all')
    plt.clf()
    plt.cla()


def make_mag_field_scatter_plots():
    fontsize = 8

    a, b, c = get_wfanew_alternate()

    magha, magha_p, magha_full_line = a.T, b.T, c.T

    interesting_ltaus = [-2, -4.5]

    ltau_indice = list()

    for interesting_ltau in interesting_ltaus:
        ltau_indice.append(np.argmin(np.abs(ltau500 - interesting_ltau)))

    ltau_indice = np.array(ltau_indice)

    _, _, _, _, _, mask = get_fov_data()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output.nc', 'r')

    a0 = f['blong'][0, 0:17, :, ltau_indice[0]].T
    a1 = f['blong'][0, 0:17, :, ltau_indice[1]].T

    f.close()

    fontsize = 8

    fig, axs = plt.subplots(2, 2, figsize=(7, 3.5))

    a, b = np.where(a0 >= 0)
    c, d = np.where(a0 < 0)

    axs[0][0].scatter(np.abs(a0[c, d]), np.abs(a1[c, d]), s=1, color='royalblue')
    axs[0][0].scatter(np.abs(a0[a, b]), np.abs(a1[a, b]), s=1, color='red')

    maxval = np.abs(a0).max().astype(np.int64) + 1
    axs[0][0].plot(range(maxval), range(maxval), color='darkorange', linestyle='--')

    axs[0][1].scatter(np.abs(a1[c, d]), np.abs(magha[c, d]), s=1, color='royalblue')
    axs[0][1].scatter(np.abs(a1[a, b]), np.abs(magha[a, b]), s=1, color='red')
    maxval = np.abs(a1).max().astype(np.int64) + 1
    axs[0][1].plot(range(maxval), range(maxval), color='darkorange', linestyle='--')

    axs[1][0].scatter(np.abs(a0[c, d]), np.abs(magha_p[c, d]), s=1, color='royalblue')
    axs[1][0].scatter(np.abs(a0[a, b]), np.abs(magha_p[a, b]), s=1, color='red')
    maxval = np.abs(a0).max().astype(np.int64) + 1
    axs[1][0].plot(range(maxval), range(maxval), color='darkorange', linestyle='--')

    axs[1][1].scatter(np.abs(a0[c, d]), np.abs(magha_full_line[c, d]), s=1, color='royalblue')
    axs[1][1].scatter(np.abs(a0[a, b]), np.abs(magha_full_line[a, b]), s=1, color='red')
    maxval = np.abs(a0).max().astype(np.int64) + 1
    axs[1][1].plot(range(maxval), range(maxval), color='darkorange', linestyle='--')

    for i in range(2):
        for j in range(2):
            axs[i][j].tick_params(axis='x', labelsize=fontsize)
            axs[i][j].tick_params(axis='y', labelsize=fontsize)
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(50))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(50))

    axs[0][0].set_xlabel(r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$2) [G]', fontsize=fontsize)
    axs[0][1].set_xlabel(r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5) [G]', fontsize=fontsize)
    axs[1][0].set_xlabel(r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$2) [G]', fontsize=fontsize)
    axs[1][1].set_xlabel(r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$2) [G]', fontsize=fontsize)
    axs[0][0].set_ylabel(r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5) [G]', fontsize=fontsize)
    axs[0][1].set_ylabel(r'$|B_{\mathrm{LOS}}|$ WFA (H$\alpha$ core) [G]', fontsize=fontsize)
    axs[1][0].set_ylabel(r'$|B_{\mathrm{LOS}}|$ WFA (H$\alpha$ wing) [G]', fontsize=fontsize)
    axs[1][1].set_ylabel(r'$|B_{\mathrm{LOS}}|$ WFA (H$\alpha\pm1.5\AA$) [G]', fontsize=fontsize)

    axs[0][0].text(
        0.05, 0.85,
        '(a)',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )

    axs[0][1].text(
        0.05, 0.85,
        '(b)',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )

    axs[1][0].text(
        0.05, 0.85,
        '(c)',
        transform=axs[1][0].transAxes,
        fontsize=fontsize
    )

    axs[1][1].text(
        0.05, 0.85,
        '(d)',
        transform=axs[1][1].transAxes,
        fontsize=fontsize
    )
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.99, top=0.95, wspace=0.25, hspace=0.35)

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'MagScatter.pdf', format='pdf', dpi=300)


if __name__ == '__main__':
    # points = [
    #     (12, 49),
    #     (12, 40),
    #     (12, 34),
    #     (12, 31),
    #     (12, 18),
    #     (8, 53),
    #     (8, 50),
    #     (8, 37),
    #     (8, 31),
    #     (8, 9),
    # ]
    # colors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green', 'darkslateblue', 'purple', 'mediumvioletred', 'darkolivegreen']
    # make_fov_plots(points, colors)
    # points = [
    #     49,
    #     40,
    #     34,
    #     31,
    #     18
    # ]
    # colors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown']
    # factor_ca_list = [
    #     (2, 2, 1),
    #     (2, 2, 1),
    #     (2, 2, 1),
    #     (2, 2, 1),
    #     (2, 2, 1)
    # ]
    # factor_ha_list = [
    #     (2, 1),
    #     (2, 1),
    #     (2, 1),
    #     (2, 1),
    #     (1.5, 1)
    # ]
    # cut_indice = 12
    # plot_stokes_parameters(cut_indice, points, colors)
    # plot_spatial_variation_of_profiles(cut_indice, points, colors, factor_ca_list, factor_ha_list)
    # cut_indice = 8
    # points = [
    #     53,
    #     50,
    #     37,
    #     31,
    #     9
    # ]
    # factor_ca_list = [
    #     (1, 1, 1),
    #     (1, 1, 1),
    #     (1, 1, 1),
    #     (3, 3, 1),
    #     (1, 1, 1)
    # ]
    # factor_ha_list = [
    #     (3, 1),
    #     (3, 1),
    #     (3, 1),
    #     (3, 1),
    #     (3, 1)
    # ]
    # colors = ['green', 'darkslateblue', 'purple', 'mediumvioletred', 'darkolivegreen']
    # plot_stokes_parameters(cut_indice, points, colors)
    # plot_spatial_variation_of_profiles(cut_indice, points, colors, factor_ca_list, factor_ha_list)
    # plot_profiles()
    points = [
        (12, 49),
        (12, 40),
        (12, 34),
        (12, 31),
        (12, 18),
        (8, 53),
        (8, 50),
        (8, 37),
        (8, 31),
        (8, 9),
    ]
    colors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green', 'darkslateblue', 'purple', 'mediumvioletred', 'darkolivegreen']
    make_output_param_plots(points, colors)
    # plot_mag_field_compare()
    # plot_mag_field_compare_new(points, colors)
    # make_mag_field_scatter_plots()
    # points = [
    #     (12, 49),
    #     (12, 40),
    #     (12, 34),
    #     (12, 31),
    #     (12, 18),
    #     (8, 53),
    #     (8, 50),
    #     (8, 37),
    #     (8, 31),
    #     (8, 9),
    # ]
    # colors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green', 'darkslateblue', 'purple', 'mediumvioletred', 'darkolivegreen']
    # make_quality_of_fits(points, colors)
    # make_atmos_for_response_functions(points)
    # mean_profile_inversion_plots()
    # make_legend()
    # points = [
    #     (12, 49),
    #     (12, 40),
    #     (12, 34),
    #     (12, 31),
    #     (12, 18),
    #     (8, 53),
    #     (8, 50),
    #     (8, 37),
    #     (8, 31),
    #     (8, 9),
    # ]
    # colors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green', 'darkslateblue', 'purple', 'mediumvioletred', 'darkolivegreen']
    # base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')
    # filenames = ['combined_output_ha_profs_synth.nc', 'combined_output_ha_profs_synth_sublevels_4_lines.nc', 'combined_output_ha_profs_synth_sublevels_7_lines.nc']
    # names = ['H_6_geff_1.048', 'H_substructure_4_lines', 'H_substructure_7_lines']
    # for filename, name in zip(filenames, names):
    #     make_forward_synthesis_plots(base_path / filename, points, colors, name)
