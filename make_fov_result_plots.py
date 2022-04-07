import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict
import queue
from matplotlib.colors import LinearSegmentedColormap
from weak_field_approx import prepare_calculate_blos


processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

ca_ha_data_file = processed_inputs / 'aligned_Ca_Ha_stic_profiles.nc'

me_data_file = processed_inputs / 'me_results_6569.nc'

wfa_8542_data_file = processed_inputs / 'wfa_8542.nc'

points = [
    57,
    49,
    40,
    34,
    31,
    18,
    8
]


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


def plot_stokes_parameters():

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    fig, axs = plt.subplots(2, 2, figsize=(7, 4.5))

    colors = ["red", "yellow", "white", "blue", "green"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    X, Y = np.meshgrid(fcaha['wav'][ind[0:306]], np.arange(0, 60 * 0.38, 0.38))
    im00 = axs[0][0].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[0:306], 0], cmap='gray', shading='gouraud')
    im01 = axs[0][1].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[0:306], 3] * 100 / fcaha['profiles'][0, 12, :, ind[0:306], 0], cmap=cmap1, shading='gouraud', vmin=-10, vmax=10)
    im10 = axs[1][0].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[0:306], 1] * 100 / fcaha['profiles'][0, 12, :, ind[0:306], 0], cmap=cmap1, shading='gouraud', vmin=-2, vmax=2)
    im11 = axs[1][1].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[0:306], 2] * 100 / fcaha['profiles'][0, 12, :, ind[0:306], 0], cmap=cmap1, shading='gouraud', vmin=-2, vmax=2)

    colors = ['violet', 'indigo', 'blue', 'black', 'orange', 'brown', 'red']

    for point, color in zip(points, colors):
        for i in range(1):
            for j in range(2):
                axs[i][j].plot(fcaha['wav'][ind[0:306]], np.ones_like(fcaha['wav'][ind[0:306]]) * point * 0.38, color=color, linestyle='--', linewidth=0.5)

    fig.colorbar(im00, ax=axs[0][0])
    fig.colorbar(im01, ax=axs[0][1])
    fig.colorbar(im10, ax=axs[1][0])
    fig.colorbar(im11, ax=axs[1][1])

    axs[0][0].set_ylabel('Slit Position [arcsec]')
    axs[1][0].set_ylabel('Slit Position [arcsec]')
    axs[0][0].set_title('Intensity [Normalised]')

    axs[0][1].set_title(r'$V/I$ [%]')

    axs[1][0].set_title(r'$Q/I$ [%]')

    axs[1][1].set_title(r'$U/I$ [%]')

    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticklabels([])

    axs[0][0].set_xticks([8536, 8538, 8540, 8542, 8544])
    axs[0][0].set_xticklabels([8536, 8538, 8540, 8542, 8544])

    axs[0][1].set_xticks([8536, 8538, 8540, 8542, 8544])
    axs[0][1].set_xticklabels([8536, 8538, 8540, 8542, 8544])

    axs[1][0].set_xticks([8536, 8538, 8540, 8542, 8544])
    axs[1][0].set_xticklabels([8536, 8538, 8540, 8542, 8544])

    axs[1][1].set_xticks([8536, 8538, 8540, 8542, 8544])
    axs[1][1].set_xticklabels([8536, 8538, 8540, 8542, 8544])

    axs[1][0].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    axs[1][1].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')

    # fig.suptitle(r'Ca II 8542 $\mathrm{\AA}$')

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(0.25))

    fig.tight_layout()

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'CaII_Stokes_12.pdf', format='pdf', dpi=300)

    fig, axs = plt.subplots(2, 2, figsize=(7, 4.5))

    colors = ["red", "yellow", "white", "blue", "green"]

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    X, Y = np.meshgrid(fcaha['wav'][ind[306:]], np.arange(0, 60 * 0.38, 0.38))
    im00 = axs[0][0].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[306:], 0], cmap='gray', shading='gouraud')
    im01 = axs[0][1].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[306:], 3] * 100 / fcaha['profiles'][0, 12, :, ind[306:], 0], cmap=cmap1, shading='gouraud', vmin=-6, vmax=6)
    im10 = axs[1][0].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[306:], 1] * 100 / fcaha['profiles'][0, 12, :, ind[306:], 0], cmap=cmap1, shading='gouraud', vmin=-2, vmax=2)
    im11 = axs[1][1].pcolormesh(X, Y, fcaha['profiles'][0, 12, :, ind[306:], 2] * 100 / fcaha['profiles'][0, 12, :, ind[306:], 0], cmap=cmap1, shading='gouraud', vmin=-3, vmax=3)

    colors = ['violet', 'indigo', 'blue', 'black', 'orange', 'brown', 'red']

    for point, color in zip(points, colors):
        for i in range(1):
            for j in range(2):
                axs[i][j].plot(fcaha['wav'][ind[306:]], np.ones_like(fcaha['wav'][ind[306:]]) * point * 0.38, color=color, linestyle='--', linewidth=0.5)

    fig.colorbar(im00, ax=axs[0][0])
    fig.colorbar(im01, ax=axs[0][1])
    fig.colorbar(im10, ax=axs[1][0])
    fig.colorbar(im11, ax=axs[1][1])

    axs[0][0].set_ylabel('Slit Position [arcsec]')
    axs[1][0].set_ylabel('Slit Position [arcsec]')
    axs[0][0].set_title('Intensity [Normalised]')

    axs[0][1].set_title(r'$V/I$ [%]')

    axs[1][0].set_title(r'$Q/I$ [%]')

    axs[1][1].set_title(r'$U/I$ [%]')

    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticklabels([])

    axs[1][0].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    axs[1][1].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')

    # fig.suptitle(r'H$\mathrm{\alpha}$')

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(0.25))

    fig.tight_layout()

    # plt.show()

    write_path = Path('/home/harsh/Spinor Paper/')
    fig.savefig(write_path / 'Ha_Stokes_12.pdf', format='pdf', dpi=300)

    fcaha.close()


def plot_profiles():
    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    plt.pcolormesh(fcaha['profiles'][0, 12, :, ind[0:306], 0], cmap='gray', shading='gouraud')

    plt.show()

    fcaha.close()


def plot_spatial_variation_of_profiles():

    fcaha = h5py.File(ca_ha_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0]!=0)[0]

    fig, axs = plt.subplots(2, 2, figsize=(7, 4.5))

    colors = ['violet', 'indigo', 'blue', 'black', 'orange', 'brown', 'red']

    for point, color in zip(points, colors):
        axs[0][0].plot(fcaha['wav'][ind[0:306]], fcaha['profiles'][0, 12, point, ind[0:306], 0], color=color, linewidth=0.5)
        axs[0][1].plot(fcaha['wav'][ind[0:306]], fcaha['profiles'][0, 12, point, ind[0:306], 3] / fcaha['profiles'][0, 12, point, ind[0:306], 0], color=color, linewidth=0.5)

    for point, color in zip(points, colors):
        axs[1][0].plot(fcaha['wav'][ind[306:]], fcaha['profiles'][0, 12, point, ind[306:], 0], color=color, linewidth=0.5)
        axs[1][1].plot(fcaha['wav'][ind[306:]], fcaha['profiles'][0, 12, point, ind[306:], 3] / fcaha['profiles'][0, 12, point, ind[306:], 0], color=color, linewidth=0.5)

    axs[0][0].set_ylabel(r'$I/I_{c}$')
    axs[1][0].set_ylabel(r'$I/I_{c}$')
    axs[0][1].set_ylabel(r'$V/I$')
    axs[1][1].set_ylabel(r'$V/I$')

    axs[0][0].set_xticks([8536, 8538, 8540, 8542, 8544, 8546])
    axs[0][0].set_xticklabels([8536, 8538, 8540, 8542, 8544, 8546])

    axs[0][1].set_xticks([8536, 8538, 8540, 8542, 8544, 8546])
    axs[0][1].set_xticklabels([8536, 8538, 8540, 8542, 8544, 8546])

    axs[1][0].set_xticks([6560, 6562, 6564, 6566, 6568, 6570])
    axs[1][0].set_xticklabels([6560, 6562, 6564, 6566, 6568, 6570])

    axs[1][1].set_xticks([6560, 6562, 6564, 6566, 6568, 6570])
    axs[1][1].set_xticklabels([6560, 6562, 6564, 6566, 6568, 6570])

    axs[1][0].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    axs[1][1].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(0.25))

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(0.01))

    axs[0][1].set_ylim(-0.13, 0.13)
    axs[1][1].set_ylim(-0.07, 0.07)

    fig.tight_layout()

    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'SpatialVariationProfiles.pdf', format='pdf', dpi=300)

    fcaha.close()

    plt.close('all')

    plt.clf()

    plt.cla()


def make_output_param_plots():

    fontsize = 8

    _, _, _, _, _, mask = get_fov_data()

    interesting_ltaus = [-5, -3, -1]

    ltau_indice = list()

    for interesting_ltau in interesting_ltaus:
        ltau_indice.append(np.argmin(np.abs(ltau500 - interesting_ltau)))

    ltau_indice = np.array(ltau_indice)

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output.nc', 'r')

    fig, axs = plt.subplots(3, 3, figsize=(3.5, 7))

    ind = np.where((ltau500 >= -1) & (ltau500 <= 0))[0]

    a, b = np.where(mask[0, 0] == 1)

    calib_vlos = np.mean(f['vlos'][()][0, a, b][:, ind])

    X, Y = np.meshgrid(np.arange(0, 17 * 0.38, 0.38), np.arange(0, 60 * 0.38, 0.38))

    im00 = axs[0][0].pcolormesh(X, Y, f['temp'][0, 0:17, :, ltau_indice[0]].T / 1e3, cmap='hot', shading='gouraud')
    im01 = axs[0][1].pcolormesh(X, Y, f['temp'][0, 0:17, :, ltau_indice[1]].T / 1e3, cmap='hot', shading='gouraud')
    im02 = axs[0][2].pcolormesh(X, Y, f['temp'][0, 0:17, :, ltau_indice[2]].T / 1e3, cmap='hot', shading='gouraud')

    im10 = axs[1][0].pcolormesh(X, Y, (f['vlos'][0, 0:17, :, ltau_indice[0]].T - calib_vlos) / 1e5, cmap='bwr', vmin=-5, vmax=5, shading='gouraud')
    im11 = axs[1][1].pcolormesh(X, Y, (f['vlos'][0, 0:17, :, ltau_indice[1]].T - calib_vlos) / 1e5, cmap='bwr', vmin=-5, vmax=5, shading='gouraud')
    im12 = axs[1][2].pcolormesh(X, Y, (f['vlos'][0, 0:17, :, ltau_indice[2]].T - calib_vlos) / 1e5, cmap='bwr', vmin=-5, vmax=5, shading='gouraud')

    im20 = axs[2][0].pcolormesh(X, Y, f['vturb'][0, 0:17, :, ltau_indice[0]].T / 1e5, cmap='copper', vmin=0, vmax=5, shading='gouraud')
    im21 = axs[2][1].pcolormesh(X, Y, f['vturb'][0, 0:17, :, ltau_indice[1]].T / 1e5, cmap='copper', vmin=0, vmax=5, shading='gouraud')
    im22 = axs[2][2].pcolormesh(X, Y, f['vturb'][0, 0:17, :, ltau_indice[2]].T / 1e5, cmap='copper', vmin=0, vmax=5, shading='gouraud')

    cbaxes = inset_axes(
        axs[0][0],
        width="100%",
        height="5%",
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
        width="100%",
        height="5%",
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
        width="100%",
        height="5%",
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
        width="10%",
        height="100%",
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
        width="10%",
        height="100%",
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

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[0][2].set_xticklabels([])
    axs[0][2].set_yticklabels([])

    axs[1][0].set_xticklabels([])
    axs[1][1].set_xticklabels([])
    axs[1][1].set_yticklabels([])
    axs[1][2].set_xticklabels([])
    axs[1][2].set_yticklabels([])

    axs[2][1].set_yticklabels([])
    axs[2][2].set_yticklabels([])

    for i in range(3):
        for j in range(3):
            color = 'black'
            if i == 2:
                color = 'white'
            axs[i][j].contour(X, Y, mask[0, 0].T, levels=0, colors=color, linewidths=0.5)

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

    axs[0][0].set_ylabel('x [arcsec]')
    axs[1][0].set_ylabel('x [arcsec]')
    axs[2][0].set_ylabel('x [arcsec]')

    axs[2][0].set_xlabel('y [arcsec]')
    axs[2][1].set_xlabel('y [arcsec]')
    axs[2][2].set_xlabel('y [arcsec]')

    axs[0][0].text(
        0.05, 1.27,
        r'$\log \tau_{\mathrm{500}}=-5$',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.05, 1.27,
        r'$\log \tau_{\mathrm{500}}=-3$',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[0][2].text(
        0.05, 1.27,
        r'$\log \tau_{\mathrm{500}}=-1$',
        transform=axs[0][2].transAxes,
        fontsize=fontsize
    )

    plt.subplots_adjust(left=0.15, bottom=0.07, right=0.87, top=0.9, wspace=0.2, hspace=0.1)

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

    print (magha.min(), magha.max())
    print (magha_p.min(), magha_p.max())
    return magha, magha_p


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

    fcaha.close()

    print (magha.min(), magha.max())
    print (magha_p.min(), magha_p.max())

    return magha, magha_p


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


def plot_mag_field_compare_new():

    fontsize = 8

    a, b = get_wfanew_alternate()

    magha, magha_p = a.T, b.T

    interesting_ltaus = [0, -2, -3, -5]

    ltau_indice = list()

    for interesting_ltau in interesting_ltaus:
        ltau_indice.append(np.argmin(np.abs(ltau500 - interesting_ltau)))

    ltau_indice = np.array(ltau_indice)

    _, _, _, _, _, mask = get_fov_data()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')

    f = h5py.File(base_path / 'combined_output.nc', 'r')

    fig, axs = plt.subplots(2, 5, figsize=(7, 7))

    a0 = f['blong'][0, 0:17, :, ltau_indice[0]].T
    a1 = f['blong'][0, 0:17, :, ltau_indice[1]].T
    a2 = f['blong'][0, 0:17, :, ltau_indice[2]].T
    a3 = f['blong'][0, 0:17, :, ltau_indice[3]].T

    colors = ["saddlebrown", "peru", "white", "green", "blue"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    cmap = cmap1  #RdGy

    X, Y = np.meshgrid(np.arange(0, 17 * 0.38, 0.38), np.arange(0, 60 * 0.38, 0.38))

    im00 = axs[0][0].pcolormesh(X, Y, a1, cmap=cmap, shading='gouraud', vmin=-1000, vmax=1000)

    im01 = axs[0][1].pcolormesh(X, Y, a2, cmap=cmap, shading='gouraud', vmin=-900, vmax=900)

    im02 = axs[0][2].pcolormesh(X, Y, a3, cmap=cmap, shading='gouraud', vmin=-600, vmax=600)

    im03 = axs[0][3].pcolormesh(X, Y, magha, cmap=cmap, shading='gouraud', vmin=-400, vmax=400)

    im04 = axs[0][4].pcolormesh(X, Y, magha_p, cmap=cmap, shading='gouraud', vmin=-700, vmax=700)

    im10 = axs[1][0].pcolormesh(X, Y, np.abs(a1) - np.abs(a0), cmap='bwr', shading='gouraud', vmin=-300, vmax=300)

    im11 = axs[1][1].pcolormesh(X, Y, np.abs(a2) - np.abs(a1), cmap='bwr', shading='gouraud', vmin=-400, vmax=400)

    im12 = axs[1][2].pcolormesh(X, Y, np.abs(a3) - np.abs(a2), cmap='bwr', shading='gouraud', vmin=-500, vmax=500)

    im13 = axs[1][3].pcolormesh(X, Y, np.abs(magha) - np.abs(a3), cmap='bwr', shading='gouraud', vmin=-250, vmax=250)

    im14 = axs[1][4].pcolormesh(X, Y, np.abs(magha_p) - np.abs(a1), cmap='bwr', shading='gouraud', vmin=-600, vmax=600)

    for i in range(2):
        for j in range(5):
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
        ticks=[-300, 0, 300],
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
        ticks=[-500, 0, 500],
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
        ticks=[-500, 0, 500],
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

    axs[1][1].set_yticklabels([])
    axs[1][2].set_yticklabels([])
    axs[1][3].set_yticklabels([])
    axs[1][4].set_yticklabels([])

    axs[0][0].set_yticks([0, 5, 10, 15, 20])
    axs[0][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs[0][1].set_yticks([0, 5, 10, 15, 20])
    axs[0][2].set_yticks([0, 5, 10, 15, 20])
    axs[0][3].set_yticks([0, 5, 10, 15, 20])
    axs[0][4].set_yticks([0, 5, 10, 15, 20])
    axs[1][0].set_yticks([0, 5, 10, 15, 20])
    axs[1][0].set_yticklabels([0, 5, 10, 15, 20], fontsize=fontsize)
    axs[1][1].set_yticks([0, 5, 10, 15, 20])
    axs[1][2].set_yticks([0, 5, 10, 15, 20])
    axs[1][3].set_yticks([0, 5, 10, 15, 20])
    axs[1][4].set_yticks([0, 5, 10, 15, 20])

    axs[0][0].set_xticks([0, 5])
    axs[0][1].set_xticks([0, 5])
    axs[0][2].set_xticks([0, 5])
    axs[0][3].set_xticks([0, 5])
    axs[0][4].set_xticks([0, 5])

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

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][3].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][4].yaxis.set_minor_locator(MultipleLocator(1))

    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][4].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][3].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][4].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_ylabel('x [arcsec]', fontsize=fontsize)
    axs[1][0].set_ylabel('x [arcsec]', fontsize=fontsize)

    axs[1][0].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][1].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][2].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][3].set_xlabel('y [arcsec]', fontsize=fontsize)
    axs[1][4].set_xlabel('y [arcsec]', fontsize=fontsize)

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
        r'$\log \tau_{\mathrm{500}}=-5$',
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
        r'WFA (H$\alpha \pm 0.35 \AA$)',
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
        0.15, 1.2,
        r'WFA (H$\alpha$ Wing)',
        transform=axs[0][4].transAxes,
        fontsize=fontsize
    )
    axs[0][4].text(
        0.05, 0.95,
        r'(e)',
        transform=axs[0][4].transAxes,
        fontsize=fontsize
    )

    axs[1][0].text(
        0.05, 0.95,
        r'|(a)| - |$B_{\log\tau_{\mathrm{500} = 0}}$|',
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

    plt.subplots_adjust(left=0.1, bottom=0.07, right=0.96, top=0.9, wspace=0.2, hspace=0.2)

    write_path = Path('/home/harsh/Spinor Paper/')

    fig.savefig(write_path / 'MagneticField.pdf', format='pdf', dpi=300)

    # plt.show()

    f.close()


if __name__ == '__main__':
    # make_fov_plots()
    # plot_stokes_parameters()
    # plot_profiles()
    # plot_spatial_variation_of_profiles()
    make_output_param_plots()
    # plot_mag_field_compare()
    # plot_mag_field_compare_new()
