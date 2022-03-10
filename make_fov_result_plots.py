import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

ca_ha_data_file = processed_inputs / 'aligned_Ca_Ha_stic_profiles.nc'

me_data_file = processed_inputs / 'me_results_6569.nc'

wfa_8542_data_file = processed_inputs / 'wfa_8542.nc'


def get_fov_data():
    data = np.zeros((3, 2, 17, 60), dtype=np.float64)

    fcaha = h5py.File(ca_ha_data_file, 'r')

    fme = h5py.File(me_data_file, 'r')

    fwfa = h5py.File(wfa_8542_data_file, 'r')

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

    ind_8542_core = ind[np.argmin(np.abs(fcaha['wav'][ind] - 8542.09))]

    ind_8542_wing = ind[np.argmin(np.abs(fcaha['wav'][ind] - 8534.288))]

    ind_6563_core = ind[np.argmin(np.abs(fcaha['wav'][ind] - 6562.8))]

    ind_6563_wing = ind[np.argmin(np.abs(fcaha['wav'][ind] - 6569.826))]

    data[0][0] = fcaha['profiles'][0, :, :, ind_8542_wing, 0]
    data[0][1] = fcaha['profiles'][0, :, :, ind_8542_core, 0]
    data[1][0] = fcaha['profiles'][0, :, :, ind_6563_wing, 0]
    data[1][1] = fcaha['profiles'][0, :, :, ind_6563_core, 0]
    data[2][0] = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])
    data[2][1] = fwfa['blos_gauss'][()]

    aa, bb, cc, dd = fcaha['wav'][ind_8542_wing], fcaha['wav'][ind_8542_core], fcaha['wav'][ind_6563_wing], fcaha['wav'][ind_6563_core]

    fwfa.close()

    fme.close()

    fcaha.close()

    return data, aa, bb, cc, dd


def make_fov_plots():
    data, wing_ca, core_ca, wing_ha, core_ha = get_fov_data()

    fig, axs = plt.subplots(3, 2, figsize=(7, 3.3))

    extent = [0, 22.8, 0, 6.46]

    for i in range(3):
        for j in range(2):
            if i == 2:
                if j == 0:
                    vmin = -600
                    vmax = 600
                else:
                    vmin=
                axs[i][j].imshow(data[i][j], cmap='RdGy', origin='lower', extent=extent)
            else:
                axs[i][j].imshow(data[i][j], cmap='RdGy', origin='lower', extent=extent)

    axs[0][0].text(
        0.01, 0.8,
        r'(a) Ca II 8542 $\mathrm{{\AA}}$ {} $\mathrm{{\AA}}$'.format(
            np.round(wing_ca - 8542.09), 2
        ),
        transform=axs[0][0].transAxes,
        color='black'
    )
    axs[0][1].text(
        0.01, 0.8,
        r'(b) Ca II 8542 $\mathrm{{\AA}}$ core',
        transform=axs[0][1].transAxes,
        color='white'
    )
    axs[1][0].text(
        0.01, 0.8,
        r'(c) H$\alpha$ +{} $\mathrm{{\AA}}$'.format(
            np.round(wing_ha - 6562.8), 2
        ),
        transform=axs[1][0].transAxes,
        color='black'
    )
    axs[1][1].text(
        0.01, 0.8,
        r'(d) H$\alpha$ core',
        transform=axs[1][1].transAxes,
        color='white'
    )
    im20 = axs[2][0].text(
        0.01, 0.8,
        r'(e) $B_{{\mathrm{LOS}}}$ (Fe I 6569 $\AA$) [G]',
        transform=axs[2][0].transAxes,
        color='black'
    )

    im21 = axs[2][1].text(
        0.01, 0.8,
        r'(e) $B_{{\mathrm{LOS}}}$ (Ca II 8542 $\AA$) [G]',
        transform=axs[2][1].transAxes,
        color='black'
    )

    cbaxes = inset_axes(
        axs[2][0],
        width="50%",
        height="8%",
        loc=1,
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[-300, 0, 300],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='white')

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[2][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[2][1].yaxis.set_minor_locator(MultipleLocator(1))
    #
    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2][1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].tick_params(direction='in', which='both', color='white')
    axs[0][1].tick_params(direction='in', which='both', color='white')
    axs[1][0].tick_params(direction='in', which='both', color='white')
    axs[1][1].tick_params(direction='in', which='both', color='white')
    axs[2][0].tick_params(direction='in', which='both', color='white')
    axs[2][1].tick_params(direction='in', which='both', color='white')

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[1][0].set_xticklabels([])
    axs[1][1].set_xticklabels([])
    axs[2][0].set_xticks([0, 10, 20])
    axs[2][0].set_xticklabels([0, 10, 20])
    axs[2][1].set_xticks([0, 10, 20])
    axs[2][1].set_xticklabels([0, 10, 20])
    axs[0][0].set_yticks([0, 2, 4])
    axs[0][0].set_yticklabels([0, 2, 4])
    axs[1][0].set_yticks([0, 2, 4])
    axs[1][0].set_yticklabels([0, 2, 4])
    axs[2][0].set_yticks([0, 2, 4])
    axs[2][0].set_yticklabels([0, 2, 4])
    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticklabels([])
    axs[2][1].set_yticklabels([])

    axs[0][0].set_ylabel('y [arcsec]')
    axs[1][0].set_ylabel('y [arcsec]')
    axs[2][0].set_ylabel('y [arcsec]')

    axs[2][0].set_xlabel('x [arcsec]')
    axs[2][1].set_xlabel('x [arcsec]')

    plt.subplots_adjust(left=0.05, bottom=0.13, right=1, top=1, hspace=0.0, wspace=0.0)
    plt.show()


if __name__ == '__main__':
    make_fov_plots()
