import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

    fig, axs = plt.subplots(3, 2, figsize=(7, 7*51/120))

    extent = [0, 22.8, 0, 6.46]

    for i in range(3):
        for j in range(2):
            axs[i][j].imshow(data[i][j], cmap='gray', origin='lower', extent=extent)
            # axs[i][j].text(
            #
            # )

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

    plt.subplots_adjust(left=0.15, bottom=0.15, right=1, top=1, hspace=0.0, wspace=0.0)
    plt.show()


if __name__ == '__main__':
    make_fov_plots()
