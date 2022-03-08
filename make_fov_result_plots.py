import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

ca_ha_data_file = processed_inputs / 'aligned_Ca_Ha_stic_profiles.nc'

me_data_file = processed_inputs / 'me_results_6569.nc'

wfa_8542_data_file = processed_inputs / 'me_results_6569.nc'


def get_fov_data():
    data = np.zeros((2, 3, 17, 60), dtype=np.float64)

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
    data[1][0] = fcaha['profiles'][0, :, :, ind_6563_core, 0]
    data[0][2] = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])
    data[1][2] = fwfa['blos_gauss'][()]
    fwfa.close()

    fme.close()

    fcaha.close()

    return data, fcaha['wav'][ind_8542_wing], fcaha['wav'][ind_8542_core], fcaha['wav'][ind_6563_wing], fcaha['wav'][ind_6563_core]


def make_fov_plots():
    data, wing_ca, core_ca, wing_ha, core_ha = get_fov_data()

    fig, axs = plt.subplots(2, 3, figsize=(7, 7 * 120/51))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.0, wspace=0.0)

    for i in range(2):
        for j in range(3):
            pass