import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


base_input_path = Path(
    '/home/harsh/CourseworkRepo/2008 Sp Data/final_data/Ca_x_30_y_18_2_20_250_280'
)

second_path = base_input_path / 'plots_v1'

profile_files = [
    second_path / 'Ca_x_30_y_18_2_20_250_280_cycle_1_t_5_vl_1_vt_4_blong_1_profs.nc'
]

atmos_files = [
    # base_input_path / 'falc_vlos_guess_cycle_1_t_5_vl_2_vt_2_atmos.nc',
    # base_input_path / 'v10_falc_vlos_guess_cycle_2_t_5_vl_5_vt_5_atmos.nc'
    second_path / 'Ca_x_30_y_18_2_20_250_280_cycle_1_t_5_vl_1_vt_4_blong_1_atmos.nc'
]

observed_file = Path(
    base_input_path / 'Ca_x_30_y_18_2_20_250_280.nc'
)

profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
indices = np.where(observed['profiles'][0, 0, 0, :, 0] != 0)[0]

# indices_ca_non_zero = np.arange(4, 1860, 4)
# indices_ha_non_zero = np.arange(4, 1804, 4) + 1860
# indices_ca = np.arange(1860)
# indices_ha = np.arange(1804) + 1860

write_path = second_path # / 'plots_alternate'

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

# coordinates = [
#     (10, 258),
#     (11, 256),
#     (11, 264),
#     (12, 262),
#     (12, 263),
#     (12, 264),
#     (13, 262),
#     (13, 263),
#     (13, 269),
#     (19, 275)
# ]

for i in range(18):
    for j in range(30):

# for j in range(10):

        plt.close('all')
        plt.clf()
        plt.cla()

        fig, axs = plt.subplots(3, 2)

        # ------- Cycle 1 --------------------

        # plotting the observed profile
        axs[0][0].plot(
            observed['wav'][indices],
            observed['profiles'][0, i, j, indices, 0],
            color=red,
            linewidth=0.5
        )

        axs[0][0].plot(
            profiles[0]['wav'][()],
            profiles[0]['profiles'][0, i, j, :, 0],
            color=green,
            linewidth=0.5
        )

        axs[0][0].set_ylim(0, 1)

        axs[0][1].plot(
            observed['wav'][indices],
            observed['profiles'][0, i, j, indices, 3] / observed['profiles'][0, i, j, indices, 0],
            color=red,
            linewidth=0.5
        )

        axs[0][1].plot(
            profiles[0]['wav'][()],
            profiles[0]['profiles'][0, i, j, :, 3] / profiles[0]['profiles'][0, i, j, :, 0],
            color=green,
            linewidth=0.5
        )

        # plot inverted temperature profile
        axs[1][0].plot(
            atmos[0]['ltau500'][0][i][j],
            atmos[0]['temp'][0][i][j],
            color=green,
            linewidth=0.5
        )

        # plot inverted Vlos profile
        axs[1][1].plot(
            atmos[0]['ltau500'][0][i][j],
            atmos[0]['vlos'][0][i][j] / 1e5,
            color=green,
            linewidth=0.5
        )

        # plot inverted Vturb profile
        axs[2][0].plot(
            atmos[0]['ltau500'][0][i][j],
            atmos[0]['vturb'][0][i][j] / 1e5,
            color=green,
            linewidth=0.5
        )

        axs[2][1].plot(
            atmos[0]['ltau500'][0][i][j],
            atmos[0]['blong'][0][i][j],
            color=green,
            linewidth=0.5
        )

        fig.tight_layout()

        # plt.show()
        plt.savefig(
            write_path / 'plot_{}_{}.png'.format(i + 2, j + 250),
            format='png',
            dpi=1200
        )

