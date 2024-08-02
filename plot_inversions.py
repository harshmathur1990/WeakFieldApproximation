import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# from model_prof_tools import *


# base_input_path = Path(
#     '/Users/harshmathur/CourseworkRepo/2008 Sp Data'
# )
base_input_path = Path(
    '/Volumes/Harsh 9599771751/Oslo Work'
)

vlos_negative = base_input_path / 'vlos_negative'
profile_files = [
    vlos_negative / 'cycle_1_vlos_initial_profs_sw_t_7_vt_0_vl_1.nc',
    vlos_negative / 'cycle_2_vlos_initial_profs_cw_t_7_vt_0_vl_1.nc',
    vlos_negative / 'cycle_3_alt_vlos_initial_profs_sw_t_7_vt_0_vl_2.nc',
    vlos_negative / 'cycle_4_alt_vlos_initial_profs_sw_t_7_vt_0_vl_3.nc',
    vlos_negative / 'cycle_5_alt_vlos_initial_profs_sw_t_7_vt_5_vl_3.nc'
]
atmos_files = [
    vlos_negative / 'cycle_1_vlos_initial_atmos_sw_t_7_vt_0_vl_1.nc',
    vlos_negative / 'cycle_2_vlos_initial_atmos_cw_t_7_vt_0_vl_1.nc',
    vlos_negative / 'cycle_3_alt_vlos_initial_atmos_sw_t_7_vt_0_vl_2.nc',
    vlos_negative / 'cycle_4_alt_vlos_initial_atmos_sw_t_7_vt_0_vl_3.nc',
    vlos_negative / 'cycle_5_alt_vlos_initial_atmos_sw_t_7_vt_5_vl_3.nc'
]
# profile_files = [
#     fts_atlas_path / 'fts_atlas_cycle_1_t_7_vt_0_vlos_0_profs.pro',
#     fts_atlas_path / 'fts_atlas_cycle_2_cw_t_7_vt_0_vlos_3_profs.pro',
#     fts_atlas_path / 'fts_atlas_cycle_3_t_7_vt_5_vlos_3_profs.pro'
# ]
# atmos_files = [
#     fts_atlas_path / 'fts_atlas_cycle_1_t_7_vt_0_vlos_0_atmos.mod',
#     fts_atlas_path / 'fts_atlas_cycle_2_cw_t_7_vt_0_vlos_3_atmos.mod',
#     fts_atlas_path / 'fts_atlas_cycle_3_t_7_vt_5_vlos_3_atmos.mod'
# ]
observed_file = base_input_path / 'merged_rps.nc'
falc_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/model_atmos/falc_nicole_for_stic.nc'
)

# filetype_prof, nx, ny, nlam = check_prof(str(profile_files[0]))
# filetype_model, nx, ny, nz = check_model(str(atmos_files[0]))

# profiles = [
#     np.array(
#         read_prof(
#             filename,
#             filetype_prof,
#             nx, ny, nlam, 0, 0
#         )
#     ) for filename in profile_files
# ]
# atmos = [
#     np.array(
#         read_model(
#             filename,
#             filetype_model,
#             nx, ny, nz, 0, 0
#         )
#     ) for filename in atmos_files
# ]
# i_index = np.arange(0, 1856, 4)

profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
falc = h5py.File(falc_file, 'r')
indices = np.where(observed['profiles'][0, 0, 0, :-1, 0] != 0)[0]
write_path = Path(
    vlos_negative / 'plot_alternate'
)

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

for i in range(45):
    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(5, 4)
    # ------- Cycle 1 --------------------

    for j in range(5):
        # plotting the observed profile
        axs[j][0].plot(
            observed['wav'][indices],
            observed['profiles'][0, 0, i, indices, 0],
            color=red,
            linewidth=0.5
        )

        # plotting the inverted profile
        axs[j][0].plot(
            profiles[j]['wav'][:-1],
            profiles[j]['profiles'][0, 0, i, :-1, 0],
            color=green,
            linewidth=0.5
        )

        axs[j][0].plot(
            observed['wav'][indices],
            observed['profiles'][0, 0, 3, indices, 0],
            color=brown,
            linewidth=0.5
        )

        axs[j][0].set_ylim(0, 0.3)

        # plot FALC temperature profile
        axs[j][1].plot(
            falc['ltau500'][0][0][0],
            falc['temp'][0][0][0],
            color=red,
            linewidth=0.5
        )

        # plot inverted temperature profile
        axs[j][1].plot(
            atmos[j]['ltau500'][0][0][i],
            atmos[j]['temp'][0][0][i],
            color=green,
            linewidth=0.5
        )

        axs[j][1].plot(
            atmos[j]['ltau500'][0][0][3],
            atmos[j]['temp'][0][0][3],
            color=brown,
            linewidth=0.5
        )

        axs[j][1].set_ylim(3400, 11000)

        # plot FALC Vlos profile
        axs[j][2].plot(
            falc['ltau500'][0][0][0],
            falc['vlos'][0][0][0] / 1e5,
            color=red,
            linewidth=0.5
        )

        # plot inverted Vlos profile
        axs[j][2].plot(
            atmos[j]['ltau500'][0][0][i],
            atmos[j]['vlos'][0][0][i] / 1e5,
            color=green,
            linewidth=0.5
        )

        axs[j][2].plot(
            atmos[j]['ltau500'][0][0][3],
            atmos[j]['vlos'][0][0][3] / 1e5,
            color=brown,
            linewidth=0.5
        )

        axs[j][2].set_ylim(-5, 11)

        # plot FALC Vturb profile
        axs[j][3].plot(
            falc['ltau500'][0][0][0],
            falc['vturb'][0][0][0] / 1e5,
            color=red,
            linewidth=0.5
        )

        # plot inverted Vturb profile
        axs[j][3].plot(
            atmos[j]['ltau500'][0][0][i],
            atmos[j]['vturb'][0][0][i] / 1e5,
            color=green,
            linewidth=0.5
        )

        axs[j][3].plot(
            atmos[j]['ltau500'][0][0][3],
            atmos[j]['vturb'][0][0][3] / 1e5,
            color=brown,
            linewidth=0.5
        )

        axs[j][3].set_ylim(0, 8)

    fig.tight_layout()

    plt.savefig(write_path / 'plot_{}.png'.format(i), format='png', dpi=300)

    # ------- Cycle 1 --------------------

    # ------- Cycle 2 --------------------

    # plotting the observed profile
    # axs[1][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, i, indices, 0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plotting the inverted profile
    # axs[1][0].plot(
    #     observed['wav'],
    #     profiles[1][0, 0, i, :, 0],
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[1][0].plot(
    # #     observed['wavelength'][indices],
    # #     observed['profiles'][0, 0, 3, :, 0][indices],
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # # axs[1][0].set_ylim(0, 0.3)
    # # plot FALC temperature profile
    # axs[1][1].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['temp'][0][0][0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted temperature profile
    # axs[1][1].plot(
    #     atmos[1][150:300],
    #     atmos[1][300:450],
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[1][1].plot(
    # #     atmos[1]['ltau500'][0][0][3],
    # #     atmos[1]['temp'][0][0][3],
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # axs[1][1].set_ylim(3400, 11000)
    # # plot FALC Vlos profile
    # axs[1][2].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vlos'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vlos profile
    # axs[1][2].plot(
    #     atmos[1][150:300],
    #     atmos[1][900:1050] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[1][2].plot(
    # #     atmos[1]['ltau500'][0][0][3],
    # #     atmos[1]['vlos'][0][0][3] / 1e5,
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # axs[1][2].set_ylim(-20, 100)

    # # plot FALC Vturb profile
    # axs[1][3].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vturb'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vturb profile
    # axs[1][3].plot(
    #     atmos[1][150:300],
    #     atmos[1][1050:1200] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[1][3].plot(
    # #     atmos[1]['ltau500'][0][0][3],
    # #     atmos[1]['vturb'][0][0][3] / 1e5,
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # axs[1][3].set_ylim(0, 80)

    # # ------- Cycle 2 --------------------

    # # ------- Cycle 3 --------------------

    # # plotting the observed profile
    # axs[2][0].plot(
    #     observed['wavelength'],
    #     observed['intensity'][0, 0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plotting the inverted profile
    # axs[2][0].plot(
    #     observed['wavelength'],
    #     profiles[2][i_index],
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[2][0].plot(
    # #     observed['wavelength'][indices],
    # #     observed['profiles'][0, 0, 3, :, 0][indices],
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # # axs[2][0].set_ylim(0, 0.3)

    # # plot FALC temperature profile
    # axs[2][1].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['temp'][0][0][0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted temperature profile
    # axs[2][1].plot(
    #     atmos[2][150:300],
    #     atmos[2][300:450],
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[2][1].plot(
    # #     atmos[2]['ltau500'][0][0][3],
    # #     atmos[2]['temp'][0][0][3],
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # axs[2][1].set_ylim(3400, 11000)

    # # plot FALC Vlos profile
    # axs[2][2].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vlos'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vlos profile
    # axs[2][2].plot(
    #     atmos[2][150:300],
    #     atmos[2][900:1050] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[2][2].plot(
    # #     atmos[2]['ltau500'][0][0][3],
    # #     atmos[2]['vlos'][0][0][3] / 1e5,
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # axs[2][2].set_ylim(-20, 100)

    # # plot FALC Vturb profile
    # axs[2][3].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vturb'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vturb profile
    # axs[2][3].plot(
    #     atmos[2][150:300],
    #     atmos[2][1050:1200] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # # axs[2][3].plot(
    # #     atmos[2]['ltau500'][0][0][3],
    # #     atmos[2]['vturb'][0][0][3] / 1e5,
    # #     color=brown,
    # #     linewidth=0.5
    # # )

    # axs[2][3].set_ylim(0, 80)

    # ------- Cycle 3 --------------------

    # fig.tight_layout()

    # plt.savefig(write_path / 'inversion.png', format='png', dpi=300)
