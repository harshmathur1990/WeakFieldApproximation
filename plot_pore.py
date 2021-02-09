import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


base_input_path = Path(
    '/Users/harshmathur/Documents/CourseworkRepo/2008 Sp Data/final_data/stic_10_profiles'
)
# profile_files = [
#     base_input_path / 'cycle_1_profs_sw_t_7_vt_0_vl_3.nc',
#     base_input_path / 'cycle_2_profs_w_16_23_t_7_vt_0_vl_3.nc',
#     base_input_path / 'cycle_3_profs_sw_t_7_vt_5_vl_3.nc'
# ]
# atmos_files = [
#     base_input_path / 'cycle_1_atmos_sw_t_7_vt_0_vl_3.nc',
#     base_input_path / 'cycle_2_atmos_w_16_23_t_7_vt_0_vl_3.nc',
#     base_input_path / 'cycle_3_atmos_sw_t_7_vt_5_vl_3.nc'
# ]
# profile_files = [x 
#     base_input_path / 'cycle_1_fl_profs_sw_t_7_vt_0_vl_3.nc',
#     base_input_path / 'cycle_2_fl_profs_cw_t_7_vt_0_vl_3.nc',
#     base_input_path / 'cycle_3_fl_profs_sw_t_7_vt_5_vl_3.nc'
# ]
profile_files = [
    # base_input_path / 'falc_vlos_guess_cycle_1_t_5_vl_2_vt_2_profs.nc',
    # base_input_path / 'v10_falc_vlos_guess_cycle_2_t_5_vl_5_vt_5_profs.nc'
    base_input_path / 'observed_interesting_profiles_ca_ha_t_5_vl_3_vt_3_blong_2_profs.nc'
]
# atmos_files = [
#     base_input_path / 'cycle_1_fl_atmos_sw_t_7_vt_0_vl_3.n',
#     base_input_path / 'cycle_2_fl_atmos_cw_t_7_vt_0_vl_3.nc',
#     base_input_path / 'cycle_3_fl_atmos_sw_t_7_vt_5_vl_3.nc'
# ]
atmos_files = [
    # base_input_path / 'falc_vlos_guess_cycle_1_t_5_vl_2_vt_2_atmos.nc',
    # base_input_path / 'v10_falc_vlos_guess_cycle_2_t_5_vl_5_vt_5_atmos.nc'
    base_input_path / 'observed_interesting_profiles_ca_ha_t_5_vl_3_vt_3_blong_2_atmos.nc'
]
# observed_file = Path(
#     '/Volumes/Harsh 9599771751/Oslo Work/merged_rps.nc'
# )
# observed_file = Path(
#     '/home/harsh/CourseworkRepo/2008 Sp Data/final_data/FOV_19_35_1_20_250_285.nc'
# )
observed_file = Path(
    base_input_path / 'observed_interesting_profiles_ca_ha.nc'
)
# falc_file = Path(
#     base_input_path / 'falc_nicole_for_stic.nc'
# )
# median_prof_file = Path(
#     '/home/harsh/CourseworkRepo/2008 Sp Data/buerro_approach_to_stray_light/corrected_data_buererro_for_stic.nc'
# )

# median_atmos_file = Path(
#     '/home/harsh/CourseworkRepo/2008 Sp Data/buerro_approach_to_stray_light/no_vturb_initial/stic_try/plots_v2/falc_no_vturb_cycle_1_t_5_vl_2_vt_2_atmos.nc'
# )
profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
# falc = h5py.File(falc_file, 'r')
indices_ca_non_zero = np.arange(4, 1860, 4)
indices_ha_non_zero = np.arange(4, 1804, 4) + 1860
indices_ca = np.arange(1860)
indices_ha = np.arange(1804) + 1860
# median_prof = h5py.File(median_prof_file, 'r')
# median_atmos = h5py.File(median_atmos_file, 'r')
# write_path = Path(
#     base_input_path / 'plots_v5'
# )
write_path = base_input_path # / 'plots_alternate'

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

coordinates = [
    (10, 258),
    (11, 256),
    (11, 264),
    (12, 262),
    (12, 263),
    (12, 264),
    (13, 262),
    (13, 263),
    (13, 269),
    (19, 275)
]
# for i in range(1, 20):
#     for j in range(250, 285):

for j in range(10):

        i = 1
        plt.close('all')
        plt.clf()
        plt.cla()

        fig, axs = plt.subplots(2, 4)

        # ------- Cycle 1 --------------------

        # plotting the observed profile
        axs[0][0].plot(
            observed['wav'][indices_ca_non_zero],
            observed['profiles'][0, i-1, j, indices_ca_non_zero, 0],
            color=red,
            linewidth=0.5
        )
        #import ipdb;ipdb.set_trace()
        # plotting the inverted profile
        axs[0][0].plot(
            profiles[0]['wav'][indices_ca],
            profiles[0]['profiles'][0, i-1, j, indices_ca, 0],
            color=green,
            linewidth=0.5
        )


        axs[0][0].set_ylim(0, 1)

        axs[1][0].plot(
            observed['wav'][indices_ha_non_zero],
            observed['profiles'][0, i-1, j, indices_ha_non_zero, 0],
            color=red,
            linewidth=0.5
        )
        #import ipdb;ipdb.set_trace()
        # plotting the inverted profile
        axs[1][0].plot(
            profiles[0]['wav'][indices_ha],
            profiles[0]['profiles'][0, i-1, j, indices_ha, 0],
            color=green,
            linewidth=0.5
        )


        axs[1][0].set_ylim(0, 1)

        


        axs[0][1].plot(
            observed['wav'][indices_ca_non_zero],
            observed['profiles'][0, i-1, j, indices_ca_non_zero, 3] / observed['profiles'][0, i-1, j, indices_ca_non_zero, 0],
            color=red,
            linewidth=0.5
        )
        #import ipdb;ipdb.set_trace()
        # plotting the inverted profile
        axs[0][1].plot(
            profiles[0]['wav'][indices_ca],
            profiles[0]['profiles'][0, i-1, j, indices_ca, 3] / profiles[0]['profiles'][0, i-1, j, indices_ca, 0],
            color=green,
            linewidth=0.5
        )


        axs[1][1].plot(
            observed['wav'][indices_ha_non_zero],
            observed['profiles'][0, i-1, j, indices_ha_non_zero, 3] / observed['profiles'][0, i-1, j, indices_ha_non_zero, 0],
            color=red,
            linewidth=0.5
        )
        #import ipdb;ipdb.set_trace()
        # plotting the inverted profile
        axs[1][1].plot(
            profiles[0]['wav'][indices_ha],
            profiles[0]['profiles'][0, i-1, j, indices_ha, 3] / profiles[0]['profiles'][0, i-1, j, indices_ha, 0],
            color=green,
            linewidth=0.5
        )


        
        # plot inverted temperature profile
        axs[0][2].plot(
            atmos[0]['ltau500'][0][i-1][j],
            atmos[0]['temp'][0][i-1][j],
            color=green,
            linewidth=0.5
        )
       

        # plot inverted Vlos profile
        axs[0][3].plot(
            atmos[0]['ltau500'][0][i-1][j],
            atmos[0]['vlos'][0][i-1][j] / 1e5,
            color=green,
            linewidth=0.5
        )

        # plot inverted Vturb profile
        axs[1][2].plot(
            atmos[0]['ltau500'][0][i-1][j],
            atmos[0]['vturb'][0][i-1][j] / 1e5,
            color=green,
            linewidth=0.5
        )

        axs[1][3].plot(
            atmos[0]['ltau500'][0][i-1][j],
            atmos[0]['blong'][0][i-1][j],
            color=green,
            linewidth=0.5
        )

        fig.tight_layout()

        # plt.show()
        plt.savefig(
            write_path / 'plot_{}_{}.png'.format(*coordinates[j]),
            format='png',
            dpi=1200
        )

