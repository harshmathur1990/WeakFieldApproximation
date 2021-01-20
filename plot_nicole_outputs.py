import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from model_prof_tools import *

base_input_path = Path(
    '/home/harsh/Spinor_2008/pore_nicole_v1'
)

profile_files = [
    base_input_path / 'pore_cycle_1_t_5_vl_1_vt_1_bz_0_profs.mod',
    base_input_path / 'pore_cycle_2_t_5_vl_1_vt_1_bz_1_profs.mod',
    base_input_path / 'pore_cycle_3_t_5_vl_2_vt_2_bz_2_profs.mod'
]

atmos_files = [
    base_input_path / 'pore_cycle_1_t_5_vl_1_vt_1_bz_0_atmos.h5',
    base_input_path / 'pore_cycle_2_t_5_vl_1_vt_1_bz_1_atmos.h5',
    base_input_path / 'pore_cycle_3_t_5_vl_2_vt_2_bz_2_atmos.h5'
    
]

observed_file = Path(
    base_input_path / 'PoreProfile_Nicole.nc'
)
falc_file = Path(
    '/home/harsh/OsloAnalysis/falc_nicole_for_stic.nc'
)
# median_prof_file = Path(
#     '/home/harsh/CourseworkRepo/2008 Sp Data/buerro_approach_to_stray_light/corrected_data_buererro_for_stic.nc'
# )

# median_atmos_file = Path(
#     '/home/harsh/CourseworkRepo/2008 Sp Data/buerro_approach_to_stray_light/no_vturb_initial/stic_try/plots_v2/falc_no_vturb_cycle_1_t_5_vl_2_vt_2_atmos.nc'
# )
filetype, nx, ny, nlam = check_prof(profile_files[0])
profiles = np.zeros((len(profile_files), nx, ny, nlam, 4))
for index, profile_file in enumerate(profile_files):
    for ix in range(nx):
        for iy in range(ny):
            profiles[index][ix][iy] = np.array(
                read_prof(
                    profile_file,
                    filetype,
                    nx,
                    ny,
                    nlam,
                    ix,
                    iy
                )
            ).reshape(nlam, 4)
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
falc = h5py.File(falc_file, 'r')
indices = np.where(observed['intensity'][0, 0, :] != 0)[0]
# median_prof = h5py.File(median_prof_file, 'r')
# median_atmos = h5py.File(median_atmos_file, 'r')
# write_path = Path(
#     base_input_path / 'plots_v5'
# )
write_path = base_input_path # / 'plots_alternate'

red = '#ff4646'
pink = '#ff8585'
brown = '#ffb396'
yellow = '#fff5c0'
green = '#295939'
dark_brown = '#433520'

size = plt.rcParams['lines.markersize']

fontP = FontProperties()
fontP.set_size('xx-small')

color_list = [pink, brown, green]

for i in range(1):

    for j in range(1):

        plt.close('all')
        plt.clf()
        plt.cla()

        fig, axs = plt.subplots(3, 2)

        axs[0][0].plot(
            observed['wavelength'][indices],
            observed['intensity'][i, j, indices],
            color=red,
            linewidth=0.01,
            label='Observed'
        )

        axs[0][0].scatter(
            observed['wavelength'][indices],
            observed['intensity'][i, j, indices],
            color=red,
            s=size / 500
        )

        for k in range(3):
            axs[0][0].plot(
                observed['wavelength'][()],
                profiles[k, i, j, :, 0],
                color=color_list[k],
                linewidth=0.01,
                label='cycle {}'.format(k + 1)
            )

            axs[0][0].scatter(
                observed['wavelength'][()],
                profiles[k, i, j, :, 0],
                color=color_list[k],
                s=size / 500
            )

        axs[0][0].set_ylim(0, 1)

        axs[0][1].plot(
            observed['wavelength'][indices],
            observed['stokes_V'][i, j, indices] / observed['intensity'][i, j, indices],
            color=red,
            linewidth=0.5
        )

        for k in range(3):
            axs[0][1].plot(
                observed['wavelength'][()],
                profiles[k, i, j, :, 3] / profiles[k, i, j, :, 0],
                color=color_list[k],
                linewidth=0.5
            )

        for k in range(3):
            axs[1][0].plot(
                atmos[k]['tau'][i, j],
                atmos[k]['temperature'][i, j],
                color=color_list[k],
                linewidth=0.5
            )

        axs[1][0].set_ylim(3000, 20000)

        for k in range(3):
            axs[1][1].plot(
                atmos[k]['tau'][i, j],
                atmos[k]['velocity_z'][i, j] / 1e5,
                color=color_list[k],
                linewidth=0.5
            )

        axs[1][1].set_ylim(-5, 5)
        
        for k in range(3):
            axs[2][0].plot(
                atmos[k]['tau'][i, j],
                atmos[k]['velocity_turbulent'][i, j] / 1e5,
                color=color_list[k],
                linewidth=0.5
            )

        for k in range(3):
            axs[2][1].plot(
                atmos[k]['tau'][i, j],
                atmos[k]['B_z'][i, j],
                color=color_list[k],
                linewidth=0.5
            )

        fig.tight_layout()

        axs[0][0].legend(loc='upper right', prop=fontP)

        plt.savefig(
            write_path / 'plot_{}_{}.png'.format(i, j),
            format='png',
            dpi=1200
        )

