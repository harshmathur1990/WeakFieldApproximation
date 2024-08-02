import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib


base_input_path = Path(
    '/home/harsh/Spinor_2008/Ca_x_30_y_18_2_20_250_280'
)

second_path = base_input_path / 'plots_v2'
#
profile_files = [

    second_path / 'Ca_x_30_y_18_2_20_250_280_cycle_1_t_5_vl_1_vt_4_blong_1_profs.nc'
]

atmos_files = [
    
    second_path / 'Ca_x_30_y_18_2_20_250_280_cycle_1_t_5_vl_1_vt_4_blong_1_atmos.nc'
]

observed_file = Path(
    base_input_path / 'Ca_x_30_y_18_2_20_250_280.nc'
)

profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
indices = np.where(observed['profiles'][0, 0, 0, :, 0] != 0)[0]
write_path = second_path

wing_indice = 4

core_indice = 1208

vdoppler_wing = np.round(
    (observed['wav'][wing_indice] - 8542.12) * 2.99792458e5 / 8542.12,
    0
)
vdoppler_core = np.round(
    (observed['wav'][core_indice] - 8542.12) * 2.99792458e5 / 8542.12,
    0
)

wing_image = observed['profiles'][0, :, :, wing_indice, 0]

core_image = observed['profiles'][0, :, :, core_indice, 0]

blong_image = atmos[0]['blong'][0, :, :, 0]

vlos_image = atmos[0]['vlos'][0, :, :, 0] / 1e5

i, j = 10, 13

vlos_image[:, :] = vlos_image - vlos_image[i, j]

tau = atmos[0]['ltau500'][0, 0, 0, :]

indices_wing = np.where((tau >= -1) & (tau <= 0))[0]

indices_core = np.where((tau >= -5) & (tau <= -3))[0]

wing_temperature_image = np.mean(
    atmos[0]['temp'][0, :, :, indices_wing],
    2
)

core_temperature_image = np.mean(
    atmos[0]['temp'][0, :, :, indices_core],
    2
)

extent = [250, 280, 2, 20]

plt.close('all')

plt.clf()

plt.cla()

fig, axs = plt.subplots(2, 3, figsize=(11,4))

axs[0][0].imshow(wing_image, cmap='gray', origin='lower', interpolation='none', extent=extent)

axs[0][1].imshow(core_image, cmap='gray', origin='lower', interpolation='none', extent=extent)

im02 = axs[0][2].imshow(blong_image, cmap='gray', origin='lower', interpolation='none', extent=extent)

im10 = axs[1][0].imshow(wing_temperature_image, cmap='hot', origin='lower', interpolation='none', extent=extent)

im11 = axs[1][1].imshow(core_temperature_image, cmap='hot', origin='lower', interpolation='none', extent=extent)

im12 = axs[1][2].imshow(vlos_image, cmap='bwr',interpolation='nearest', extent=extent, vmax=4, vmin=-4, aspect='equal', origin='lower')

mask = np.loadtxt('/home/harsh/Spinor_2008/Ca_x_30_y_18_2_20_250_280/Fe_ME/mask_pore.txt')

mask = mask.astype(np.int64)

X, Y = np.meshgrid(np.arange(250, 280), np.arange(2, 20))

axs[0][0].contour(X, Y, mask, levels=0)

axs[0][1].contour(X, Y, mask, levels=0)

axs[0][2].contour(X, Y, mask, levels=0)

axs[1][0].contour(X, Y, mask, levels=0)

axs[1][1].contour(X, Y, mask, levels=0)

axs[1][2].contour(X, Y, mask, levels=0)

axs[0][0].set_xlabel('(a)')

axs[0][1].set_xlabel('(b)')

axs[0][2].set_xlabel('(c)')

axs[1][0].set_xlabel('(d)')

axs[1][1].set_xlabel('(e)')

axs[1][2].set_xlabel('(f)')

fig.colorbar(im02, ax=axs[0][2])

fig.colorbar(im10, ax=axs[1][0])

fig.colorbar(im11, ax=axs[1][1])

fig.colorbar(im12, ax=axs[1][2])

fig.suptitle('Inversions Result from Ca II 8542 line')

fig.tight_layout()

plt.tight_layout()

plt.savefig(str(write_path / 'fov_map.png'), format='png', dpi=1200)

plt.show()
