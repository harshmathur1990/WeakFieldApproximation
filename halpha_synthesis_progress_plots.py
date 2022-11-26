import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from weak_field_approx import prepare_calculate_blos_vlos_gradient
from skimage.transform import resize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


def get_data(recalculate=False):

    if not recalculate:
        if os.path.exists('cache.h5'):
            fd = h5py.File('cache.h5', 'r')
            data = fd['data'][()]
            fd.close()
            return data

    base_path_multi3d_no_subs = Path('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/BifrostRun/')
    base_path_multi3d_subs = Path('/home/harsh/BifrostRun_fast_Access/MULTI3D_s_385_x_180_y_60_dimx_180_dim_y_180_step_x_3_step_y_3/')

    f1 = h5py.File(base_path_multi3d_no_subs / 'MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_supplementary_outputs.nc', 'r')

    data = np.zeros((3, 4, 504, 504), dtype=np.float64)

    data[0][0] = f1['profiles_H'][0, :, :, 400, 0]
    data[0][1] = f1['profiles_H'][0, :, :, 400, 3] / f1['profiles_H'][0, :, :, 400, 0]

    f2 = h5py.File(base_path_multi3d_no_subs / 'BIFROST_en024048_hion_snap_385_0_504_0_504_-500000.0_3000000.0_supplementary_outputs.nc', 'r')
    data[0][2] = f2['profiles_CaIR'][0, :, :, 400, 0]
    data[0][3] = f2['profiles_CaIR'][0, :, :, 400, 3] / f2['profiles_CaIR'][0, :, :, 400, 0]

    actual_calculate_blos = prepare_calculate_blos_vlos_gradient(f1['profiles_H'], wavelength_arr=f1['wave_H'][()] / 10, lambda0=656.28, lambda_range=0.35 / 10, g_eff=1.048)
    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
    data[1][0] = np.fromfunction(vec_actual_calculate_blos, shape=(504, 504))

    actual_calculate_blos = prepare_calculate_blos_vlos_gradient(f2['profiles_CaIR'], wavelength_arr=f2['wave_CaIR'][()] / 10, lambda0=854.209, lambda_range=0.4 / 10, g_eff=1.1)
    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
    data[1][1] = np.fromfunction(vec_actual_calculate_blos, shape=(504, 504))

    f3 = h5py.File(base_path_multi3d_subs / 'combined_output_profs.h5', 'r')

    data[1][2] = resize(f3['stokes_I'][528, :, :], (504, 504), order=3)
    data[1][3] = resize(f3['stokes_V'][528, :, :] / f3['stokes_I'][528, :, :], (504, 504), order=3)

    grid = np.arange(-16, 2, 0.1)

    index = np.argmin(np.abs(grid + 11))
    data[2][0] = f2['b_z_ltau_strata'][0, :, :, index] * 1e4

    index = np.argmin(np.abs(grid + 10))
    data[2][1] = f2['b_z_ltau_strata'][0, :, :, index] * 1e4

    data[2][2] = resize(f1['profiles_H'][0, 180:180+183, 60:60+183, 400, 0], (504, 504), order=3)
    data[2][3] = resize(f1['profiles_H'][0, 180:180 + 183, 60:60 + 183, 400, 3] / f1['profiles_H'][0, 180:180 + 183, 60:60 + 183, 400, 0], (504, 504), order=3)

    f1.close()
    f2.close()
    f3.close()

    fd = h5py.File('cache.h5', 'w')
    fd['data'] = data
    fd.close()

    return data


def make_progress_plots():
    data = get_data()

    fontsize = 10

    fig, axs = plt.subplots(3, 4, figsize=(7, 5.25))

    for i in range(3):
        for j in range(4):
            axs[i][j].set_xticks([])
            axs[i][j].set_xticklabels([])
            axs[i][j].set_yticks([])
            axs[i][j].set_yticklabels([])

            plotted = False

            if i == 0:
                if j == 1:
                    im = axs[i][j].imshow(data[i][j] * 100, vmin=-0.6, vmax=0.6, cmap='BrBG', origin='lower')
                    plotted = True
                if j == 3:
                    im = axs[i][j].imshow(data[i][j] * 100, vmin=-13, vmax=13, cmap='BrBG', origin='lower')
                    plotted = True
            if i == 1:
                if j in [0, 1]:
                    im = axs[i][j].imshow(data[i][j], vmin=-600, vmax=600, cmap='Spectral', origin='lower')
                    plotted = True
                if j == 3:
                    im = axs[i][j].imshow(data[i][j] * 100, vmin=-0.3, vmax=0.3, cmap='BrBG', origin='lower')
                    plotted = True
            if i == 2:
                if j in [0, 1]:
                    im = axs[i][j].imshow(data[i][j], vmin=-600, vmax=600, cmap='Spectral', origin='lower')
                    plotted = True
                if j == 3:
                    im = axs[i][j].imshow(data[i][j] * 100, vmin=-0.3, vmax=0.3, cmap='BrBG', origin='lower')
                    plotted = True

            if plotted:
                cbaxes = inset_axes(
                    axs[i][j],
                    width="70%",
                    height="3%",
                    loc=1,
                    borderpad=0.5
                )
                cbar = fig.colorbar(
                    im,
                    cax=cbaxes,
                    # ticks=[-500, 0, 500],
                    orientation='horizontal'
                )

                # cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.tick_params(labelsize=fontsize, colors='black')

                cbar.ax.yaxis.set_ticks_position('left')

            else:
                im = axs[i][j].imshow(data[i][j], cmap='gray', origin='lower')

            if (i == 1 and j in [2, 3]) or (i == 2 and j in [2, 3]):
                pass
            else:
                color = 'black'
                if i == 0 and j in [0, 2]:
                    color = 'white'

                axs[i][j].plot([60, 60+183, 60+183, 60, 60], [180, 180, 180+183, 180+183, 180], color=color)

    axs[0][0].text(
        0.02, 0.13,
        r'(a) H$\alpha$ core Stokes $I$, 3D pops',
        transform=axs[0][0].transAxes,
        fontsize=fontsize - 2,
        color='white'
    )
    axs[0][0].text(
        0.05, 0.05,
        r'with substructure using RH',
        transform=axs[0][0].transAxes,
        fontsize=fontsize - 2,
        color='white'
    )
    axs[0][1].text(
        0.02, 0.13,
        r'(b) H$\alpha$ Stokes $V$/$I$ [%]',
        transform=axs[0][1].transAxes,
        fontsize=fontsize - 2,
        color='black'
    )
    axs[0][1].text(
        0.05, 0.05,
        r'3D pops, substructure RH',
        transform=axs[0][1].transAxes,
        fontsize=fontsize - 2,
        color='black'
    )

    axs[0][2].text(
        0.02, 0.13,
        r'(c) Ca II 8542 $\mathrm{\AA}$ core',
        transform=axs[0][2].transAxes,
        fontsize=fontsize - 2,
        color='white'
    )
    axs[0][2].text(
        0.02, 0.05,
        r'Stokes $I$, RHF1D',
        transform=axs[0][2].transAxes,
        fontsize=fontsize - 2,
        color='white'
    )

    axs[0][3].text(
        0.02, 0.13,
        r'(c) Ca II 8542 $\mathrm{\AA}$',
        transform=axs[0][3].transAxes,
        fontsize=fontsize - 2,
        color='black'
    )
    axs[0][3].text(
        0.02, 0.05,
        r'Stokes $V$/$I$ [%], RHF1D',
        transform=axs[0][3].transAxes,
        fontsize=fontsize - 2,
        color='black'
    )

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.0, hspace=0.0)
    plt.show()


if __name__ == '__main__':
    make_progress_plots()
