import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/python/')
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from specutils.utils.wcs_utils import vac_to_air
from astropy import units as u
from pathlib import Path
import rhanalyze

rh_output_path = Path('/home/harsh/BifrostRun/RH_HA_outputs/')
base_path = Path('/home/harsh/BifrostRun/')
rh_bifrost_output = Path('/home/harsh/BifrostRun/Rh15d_Bifrost_output/output_ray.hdf5')

def make_comparison_plots(filename_x_y_list, catalog_plot=True, falc_plot=True, bifrost_plot=False, bifrost_coords=None):

    catalog = np.loadtxt('/home/harsh/CourseworkRepo/WFAComparison/catalog_6563.txt')

    ind_2 = np.where((catalog[:, 0] >= 6559.77655) & (catalog[:, 0] <= 6559.77655 + 1780 * 0.00561))[0]

    cwd = os.getcwd()

    os.chdir(rh_output_path)

    out = rhanalyze.rhout()

    os.chdir(cwd)

    ind_3 = np.where((out.spectrum.waves >= 655.977655) & (out.spectrum.waves <= (6559.77655 + 1780 * 0.00561) / 10))[0]

    wv3 = out.spectrum.waves[ind_3] * 10

    plt.close('all')

    plt.clf()

    plt.cla()

    if falc_plot is True:
        plt.plot(wv3, out.rays[0].I[ind_3] / out.rays[0].I[ind_3][0], label='RH 6 level atom')

    if catalog_plot is True:
        plt.plot(catalog[ind_2, 0], catalog[ind_2, 1] / catalog[ind_2, 1][6], label='BASS 2000')

    for indice, (filename, x, y, wavefile, label) in enumerate(filename_x_y_list):
        f = h5py.File(base_path / filename, 'r')

        wave = np.loadtxt(base_path / wavefile)

        if len(wave.shape) == 2:
            wave = wave[:, 1][::-1]
        else:
            wave = wave[::-1]

        wv = vac_to_air(wave * u.angstrom)

        wv = np.array(wv)

        ind = np.where((wv >= 6559.77655) & (wv <= 6559.77655 + 1780 * 0.00561))[0]

        print(ind)

        plt.plot(wv[ind], f['stokes_I'][ind, x, y] / f['stokes_I'][ind, x, y][-2], label=label)

        f.close()

    if bifrost_plot is True:
        f = h5py.File(rh_bifrost_output, 'r')
        wv = f['wavelength'][()] * 10
        ind = np.where((wv >= 6559.77655) & (wv <= 6559.77655 + 1780 * 0.00561))[0]
        for (xx, yy) in bifrost_coords:
            plt.plot(wv[ind], f['intensity'][xx, yy, ind] / f['intensity'][xx, yy, ind][0], label='Bifrost {}x{} RH'.format(xx, yy))
        f.close()

    plt.xlabel(r'$\lambda$ [$\mathrm{\AA}$]')

    plt.ylabel(r'I/$I_{c}$')

    plt.gcf().set_size_inches(7, 5, forward=True)

    plt.legend()

    write_path = base_path

    plt.gcf().savefig(write_path / 'RH_BASS2000_PORTA.pdf', format='pdf', dpi=300)

    plt.gcf().savefig(write_path / 'RH_BASS2000_PORTA.png', format='png', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


def make_line_core_image(porta_file, wavefile, start_x, end_x, start_y, end_y):
    f = h5py.File(base_path / porta_file, 'r')

    wave = np.loadtxt(base_path / wavefile)

    fa = h5py.File(base_path / 'bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0.nc', 'r')

    if len(wave.shape) == 2:
        wave = wave[:, 1][::-1]
    else:
        wave = wave[::-1]

    wv = vac_to_air(wave * u.angstrom)

    wv = np.array(wv)

    line_core_ind = np.argmin(np.abs(wv - 6562.8))

    print(line_core_ind)

    fb = h5py.File(rh_bifrost_output, 'r')
    wvb = fb['wavelength'][()] * 10

    line_core_bind = np.argmin(np.abs(wvb - 6562.8))

    ind = np.where((wvb >= 6559.77655) & (wvb <= 6559.77655 + 1780 * 0.00561))[0]

    intensity = fb['intensity'][start_x:end_x, start_y:end_y, line_core_bind]

    a, b = np.where(intensity > 2)

    intensity[a, b] = 0

    max_norm = fb['intensity'][start_x:end_x, start_y:end_y, ind]

    a, b, c = np.where(max_norm > 2)

    max_norm[a, b, c] = 0

    maxval = np.max(max_norm)

    intensity /= maxval

    fig, axs = plt.subplots(1, 3, figsize=(7, 3.5))

    axs[0].imshow(f['stokes_I'][line_core_ind] / np.max(f['stokes_I'][-2]), cmap='gray', origin='lower')

    axs[1].imshow(intensity, cmap='gray', origin='lower')

    ind_z = np.argmin(np.abs(fa['z'][0, 0, 0] / 1e3 - 2500))
    axs[2].imshow(fa['temperature'][0, 200:261, 200:261, ind_z], cmap='gray', origin='lower')

    axs[0].set_title('PORTA')

    axs[1].set_title('RH15D')

    axs[0].set_ylabel('pixels')

    axs[0].set_xlabel('pixels')

    axs[1].set_xlabel('pixels')

    plt.subplots_adjust(left=0.05, bottom=0.2, right=1, top=0.9, wspace=0.1, hspace=0.1)

    write_path = base_path

    fig.savefig(write_path / 'RH_PORTA_FOV.pdf', format='pdf', dpi=300)

    fig.savefig(write_path / 'RH_PORTA_FOV.png', format='png', dpi=300)

    plt.show()

    fb.close()

    f.close()



def make_compare_two_frequency_grids():

    base_path = Path('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/')

    f1 = h5py.File(base_path / 'MULTI3D_H_Bifrost_s_385_0_0_3_1_791_profs.h5', 'r')

    f2 = h5py.File(base_path / 'MULTI3D_H_Bifrost_s_385_0_0_3_1_441_profs.h5', 'r')

    print(f1['stokes_I'].shape)

    print(f2['stokes_I'].shape)
    wbpath = Path('/home/harsh/CourseworkRepo/PORTA/tests/multilevel-module/')

    wave_791 = np.loadtxt(wbpath / 'wave_791.txt')

    wave_1141 = np.loadtxt(wbpath / 'wave_441.txt')

    wave_791 = wave_791[:, 1][::-1]

    wave_1141 = wave_1141[:, 1][::-1]

    wv_791 = np.array(vac_to_air(wave_791 * u.angstrom))

    wv_1141 = np.array(vac_to_air(wave_1141 * u.angstrom))

    ind_791 = np.where((wv_791 >= 6560) & (wv_791 <= 6566))[0]

    ind_1141 = np.where((wv_1141 >= 6560) & (wv_1141 <= 6566))[0]

    fig, axs = plt.subplots(2, 1, figsize=(7, 7))

    axs[0].plot(wv_791[ind_791], f1['stokes_I'][ind_791, 1, 1], color='green')

    axs[0].plot(wv_1141[ind_1141], f2['stokes_I'][ind_1141, 1, 1], color='black')

    axs[1].plot(wv_791[ind_791], f1['stokes_V'][ind_791, 1, 1] / f1['stokes_I'][ind_791, 1, 1], color='green')

    axs[1].plot(wv_1141[ind_1141], f2['stokes_V'][ind_1141, 1, 1] / f2['stokes_I'][ind_1141, 1, 1], color='black')

    axs[0].scatter(wv_791[ind_791], f1['stokes_I'][ind_791, 1, 1], color='green', s=6)

    axs[0].scatter(wv_1141[ind_1141], f2['stokes_I'][ind_1141, 1, 1], color='black', s=6)

    axs[1].scatter(wv_791[ind_791], f1['stokes_V'][ind_791, 1, 1] / f1['stokes_I'][ind_791, 1, 1], color='green', s=6)

    axs[1].scatter(wv_1141[ind_1141], f2['stokes_V'][ind_1141, 1, 1] / f2['stokes_I'][ind_1141, 1, 1], color='black',
                   s=6)

    fig.tight_layout()

    fig.savefig(base_path / 'freq_compaare.pdf', format='pdf')


if __name__ == '__main__':
    # make_comparison_plots(
    #     [
    #         ('H_FALC_11_11_profs.h5', 5, 5, 'output_falc.txt', 'FALC 5x5 PORTA'),
    #         ('H_FALC_profs_3x3.h5', 1, 1, 'output_falc.txt', 'FALC 3x3 PORTA'),
    #         ('H_FALC_21_21_profs.h5', 10, 10, 'output_falc.txt', 'FALC 21x21 PORTA'),
    #         ('H_FALC_61_61_profs.h5', 30, 30, 'output_falc.txt', 'FALC 61x61 PORTA'),
            # ('combined_output_profs.h5', 100, 100, 'wave_ha.txt', 'Bifrost 61x61 PORTA')
        # ],
        # catalog_plot=True,
        # falc_plot=True,
        # bifrost_plot=True,
        # bifrost_coords=[(230, 230)]
    # )

    # make_line_core_image('H_Bifrost_180_60_61_3_profs.h5', 'wave_ha.txt', 200, 261, 200, 261)

    make_compare_two_frequency_grids()
