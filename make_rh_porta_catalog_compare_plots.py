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


if __name__ == '__main__':
    make_comparison_plots(
        [
            # ('H_FALC_11_11_profs.h5', 5, 5, 'output_falc.txt', 'FALC 5x5 PORTA'),
            # ('H_FALC_profs_3x3.h5', 1, 1, 'output_falc.txt', 'FALC 3x3 PORTA'),
            # ('H_FALC_21_21_profs.h5', 10, 10, 'output_falc.txt', 'FALC 21x21 PORTA'),
            # ('H_FALC_61_61_profs.h5', 30, 30, 'output_falc.txt', 'FALC 61x61 PORTA'),
            ('H_Bifrost_200_261_200_261_profs_3.h5', 30, 30, 'wave_ha.txt', 'Bifrost 61x61 PORTA')
        ],
        catalog_plot=True,
        falc_plot=True,
        bifrost_plot=True,
        bifrost_coords=[(230, 230)]
    )
