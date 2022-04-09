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


def make_comparison_plots(filename_x_y_list):

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

    plt.plot(wv3, out.rays[0].I[ind_3] / out.rays[0].I[ind_3][0], label='RH 6 level atom')

    plt.plot(catalog[ind_2, 0], catalog[ind_2, 1] / catalog[ind_2, 1][6], label='BASS 2000')

    for (filename, x, y, name) in filename_x_y_list:
        f = h5py.File(base_path / filename, 'r')

        wave = np.loadtxt(base_path / 'output_falc.txt')

        wave = wave[:, 1][::-1]

        wv = vac_to_air(wave * u.angstrom)

        wv = np.array(wv)

        ind = np.where((wv >= 6559.77655) & (wv <= 6559.77655 + 1780 * 0.00561))[0]

        plt.plot(wv[ind], f['stokes_I'][ind, x, y] / f['stokes_I'][ind, x, y][-2], label='FALC {} PORTA'.format(name))

        plt.xlabel(r'$\lambda$ [$\mathrm{\AA}$]')

        plt.ylabel(r'I/$I_{c}$')

        plt.legend()

        plt.gcf().set_size_inches(7, 5, forward=True)

        f.close()

    write_path = base_path

    plt.gcf().savefig(write_path / 'RH_BASS2000_PORTA.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


if __name__ == '__main__':
    make_comparison_plots(
        [
            ('H_FALC_profs_11x11.h5', 5, 5, '11x11'),
            ('H_FALC_profs_3x3.h5', 1, 1, '3x3')
        ]
    )
