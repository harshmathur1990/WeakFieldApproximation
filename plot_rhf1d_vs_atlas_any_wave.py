import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/rhv2src/python')
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
import os
import rh
import numpy as np
import matplotlib.pyplot as plt
import h5py
from stray_light_approximation import *


def make_plot(name, cw):
    catalog = np.loadtxt(
        '/home/harsh/CourseworkRepo/WFAComparison/catalog_{}.txt'.format(
            cw
        )
    )

    os.chdir("/home/harsh/CourseworkRepo/rh/rhv2src/rhf1d/run")

    out = rh.readOutFiles()

    wave = np.array(out.spect.lambda0)

    wave *= 10

    sel_wave_indice = np.where((wave >= cw - 2) & (wave <= cw + 2))

    sel_wave = wave[sel_wave_indice]

    intensity = np.array(out.ray.I)

    get_indice_catalog = prepare_get_indice(catalog[:, 0])

    vec_get_indice_catalog = np.vectorize(get_indice_catalog)

    catalog_indice = vec_get_indice_catalog(sel_wave)

    norm_line, norm_atlas, atlas_wave = normalise_profiles(
        intensity[sel_wave_indice], sel_wave,
        catalog[:, 1][catalog_indice], catalog[:, 0][catalog_indice],
        wave[sel_wave_indice][-1]
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(sel_wave, norm_line, label='synthesis')

    plt.plot(atlas_wave, norm_atlas, label='BASS2000')

    plt.xlabel('Wavelength (Angstrom)')

    plt.ylabel('Normalised Intensity')

    plt.legend()

    fig = plt.gcf()

    fig.set_size_inches(6, 4, forward=True)

    plt.savefig('{}.eps'.format(name), dpi=300, format='eps')

    plt.savefig('{}.png'.format(name), dpi=300, format='png')

    plt.show()
