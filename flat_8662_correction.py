import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/rhv2src/python')
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
import h5py
import sunpy.io.fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stray_light_approximation import *


base_path = Path(
    '/home/harsh/Flat Data 8662/'
)

write_path = base_path / 'flat_calculation'

raw_flat_filename = '100306_FLAT.fits'

raw_dark_frame = '110524_DARK.fits'

catalog_base = Path(
    '/home/harsh/CourseworkRepo/WFAComparison'
)

ca_ir_catalog_file = 'catalog_8662.txt'

raw_fringe_frame = '101042_FLAT.fits'

spectral_dispersion_8542 = 10.35 # \AA per binned pixel (13.5 * 2 um)

def get_dark_frame():

    mean_dark = np.zeros((1024, 1024), dtype=np.float64)

    data, header = sunpy.io.fits.read(base_path / raw_dark_frame)[0]

    mean_dark = np.mean(data, 0)

    sunpy.io.fits.write(write_path / 'dark_master.fits', mean_dark, dict(), overwrite=True)

    return mean_dark


def get_dark_corrected_tilt_uncorrected_flat(mean_dark):

    dark_corrected_flat = np.zeros((1024, 1024), dtype=np.float64)

    data, header = sunpy.io.fits.read(base_path / raw_flat_filename)[0]
    all_mod = np.mean(data, 0)

    dark_corrected_flat = all_mod - mean_dark

    return dark_corrected_flat


def get_dark_corrected_tilt_uncorrected_fringe_flat(mean_dark):

    dark_corrected_flat = np.zeros((1024, 1024), dtype=np.float64)

    data, header = sunpy.io.fits.read(base_path / raw_fringe_frame)[0]
    all_mod = np.mean(data, 0)

    dark_corrected_flat = all_mod - mean_dark

    return dark_corrected_flat


def get_x_shift(dark_corrected_flat):

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.figure('Click on the slit profile (0, 1024) to trace')

    plt.imshow(dark_corrected_flat, cmap='gray', origin='lower')

    point = np.array(plt.ginput(50, 600))

    a, b = np.polyfit(point[:, 0], point[:, 1], 1)

    y1 = a * np.arange(1024) + b

    y1 = ((y1.max() + y1.min()) / 2) - y1

    plt.close('all')

    plt.clf()

    plt.cla()

    return y1


def get_y_shift(x_corrected_flat):

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.figure('Click on the line (0, 1024) profile to trace')

    plt.imshow(x_corrected_flat, cmap='gray', origin='lower')

    point = np.array(plt.ginput(50, 600))

    a, b = np.polyfit(point[:, 1], point[:, 0], 1)

    y1 = a * np.arange(1024) + b

    y1 = ((y1.max() + y1.min()) / 2) - y1

    plt.close('all')

    plt.clf()

    plt.cla()

    return y1


def apply_x_shift(dark_corrected_flat, y1):
    result = dark_corrected_flat.copy()

    for i in np.arange(dark_corrected_flat.shape[1]):
        scipy.ndimage.shift(
            dark_corrected_flat[0:dark_corrected_flat.shape[0], i],
            y1[i],
            result[0:dark_corrected_flat.shape[0], i],
            mode='nearest'
        )

    plt.imshow(result, cmap='gray', origin='lower')

    plt.show()

    return result


def apply_y_shift(dark_corrected_flat, y1):
    result = dark_corrected_flat.copy()

    for i in np.arange(dark_corrected_flat.shape[0]):
        scipy.ndimage.shift(
            dark_corrected_flat[i, :],
            y1[i],
            result[i, :],
            mode='nearest'
        )

    plt.imshow(result, cmap='gray', origin='lower')

    plt.show()

    return result


def adhoc_y_shift(dark_corrected_flat, y1):
    result = dark_corrected_flat.copy()

    for i in np.arange(dark_corrected_flat.shape[0]//2):
        scipy.ndimage.shift(
            dark_corrected_flat[i, :],
            y1[i],
            result[i, :],
            mode='nearest'
        )

    plt.imshow(result, cmap='gray', origin='lower')

    plt.show()

    return result


def get_x_y_shifts_and_flat_master():

    mean_dark = get_dark_frame()

    dark_corrected_flat = get_dark_corrected_tilt_uncorrected_flat(mean_dark)

    x_shifts = get_x_shift(dark_corrected_flat)

    x_corrected_flat = apply_x_shift(dark_corrected_flat, x_shifts)

    y_shifts = get_y_shift(x_corrected_flat)

    y_corrected_flat = apply_y_shift(x_corrected_flat, y_shifts)

    np.savetxt(write_path / 'x_shifts.txt', x_shifts)

    np.savetxt(write_path / 'y_shifts.txt', y_shifts)

    dark_corrected_fringe = get_dark_corrected_tilt_uncorrected_fringe_flat(mean_dark)

    x_corrected_fringe = apply_x_shift(dark_corrected_fringe, x_shifts)

    y_corrected_fringe = apply_y_shift(x_corrected_fringe, y_shifts)

    med_prof_im = y_corrected_flat[:-1] / norm_y_corrected_fringe[1:]

    median_profile = np.median(med_prof_im, 0)

    plt.plot(median_profile)

    plt.show()

    np.savetxt(write_path / 'median_profile_uncorrected.txt', median_profile)

    flat_master = y_corrected_flat / median_profile

    plt.imshow(flat_master, cmap='gray', origin='lower')

    plt.show()

    sunpy.io.fits.write(write_path / 'flat_master.fits', flat_master, dict(), overwrite=True)

    return x_shifts, y_shifts, adhoc_shifts_0_512, flat_master


def correct_flat_frames(
    mean_dark,
    x_shifts,
    y_shifts,
    flat_master
):

    data, header = sunpy.io.fits.read(base_path / raw_flat_filename)[0]

    corrected_flat = np.zeros_like(data)

    for i in range(data.shape[0]):
        dark_corrected_mod = np.subtract(
            data[i],
            mean_dark
        )

        x_corrected_mod = apply_x_shift(dark_corrected_mod, x_shifts)

        y_corrected_mod = apply_y_shift(x_corrected_mod, y_shifts)

        flat_corrected_mod = y_corrected_mod / flat_master

        corrected_flat[i] = flat_corrected_mod

    sunpy.io.fits.write(write_path / 'HA_corrected_flat.fits', corrected_flat, dict(), overwrite=True)

    return corrected_flat


def get_corrected_median_profile(corrected_flat):

    corrected_median_profile = np.median(corrected_flat, (0, 1))

    np.savetxt(write_path / 'Corrected_median_profile.txt', corrected_median_profile)

    return np.median(both_beams, (0, 1))


def compare_corrected_median_profile_with_atlas(corrected_median_profile):

    catalog = np.loadtxt(catalog_base / ca_ir_catalog_file)

    a, b = np.polyfit([537, 558], [8662.172, 8661.992], 1)

    wave = a * np.arange(1024) + b

    #wave[1] - wave[0] = 0.0085
    norm_line, norm_atlas, _ = normalise_profiles(
        corrected_median_profile[0:800],
        wave[0:800],
        catalog[:, 1],
        catalog[:, 0],
        cont_wave=wave[0]
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(wave[0:800] - 8662.14, norm_line, label='Flat Profile')

    plt.plot(wave[0:800] - 8662.14, norm_atlas, label='BASS 2000 Atlas')

    plt.xlabel(r'$\Delta \lambda (\AA)$')

    plt.ylabel(r'$I/I_{c}$')

    plt.legend()

    fig = plt.gcf()

    fig.set_size_inches(6, 4, forward=True)

    fig.tight_layout()

    fig.savefig(write_path / 'FlatProfileVsBass2000.png', format='png', dpi=300)

    fig.savefig(write_path / 'FlatProfileVsBass2000.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()
