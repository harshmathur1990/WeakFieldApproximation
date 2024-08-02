import sys
import h5py
import sunpy.io.fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stray_light_approximation import *


base_path = Path(
    '/Volumes/HarshHDD-Data/Documents/CourseworkRepo/2008 Sp Data/'
)

write_path = base_path / 'halpha_flat_master_calc'

raw_flat_filename = '081204.093615.0.ccc01.c-hrt.flat01.flip.fits'

ha_wave_file = 'wave_halpha.txt'

catalog_base = Path(
    '/Volumes/HarshHDD-Data/Documents/CourseworkRepo/WFAComparison'
)

ha_catalog_file = 'catalog_6563.txt'

mu_1_file = 'falc_nicole_for_stic_mu_1_ha.nc'

mu_0p536_file = 'falc_nicole_for_stic_mu_0p536_ha.nc'

ha_data_file = 'alignedspectra_scan1_map01_Ha.fits'

def get_dark_frame():

    mean_dark = np.zeros((1024, 512), dtype=np.float64)

    for i in [1, 2, 3, 4, 21, 22, 23, 24]:
        data, header = sunpy.io.fits.read(base_path / raw_flat_filename)[i]
        all_mod = np.mean(data, 0)
        mean_dark = np.add(mean_dark, all_mod)

    mean_dark /= 8

    sunpy.io.fits.write(write_path / 'dark_master.fits', mean_dark, dict(), overwrite=True)

    return mean_dark


def get_dark_corrected_tilt_uncorrected_flat(mean_dark):

    dark_corrected_flat = np.zeros((1024, 512), dtype=np.float64)

    for i in range(5, 21):
        data, header = sunpy.io.fits.read(base_path / raw_flat_filename)[i]
        all_mod = np.mean(data, 0)
        dark_corrected_flat = np.add(
            dark_corrected_flat,
            np.subtract(
                all_mod,
                mean_dark
            )
        )

    dark_corrected_flat /= 16

    return dark_corrected_flat


def get_x_shift(dark_corrected_flat):
    plt.figure('Click on the slit profile (0, 512) to trace')

    plt.imshow(dark_corrected_flat, cmap='gray', origin='lower')

    point = np.array(plt.ginput(2, 600))

    a, b = np.polyfit(point[:, 0], point[:, 1], 1)

    y1 = a * np.arange(512) + b

    y1 = ((y1.max() + y1.min()) / 2) - y1

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.figure('Click on the slit profile (512, 1024) to trace')

    plt.imshow(dark_corrected_flat, cmap='gray', origin='lower')

    point = np.array(plt.ginput(2, 600))

    a, b = np.polyfit(point[:, 0], point[:, 1], 1)

    y2 = a * np.arange(512) + b

    y2 = ((y2.max() + y2.min()) / 2) - y2

    plt.close('all')

    plt.clf()

    plt.cla()

    return y1, y2


def get_y_shift(x_corrected_flat):
    plt.figure('Click on the line (0, 512) profile to trace')

    plt.imshow(x_corrected_flat, cmap='gray', origin='lower')

    point = np.array(plt.ginput(2, 600))

    a, b = np.polyfit(point[:, 1], point[:, 0], 1)

    y1 = a * np.arange(512) + b

    y1 = ((y1.max() + y1.min()) / 2) - y1

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.figure('Click on the line (512, 1024) profile to trace')

    plt.imshow(x_corrected_flat, cmap='gray', origin='lower')

    point = np.array(plt.ginput(2, 600))

    a, b = np.polyfit(point[:, 1], point[:, 0], 1)

    y2 = a * np.arange(512, 1024) + b

    y2 = ((y2.max() + y2.min()) / 2) - y2

    plt.close('all')

    plt.clf()

    plt.cla()

    return y1, y2


def apply_x_shift(dark_corrected_flat, y1, y2):
    result = dark_corrected_flat.copy()

    for i in np.arange(dark_corrected_flat.shape[1]):
        scipy.ndimage.shift(
            dark_corrected_flat[0:dark_corrected_flat.shape[0]//2, i],
            y1[i],
            result[0:dark_corrected_flat.shape[0]//2, i],
            mode='nearest'
        )

    for i in np.arange(dark_corrected_flat.shape[1]):
        scipy.ndimage.shift(
            dark_corrected_flat[dark_corrected_flat.shape[0]//2:, i],
            y2[i],
            result[dark_corrected_flat.shape[0]//2:, i],
            mode='nearest'
        )


    plt.imshow(result, cmap='gray', origin='lower')

    plt.show()

    return result


def apply_y_shift(dark_corrected_flat, y1, y2):
    result = dark_corrected_flat.copy()

    for i in np.arange(dark_corrected_flat.shape[0]//2):
        scipy.ndimage.shift(
            dark_corrected_flat[i, :],
            y1[i],
            result[i, :],
            mode='nearest'
        )

    for i in np.arange(dark_corrected_flat.shape[0]//2, dark_corrected_flat.shape[0]):
        scipy.ndimage.shift(
            dark_corrected_flat[i, :],
            y2[i-dark_corrected_flat.shape[0]//2],
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

    x_corrected_flat = apply_x_shift(dark_corrected_flat, *x_shifts)

    y_shifts = get_y_shift(x_corrected_flat)

    y_corrected_flat = apply_y_shift(x_corrected_flat, *y_shifts)

    adhoc_shifts_0_512 = np.ones(512) * -6

    adhoc_shifted = adhoc_y_shift(y_corrected_flat, adhoc_shifts_0_512)

    np.savetxt(write_path / 'x_shifts.txt', x_shifts)

    np.savetxt(write_path / 'y_shifts.txt', y_shifts)

    np.savetxt(write_path / 'adhoc_y_0_512_shift.txt', adhoc_shifts_0_512)

    cropped_frame = np.zeros((800, 512), dtype=np.float64)

    cropped_frame[0:400] = adhoc_shifted[30:430]

    cropped_frame[400:] = adhoc_shifted[520:920]

    median_profile = np.median(cropped_frame, 0)

    plt.plot(median_profile)

    plt.show()

    np.savetxt(write_path / 'median_profile_uncorrected.txt', median_profile)

    flat_master = adhoc_shifted / median_profile

    plt.imshow(flat_master, cmap='gray', origin='lower')

    plt.show()

    sunpy.io.fits.write(write_path / 'flat_master.fits', flat_master, dict(), overwrite=True)

    sunpy.io.fits.write(write_path / 'only_shift_corrected_but_not_flat_master_corrected.fits', adhoc_shifted, dict(), overwrite=True)

    return x_shifts, y_shifts, adhoc_shifts_0_512, flat_master


def correct_flat_frames(
    mean_dark,
    x_shifts,
    y_shifts,
    adhoc_shifts_0_512,
    flat_master
):
    
    corrected_flat = np.zeros((16, 1024, 512), dtype=np.float64)

    for i in range(5, 21):
        data, header = sunpy.io.fits.read(base_path / raw_flat_filename)[i]
        all_mod = np.mean(data, 0)
        dark_corrected_mod = np.subtract(
            all_mod,
            mean_dark
        )

        x_corrected_mod = apply_x_shift(dark_corrected_mod, *x_shifts)

        y_corrected_mod = apply_y_shift(x_corrected_mod, *y_shifts)

        adhoc_corrected_mod = adhoc_y_shift(y_corrected_mod, adhoc_shifts_0_512)

        flat_corrected_mod = adhoc_corrected_mod / flat_master

        corrected_flat[i - 5] = flat_corrected_mod

    sunpy.io.fits.write(write_path / 'HA_corrected_flat.fits', corrected_flat, dict(), overwrite=True)

    return corrected_flat


def get_corrected_median_profile(corrected_flat):
    both_beams = np.zeros((16, 512, 512), dtype=np.float64)

    both_beams = np.add(
        corrected_flat[:, 0:512],
        corrected_flat[:, 512:]
    )

    corrected_median_profile = np.median(both_beams, (0, 1))

    np.savetxt(write_path / 'Corrected_median_profile.txt', corrected_median_profile)

    return np.median(both_beams, (0, 1))


def do_straylight_calibration(corrected_median_profile):
    
    wave_ha = np.loadtxt(base_path / ha_wave_file)[32:482][::-1][:-6]

    ordered_median_profile = corrected_median_profile[32:482][::-1][6:]

    catalog = np.loadtxt(catalog_base / ha_catalog_file)

    norm_line, norm_atlas, atlas_wave = normalise_profiles(
        ordered_median_profile,
        wave_ha,
        catalog[:, 1],
        catalog[:, 0],
        cont_wave=wave_ha[-1]
    )

    a, b = np.polyfit([0, norm_atlas.size-1], [norm_atlas[0], norm_atlas[-1]], 1)

    atlas_slope = a * np.arange(norm_atlas.size) + b

    atlas_slope /= atlas_slope.max()

    a, b = np.polyfit([0, norm_line.size-1], [norm_line[0], norm_line[-1]], 1)

    line_slope = a * np.arange(norm_line.size) + b

    line_slope /= line_slope.max()

    multiplicative_factor = atlas_slope / line_slope

    multiplicative_factor /= multiplicative_factor.max()

    norm_line *= multiplicative_factor

    plt.plot(wave_ha, norm_line, label='line')

    plt.plot(wave_ha, norm_atlas, label='atlas')

    plt.legend()

    plt.show()

    np.savetxt(write_path / 'multiplicative_factor.txt', multiplicative_factor)

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        norm_line, norm_atlas
    )

    f = h5py.File(write_path / 'straylight_ha_using_flat_profiles.h5', 'w')

    f['wave_ha'] = wave_ha

    f['corrected_median_profile'] = corrected_median_profile

    f['median_indice'] = '[32:482][::-1][6:]'

    f['bass2000_catalog_ha'] = catalog

    f['norm_line'] = norm_line

    f['norm_atlas'] = norm_atlas

    f['atlas_wave'] = atlas_wave

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f.close()

    return result, result_atlas, fwhm, sigma, k_values, norm_line, norm_atlas, wave_ha, multiplicative_factor


def get_raw_halpha_data():

    raw_data = np.zeros((20, 4, 512, 512), dtype=np.float64)

    for i in range(20):
        data, header = sunpy.io.fits.read(base_path / ha_data_file)[i + 1]
        raw_data[i] = data

    return raw_data[:, :, :, 32:482][:, :, :, ::-1][:, :, :, :-6]


def make_0p536_atlas_file(wave_ha, norm_atlas):
    f1 = h5py.File(write_path / mu_1_file, 'r')

    f2 = h5py.File(write_path / mu_0p536_file, 'r')

    get_indice = prepare_get_indice(f1['wav'][()])

    vec_get_indice = np.vectorize(get_indice)

    synth_indice = vec_get_indice(wave_ha)

    zp536_factor = f2['profiles'][0, 0, 0, synth_indice, 0] / f1['profiles'][0, 0, 0, synth_indice, 0]

    f1.close()

    f2.close()

    zp536_factor /= zp536_factor.max()

    atlas_at_0p536 = norm_atlas * zp536_factor

    norm_atlas_at_0p536 = atlas_at_0p536 / atlas_at_0p536[-1]

    raw_data = get_raw_halpha_data()

    median_profile = np.median(
        raw_data[2:18, 0, 270:400],
        (0, 1)
    )

    norm_median = median_profile / median_profile[-1]

    plt.plot(norm_median, label='median')

    plt.plot(norm_atlas_at_0p536, label='atlas at 0.536')

    plt.legend()

    plt.show()

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        norm_median, norm_atlas_at_0p536
    )

    f = h5py.File(write_path / 'straylight_ha_using_median_profiles_with_atlas_at_0p536_estimated_profile.h5', 'w')

    f['wave_ha'] = wave_ha

    f['norm_atlas'] = norm_atlas

    f['0p536_factor'] = zp536_factor

    f['atlas_at_0p536'] = atlas_at_0p536

    f['norm_atlas_at_0p536'] = norm_atlas_at_0p536

    f['median_indice'] = '[2:18, 0, 270:400]'

    f['median_profile'] = median_profile

    f['norm_median'] = norm_median

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f.close()

    return result, result_atlas, fwhm, sigma, k_values, norm_median, norm_atlas_at_0p536, wave_ha, zp536_factor
