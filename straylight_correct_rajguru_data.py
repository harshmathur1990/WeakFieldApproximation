import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
import scipy.io
import h5py
import numpy as np
import sunpy.io.fits
from pathlib import Path
from prepare_data import *
from stray_light_approximation import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


falc_file_path = Path(
    '/home/harsh/CourseworkRepo/stic/run/falc_nicole_for_stic.nc'
)

base_path = Path('/home/harsh/Rajguru Data')
write_path = base_path
falc_mu_1_file = base_path / 'falc_Fe_mu_1.nc'
falc_mu_0p923_file = base_path / 'falc_Fe_mu_0.923.nc'

catalog_6173_file = base_path / 'catalog_6173.txt'
catalog_7090_file = base_path / 'catalog_7090.txt'
wavefile_6173 = base_path / 'lambda_6173.sav'
wavefile_7090 = base_path / 'lambda_7090.sav'
profiles_file = base_path / 'ibis_lineprofiles.sav'

wave_6173_data = scipy.io.readsav(wavefile_6173)
wave_7090_data = scipy.io.readsav(wavefile_7090)
profiles = scipy.io.readsav(profiles_file)

catalog_6173 = np.loadtxt(catalog_6173_file)
catalog_7090 = np.loadtxt(catalog_7090_file)
wave_6173 = 6173.34 - wave_6173_data['lambda_6173_int'][12] + wave_6173_data['lambda_6173_int']
wave_7090 = 7090.4 - wave_7090_data['lambda_7090_int'][12] + wave_7090_data['lambda_7090_int']

qs_6173 = profiles['fe6173_qsunav']
qs_7090 = profiles['fe7090_qsunav']

umbra_6173 = profiles['fe6173_umbra']  # (4, 23)
penumbra_6173 = profiles['fe6173_penumbra']  # (4, 23)
umbra_7090 = profiles['fe7090_umbra']  # (23)
penumbra_7090 = profiles['fe7090_penumbra']  # (23)


ltau = np.array(
    [
        -8.       , -7.78133  , -7.77448  , -7.76712  , -7.76004  ,
        -7.75249  , -7.74429  , -7.7356   , -7.72638  , -7.71591  ,
        -7.70478  , -7.69357  , -7.68765  , -7.68175  , -7.67589  ,
        -7.66997  , -7.66374  , -7.65712  , -7.64966  , -7.64093  ,
        -7.63093  , -7.6192   , -7.6053   , -7.58877  , -7.56925  ,
        -7.54674  , -7.52177  , -7.49317  , -7.4585   , -7.41659  ,
        -7.36725  , -7.31089  , -7.24834  , -7.18072  , -7.1113   ,
        -7.04138  , -6.97007  , -6.89698  , -6.82299  , -6.74881  ,
        -6.67471  , -6.60046  , -6.52598  , -6.45188  , -6.37933  ,
        -6.30927  , -6.24281  , -6.17928  , -6.11686  , -6.05597  ,
        -5.99747  , -5.94147  , -5.88801  , -5.84684  , -5.81285  ,
        -5.78014  , -5.74854  , -5.71774  , -5.68761  , -5.65825  ,
        -5.6293   , -5.60066  , -5.57245  , -5.54457  , -5.51687  ,
        -5.48932  , -5.46182  , -5.43417  , -5.40623  , -5.37801  ,
        -5.3496   , -5.32111  , -5.29248  , -5.26358  , -5.23413  ,
        -5.20392  , -5.17283  , -5.14073  , -5.1078   , -5.07426  ,
        -5.03999  , -5.00492  , -4.96953  , -4.93406  , -4.89821  ,
        -4.86196  , -4.82534  , -4.78825  , -4.75066  , -4.71243  ,
        -4.67439  , -4.63696  , -4.59945  , -4.5607   , -4.52212  ,
        -4.48434  , -4.44653  , -4.40796  , -4.36863  , -4.32842  ,
        -4.28651  , -4.24205  , -4.19486  , -4.14491  , -4.09187  ,
        -4.03446  , -3.97196  , -3.90451  , -3.83088  , -3.7496   ,
        -3.66     , -3.56112  , -3.4519   , -3.33173  , -3.20394  ,
        -3.07448  , -2.94444  , -2.8139   , -2.68294  , -2.55164  ,
        -2.42002  , -2.28814  , -2.15605  , -2.02377  , -1.89135  ,
        -1.7588   , -1.62613  , -1.49337  , -1.36127  , -1.23139  ,
        -1.10699  , -0.99209  , -0.884893 , -0.782787 , -0.683488 ,
        -0.584996 , -0.485559 , -0.383085 , -0.273456 , -0.152177 ,
        -0.0221309,  0.110786 ,  0.244405 ,  0.378378 ,  0.51182  ,
        0.64474  ,  0.777188 ,  0.909063 ,  1.04044  ,  1.1711
    ]
)

@np.vectorize
def get_relative_velocity_6173(wavelength):
    return wavelength - 6173.34


@np.vectorize
def get_relative_velocity_7090(wavelength):
    return wavelength - 7090.4


def compare_falc_synthesis_with_bass():
    f = h5py.File(falc_mu_1_file, 'r')
    
    norm_line_6173, norm_atlas_6173, _ = normalise_profiles(
        f['profiles'][0, 0, 0, 0:148, 0],
        f['wav'][0:148],
        catalog_6173[:, 1],
        catalog_6173[:, 0],
        f['wav'][0]
    )

    norm_line_7090, norm_atlas_7090, _ = normalise_profiles(
        f['profiles'][0, 0, 0, 148:, 0],
        f['wav'][148:],
        catalog_7090[:, 1],
        catalog_7090[:, 0],
        f['wav'][148]
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(4, 6))

    gs = gridspec.GridSpec(2, 1)

    axs = fig.add_subplot(gs[0])

    axs.plot(f['wav'][0:148], norm_line_6173, label='FALC using STiC', color='blue')

    axs.plot(f['wav'][0:148], norm_atlas_6173, label='BASS 2000', color='brown')

    handles, labels = axs.get_legend_handles_labels()

    plt.legend(
        handles,
        labels,
        ncol=2,
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc='lower left',
        mode="expand",
        borderaxespad=0.
    )

    axs.set_xlabel(r'$\lambda (\AA)$')
    axs.set_ylabel(r'$I/I_{c 6173 \AA}$')

    axs = fig.add_subplot(gs[1])

    axs.plot(f['wav'][148:], norm_line_7090, color='blue')

    axs.plot(f['wav'][148:], norm_atlas_7090, color='brown')

    axs.set_xlabel(r'$\lambda (\AA)$')
    axs.set_ylabel(r'$I/I_{c 7090 \AA}$')

    fig.tight_layout()

    fig.savefig(
        'FALC_mu_1_vs_BASS.pdf',
        format='pdf',
        dpi=300
    )

    f.close()

    plt.show()

    plt.close('all')

    plt.clf()

    plt.cla()


def correct_for_straylight():

    f1 = h5py.File(falc_mu_0p923_file, 'r')

    norm_line_6173, norm_atlas_6173, _ = normalise_profiles(
        qs_6173,
        wave_6173,
        f1['profiles'][0, 0, 0, 0:148, 0],
        f1['wav'][0:148],
        cont_wave=wave_6173[0]
    )

    a, b = np.polyfit([0, norm_atlas_6173.size-1], [norm_atlas_6173[0], norm_atlas_6173[-1]], 1)

    atlas_slope = a * np.arange(norm_atlas_6173.size) + b

    atlas_slope /= atlas_slope.max()

    a, b = np.polyfit([0, norm_line_6173.size-1], [norm_line_6173[0], norm_line_6173[-1]], 1)

    line_slope = a * np.arange(norm_line_6173.size) + b

    line_slope /= line_slope.max()

    multiplicative_factor = atlas_slope / line_slope

    multiplicative_factor /= multiplicative_factor.max()

    norm_line_6173 *= multiplicative_factor

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        norm_line_6173,
        norm_atlas_6173,
        continuum=1.0,
        indices=None
    )

    stray_corrected_median_6173 = qs_6173 * multiplicative_factor

    stray_corrected_median_6173 = (
        stray_corrected_median_6173 - (
            (
                np.unravel_index(
                    np.argmin(result),
                    result.shape
                )[1] / 100
            ) * stray_corrected_median_6173[0]
        )
    ) / (
        1 - (
            np.unravel_index(
                np.argmin(result),
                result.shape
            )[1] / 100
        )
    )

    stic_cgs_calib_factor_6173 = stray_corrected_median_6173[0] / f1['profiles'][0, 0, 0, 0, 0]

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_6173.h5',
        'w'
    )

    f['wave_6173'] = wave_6173

    f['correction_factor'] = multiplicative_factor

    f['atlas_at_0p923'] = f1['profiles'][0, 0, 0, 0:148, 0]

    f['norm_atlas_6173'] = norm_atlas_6173

    f['qs_6173'] = qs_6173

    f['norm_line_6173'] = norm_line_6173

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f['straylight_value'] = np.unravel_index(np.argmin(result), result.shape)[1]

    f['broadening_in_km_sec'] = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]] * (wave_6173[1] - wave_6173[0]) * 2.99792458e5/ 6173.34

    f['stray_corrected_median_6173'] = stray_corrected_median_6173

    f['stic_cgs_calib_factor_6173'] = stic_cgs_calib_factor_6173

    f.close()

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(wave_6173, stray_corrected_median_6173 / stic_cgs_calib_factor_6173, label='Stray Corrected Median')

    plt.plot(f1['wav'][0:148], f1['profiles'][0, 0, 0, 0:148, 0], label='Atlas')

    plt.legend()

    plt.show()

    
    #-------------------------------------------------------------------------#


    norm_line_7090, norm_atlas_7090, _ = normalise_profiles(
        qs_7090,
        wave_7090,
        f1['profiles'][0, 0, 0, 148:, 0],
        f1['wav'][148:],
        cont_wave=wave_7090[0]
    )

    a, b = np.polyfit([0, norm_atlas_7090.size-1], [norm_atlas_7090[0], norm_atlas_7090[-1]], 1)

    atlas_slope = a * np.arange(norm_atlas_7090.size) + b

    atlas_slope /= atlas_slope.max()

    a, b = np.polyfit([0, norm_line_7090.size-1], [norm_line_7090[0], norm_line_7090[-1]], 1)

    line_slope = a * np.arange(norm_line_7090.size) + b

    line_slope /= line_slope.max()

    multiplicative_factor = atlas_slope / line_slope

    multiplicative_factor /= multiplicative_factor.max()

    norm_line_7090 *= multiplicative_factor

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        norm_line_7090,
        norm_atlas_7090,
        continuum=1.0,
        indices=None
    )

    stray_corrected_median_7090 = qs_7090 * multiplicative_factor

    stray_corrected_median_7090 = (
        stray_corrected_median_7090 - (
            (
                np.unravel_index(
                    np.argmin(result),
                    result.shape
                )[1] / 100
            ) * stray_corrected_median_7090[0]
        )
    ) / (
        1 - (
            np.unravel_index(
                np.argmin(result),
                result.shape
            )[1] / 100
        )
    )

    stic_cgs_calib_factor_7090 = stray_corrected_median_7090[0] / f1['profiles'][0, 0, 0, 148, 0]

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_7090.h5',
        'w'
    )

    f['wave_7090'] = wave_7090

    f['correction_factor'] = multiplicative_factor

    f['atlas_at_0p923'] = f1['profiles'][0, 0, 0, 148:, 0]

    f['norm_atlas_7090'] = norm_atlas_7090

    f['qs_7090'] = qs_7090

    f['norm_line_7090'] = norm_line_7090

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f['straylight_value'] = np.unravel_index(np.argmin(result), result.shape)[1]

    f['broadening_in_km_sec'] = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]] * (wave_6173[1] - wave_6173[0]) * 2.99792458e5/ 6173.34

    f['stray_corrected_median_7090'] = stray_corrected_median_7090

    f['stic_cgs_calib_factor_7090'] = stic_cgs_calib_factor_7090

    f.close()

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(wave_7090, stray_corrected_median_7090 / stic_cgs_calib_factor_7090, label='Stray Corrected Median')

    plt.plot(f1['wav'][148:], f1['profiles'][0, 0, 0, 148:, 0], label='Atlas')

    plt.legend()

    plt.show()

    f1.close()

    return stray_corrected_median_6173, stic_cgs_calib_factor_6173, stray_corrected_median_7090, stic_cgs_calib_factor_7090


def correct_for_straylight(data, straylight_factor, multiplicative_factor=None):
    
    #data must be of shape (t, x, y, lambda, stokes) with t, x, y optional
    # straylight_factor must be between 0-100

    result = data.copy()

    if multiplicative_factor is not None:
        if result.ndim == 5:
            result[:, :, :, :, 0] = result[:, :, :, :, 0] * multiplicative_factor
        elif result.ndim == 4:
            result[:, :, :, 0] = result[:, :, :, 0] * multiplicative_factor
        elif result.ndim == 3:
            result[:, :, 0] = result[:, :, 0] * multiplicative_factor
        elif result.ndim == 2:
            result[:, 0] = result[:, 0] * multiplicative_factor
        elif result.ndim == 1:
            result = result * multiplicative_factor

    if result.ndim == 5:
            result[:, :, :, :, 0] = (result[:, :, :, :, 0] - straylight_factor * result[:, :, :, 0, 0][:, :, :, np.newaxis]) / (1 - straylight_factor)
    elif result.ndim == 4:
        result[:, :, :, 0] = (result[:, :, :, 0] - straylight_factor * result[:, :, 0, 0][:, :, np.newaxis]) / (1 - straylight_factor)
    elif result.ndim == 3:
        result[:, :, 0] = (result[:, :, 0] - straylight_factor * result[:, 0, 0][:, np.newaxis]) / (1 - straylight_factor)
    elif result.ndim == 2:
        result[:, 0] = (result[:, 0] - straylight_factor * result[0, 0]) / (1 - straylight_factor)
    elif result.ndim == 1:
        result = (result - straylight_factor * result[0]) / (1 - straylight_factor)

    return result


def make_umbra_inversion_files():
    
    wfe1, ife1 = findgrid(wave_6173, (wave_6173[10] - wave_6173[9]) * 0.25, extra=8)

    wfe2, ife2 = findgrid(wave_7090, (wave_7090[10] - wave_7090[9]) * 0.25, extra=8)

    fe1 = sp.profile(nx=1, ny=1, ns=4, nw=wfe1.size)

    fe2 = sp.profile(nx=1, ny=1, ns=4, nw=wfe2.size)

    fe1.wav[:] = wfe1[:]

    fe2.wav[:] = wfe2[:]

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_6173.h5',
        'r'
    )

    fe1.dat[0, 0, 0, ife1, :] = correct_for_straylight(
        profiles['fe6173_umbra'].T,
        f['straylight_value'][()]  / 100,
        f['correction_factor'][()]
    ) / f['stic_cgs_calib_factor_6173'][()]

    f.close()

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_7090.h5',
        'r'
    )

    fe2.dat[0, 0, 0, ife2, 0] = correct_for_straylight(
        profiles['fe7090_umbra'],
        f['straylight_value'][()]  / 100,
        f['correction_factor'][()]
    ) / f['stic_cgs_calib_factor_7090'][()]

    f.close()

    fe1.weights[:, :] = 1e16

    fe1.weights[ife1, :] = 0.004

    fe2.weights[:, :] = 1e16

    fe2.weights[ife2, :] = 0.004

    fe = fe1 + fe2

    fe.write(
        write_path / 'umbra_stic_profiles.nc'
    )


def make_penumbra_inversion_files():
    
    wfe1, ife1 = findgrid(wave_6173, (wave_6173[10] - wave_6173[9]) * 0.25, extra=8)

    wfe2, ife2 = findgrid(wave_7090, (wave_7090[10] - wave_7090[9]) * 0.25, extra=8)

    fe1 = sp.profile(nx=1, ny=1, ns=4, nw=wfe1.size)

    fe2 = sp.profile(nx=1, ny=1, ns=4, nw=wfe2.size)

    fe1.wav[:] = wfe1[:]

    fe2.wav[:] = wfe2[:]

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_6173.h5',
        'r'
    )

    fe1.dat[0, 0, 0, ife1, :] = correct_for_straylight(
        profiles['fe6173_penumbra'].T,
        f['straylight_value'][()]  / 100,
        f['correction_factor'][()]
    ) / f['stic_cgs_calib_factor_6173'][()]

    f.close()

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_7090.h5',
        'r'
    )

    fe2.dat[0, 0, 0, ife2, 0] = correct_for_straylight(
        profiles['fe7090_penumbra'],
        f['straylight_value'][()] / 100,
        f['correction_factor'][()]
    ) / f['stic_cgs_calib_factor_7090'][()]

    f.close()

    fe1.weights[:, :] = 1e16

    fe1.weights[ife1, :] = 0.004

    fe2.weights[:, :] = 1e16

    fe2.weights[ife2, :] = 0.004

    fe = fe1 + fe2

    fe.write(
        write_path / 'penumbra_stic_profiles.nc'
    )


def make_median_profile_inversion_files():
    
    wfe1, ife1 = findgrid(wave_6173, (wave_6173[10] - wave_6173[9]) * 0.25, extra=8)

    wfe2, ife2 = findgrid(wave_7090, (wave_7090[10] - wave_7090[9]) * 0.25, extra=8)

    fe1 = sp.profile(nx=1, ny=1, ns=4, nw=wfe1.size)

    fe2 = sp.profile(nx=1, ny=1, ns=4, nw=wfe2.size)

    fe1.wav[:] = wfe1[:]

    fe2.wav[:] = wfe2[:]

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_6173.h5',
        'r'
    )

    fe1.dat[0, 0, 0, ife1, 0] = correct_for_straylight(
        profiles['fe6173_qsunav'],
        f['straylight_value'][()]  / 100,
        f['correction_factor'][()]
    ) / f['stic_cgs_calib_factor_6173'][()]

    f.close()

    f = h5py.File(
        write_path / 'rajguru_data_mu_0.923_stray_approximated_wave_7090.h5',
        'r'
    )

    fe2.dat[0, 0, 0, ife2, 0] = correct_for_straylight(
        profiles['fe7090_qsunav'],
        f['straylight_value'][()]  / 100,
        f['correction_factor'][()]
    ) / f['stic_cgs_calib_factor_7090'][()]

    f.close()

    fe1.weights[:, :] = 1e16

    fe1.weights[ife1, 0] = 0.004

    fe2.weights[:, :] = 1e16

    fe2.weights[ife2, 0] = 0.004

    fe = fe1 + fe2

    fe.write(
        write_path / 'median_profile_stic_profiles.nc'
    )


def make_initial_atmos_umbra_penumbra():
    f = h5py.File(falc_file_path, 'r')

    m = sp.model(nx=2, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0, 0]

    m.vlos[:, :, :] = f['vlos'][0, 0, 0]

    m.vturb[:, :, :] = f['vturb'][0, 0, 0]

    m.write('falc_2_1.nc')


def compare_inversions(
    input_profile_file,
    output_profile_file,
    output_atmos_file,
    index
):

    base_path = Path('/home/harsh/CourseworkRepo/stic/run_rajguru_active')

    size = plt.rcParams['lines.markersize']

    median_profile_f = h5py.File(
        write_path / 'median_profile_stic_profiles.nc',
        'r'
    )

    median_atmos_f = h5py.File(
        '/home/harsh/CourseworkRepo/stic/run_rajguru/median_profile_stic_profiles_cycle_1_t_3_vl_1_vt_0_atmos.nc',
        'r'
    )

    input_profile_f = h5py.File(
        base_path / input_profile_file,
        'r'
    )

    output_profile_f = h5py.File(
        base_path / output_profile_file,
        'r'
    )

    output_atmos_f = h5py.File(
        base_path / output_atmos_file,
        'r'
    )

    wave_6173 = get_relative_velocity_6173(median_profile_f['wav'][0:148])

    wave_7090 = get_relative_velocity_7090(median_profile_f['wav'][()])

    indice_6173 = np.where(median_profile_f['profiles'][0, 0, 0, 0:148, 0] != 0)[0]

    indice_7090 = np.where(median_profile_f['profiles'][0, 0, 0, 148:, 0] != 0)[0] + 148

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(9, 6))

    gs = gridspec.GridSpec(3, 3)

    axs = fig.add_subplot(gs[0])

    axs.scatter(
        wave_6173[indice_6173],
        input_profile_f['profiles'][0, 0, index, indice_6173, 0],
        color='brown',
        s=size/3
    )

    axs.scatter(
        wave_7090[indice_7090],
        input_profile_f['profiles'][0, 0, index, indice_7090, 0],
        color='#097969',
        s=size/3
    )

    axs.plot(
        wave_6173[indice_6173],
        input_profile_f['profiles'][0, 0, index, indice_6173, 0],
        linewidth=0.5,
        linestyle='dashed',
        color='brown',
    )

    axs.plot(
        wave_7090[indice_7090],
        input_profile_f['profiles'][0, 0, index, indice_7090, 0],
        linewidth=0.5,
        linestyle='dashdot',
        color='#097969',
    )

    axs.scatter(
        wave_6173[indice_6173],
        output_profile_f['profiles'][0, 0, index, indice_6173, 0],
        s=size/3,
        color='blue',
    )

    axs.scatter(
        wave_7090[indice_7090],
        output_profile_f['profiles'][0, 0, index, indice_7090, 0],
        color='blue',
        s=size/3
    )

    axs.plot(
        wave_6173[indice_6173],
        output_profile_f['profiles'][0, 0, index, indice_6173, 0],
        linewidth=0.5,
        linestyle='dashed',
        color='blue'
    )

    axs.plot(
        wave_7090[indice_7090],
        output_profile_f['profiles'][0, 0, index, indice_7090, 0],
        linewidth=0.5,
        linestyle='dashdot',
        color='blue'
    )

    axs.set_xlabel(r'$\lambda (\AA)$')
    axs.set_ylabel(r'$I/I_{c}$')

    for stokes, label in zip(range(1, 4), ['Q', 'U', 'V']):

        axs = fig.add_subplot(gs[stokes])

        axs.scatter(
            wave_6173[indice_6173],
            input_profile_f['profiles'][0, 0, index, indice_6173, stokes] / 
            input_profile_f['profiles'][0, 0, index, indice_6173, 0],
            color='brown',
            s=size/3
        )

        axs.plot(
            wave_6173[indice_6173],
            input_profile_f['profiles'][0, 0, index, indice_6173, stokes] / 
            input_profile_f['profiles'][0, 0, index, indice_6173, 0],
            linewidth=0.5,
            color='brown',
        )

        axs.scatter(
            wave_6173[indice_6173],
            output_profile_f['profiles'][0, 0, index, indice_6173, stokes] /
            output_profile_f['profiles'][0, 0, index, indice_6173, 0],
            color='blue',
            s=size/3
        )

        axs.plot(
            wave_6173[indice_6173],
            output_profile_f['profiles'][0, 0, index, indice_6173, stokes] /
            output_profile_f['profiles'][0, 0, index, indice_6173, 0],
            linewidth=0.5,
            color='blue'
        )

        axs.set_xlabel(r'$\lambda (\AA)$')

        axs.set_ylabel(r'${}/I$'.format(label))

    axs = fig.add_subplot(gs[4])

    axs.plot(
        ltau,
        median_atmos_f['temp'][0, 0, 0] / 1e3,
        color='orange',
        # linewidth=0.05
    )

    axs.plot(
        ltau,
        output_atmos_f['temp'][0, 0, index] / 1e3,
        color='blue',
        # linewidth=0.05
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$T[kK]$')

    axs = fig.add_subplot(gs[5])

    axs.plot(
        ltau,
        median_atmos_f['vlos'][0, 0, 0] / 1e5,
        color='orange',
        # linewidth=0.05
    )

    axs.plot(
        ltau,
        output_atmos_f['vlos'][0, 0, index] / 1e5,
        color='blue',
        # linewidth=0.05
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$V_{LOS}[kms^{-1}]$')


    axs = fig.add_subplot(gs[6])

    axs.plot(
        ltau,
        output_atmos_f['blong'][0, 0, index],
        color='blue',
        # linewidth=0.05
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$B_{long}[G]$')

    axs = fig.add_subplot(gs[7])

    axs.plot(
        ltau,
        output_atmos_f['bhor'][0, 0, index],
        color='blue',
        # linewidth=0.05
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$B_{hor}[G]$')


    axs = fig.add_subplot(gs[8])

    axs.plot(
        ltau,
        output_atmos_f['azi'][0, 0, index] * 180 / np.pi,
        color='blue',
        # linewidth=0.05
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$Azi[degree]$')

    fig.tight_layout()

    fig.savefig(
        write_path / 'InversionComparison_{}.pdf'.format(
            index
        ),
        dpi=300,
        format='pdf'
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    median_profile_f.close()

    median_atmos_f.close()

    input_profile_f.close()

    output_profile_f.close()

    output_atmos_f.close()


def compare_median_profile():
    
    size = plt.rcParams['lines.markersize']

    median_input_profile_f = h5py.File(
        write_path / 'median_profile_stic_profiles.nc',
        'r'
    )

    median_output_atmos_f = h5py.File(
        '/home/harsh/CourseworkRepo/stic/run_rajguru/median_profile_stic_profiles_cycle_1_t_4_vl_1_vt_0_atmos.nc',
        'r'
    )

    median_output_profile_f = h5py.File(
        '/home/harsh/CourseworkRepo/stic/run_rajguru/median_profile_stic_profiles_cycle_1_t_4_vl_1_vt_0_profs.nc',
        'r'
    )

    wave_6173 = get_relative_velocity_6173(median_input_profile_f['wav'][0:148])

    wave_7090 = get_relative_velocity_7090(median_input_profile_f['wav'][()])

    indice_6173 = np.where(median_input_profile_f['profiles'][0, 0, 0, 0:148, 0] != 0)[0]

    indice_7090 = np.where(median_input_profile_f['profiles'][0, 0, 0, 148:, 0] != 0)[0] + 148

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(2, 2)

    axs = fig.add_subplot(gs[0])

    axs.scatter(
        wave_6173[indice_6173],
        median_input_profile_f['profiles'][0, 0, 0, indice_6173, 0],
        s=size/2,
        color='orange'
    )

    axs.plot(
        wave_6173[indice_6173],
        median_input_profile_f['profiles'][0, 0, 0, indice_6173, 0],
        linewidth=0.5,
        color='orange'
    )

    axs.scatter(
        wave_6173[indice_6173],
        median_output_profile_f['profiles'][0, 0, 0, indice_6173, 0],
        s=size/2,
        color='blue'
    )

    axs.plot(
        wave_6173[indice_6173],
        median_output_profile_f['profiles'][0, 0, 0, indice_6173, 0],
        linewidth=0.5,
        color='blue'
    )

    axs.set_xlabel(r'$\lambda (\AA)$')
    axs.set_ylabel(r'$I/I_{c}$')

    axs = fig.add_subplot(gs[1])

    axs.scatter(
        wave_7090[indice_7090],
        median_input_profile_f['profiles'][0, 0, 0, indice_7090, 0],
        s=size/2,
        color='orange'
    )

    axs.plot(
        wave_7090[indice_7090],
        median_input_profile_f['profiles'][0, 0, 0, indice_7090, 0],
        linewidth=0.5,
        color='orange'
    )

    axs.scatter(
        wave_7090[indice_7090],
        median_output_profile_f['profiles'][0, 0, 0, indice_7090, 0],
        s=size/2,
        color='blue'
    )

    axs.plot(
        wave_7090[indice_7090],
        median_output_profile_f['profiles'][0, 0, 0, indice_7090, 0],
        linewidth=0.5,
        color='blue'
    )

    axs.set_xlabel(r'$\lambda (\AA)$')
    axs.set_ylabel(r'$I/I_{c}$')

    axs = fig.add_subplot(gs[2])

    axs.plot(
        ltau,
        median_output_atmos_f['temp'][0, 0, 0] / 1e3,
        color='blue'
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$T[kK]$')

    axs = fig.add_subplot(gs[3])

    axs.plot(
        ltau,
        median_output_atmos_f['vlos'][0, 0, 0] / 1e5,
        color='blue'
    )

    fig.tight_layout()

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$V_{LOS}[kms^{-1}]$')

    fig.tight_layout()

    fig.savefig(
        write_path / 'MedianProfileComparison.pdf',
        dpi=300,
        format='pdf'
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    median_input_profile_f.close()

    median_output_profile_f.close()

    median_output_atmos_f.close()
