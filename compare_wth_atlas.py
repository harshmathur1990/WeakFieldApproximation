from pathlib import Path
from helita.sim import rh15d
import numpy as np
import matplotlib.pyplot as plt


AIR_TO_VACCUM_LIMIT = 199.9352


def get_indices_for_wavelength(wave, wav_start, wav_end):

    if not isinstance(wave, np.ndarray):
        wavedata = wave.data
    else:
        wavedata = wave

    return np.where((wavedata >= wav_start) & (wavedata < wav_end))[0]


def get_wavelength_points(wave, indices):
    return wave.data[indices]


def get_intensity_points(intensity, indices, x=0, y=0):
    return intensity.data[x][y][indices]


def get_catalog_intensity(
    wav_start,
    wav_end,
    base_dir=None,
    catalog_filename=None
):

    if not base_dir:
        base_dir = Path('.')

    if not catalog_filename:
        catalog_filename = 'catalog_6563.txt'

    catalog_file_path = base_dir / catalog_filename

    catalog_6563 = np.loadtxt(catalog_file_path)

    # vaccum_wavelengths = air_to_vaccum(catalog_6563.T[0] / 10)

    vaccum_wavelengths = catalog_6563.T[0] / 10

    indices = get_indices_for_wavelength(
        vaccum_wavelengths,
        wav_start,
        wav_end
    )

    return vaccum_wavelengths[indices], catalog_6563.T[1][indices]


def index_and_continuous_wavelength(wave, cont_wave=658.8174):

    index_cont = np.argmin(np.abs(wave.data - cont_wave))

    return index_cont, wave.data[index_cont]


def normalise_intensity(intensity_points, continous_intensity):
    return intensity_points / continous_intensity


def read_output_model(base_dir=None):

    if not base_dir:
        base_dir = Path('.')

    return rh15d.Rh15dout(fdir=base_dir.absolute())


def get_normalised_model_intensities(
    base_dir=None,
    wav_start=653,
    wav_end=659,
    cont_wave=658.8174,
    x=0,
    y=0
):

    out = read_output_model(base_dir=base_dir)

    wave = out.ray.wavelength

    intensity = out.ray.intensity

    indices = get_indices_for_wavelength(
        wave,
        wav_start,
        wav_end
    )

    model_wavelength_points = get_wavelength_points(
        wave,
        indices
    )

    model_intensity_points = get_intensity_points(
        intensity,
        indices,
        x,
        y
    )

    index_continuum, continuum_wavelength = index_and_continuous_wavelength(
        wave,
        cont_wave
    )

    continous_intensity = get_intensity_points(
        intensity,
        index_continuum,
        x,
        y
    )

    normalised_model_intensity = normalise_intensity(
        model_intensity_points,
        continous_intensity
    )

    return model_wavelength_points, \
        normalised_model_intensity, continuum_wavelength


def get_normalised_catalog_intensities(
    wav_start,
    wav_end,
    continuum_wavelength=658.8174,
    base_dir=None,
    catalog_filename=normalise_intensity
):

    catalog_wavelength, catalog_intensity = get_catalog_intensity(
        wav_start=wav_start,
        wav_end=wav_end,
        base_dir=base_dir,
        catalog_filename=catalog_filename
    )

    index = np.argmin(np.abs(catalog_wavelength - continuum_wavelength))

    catalog_continuum_intensity = catalog_intensity[index]

    return catalog_wavelength, \
        normalise_intensity(catalog_intensity, catalog_continuum_intensity)


def get_normalised_intensities(
    model_base_dir,
    catalog_base_dir=None,
    wav_start=653,
    wav_end=659,
    cont_wave=658.8174,
    x=0,
    y=0,
    catalog_filename='catalog_6563.txt'
):

    model_wavelength, model_intensity, continuum_wavelength = get_normalised_model_intensities(
        base_dir=model_base_dir,
        wav_start=653,
        wav_end=659,
        cont_wave=658.8174,
        x=0,
        y=0
    )

    catalog_wavelength, catalog_intensity = get_normalised_catalog_intensities(
        wav_start=wav_start,
        wav_end=wav_end,
        continuum_wavelength=continuum_wavelength,
        base_dir=catalog_base_dir,
        catalog_filename=catalog_filename
    )

    return model_wavelength, model_intensity, \
        catalog_wavelength, catalog_intensity, continuum_wavelength


def air_to_vaccum(wavelengths):

    wavelengths = wavelengths.copy()

    lambda_vaccum = np.zeros_like(wavelengths)

    for index, wave in enumerate(wavelengths):
        if wave >= AIR_TO_VACCUM_LIMIT:
            wave_square = (1e7 / wave)**2

            increase = 1.0000834213 + 2.406030e6 / \
                (1.30E+10 - wave_square) + 1.5997e4 /\
                (3.89e9 - wave_square)

            lambda_vaccum[index] = wave * increase
        else:
            lambda_vaccum[index] = wave

    return lambda_vaccum


def plot_model_and_catalog_intensities(
    model_wavelength,
    model_intensity,
    catalog_wavelength,
    catalog_intensity,
    continuum_wavelength=658.8174,
    mag_field=None,
    mode=None,
    filename=None
):

    plt.plot(
        model_wavelength,
        model_intensity,
        label='FALC'
    )

    plt.plot(
        catalog_wavelength,
        catalog_intensity,
        label='Atlas Spectrum'
    )

    plt.ylabel(
        r'I/$I_{{ref}}$'.format(
            continuum_wavelength
        )
    )

    plt.xlabel('Wavelength (Units: nm)')

    plt.title(
        'FALC vs Atlas (Magnetic Field: {} Guass Vertical, Mode: {})'.format(
            mag_field, mode
        )
    )

    plt.legend()

    plt.tight_layout()

    if not filename:
        plt.show()
    else:
        plt.savefig(filename, dpi=900, format='png')

    plt.clf()
    plt.cla()


def do_everything(path_to_folder, path_save_file, mag_field, mode):
    model_wavelength, model_intensity, catalog_wavelength, catalog_intensity, continuum_wavelength = get_normalised_intensities(
        model_base_dir=path_to_folder,
        catalog_base_dir=None,
        wav_start=653,
        wav_end=659,
        cont_wave=658.8174,
        x=0,
        y=0,
        catalog_filename='catalog_6563.txt'
    )

    plot_model_and_catalog_intensities(
        model_wavelength,
        model_intensity,
        catalog_wavelength,
        catalog_intensity,
        continuum_wavelength=658.8174,
        mag_field=mag_field,
        mode=mode,
        filename=path_save_file
    )
