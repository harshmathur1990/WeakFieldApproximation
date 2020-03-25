from pathlib import Path
from helita.sim import rh15d
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def get_indices_for_wavelength(wave, wav_start, wav_end):

    return np.where((wave.data >= wav_start) & (wave.data < wav_end))[0]


def get_wavelength_points(wave, indices):
    return wave.data[indices]


def get_intensity_points(intensity, indices, x=0, y=0):
    return intensity.data[x][y][indices]


def get_catalog_interpolation_function(
    base_dir=None,
    catalog_filename=None
):

    if not base_dir:
        base_dir = Path('.')

    if not catalog_filename:
        catalog_filename = 'catalog_6563.txt'

    catalog_file_path = base_dir / catalog_filename

    catalog_6563 = np.loadtxt(catalog_file_path)

    f = scipy.interpolate.interp1d(catalog_6563.T[0], catalog_6563.T[1])

    return f


def get_catalog_intensity(
    wavelength_points,
    base_dir=None,
    catalog_filename=None
):

    f = get_catalog_interpolation_function(
        base_dir=base_dir,
        catalog_filename=catalog_filename
    )

    return f(wavelength_points * 10)


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
    wavelength_points,
    continuum_wavelength=658.8174,
    base_dir=None,
    catalog_filename=None
):

    catalog_intensity = get_catalog_intensity(
        wavelength_points,
        base_dir=base_dir,
        catalog_filename=catalog_filename
    )

    catalog_continuum_intensity = get_catalog_intensity(
        continuum_wavelength,
        base_dir=base_dir,
        catalog_filename=catalog_filename
    )

    return normalise_intensity(catalog_intensity, catalog_continuum_intensity)


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

    wavelength_points, model_intensity, continuum_wavelength = get_normalised_model_intensities(
        base_dir=model_base_dir,
        wav_start=653,
        wav_end=659,
        cont_wave=658.8174,
        x=0,
        y=0
    )

    catalog_intensity = get_normalised_catalog_intensities(
        wavelength_points,
        continuum_wavelength=continuum_wavelength,
        base_dir=catalog_base_dir,
        catalog_filename=catalog_filename
    )

    return wavelength_points, model_intensity, \
        catalog_intensity, continuum_wavelength


def plot_model_and_catalog_intensities(
    wavelength_points,
    model_intensity,
    catalog_intensity,
    continuum_wavelength=658.8174,
    mag_field=None
):

    plt.plot(
        wavelength_points,
        model_intensity,
        label='FALC {}'.format(
            'FIELD FREE' if mag_field is None else '{} nm'.format(mag_field)
        )
    )

    plt.plot(
        wavelength_points,
        catalog_intensity,
        label='Atlas Spectrum'
    )

    plt.ylabel(
        'Normalised intensity w.r.t continuum wavelength {} nm'.format(
            continuum_wavelength
        )
    )

    plt.xlabel('Wavelength in nm')

    plt.title('FALC vs Atlas')

    plt.legend()

    plt.tight_layout()

    plt.show()
