import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops


def get_drift_data(
    data,
    ref_centroid_y,
    ref_centroid_x,
    arcsec_per_mm=3.3,
    drift_y=None,
    drift_x=None
):

    k = -2

    if drift_y is None:
        drift_y = list()

    if drift_x is None:
        drift_x = list()

    for i in range(data.shape[0]):
        mean = data[i].mean()
        std = data[i].std()
        label_image = data[i] < (mean +(k *std))
        regions = regionprops(label_image.astype(np.int64), intensity_image=data[i])
        centroid_y, centroid_x = regions[0].centroid
        drift_y.append(centroid_y - ref_centroid_y)
        drift_x.append(centroid_x - ref_centroid_x)

    return drift_y, drift_x


def get_all_drifts(data_list):
    k = -2
    arcsec_per_mm=3.3
    mean = data_list[0][0].mean()
    std = data_list[0][0].std()

    ref_label_image = data_list[0][0] < (mean + (k *std))

    regions = regionprops(ref_label_image.astype(np.int64), intensity_image=data_list[0][0])

    centroid_y, centroid_x = regions[0].centroid

    drift_y = list()

    drift_x = list()

    for data in data_list:
        get_drift_data(data, centroid_y, centroid_x, drift_y=drift_y, drift_x=drift_x)

    drift_y, drift_x = np.array(drift_y), np.array(drift_x)

    drift_in_mm_y = drift_y * 6.5e-3

    drift_in_mm_x = drift_x * 6.5e-3

    drift_in_arcsec_y = drift_in_mm_y * arcsec_per_mm

    drift_in_arcsec_x = drift_in_mm_x * arcsec_per_mm

    drift_in_mm_y -= drift_in_mm_y.mean()

    drift_in_mm_x -= drift_in_mm_x.mean()

    drift_in_arcsec_y -= drift_in_arcsec_y.mean()

    drift_in_arcsec_x -= drift_in_arcsec_y.mean()

    return drift_in_mm_y, drift_in_mm_x, drift_in_arcsec_y, drift_in_arcsec_x


def save_power_spectra(drift_in_arcsec_y, drift_in_arcsec_x):

    plt.close('all')

    plt.clf()

    plt.cla()

    fft_drift_y = np.fft.fft(drift_in_arcsec_y)

    fft_drift_x = np.fft.fft(drift_in_arcsec_x)

    fftfreq = np.fft.fftfreq(fft_drift_x.shape[0], 1)

    ind = np.where(fftfreq > 0)

    fig, axs = plt.subplots(2, 1, figsize=(19.2, 10.8))

    axs[0].plot(fftfreq[ind], np.abs(fft_drift_x[ind]), color='#3fC1C9')

    axs[1].plot(fftfreq[ind], np.abs(fft_drift_y[ind]), color='#3fC1C9')

    axs[0].set_xlim(0, 0.05)

    axs[1].set_xlim(0, 0.05)

    axs[0].set_xticklabels([])

    axs[1].set_xlabel('Frequency (Hz)')

    axs[0].set_ylabel('Power')

    axs[1].set_ylabel('Power')

    axs[0].set_title('East / RA')

    axs[1].set_title('North / DEC')

    fig.tight_layout()

    plt.savefig('fft_sunspot.eps', format='eps', dpi=300)


def save_power_spectra_sensor_drift(outfile, filename_ra, filename_dec, xlim_ra=None, xlim_dec=None):

    data_ra = np.loadtxt(filename_ra)

    data_dec = np.loadtxt(filename_dec)

    drift_ra = data_ra[:, 1] - data_ra[:, 2]

    drift_dec = data_dec[:, 1] - data_dec[:, 2]

    plt.close('all')

    plt.clf()

    plt.cla()

    fft_drift_y = np.fft.fft(drift_dec)

    fft_drift_x = np.fft.fft(drift_ra)

    fftfreq_ra = np.fft.fftfreq(fft_drift_x.shape[0], (data_ra[1, 0]-data_ra[0, 0]) * 60)

    fftfreq_dec = np.fft.fftfreq(fft_drift_y.shape[0], (data_dec[1, 0]-data_dec[0, 0]) * 60)

    ind_ra = np.where(fftfreq_ra > 0)

    ind_dec = np.where(fftfreq_dec > 0)

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(fftfreq_ra[ind_ra], np.abs(fft_drift_x[ind_ra]), color='#3fC1C9')

    axs[1].plot(fftfreq_dec[ind_dec], np.abs(fft_drift_y[ind_dec]), color='#3fC1C9')

    if xlim_ra is not None:
        axs[0].set_xlim(0, xlim_ra)
    
    if xlim_dec is not None:
        axs[1].set_xlim(0, xlim_dec)

    # axs[0].set_xticklabels([])

    axs[1].set_xlabel('Frequency (Hz)')

    axs[0].set_ylabel('Power')

    axs[1].set_ylabel('Power')

    axs[0].set_title('East / RA')

    axs[1].set_title('North / DEC')

    fig.tight_layout()

    plt.savefig('{}.eps'.format(outfile), format='eps', dpi=300)


def save_power_spectra_sensor_observation(outfile, filename, xlim=None):

    data = np.loadtxt(filename)

    data[:, 1:] *= 5.5

    drift_ra = np.loadtxt(filename)[:, 1]

    drift_dec = np.loadtxt(filename)[:, 2]

    plt.close('all')

    plt.clf()

    plt.cla()

    fft_drift_y = np.fft.fft(drift_dec)

    fft_drift_x = np.fft.fft(drift_ra)

    fftfreq = np.fft.fftfreq(fft_drift_x.shape[0], data[1, 0] - data[0, 0])

    ind = np.where(fftfreq > 0)

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(fftfreq[ind], np.abs(fft_drift_x[ind]), color='#3fC1C9')

    axs[1].plot(fftfreq[ind], np.abs(fft_drift_y[ind]), color='#3fC1C9')

    if xlim is not None:
        axs[0].set_xlim(0, xlim)
        axs[1].set_xlim(0, xlim)

    axs[0].set_xticklabels([])

    axs[1].set_xlabel('Frequency (Hz)')

    axs[0].set_ylabel('Power')

    axs[1].set_ylabel('Power')

    axs[0].set_title('East / RA')

    axs[1].set_title('North / DEC')

    fig.tight_layout()

    plt.savefig('{}.eps'.format(outfile), format='eps', dpi=300)
