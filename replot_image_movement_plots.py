import numpy as np
import matplotlib.pyplot as plt


# size = plt.rcParams['lines.markersize']


def make_plots(filename, ra_filename, dec_filename, timestring, rad, dad, imagescale=5.5):

    plt.close('all')
    plt.clf()
    plt.cla()

    plt.rc('font', size=24)

    ra_drift = np.loadtxt(ra_filename)

    dec_drift = np.loadtxt(dec_filename)

    fig, axs = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)

    axs[0].scatter(ra_drift[:, 0], ra_drift[:, 1] * imagescale, color='#364f6B')
    axs[0].plot(ra_drift[:, 0], ra_drift[:, 2] * imagescale, label='drift={} arcsec/min'.format(rad), color='#3fC1C9')

    axs[1].scatter(dec_drift[:, 0], dec_drift[:, 1] * imagescale, color='#364f6B')
    axs[1].plot(dec_drift[:, 0], dec_drift[:, 2] * imagescale, label='drift={} arcsec/min'.format(dad), color='#3fC1C9')

    axs[0].set_xlabel('Time in Minutes')
    axs[1].set_xlabel('Time in Minutes')

    axs[0].set_ylabel('Arcseconds')
    axs[1].set_ylabel('Arcseconds')

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    fig.suptitle('Drift Measurement at {} '.format(timestring))

    fig.tight_layout()

    plt.savefig('{}.eps'.format(filename), format='eps')


def make_plots_deviation(filename, ra_filename, dec_filename, timestring, imagescale=5.5):

    plt.close('all')
    plt.clf()
    plt.cla()

    plt.rc('font', size=24)

    ra_drift = np.loadtxt(ra_filename)

    dec_drift = np.loadtxt(dec_filename)

    fig, axs = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)

    axs[0].scatter(ra_drift[:, 0], (ra_drift[:, 1] - ra_drift[:, 2]) * imagescale, color='#364f6B', label='Drift Subtracted data')
    axs[0].plot(ra_drift[:, 0], 0, color='#3fC1C9')

    axs[1].scatter(dec_drift[:, 0], (dec_drift[:, 1] - dec_drift[:, 2]) * imagescale, color='#364f6B', label='Drift Subtracted data')
    axs[1].plot(dec_drift[:, 0], 0, color='#3fC1C9')

    axs[0].set_xlabel('Time in Minutes')
    axs[1].set_xlabel('Time in Minutes')

    axs[0].set_ylabel('Arcseconds')
    axs[1].set_ylabel('Arcseconds')

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    fig.suptitle('Drift Measurement at {} '.format(timestring))

    fig.tight_layout()

    plt.savefig('{}.eps'.format(filename), format='eps')


def make_closeloop_tracking_plots(filename, imagescale=5.5):

    plt.close('all')
    plt.clf()
    plt.cla()

    plt.rc('font', size=24)

    data = np.loadtxt(filename)

    fig, axs = plt.subplots(2, 1, figsize=(19.2, 10.8), dpi=100)

    axs[0].scatter(data[:, 0] / 60, data[:, 1] * imagescale, color='#364f6B')
    axs[0].plot(data[:, 0] / 60, np.zeros_like(data[:, 1]), label='reference position', color='#3fC1C9')


    axs[1].scatter(data[:, 0] / 60, data[:, 2] * imagescale, color='#364f6B')
    axs[1].plot(data[:, 0] / 60, np.zeros_like(data[:, 2]), label='reference position', color='#3fC1C9')


    axs[0].set_xticks([])
    axs[1].set_xlabel('Time in Minutes')

    axs[0].set_ylabel('Arcseconds')
    axs[1].set_ylabel('Arcseconds')

    axs[0].set_title('East / RA')
    axs[1].set_title('North / DEC')

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    fig.tight_layout()

    plt.savefig(filename+'_for_overleaf.eps', format='eps')
