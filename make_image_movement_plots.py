from datetime import timedelta
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MultipleLocator
import os
import h5py
import sunpy.io.fits
import scipy.signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.morphology import closing
from skimage.measure import regionprops


def actual_make_image_movement_plots(sensor_name, mean_values, conversion_factor, time_factor, start_time):
    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(mean_values / conversion_factor)

    pts = np.asarray(plt.ginput(2, 10))

    indices = pts[:, 0].astype(np.int64)

    total_time = time_factor * np.arange(indices[1] - indices[0]) / 60

    values = mean_values[np.arange(indices[0], indices[1])] / conversion_factor

    a, b = np.polyfit(
        total_time,
        values,
        1
    )

    y = a * total_time + b

    ra_fit_data = np.zeros((y.shape[0], 3))

    ra_fit_data[:, 0] = total_time

    ra_fit_data[:, 1] = values

    ra_fit_data[:, 2] = y

    start_data_time = start_time + timedelta(seconds=indices[0] * time_factor)

    np.savetxt(
        'Drift_{}_{}_{}_min.txt'.format(
            sensor_name,
            start_data_time.strftime('%Y-%m-%d_%H:%M:%S'),
            np.round(total_time[-1], 2)
        ),
        ra_fit_data
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(total_time, values, label='Actual Drift')

    plt.plot(
        total_time,
        y,
        label='Drift Rate={} mm/min'.format(
            np.round(
                abs(a),
                2
            )
        )
    )

    plt.xlabel('Time (minutes) from {}'.format(start_data_time.strftime('%Y-%m-%d_%H:%M:%S')))

    plt.ylabel('Movement (mm)')

    plt.title('{} Drift'.format(sensor_name))

    plt.legend()

    plt.savefig(
        'Drift_{}_{}_{}_min.png'.format(
            sensor_name,
            start_data_time.strftime('%Y-%m-%d_%H:%M:%S'),
            np.round(total_time[-1], 2)
        ),
        format='png',
        dpi=600
    )

def make_image_movement_plots(
    filename,
    start_time,
    time_in_min,
    ra_slope=0.446,
    dec_slope=0.417
):
    if (isinstance(start_time, str)):
        start_time = parser.parse(start_time)

    try:
        f = open(filename, encoding='utf-16-le')
        lines = f.readlines()
    except Exception:
        f = open(filename)
        lines = f.readlines()


    data_lines = [line for line in lines if line[0].isdigit()]

    data = np.loadtxt(data_lines)

    x = data[:, 0]

    data = data[:, 1:]

    time_factor = (time_in_min * 60000 ) / (x[-1] * 2)

    time = np.arange(data.shape[0]) * time_factor * 1000

    reshaped_data = data.reshape(data.shape[0], 200, 2)

    ra_data = reshaped_data[:, :, 1]

    dec_data = reshaped_data[:, :, 0]

    mean_ra = np.mean(ra_data, 1)

    mean_dec = np.mean(dec_data, 1)

    ra_factor = 500 * ra_slope

    dec_factor = 500 * dec_slope

    actual_make_image_movement_plots('RA', mean_ra, ra_factor, time_factor, start_time)

    actual_make_image_movement_plots('DEC', mean_dec, dec_factor, time_factor, start_time)


def make_image_tracking_plots(
    filename,
    start_time,
    time_in_min,
    ra_slope,
    dec_slope,
    start_line=0,
    end_line=None,
    image_scale=5.5
):

    size = plt.rcParams['lines.markersize']

    if (isinstance(start_time, str)):
        start_time = parser.parse(start_time)

    try:
        f = open(filename, encoding='utf-16-le')
        lines = f.readlines()
    except Exception:
        f = open(filename)
        lines = f.readlines()

    ra_lines = [line.split(' ')[3] for line in lines if line.startswith('Mean Voltage RA:')][start_line:end_line]

    dec_lines = [line.split(' ')[3] for line in lines if line.startswith('Mean Voltage DEC:')][start_line:end_line]

    reference_ra = float([line.split(' ')[2] for line in lines if line.startswith('ReferenceVoltage RA:')][0])

    reference_dec = float([line.split(' ')[2] for line in lines if line.startswith('ReferenceVoltage DEC:')][0])

    ra_data = np.loadtxt(ra_lines)

    dec_data = np.loadtxt(dec_lines)

    ra_factor = 500 * ra_slope

    dec_factor = 500 * dec_slope

    if (time_in_min > 0):
        time_factor = (time_in_min * 60 ) / (ra_data.shape[0])
    else:
        time_factor = 1

    time = np.arange(ra_data.shape[0]) * time_factor

    ra_drift = (ra_data - reference_ra) / ra_factor

    dec_drift = (dec_data - reference_dec) / dec_factor

    ra_dec_drift_data = np.zeros((ra_drift.shape[0], 3))

    ra_dec_drift_data[:, 0] = time

    ra_dec_drift_data[:, 1] = ra_drift

    ra_dec_drift_data[:, 2] = dec_drift

    np.savetxt(
        'ra_dec_drift_tracking_{}_{}min.txt'.format(
            start_time.strftime('%Y-%m-%d_%H:%M:%S'),
            np.round(time[-1] / 60, 2)
        ),
        ra_dec_drift_data
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(2, 1)

    std_ra = np.round(ra_drift.std(), 3)

    std_dec = np.round(dec_drift.std(), 3)

    axs[0].scatter(time / 60, ra_drift, label='drift data', s=size/4, color='black')

    axs[0].plot(time / 60, np.zeros_like(time), label='reference point', color='red')

    axs[0].set_xlabel(
        'Time in Minutes from {}'.format(
            start_time.strftime('%Y-%m-%d_%H:%M:%S')
        )
    )

    axs[0].set_ylabel('drift in mm')

    axs[0].set_ylim(-0.7, 0.7)

    axs[0].set_title('RA Std Drift: {}'.format(std_ra))

    axs[0].legend(loc="upper right")

    axs[1].scatter(time / 60, dec_drift, label='drift data', s=size/4, color='black')

    axs[1].plot(time / 60, np.zeros_like(time), label='reference point', color='red')

    axs[1].set_xlabel(
        'Time in Minutes from {}'.format(
            start_time.strftime('%Y-%m-%d_%H:%M:%S')
        )
    )

    axs[1].set_ylabel('drift in mm')

    axs[1].set_ylim(-0.7, 0.7)

    axs[1].set_title('DEC Std Drift: {}'.format(std_dec))

    axs[1].legend(loc="upper right")

    fig.suptitle('Closed Loop Performance')

    fig.tight_layout()

    # fig.tight_layout()

    plt.savefig(
        'ra_dec_drift_tracking_{}_{}min.png'.format(
            start_time.strftime('%Y-%m-%d_%H:%M:%S'),
            np.round(time[-1] / 60, 2)
        ),
        format='png',
        dpi=1200
    )


def plot_drift_plot():

    fontsize = 10

    size = plt.rcParams['lines.markersize']

    base_path = Path('/home/harsh/AutoGuiderData/03Feb2021/Processed/')

    drift_ra_file = base_path / 'Drift_RA_2021-02-03_12:54:48_1.21_min.txt'

    drift_dec_file = base_path / 'Drift_DEC_2021-02-03_12:54:46_1.2_min.txt'

    ra_drift = np.loadtxt(drift_ra_file)

    dec_drift = np.loadtxt(drift_dec_file)

    ra_drift[:, 1] *= 5.5

    dec_drift[:, 1] *= 5.5

    ra_drift[:, 1] -= ra_drift[:, 1].max()

    dec_drift[:, 1] -= dec_drift[:, 1].min()

    a, b = np.polyfit(ra_drift[:, 0], ra_drift[:, 1], 1)

    ra_drift_fit = a * ra_drift[:, 0] + b

    a, b = np.polyfit(dec_drift[:, 0], dec_drift[:, 1], 1)

    dec_drift_fit = a * dec_drift[:, 0] + b

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(1, 2, figsize=(6.75, 3))

    axs[0].scatter(ra_drift[:, 0], ra_drift[:, 1], color='midnightblue', s=size/4)

    axs[0].plot(ra_drift[:, 0], ra_drift_fit, color='brown')

    axs[1].scatter(dec_drift[:, 0], dec_drift[:, 1], color='midnightblue', s=size/4)

    axs[1].plot(dec_drift[:, 0], dec_drift_fit, color='brown')

    axs[0].set_xlabel('Time [minutes]', fontsize=fontsize)

    axs[1].set_xlabel('Time [minutes]', fontsize=fontsize)

    axs[0].set_ylabel('Position [arcsec]', fontsize=fontsize)

    # axs[1].set_ylabel('Position [arcsec]', fontsize=fontsize)

    axs[0].set_xticks([0, 0.5, 1])
    axs[0].set_xticklabels([0, 0.5, 1])

    axs[1].set_xticks([0, 0.5, 1])
    axs[1].set_xticklabels([0, 0.5, 1])

    # axs[1].set_yticks([0, 2, 4, 6])
    # axs[1].set_yticklabels([0, 2, 4, 6])

    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.5))

    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.5))

    axs[0].text(
        0.11, 0.94,
        r'Drift = 6.93 arcsec $\mathrm{minutes^{-1}}$',
        transform=axs[0].transAxes,
        fontsize=fontsize
    )

    axs[1].text(
        0.11, 0.94,
        r'Drift = 5.01 arcsec $\mathrm{minutes^{-1}}$',
        transform=axs[1].transAxes,
        fontsize=fontsize
    )

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.98, wspace=0.2, hspace=0.2)

    write_path = Path('/home/harsh/AutoGuiderPaper/')

    fig.savefig(write_path / 'drift_measurement.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


def plot_dark_data():

    dark_data = np.array(
        [
            -14.7479, -13.459, -11.3645, 7.3249, 8.93605, 20.053,
            32.7811, 28.5921, 9.90275, 4.10259, -10.72, 5.06928,
            11.675, -11.3645, -8.46441, 9.09717, 14.0917, 13.9306,
            20.8586, 12.9639, -5.56434, -7.49772, -16.198, -20.2258,
            -0.891988, 22.4698, 15.5418, 14.7362, 13.2862, 7.00267,
            -10.5589, -18.4536, -7.98107, -22.6426, -19.098, -16.8424,
            3.29701, 15.5418, 21.9864, 10.3861, 14.414, 7.96936, 7.64713,
            -22.8037, -6.53103, -2.98649, 2.00809, 7.96936, 15.0584,
            29.7199, -2.98649, 6.35821, 7.00267, -17.9702, -16.6813,
            -20.7092, 12.8028, -6.2088, 18.1196, 19.0863
        ]
    )

    drift_data = dark_data * 5.5 / (500 * 0.543)

    fontsize = 8

    plt.close('all')

    plt.cla()

    plt.clf()

    plt.scatter(np.arange(dark_data.shape[0]), dark_data, color='black')

    plt.gca().set_xticks([0, 10, 20, 30, 40, 50, 60])
    plt.gca().set_xticklabels([0, 10, 20, 30, 40, 50, 60], fontsize=fontsize)

    plt.gca().set_yticks([-30, -20, -10, 0, 10, 20, 30])
    plt.gca().set_yticklabels([-30, -20, -10, 0, 10, 20, 30], fontsize=fontsize)

    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

    plt.plot(np.ones_like(dark_data) * dark_data.mean(), color='red')

    plt.ylim(-35, 35)
    plt.xlim(-1, 61)

    fig = plt.gcf()

    fig.set_size_inches(7, 3, forward=True)

    plt.xlabel('Samples [number]', fontsize=fontsize)

    plt.ylabel('Voltage [mVolts]', fontsize=fontsize)

    plt.gca().text(
        0.4, 0.9,
        'Std: {} mVolts or 0.3 arcsec'.format(np.round(dark_data.std(), 2)),
        transform=plt.gca().transAxes,
        fontsize=fontsize
    )

    axs2 = plt.gca().twinx()
    axs2.set_ylim(-0.7, 0.7)
    axs2.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    axs2.set_yticklabels([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6], fontsize=fontsize)
    axs2.yaxis.set_minor_locator(MultipleLocator(0.05))
    axs2.set_ylabel('Position [arcsec]', fontsize=fontsize)

    plt.subplots_adjust(left=0.07, bottom=0.13, right=0.93, top=0.99, wspace=0., hspace=0.0)
    write_path = Path('/home/harsh/AutoGuiderPaper/')

    fig.savefig(write_path / 'darkdata.pdf', format='pdf', dpi=300)


def make_image_tracking_plots_new(
        filename,
        ylim1,
        ylim2,
        xticks,
        yticks1,
        yticks2,
        label=None,
        ra_slope=0.446,
        dec_slope=0.417,
        start_line=0,
        end_line=None,
        image_scale=5.5,
        time_factor=1,
        scatter_factor=4,
        savefilename=None,
):

    fontsize = 6

    size = plt.rcParams['lines.markersize']

    try:
        f = open(filename, encoding='utf-16-le')
        lines = f.readlines()
    except Exception:
        f = open(filename)
        lines = f.readlines()

    ra_lines = [line.split(' ')[3] for line in lines if line.startswith('Mean Voltage RA:')][start_line:end_line]

    dec_lines = [line.split(' ')[3] for line in lines if line.startswith('Mean Voltage DEC:')][start_line:end_line]

    reference_ra = float([line.split(' ')[2] for line in lines if line.startswith('ReferenceVoltage RA:')][0])

    reference_dec = float([line.split(' ')[2] for line in lines if line.startswith('ReferenceVoltage DEC:')][0])

    ra_data = np.loadtxt(ra_lines)

    dec_data = np.loadtxt(dec_lines)

    ra_factor = 500 * ra_slope

    dec_factor = 500 * dec_slope

    time = np.arange(ra_data.shape[0]) * time_factor

    ra_drift = (ra_data - reference_ra) / ra_factor

    dec_drift = (dec_data - reference_dec) / dec_factor

    ra_dec_drift_data = np.zeros((ra_drift.shape[0], 3))

    ra_dec_drift_data[:, 0] = time

    ra_dec_drift_data[:, 1] = ra_drift

    ra_dec_drift_data[:, 2] = dec_drift

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(2, 1, figsize=(3, 2))

    axs[0].scatter(time / 60, ra_drift * image_scale, s=size/scatter_factor, color='black')

    axs[0].plot(time / 60, np.zeros_like(time), color='red')

    axs[0].set_ylabel('Position [arcsec]',
                      fontsize=fontsize)

    axs[1].scatter(time / 60, dec_drift * image_scale, s=size/scatter_factor, color='black')

    axs[1].plot(time / 60, np.zeros_like(time), color='red')

    print(
        np.sqrt(
            np.mean(
                (dec_drift * image_scale)**2
            )
        )
    )
    print(np.sqrt(np.mean((ra_drift * image_scale)**2)))
    axs[1].set_xlabel(
        'Time [minutes]',
        fontsize=fontsize
    )

    axs[1].set_ylabel('Position [arcsec]',
                      fontsize=fontsize)

    if label is not None:
        axs[0].text(
            -0.15, 1.1,
            label,
            transform=axs[0].transAxes,
            fontsize=fontsize
        )

        # axs[1].text(
        #     0.05, 0.9,
        #     label,
        #     transform=axs[1].transAxes,
        #     fontsize=fontsize
        # )

    axs[0].text(
        0.45, 0.9,
        'East-West',
        transform=axs[0].transAxes,
        fontsize=fontsize
    )
    axs[1].text(
        0.43, 0.85,
        'North-South',
        transform=axs[1].transAxes,
        fontsize=fontsize
    )
    axs[0].set_ylim(*ylim1)
    axs[1].set_ylim(*ylim2)

    axs[0].set_xticks([])
    axs[0].set_xticklabels([])

    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticks, fontsize=fontsize)

    axs[0].set_yticks(yticks1)
    axs[0].set_yticklabels(yticks1, fontsize=fontsize)

    axs[1].set_yticks(yticks2)
    axs[1].set_yticklabels(yticks2, fontsize=fontsize)

    plt.subplots_adjust(left=0.15, bottom=0.17, right=0.98, top=0.91, wspace=0.1, hspace=0.1)
    write_path = Path('/home/harsh/AutoGuiderPaper/')

    if savefilename is None:
        fig.savefig(write_path / '{}.pdf'.format(Path(filename).name), format='pdf', dpi=300)
    else:
        fig.savefig(write_path / savefilename, format='pdf', dpi=300)


def make_sunspot_plot():
    cwd = os.getcwd()
    os.chdir('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/AutoGuider March 2022/')
    dark, _ = sunpy.io.fits.read('dark.fit')[0]
    dark = np.mean(dark, 0)
    flat_f = h5py.File('flat_31032022.hdf5', 'r')
    tot = len(flat_f.keys()) // 2
    flat = np.zeros((tot, flat_f['Image_0'].shape[0], flat_f['Image_0'].shape[1]))
    for i in range(tot):
        flat[i] = flat_f['Image_{}'.format(i)][()]
    flat = np.mean(flat, 0)
    flat_f.close()
    data_f = h5py.File('buffer_31032022_082248_136_0_357.hdf5', 'r')
    index = 690
    plt.close('all')
    plt.clf()
    plt.cla()
    fontsize = 8
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 2))
    data = (data_f['Image_{}'.format(index)][()] - dark) * np.mean(flat-dark) / (flat - dark)
    im = axs.imshow(data, cmap='gray', origin='lower', extent=[0, data.shape[1] * 5.5 * 9/1000, 0, data.shape[0] * 5.5 * 9/1000])
    cbaxes = inset_axes(
        axs,
        width="5%",
        height="100%",
        loc='right',
        borderpad=-1.5
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        orientation='vertical'
    )
    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    axs.text(
        1.33, 0.05,
        'Intensity [digital counts]',
        transform=axs.transAxes,
        rotation=90,
        fontsize=fontsize
    )
    axs.text(
        -0.15, 0.9,
        '(b)',
        transform=axs.transAxes,
        fontsize=fontsize
    )
    axs.set_xlabel(r'$x$ [arcsec]', fontsize=fontsize)
    axs.set_ylabel(r'$y$ [arcsec]', fontsize=fontsize)
    axs.set_yticks([0, 20, 40])
    axs.set_yticklabels([0, 20, 40], fontsize=fontsize)
    axs.set_xticks([0, 20, 40, 60])
    axs.set_xticklabels([0, 20, 40, 60], fontsize=fontsize)
    axs.xaxis.set_minor_locator(MultipleLocator(5))
    axs.yaxis.set_minor_locator(MultipleLocator(5))
    plt.subplots_adjust(left=0.13, bottom=0.2, right=0.75, top=0.99, wspace=0.0, hspace=0.0)
    write_path = Path('/home/harsh/AutoGuiderPaper/')
    fig.savefig(write_path / 'sunspot_image.pdf', format='pdf', dpi=300)
    fig.savefig(write_path / 'sunspot_image.png', format='png', dpi=300)
    plt.show()
    plt.close('all')
    plt.clf()
    plt.cla()
    data_f.close()
    os.chdir(cwd)


def plot_camera_drift_plot():
    write_path = Path('/home/harsh/AutoGuiderPaper/')
    fontsize = 6
    size = plt.rcParams['lines.markersize']
    cwd = os.getcwd()
    os.chdir('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/AutoGuider March 2022/')
    dark, _ = sunpy.io.fits.read('dark.fit')[0]
    dark = np.mean(dark, 0)
    flat_f = h5py.File('flat_31032022.hdf5', 'r')
    tot = len(flat_f.keys()) // 2
    flat = np.zeros((tot, flat_f['Image_0'].shape[0], flat_f['Image_0'].shape[1]))
    for i in range(tot):
        flat[i] = flat_f['Image_{}'.format(i)][()]
    flat = np.mean(flat, 0)
    flat_f.close()
    if Path(write_path / 'buffer_31032022_082248_034_0032_237_drifts.txt').exists():
        gg = np.loadtxt(write_path / 'buffer_31032022_082248_034_0032_237_drifts.txt')
        a = gg[:, 0]
        b = gg[:, 1]
    else:
        data_f = h5py.File('buffer_31032022_082248_034_0032_237.hdf5', 'r')
        dtot = len(data_f.keys()) // 2
        a, b = np.zeros(dtot), np.zeros(dtot)
        for index in range(dtot):
            data = (data_f['Image_{}'.format(index)][()] - dark) / (flat - dark)
            mn = data.mean()
            sd = data.std()
            k = -2
            mask = np.zeros_like(data, dtype=np.int64)
            i, j = np.where(data < (mn + (k * sd)))
            mask[i, j] = 1
            mask = closing(mask)
            k = 0
            for region in regionprops(mask, data):
                a[index], b[index] = region.centroid
                print (index)
                break
        gg = np.zeros((a.size, 2), dtype=np.float64)
        gg[:, 0] = a
        gg[:, 1] = b
        np.savetxt(write_path / 'buffer_31032022_082248_034_0032_237_drifts.txt', gg)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
    time = np.arange(0, (dtot-240) * 0.5, 0.5) / 60
    y1 = (a[240:] - a[240:].min()) * 0.0495
    y2 = (b[240:] - b[240:].min()) * 0.0495
    m1, c1 = np.polyfit(time, y1, 1)
    m2, c2 = np.polyfit(time, y2, 1)
    y1d = time * m1 + c1
    y2d = time * m2 + c2
    axs[0].scatter(time, y2, s=size/4, color='black')
    axs[0].plot(time, y2d, color='red')
    axs[1].scatter(time, y1, s=size/4, color='black')
    axs[1].plot(time, y1d, color='red')
    axs[0].text(
        0.1, 0.9,
        r'Drift RA = {} '.format(np.round(m2, 2))+r'arcsec $\mathrm{minute^{-1}}$',
        transform=axs[0].transAxes,
        fontsize=fontsize
    )
    axs[1].text(
        0.1, 0.9,
        r'Drift RA = {} '.format(np.round(m1, 2))+r'arcsec $\mathrm{minute^{-1}}$',
        transform=axs[1].transAxes,
        fontsize=fontsize
    )
    axs[0].set_ylabel(r'Position [arcsec]', fontsize=fontsize)
    axs[0].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[1].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[0].set_xticks([0, 2, 4, 6, 8])
    axs[1].set_xticks([0, 2, 4, 6, 8])
    axs[0].set_xticklabels([0, 2, 4, 6, 8], fontsize=fontsize)
    axs[1].set_xticklabels([0, 2, 4, 6, 8], fontsize=fontsize)
    axs[0].set_yticks([0, 2, 4, 6])
    axs[1].set_yticks([0, 2, 4, 6, 8, 10])
    axs[0].set_yticklabels([0, 2, 4, 6], fontsize=fontsize)
    axs[1].set_yticklabels([0, 2, 4, 6, 8, 10], fontsize=fontsize)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.1, hspace=0)
    fig.savefig(write_path / 'image_drift_31030822.pdf', format='pdf', dpi=300)
    fig.savefig(write_path / 'image_drift_31030822.png', format='png', dpi=300)
    plt.close('all')
    plt.clf()
    plt.cla()
    data_f.close()
    os.chdir(cwd)


def plot_camera_closeloop_plot_and_fft():
    write_path = Path('/home/harsh/AutoGuiderPaper/')
    fontsize = 8
    size = plt.rcParams['lines.markersize']
    cwd = os.getcwd()
    os.chdir('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/AutoGuider March 2022/')
    dark, _ = sunpy.io.fits.read('dark.fit')[0]
    dark = np.mean(dark, 0)
    flat_f = h5py.File('flat_30032022.hdf5', 'r')
    tot = len(flat_f.keys()) // 2
    flat = np.zeros((tot, flat_f['Image_0'].shape[0], flat_f['Image_0'].shape[1]))
    for i in range(tot):
        flat[i] = flat_f['Image_{}'.format(i)][()]
    flat = np.mean(flat, 0)
    flat_f.close()
    data_f = None
    if Path(write_path / 'buffer_30032022_082118_17_0_0.hdf5.txt').exists():
        gg = np.loadtxt(write_path / 'buffer_30032022_082118_17_0_0.hdf5.txt')
        a = gg[:, 0]
        b = gg[:, 1]
        dtot = a.size
    else:
        data_f = h5py.File('buffer_30032022_082118_17_0_0.hdf5', 'r')
        dtot = len(data_f.keys()) // 2
        a, b = np.zeros(dtot), np.zeros(dtot)
        for index in range(dtot):
            data = (data_f['Image_{}'.format(index)][()] - dark) / (flat - dark)
            mn = data.mean()
            sd = data.std()
            k = -2
            mask = np.zeros_like(data, dtype=np.int64)
            i, j = np.where(data < (mn + (k * sd)))
            mask[i, j] = 1
            mask = closing(mask)
            k = 0
            for region in regionprops(mask, data):
                a[index], b[index] = region.centroid
                print (index)
                break
        gg = np.zeros((a.size, 2), dtype=np.float64)
        gg[:, 0] = a
        gg[:, 1] = b
        np.savetxt(write_path / 'buffer_30032022_082118_17_0_0.hdf5.txt', gg)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5))
    time = np.arange(0, dtot * 0.5, 0.5) / 60
    y1 = (a - a.mean()) * 0.0495
    y2 = (b - b.mean()) * 0.0495
    fft_y1 = np.fft.fft(y1, norm='ortho')
    fft_y2 = np.fft.fft(y2, norm='ortho')
    fftfreq = np.fft.fftfreq(y1.size, 0.5)
    ind = np.where((fftfreq > 0.004) & (fftfreq < 0.2))[0]
    axs[0][0].scatter(time, y2, s=size/4, color='black')
    axs[0][0].plot(time, np.zeros_like(y2), color='red')
    axs[0][1].scatter(time, y1, s=size/4, color='black')
    axs[0][1].plot(time, np.zeros_like(y1), color='red')
    axs[1][0].plot(1 / fftfreq[ind], np.abs(fft_y2[ind])**2)
    axs[1][1].plot(1 / fftfreq[ind], np.abs(fft_y1[ind])**2)
    axs[0][0].set_ylabel(r'Position [arcsec]', fontsize=fontsize)
    axs[0][0].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[0][1].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[1][0].set_ylabel(r'Relative Power [arbitrary units]', fontsize=fontsize)
    axs[1][0].set_xlabel(r'Period [s]', fontsize=fontsize)
    axs[1][1].set_xlabel(r'Period [s]', fontsize=fontsize)
    axs[0][0].set_xticks([0, 5, 10, 15])
    axs[0][1].set_xticks([0, 5, 10, 15])
    axs[0][0].set_xticklabels([0, 5, 10, 15], fontsize=fontsize)
    axs[0][1].set_xticklabels([0, 5, 10, 15], fontsize=fontsize)
    axs[0][0].set_yticks([-4, -2, 0, 2, 4])
    axs[0][1].set_yticks([-4, -2, 0, 2, 4])
    axs[0][0].set_yticklabels([-4, -2, 0, 2, 4], fontsize=fontsize)
    axs[0][1].set_yticklabels([-4, -2, 0, 2, 4], fontsize=fontsize)
    axs[0][0].set_ylim(-6, 6)
    axs[0][1].set_ylim(-6, 6)
    axs[1][0].set_xlim(0, 100)
    axs[1][1].set_xlim(0, 100)
    axs[1][0].set_xticks([0, 20, 40, 60, 80])
    axs[1][1].set_xticks([0, 20, 40, 60, 80])
    axs[1][0].set_xticklabels([0, 20, 40, 60, 80], fontsize=fontsize)
    axs[1][1].set_xticklabels([0, 20, 40, 60, 80], fontsize=fontsize)
    # axs[1][0].set_yticks([0, 100, 200, 300])
    # axs[1][1].set_yticks([0, 50, 100, 150])
    axs[1][0].set_yticklabels([0, 100, 200, 300], fontsize=fontsize)
    axs[1][1].set_yticklabels([0, 50, 100, 150], fontsize=fontsize)
    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(0.5))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(0.5))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(5))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(5))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(25))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(25))
    axs[0][0].text(
        0.01, 0.94,
        '(a)',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.01, 0.94,
        '(b)',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[1][0].text(
        0.01, 0.94,
        '(c)',
        transform=axs[1][0].transAxes,
        fontsize=fontsize
    )
    axs[1][1].text(
        0.01, 0.94,
        '(d)',
        transform=axs[1][1].transAxes,
        fontsize=fontsize
    )
    plt.subplots_adjust(left=0.075, bottom=0.08, right=0.99, top=0.99, wspace=0.12, hspace=0.2)
    fig.savefig(write_path / 'closeloop_17.pdf', format='pdf', dpi=300)
    fig.savefig(write_path / 'closeloop_17.png', format='png', dpi=300)
    plt.close('all')
    plt.clf()
    plt.cla()
    if data_f is not None:
        data_f.close()
    os.chdir(cwd)


def plot_camera_closeloop_plot(filename, flatfilename, kp, ki, kd, xticks=None, yticks1=None, yticks2=None, ylim1=None, ylim2=None, xminortick=None, yminortick=None):
    write_path = Path('/home/harsh/AutoGuiderPaper/')
    fontsize = 8
    size = plt.rcParams['lines.markersize']
    cwd = os.getcwd()
    os.chdir('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/AutoGuider March 2022/')
    dark, _ = sunpy.io.fits.read('dark.fit')[0]
    dark = np.mean(dark, 0)
    flat_f = h5py.File(flatfilename, 'r')
    tot = len(flat_f.keys()) // 2
    flat = np.zeros((tot, flat_f['Image_0'].shape[0], flat_f['Image_0'].shape[1]))
    for i in range(tot):
        flat[i] = flat_f['Image_{}'.format(i)][()]
    flat = np.mean(flat, 0)
    flat_f.close()
    data_f = None
    txtfile = write_path / '{}.txt'.format(filename)
    if Path(txtfile).exists():
        gg = np.loadtxt(txtfile)
        a = gg[:, 0]
        b = gg[:, 1]
        dtot = a.size
    else:
        data_f = h5py.File(filename, 'r')
        dtot = len(data_f.keys()) // 2
        a, b = np.zeros(dtot), np.zeros(dtot)
        for index in range(dtot):
            data = (data_f['Image_{}'.format(index)][()] - dark) / (flat - dark)
            mn = data.mean()
            sd = data.std()
            k = -2
            mask = np.zeros_like(data, dtype=np.int64)
            i, j = np.where(data < (mn + (k * sd)))
            mask[i, j] = 1
            mask = closing(mask)
            k = 0
            for region in regionprops(mask, data):
                a[index], b[index] = region.centroid
                print (index)
                break
        gg = np.zeros((a.size, 2), dtype=np.float64)
        gg[:, 0] = a
        gg[:, 1] = b
        np.savetxt(txtfile, gg)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
    time = np.arange(0, dtot * 0.5, 0.5) / 60
    y1 = (a - np.median(a)) * 0.0495
    y2 = (b - np.median(b)) * 0.0495
    axs[0].scatter(time, y2, s=size/4, color='black')
    axs[0].plot(time, np.zeros_like(y2), color='red')
    axs[1].scatter(time, y1, s=size/4, color='black')
    axs[1].plot(time, np.zeros_like(y1), color='red')
    axs[0].set_ylabel(r'Position [arcsec]', fontsize=fontsize)
    axs[0].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[1].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    if xticks is None:
        xticks = [0, 5, 10, 15]
    if yticks1 is None:
        yticks1 = [-4, -2, 0, 2, 4]
    if yticks2 is None:
        yticks2 = [-4, -2, 0, 2, 4]
    if ylim1 is None:
        ylim1 = (-6, 6)
    if ylim2 is None:
        ylim2 = (-6, 6)
    if xminortick is None:
        xminortick = 1
    if yminortick is None:
        yminortick = 0.5
    axs[0].set_xticks(xticks)
    axs[1].set_xticks(xticks)
    axs[0].set_xticklabels(xticks, fontsize=fontsize)
    axs[1].set_xticklabels(xticks, fontsize=fontsize)
    axs[0].set_ylim(*ylim1)
    axs[1].set_ylim(*ylim2)
    axs[0].set_yticks(yticks1)
    axs[1].set_yticks(yticks2)
    axs[0].set_yticklabels(yticks1, fontsize=fontsize)
    axs[1].set_yticklabels(yticks2, fontsize=fontsize)
    axs[0].xaxis.set_minor_locator(MultipleLocator(xminortick))
    axs[1].xaxis.set_minor_locator(MultipleLocator(xminortick))
    axs[0].yaxis.set_minor_locator(MultipleLocator(yminortick))
    axs[1].yaxis.set_minor_locator(MultipleLocator(yminortick))
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.98, wspace=0.1, hspace=0)
    fig.savefig(write_path / 'closeloop_{}_{}_{}.pdf'.format(kp, ki, kd), format='pdf', dpi=300)
    fig.savefig(write_path / 'closeloop_{}_{}_{}.png'.format(kp, ki, kd), format='png', dpi=300)
    plt.close('all')
    plt.clf()
    plt.cla()
    if data_f is not None:
        data_f.close()
    os.chdir(cwd)


def plot_camera_closeloop_plot_and_fft_alternate(filename, flatfilename, kp, ki, kd, xticks=None, yticks1=None, yticks2=None, ylim1=None, ylim2=None, xminortick=None, yminortick=None, yticks3=None, yticks4=None, yminortick3=None, yminortick4=None, ylim3=None, ylim4=None):
    write_path = Path('/home/harsh/AutoGuiderPaper/')
    fontsize = 8
    size = plt.rcParams['lines.markersize']
    cwd = os.getcwd()
    os.chdir('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/AutoGuider March 2022/')
    dark, _ = sunpy.io.fits.read('dark.fit')[0]
    dark = np.mean(dark, 0)
    flat_f = h5py.File(flatfilename, 'r')
    tot = len(flat_f.keys()) // 2
    flat = np.zeros((tot, flat_f['Image_0'].shape[0], flat_f['Image_0'].shape[1]))
    for i in range(tot):
        flat[i] = flat_f['Image_{}'.format(i)][()]
    flat = np.mean(flat, 0)
    flat_f.close()
    data_f = None
    txtfile = write_path / '{}.txt'.format(filename)
    if Path(txtfile).exists():
        gg = np.loadtxt(txtfile)
        a = gg[:, 0]
        b = gg[:, 1]
        dtot = a.size
    else:
        data_f = h5py.File(filename, 'r')
        dtot = len(data_f.keys()) // 2
        a, b = np.zeros(dtot), np.zeros(dtot)
        for index in range(dtot):
            data = (data_f['Image_{}'.format(index)][()] - dark) / (flat - dark)
            mn = data.mean()
            sd = data.std()
            k = -2
            mask = np.zeros_like(data, dtype=np.int64)
            i, j = np.where(data < (mn + (k * sd)))
            mask[i, j] = 1
            mask = closing(mask)
            k = 0
            for region in regionprops(mask, data):
                a[index], b[index] = region.centroid
                # print (index)
                break
        gg = np.zeros((a.size, 2), dtype=np.float64)
        gg[:, 0] = a
        gg[:, 1] = b
        np.savetxt(txtfile, gg)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 6))
    time = np.arange(0, dtot * 0.5, 0.5) / 60
    y1 = (a - np.median(a)) * 0.0495
    y2 = (b - np.median(b)) * 0.0495
    print(
        np.sqrt(
            np.mean(
                y1**2
            )
        )
    )
    print(
        np.sqrt(
            np.mean(
                y2 ** 2
            )
        )
    )
    fft_y1 = np.fft.fft(y1, norm='ortho')
    fft_y2 = np.fft.fft(y2, norm='ortho')
    fftfreq = np.round(np.fft.fftfreq(y1.size, 0.5), 3)
    ind = np.where((fftfreq > 0.004) & (fftfreq < 0.2))[0]
    # print(fftfreq[ind][1] - fftfreq[ind][0])
    # print(fftfreq[ind][2] - fftfreq[ind][1])
    # ind = np.where(fftfreq > 0)[0]
    # print(np.max(np.abs(fft_y1[ind])**2))
    # print(np.max(np.abs(fft_y2[ind])**2))
    # print(fftfreq[ind][np.where(np.abs(fft_y1[ind])**2 > 15)[0]])
    axs[0][0].scatter(time, y2, s=size/4, color='black')
    axs[0][0].plot(time, np.zeros_like(y2), color='red')
    axs[0][1].scatter(time, y1, s=size/4, color='black')
    axs[0][1].plot(time, np.zeros_like(y1), color='red')
    axs[1][0].plot(fftfreq[ind], np.abs(fft_y2[ind])**2)
    axs[1][1].plot(fftfreq[ind], np.abs(fft_y1[ind])**2)
    axs[0][0].set_ylabel(r'Position [arcsec]', fontsize=fontsize)
    axs[0][0].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[0][1].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[1][0].set_ylabel(r'Relative Power [arbitrary units]', fontsize=fontsize)
    axs[1][0].set_xlabel(r'Period [s]', fontsize=fontsize)
    axs[1][1].set_xlabel(r'Period [s]', fontsize=fontsize)
    if xticks is None:
        xticks = [0, 5, 10, 15]
    if yticks1 is None:
        yticks1 = [-4, -2, 0, 2, 4]
    if yticks2 is None:
        yticks2 = [-4, -2, 0, 2, 4]
    if ylim1 is None:
        ylim1 = (-6, 6)
    if ylim2 is None:
        ylim2 = (-6, 6)
    if xminortick is None:
        xminortick = 1
    if yminortick is None:
        yminortick = 0.5
    axs[0][0].set_xticks(xticks)
    axs[0][1].set_xticks(xticks)
    axs[0][0].set_xticklabels(xticks, fontsize=fontsize)
    axs[0][1].set_xticklabels(xticks, fontsize=fontsize)
    axs[0][0].set_ylim(*ylim1)
    axs[0][1].set_ylim(*ylim2)
    axs[0][0].set_yticks(yticks1)
    axs[0][1].set_yticks(yticks2)
    axs[0][0].set_yticklabels(yticks1, fontsize=fontsize)
    axs[0][1].set_yticklabels(yticks2, fontsize=fontsize)
    axs[0][0].xaxis.set_minor_locator(MultipleLocator(xminortick))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(xminortick))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(yminortick))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(yminortick))
    axs[1][0].set_xlim(0.004, 0.2)
    axs[1][1].set_xlim(0.004, 0.2)
    # axs[1][0].set_xticks([0, 20, 40, 60, 80])
    # axs[1][1].set_xticks([0, 20, 40, 60, 80])
    # axs[1][0].set_xticklabels([0, 20, 40, 60, 80], fontsize=fontsize)
    # axs[1][1].set_xticklabels([0, 20, 40, 60, 80], fontsize=fontsize)
    if yticks3 is None:
        yticks3 = [0, 50, 100, 150, 200]
    if yticks4 is None:
        yticks4 = [0, 50, 100, 150, 200]
    if yminortick3 is None:
        yminortick3 = 25
    if yminortick4 is None:
        yminortick4 = 25
    if ylim3 is None:
        ylim3 = (0, 225)
    if ylim4 is None:
        ylim4 = (0, 225)
    axs[1][0].set_yticks(yticks3)
    axs[1][1].set_yticks(yticks4)
    axs[1][0].set_ylim(*ylim3)
    axs[1][1].set_ylim(*ylim4)
    axs[1][0].set_yticklabels(yticks3, fontsize=fontsize)
    axs[1][1].set_yticklabels(yticks4, fontsize=fontsize)
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(yminortick3))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(yminortick4))
    axs[0][0].text(
        0.01, 0.94,
        '(a)',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.01, 0.94,
        '(b)',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[1][0].text(
        0.01, 0.94,
        '(c)',
        transform=axs[1][0].transAxes,
        fontsize=fontsize
    )
    axs[1][1].text(
        0.01, 0.94,
        '(d)',
        transform=axs[1][1].transAxes,
        fontsize=fontsize
    )
    plt.subplots_adjust(left=0.075, bottom=0.08, right=0.99, top=0.99, wspace=0.12, hspace=0.2)
    fig.savefig(write_path / 'closeloop_fft_{}_{}_{}.pdf'.format(kp, ki, kd), format='pdf', dpi=300)
    fig.savefig(write_path / 'closeloop_fft_{}_{}_{}.png'.format(kp, ki, kd), format='png', dpi=300)
    plt.close('all')
    plt.clf()
    plt.cla()
    if data_f is not None:
        data_f.close()
    os.chdir(cwd)


def plot_camera_drift_plot_and_fft(filename, flatfilename, xticks=None, yticks1=None, yticks2=None, ylim1=None, ylim2=None, xminortick=None, yminortick=None, yticks3=None, yticks4=None, yminortick3=None, yminortick4=None, ylim3=None, ylim4=None, flag=0):
    write_path = Path('/home/harsh/AutoGuiderPaper/')
    # write_path = Path('/Users/harshmathur/AutoGuiderPaper')
    fontsize = 8
    size = plt.rcParams['lines.markersize']
    cwd = os.getcwd()
    # os.chdir('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/AutoGuider March 2022/')
    # os.chdir('/Volumes/SeagateHarsh9599771751/AutoGuider March 2022')
    # dark, _ = sunpy.io.fits.read('dark.fit')[0]
    # dark = np.mean(dark, 0)
    # flat_f = h5py.File(flatfilename, 'r')
    # tot = len(flat_f.keys()) // 2
    # flat = np.zeros((tot, flat_f['Image_0'].shape[0], flat_f['Image_0'].shape[1]))
    # for i in range(tot):
    #     flat[i] = flat_f['Image_{}'.format(i)][()]
    # flat = np.mean(flat, 0)
    # flat_f.close()
    os.chdir('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/AutoGuider March 2022/')
    # os.chdir('/Volumes/SeagateHarsh9599771751/AutoGuider March 2022')
    dark, _ = sunpy.io.fits.read('dark.fit')[0]
    dark = np.mean(dark, 0)
    flat_f = h5py.File(flatfilename, 'r')
    tot = len(flat_f.keys()) // 2
    flat = np.zeros((tot, flat_f['Image_0'].shape[0], flat_f['Image_0'].shape[1]))
    for i in range(tot):
        flat[i] = flat_f['Image_{}'.format(i)][()]
    flat = np.mean(flat, 0)
    flat_f.close()
    data_f = None
    txtfile = write_path / '{}.txt'.format(filename)
    if Path(txtfile).exists():
        gg = np.loadtxt(txtfile)
        a = gg[:, 0]
        b = gg[:, 1]
        dtot = a.size
    else:
        data_f = h5py.File(filename, 'r')
        dtot = len(data_f.keys()) // 2
        a, b = np.zeros(dtot), np.zeros(dtot)
        for index in range(dtot):
            data = (data_f['Image_{}'.format(index)][()] - dark) / (flat - dark)
            mn = data.mean()
            sd = data.std()
            k = -2
            mask = np.zeros_like(data, dtype=np.int64)
            i, j = np.where(data < (mn + (k * sd)))
            mask[i, j] = 1
            mask = closing(mask)
            k = 0
            for region in regionprops(mask, data):
                a[index], b[index] = region.centroid
                print (index)
                break
        gg = np.zeros((a.size, 2), dtype=np.float64)
        gg[:, 0] = a
        gg[:, 1] = b
        np.savetxt(txtfile, gg)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 6))
    time = np.arange(0, dtot * 0.5, 0.5) / 60
    y1 = (a - np.min(a)) * 0.0495
    y2 = (b - np.min(b)) * 0.0495
    g1, g2, g3, g4, g5, g6 = np.polyfit(np.arange(a.size), y1, 5)
    y1l = np.arange(a.size)**5 * g1 + np.arange(a.size)**4 * g2 + np.arange(a.size)**3 * g3 + np.arange(a.size)**2 * g4 + np.arange(a.size) * g5 + g6
    h1, h2, h3, h4, h5, h6 = np.polyfit(np.arange(b.size), y2, 5)
    y2l = np.arange(b.size)**5 * h1 + np.arange(b.size)**4 * h2 + np.arange(b.size)**3 * h3 + np.arange(b.size)**2 * h4 + np.arange(b.size) * h5 + h6
    fft_y1 = np.fft.fft(y1 - y1l, norm='ortho')
    fft_y2 = np.fft.fft(y2 - y2l, norm='ortho')
    fftfreq = np.fft.fftfreq(y1.size, 0.5)
    ind = np.where((fftfreq > 0.004) & (fftfreq < 0.2))[0]
    if flag == 0:
        axs[0][0].scatter(time, y2, s=size/4, color='black')
        axs[0][0].plot(time, y2l, color='red')
        axs[0][1].scatter(time, y1, s=size/4, color='black')
        axs[0][1].plot(time, y1l, color='red')
    else:
        axs[0][0].scatter(time, y2 - y2l, s=size / 4, color='black')
        axs[0][0].plot(time, np.zeros_like(y2), color='red')
        axs[0][1].scatter(time, y1 - y1l, s=size / 4, color='black')
        axs[0][1].plot(time, np.zeros_like(y1), color='red')
    axs[1][0].plot(1 / fftfreq[ind], np.abs(fft_y2[ind])**2)
    axs[1][1].plot(1 / fftfreq[ind], np.abs(fft_y1[ind])**2)
    axs[0][0].set_ylabel(r'Position [arcsec]', fontsize=fontsize)
    axs[0][0].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[0][1].set_xlabel(r'Time [minutes]', fontsize=fontsize)
    axs[1][0].set_ylabel(r'Relative Power [arbitrary units]', fontsize=fontsize)
    axs[1][0].set_xlabel(r'Period [s]', fontsize=fontsize)
    axs[1][1].set_xlabel(r'Period [s]', fontsize=fontsize)
    if xticks is None:
        xticks = [0, 5, 10, 15]
    if yticks1 is None:
        yticks1 = [-4, -2, 0, 2, 4]
    if yticks2 is None:
        yticks2 = [-4, -2, 0, 2, 4]
    if ylim1 is None:
        ylim1 = (-6, 6)
    if ylim2 is None:
        ylim2 = (-6, 6)
    if xminortick is None:
        xminortick = 1
    if yminortick is None:
        yminortick = 0.5
    axs[0][0].set_xticks(xticks)
    axs[0][1].set_xticks(xticks)
    axs[0][0].set_xticklabels(xticks, fontsize=fontsize)
    axs[0][1].set_xticklabels(xticks, fontsize=fontsize)
    axs[0][0].set_ylim(*ylim1)
    axs[0][1].set_ylim(*ylim2)
    axs[0][0].set_yticks(yticks1)
    axs[0][1].set_yticks(yticks2)
    axs[0][0].set_yticklabels(yticks1, fontsize=fontsize)
    axs[0][1].set_yticklabels(yticks2, fontsize=fontsize)
    axs[0][0].xaxis.set_minor_locator(MultipleLocator(xminortick))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(xminortick))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(yminortick))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(yminortick))
    axs[1][0].set_xlim(0, 100)
    axs[1][1].set_xlim(0, 100)
    axs[1][0].set_xticks([0, 20, 40, 60, 80])
    axs[1][1].set_xticks([0, 20, 40, 60, 80])
    axs[1][0].set_xticklabels([0, 20, 40, 60, 80], fontsize=fontsize)
    axs[1][1].set_xticklabels([0, 20, 40, 60, 80], fontsize=fontsize)
    if yticks3 is None:
        yticks3 = [0, 10, 20, 30, 40]
    if yticks4 is None:
        yticks4 = [0, 10, 20, 30, 40]
    if yminortick3 is None:
        yminortick3 = 25
    if yminortick4 is None:
        yminortick4 = 25
    if ylim3 is None:
        ylim3 = (0, 40)
    if ylim4 is None:
        ylim4 = (0, 40)
    axs[1][0].set_yticks(yticks3)
    axs[1][1].set_yticks(yticks4)
    axs[1][0].set_ylim(*ylim3)
    axs[1][1].set_ylim(*ylim4)
    axs[1][0].set_yticklabels(yticks3, fontsize=fontsize)
    axs[1][1].set_yticklabels(yticks4, fontsize=fontsize)
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(5))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(5))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(yminortick3))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(yminortick4))
    axs[0][0].text(
        0.01, 0.94,
        '(a)',
        transform=axs[0][0].transAxes,
        fontsize=fontsize
    )
    axs[0][1].text(
        0.01, 0.94,
        '(b)',
        transform=axs[0][1].transAxes,
        fontsize=fontsize
    )
    axs[1][0].text(
        0.01, 0.94,
        '(c)',
        transform=axs[1][0].transAxes,
        fontsize=fontsize
    )
    axs[1][1].text(
        0.01, 0.94,
        '(d)',
        transform=axs[1][1].transAxes,
        fontsize=fontsize
    )
    plt.subplots_adjust(left=0.075, bottom=0.08, right=0.99, top=0.99, wspace=0.12, hspace=0.2)
    if flag == 0:
        fig.savefig(write_path / 'drift_fft_{}.pdf'.format(filename), format='pdf', dpi=300)
        fig.savefig(write_path / 'drift_fft_{}.png'.format(filename), format='png', dpi=300)
    else:
        fig.savefig(write_path / 'drift_fft_{}_flag_1.pdf'.format(filename), format='pdf', dpi=300)
        fig.savefig(write_path / 'drift_fft_{}_flag_1.png'.format(filename), format='png', dpi=300)
    plt.close('all')
    plt.clf()
    plt.cla()
    if data_f is not None:
        data_f.close()
    os.chdir(cwd)


if __name__ == '__main__':
    # plot_drift_plot()
    # plot_dark_data()
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/05Feb2021/ClosedLoop_0502_1537_44Hz.txt',
    #     label='(c) 44 Hz',
    #     ylim1=(-5, 5), ylim2=(-2, 2), xticks=[0, 1, 2, 3, 4], yticks1=[-4, -2, 0, 2, 4], yticks2=[-1, 0, 1], time_factor=0.5
    # )
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/05Feb2021/ClosedLoop_0502_1446_46Hz.txt',
    #     label='(a) 46 Hz',
    #     ylim1=(-3, 3), ylim2=(-3, 3), xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], yticks1=[-2, 0, 2], yticks2=[-2, 0, 2]
    # )
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/05Feb2021/ClosedLoop_0502_1254_48Hz.txt',
    #     label='(d) 48 Hz',
    #     ylim1=(-5, 5), ylim2=(-4, 4), xticks=[0, 1, 2, 3], yticks1=[-4, -2, 0, 2, 4], yticks2=[-2, 0, 2]
    # )
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/05Feb2021/ClosedLoop0502_1504_49Hz.txt',
    #     label='(b) 49 Hz',
    #     ylim1=(-4, 4), ylim2=(-2, 2), xticks=[0, 1, 2, 3, 4], yticks1=[-2, 0, 2], yticks2=[-1, 0, 1]
    # )
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/03Feb2021/ClosedLoop_0302_0958_1250_1s_50ms.txt',
    #     savefilename='closedLoop_0302.pdf',
    #     label='(a) 03 Feb 2021 09:58 AM',
    #     ylim1=(-6, 6), ylim2=(-4, 4), xticks=[0, 25, 50, 75, 100, 125, 150, 175], yticks1=[-4, -2, 0, 2, 4], yticks2=[-2, 0, 2], time_factor=2, scatter_factor=32
    # )
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/07Feb2021/output_070221_1hr_norml.txt',
    #     savefilename='closedloop_070221.pdf',
    #     label='(b) 07 Feb 2021 07:30 AM',
    #     ylim1=(-6, 6), ylim2=(-4, 4), xticks=[0, 10, 20, 30, 40], yticks1=[-4, -2, 0, 2, 4], yticks2=[-2, 0, 2], time_factor=1, scatter_factor=8
    # )
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/10Feb2021/output_100221_8.15am_45mts_normal.txt',
    #     savefilename='closedloop_100221.pdf',
    #     label='(b) 10 Feb 2021 08:15 AM',
    #     ylim1=(-4, 4), ylim2=(-4, 4), xticks=[0, 10, 20, 30, 40], yticks1=[-2, 0, 2], yticks2=[-2, 0, 2], time_factor=2, scatter_factor=16
    # )
    # make_image_tracking_plots_new(
    #     '/home/harsh/AutoGuiderData/11Feb2021/output_110221_8.12am_1hr_normal.txt',
    #     savefilename='closedloop_110221.pdf',
    #     label='(b) 11 Feb 2021 08:12 AM',
    #     ylim1=(-4, 4), ylim2=(-2, 2), xticks=[0, 10, 20, 30, 40, 50, 60], yticks1=[-2, 0, 2], yticks2=[-1, 0, 1], time_factor=2.5, scatter_factor=16
    # )
    # make_sunspot_plot()
    # plot_camera_drift_plot()
    # plot_camera_closeloop_plot_and_fft()
    # plot_camera_closeloop_plot('buffer_31032022_082248_085_0_0.hdf5', 'flat_31032022.hdf5', 0.85, 0, 0)
    # plot_camera_closeloop_plot('buffer_31032022_082248_076_004_0.hdf5', 'flat_31032022.hdf5', 0.76, 0.04, 0)
    # plot_camera_closeloop_plot('buffer_31032022_082248_136_0_357.hdf5', 'flat_31032022.hdf5', 1.36, 0, 3.57)
    # plot_camera_closeloop_plot('buffer_01042022_083839_102_009_267.hdf5', 'flat_31032022.hdf5', 1.02, 0.09, 2.67)
    # plot_camera_closeloop_plot('buffer_02042022_083239_119_014_374.hdf5', 'flat_31032022.hdf5', 1.19, 0.14, 3.74)
    # plot_camera_closeloop_plot('buffer_02042022_083239_056_005_396.hdf5', 'flat_31032022.hdf5', 0.56, 0.05, 3.96)
    # plot_camera_closeloop_plot('buffer_02042022_083239_034_032_237.hdf5', 'flat_31032022.hdf5', 0.34, 0.032, 2.37)
    # plot_camera_closeloop_plot('buffer_03042022_081727_0765_003_0.hdf5', 'flat_31032022.hdf5', 0.765, 0.036, 0)
    # plot_camera_closeloop_plot('buffer_03042022_081727_136_0_425.hdf5', 'flat_31032022.hdf5', 1.36, 0, 4.25)
    # plot_camera_closeloop_plot('buffer_03042022_081727_102_008_318.hdf5', 'flat_31032022.hdf5', 1.02, 0.08, 3.18)
    # plot_camera_closeloop_plot('buffer_03042022_081727_119_0119_446.hdf5', 'flat_31032022.hdf5', 1.19, 0.119, 4.46)
    # plot_camera_closeloop_plot('buffer_03042022_081727_056_004_471.hdf5', 'flat_31032022.hdf5', 0.56, 0.04, 4.71)
    # plot_camera_closeloop_plot('buffer_03042022_081727_034_002_283.hdf5', 'flat_31032022.hdf5', 0.34, 0.02, 2.83, xticks=[0, 2, 4, 6, 8])
    # plot_camera_closeloop_plot('buffer_28032022_083940_045_0216.hdf5', 'flat.hdf5', 0.45, 0.0216, 0)
    # plot_camera_closeloop_plot('buffer_29032022_084944_05_0_0.hdf5', 'flat.hdf5', 0.5, 0, 0, xticks=[0, 5, 10, 15, 20, 25, 30])
    # plot_camera_closeloop_plot('buffer_28032022_085823_07_0_0.hdf5', 'flat.hdf5', 0.7, 0, 0)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_01042022_083839_102_009_267.hdf5', 'flat_31032022.hdf5', 1.02, 0.09, 2.67, yticks3=[0, 10, 20, 30], yticks4=[0, 10, 20, 30], ylim3=[0, 30], ylim4=[0, 30], yminortick3=5, yminortick4=5)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_31032022_082248_085_0_0.hdf5', 'flat_31032022.hdf5', 0.85, 0, 0, ylim3=(0, 275), yticks3=[0, 50, 100, 150, 200, 250], yticks4=[0, 50, 100, 150, 200, 250])
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_31032022_082248_076_004_0.hdf5', 'flat_31032022.hdf5', 0.76, 0.04, 0, ylim3=(0, 400), ylim4=(0, 300), yticks3=[0, 50, 100, 150, 200, 250, 300, 350], yticks4=[0, 50, 100, 150, 200, 250])
    plot_camera_closeloop_plot_and_fft_alternate('buffer_31032022_082248_136_0_357.hdf5', 'flat_31032022.hdf5', 1.36, 0, 3.57, yticks3=[0, 5, 10, 15], yticks4=[0, 5, 10, 15], ylim3=[0, 15], ylim4=[0, 15], yminortick3=5, yminortick4=5)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_30032022_082118_17_0_0.hdf5', 'flat_30032022.hdf5', 1.7, 0, 0, yticks3=[0, 25, 50, 75], yticks4=[0, 25, 50], ylim3=[0, 75], ylim4=[0, 50], yminortick3=5, yminortick4=5)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_02042022_083239_119_014_374.hdf5', 'flat_31032022.hdf5', 1.19, 0.14, 3.74, yticks3=[0, 50, 100, 150, 200], yticks4=[0, 50, 100, 150, 200])
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_02042022_083239_056_005_396.hdf5', 'flat_31032022.hdf5', 0.56, 0.05, 3.96)
    plot_camera_closeloop_plot_and_fft_alternate('buffer_02042022_083239_034_032_237.hdf5', 'flat_31032022.hdf5', 0.34, 0.032, 2.37, yticks3=[0, 5, 10, 15], yticks4=[0, 5, 10, 15], ylim3=[0, 20], ylim4=[0, 20], yminortick3=5, yminortick4=5)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_03042022_081727_0765_003_0.hdf5', 'flat_31032022.hdf5', 0.765, 0.036, 0, ylim3=(0, 400), ylim4=(0, 300), yticks3=[0, 50, 100, 150, 200, 250, 300, 350], yticks4=[0, 50, 100, 150, 200, 250])
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_03042022_081727_136_0_425.hdf5', 'flat_31032022.hdf5', 1.36, 0, 4.25)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_03042022_081727_102_008_318.hdf5', 'flat_31032022.hdf5', 1.02, 0.08, 3.18)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_03042022_081727_119_0119_446.hdf5', 'flat_31032022.hdf5', 1.19, 0.119, 4.46)
    # plot_camera_closeloop_plot_and_fft_alternate('buffer_03042022_081727_034_002_283.hdf5', 'flat_31032022.hdf5', 0.34, 0.02, 2.83, xticks=[0, 2, 4, 6, 8])
    # plot_camera_drift_plot_and_fft('buffer_20042022_085726_drift.hdf5', 'flats_buffer_20042022_085932.hdf5', ylim1=(0, 25), ylim2=(0, 6), yticks1=[0, 5, 10, 15, 20], yticks2=[0, 2, 4], ylim3=(0, 250))
    # plot_camera_drift_plot_and_fft('buffer_20042022_083747_drift.hdf5', 'flats_buffer_20042022_085932.hdf5', ylim1=(0, 25), ylim2=(0, 6), yticks1=[0, 5, 10, 15, 20], yticks2=[0, 2, 4], ylim3=(0, 250))
    plot_camera_drift_plot_and_fft('buffer_20042022_083747_drift.hdf5', 'flats_buffer_20042022_085932.hdf5', flag=1)
    plot_camera_drift_plot_and_fft('buffer_20042022_085726_drift.hdf5', 'flats_buffer_20042022_085932.hdf5',  ylim3=(0, 250), flag=1)
    # plot_camera_drift_plot_and_fft('buffer_20042022_085726_drift.hdf5', 'flats_buffer_20042022_085932.hdf5', ylim1=(0, 25), ylim2=(0, 6), yticks1=[0, 5, 10, 15, 20], yticks2=[0, 2, 4], yticks3=[0, 10, 20, 30], yticks4=[0, 5, 10], ylim3=(0, 30), ylim4=(0, 10), yminortick3=1, yminortick4=1)
    # plot_camera_drift_plot_and_fft('buffer_20042022_083747_drift.hdf5', 'flats_buffer_20042022_085932.hdf5', ylim1=(0, 25), ylim2=(0, 6), yticks1=[0, 5, 10, 15, 20], yticks2=[0, 2, 4], yticks3=[0, 10, 20, 30], yticks4=[0, 5, 10], ylim3=(0, 30), ylim4=(0, 10), yminortick3=1, yminortick4=1)
    # plot_camera_drift_plot_and_fft('buffer_20042022_083747_drift.hdf5', 'flats_buffer_20042022_085932.hdf5', flag=1)
    # plot_camera_drift_plot_and_fft('buffer_20042022_085726_drift.hdf5', 'flats_buffer_20042022_085932.hdf5',  ylim3=(0, 250), flag=1)
