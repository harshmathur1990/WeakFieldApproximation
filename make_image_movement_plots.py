from datetime import timedelta
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np


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
