import numpy as np
import matplotlib.pyplot as plt


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def make_calibration_plots(
    ra_filename,
    dec_filename,
    ra_direction=-1,
    dec_direction=1,
    encoding='utf-16-le'
):

    plt.close('all')
    plt.clf()
    plt.cla()

    plt.rc('font', size=24)

    try:
        ra_f = open(ra_filename, encoding=encoding)
    except Exception:
        ra_f = open(ra_filename)

    ra_lines = ra_f.readlines()

    ra_data_lines = [line for line in ra_lines if line[0].isdigit()]

    ra_data = np.loadtxt(ra_data_lines)

    ra_data = ra_data[:, 1:]

    ra_data_1 = ra_data.reshape(
        ra_data.shape[0], ra_data.shape[1] // 2, 2)[:, :, 0]

    ra_data_2 = ra_data.reshape(
        ra_data.shape[0], ra_data.shape[1] // 2, 2)[:, :, 1]

    ra_mean_data_1 = np.mean(ra_data_1, axis=1)

    ra_mean_data_2 = np.mean(ra_data_2, axis=1)

    ra_indices_1 = [0, 10]

    ra_a1, ra_b1 = np.polyfit(
        ra_direction * 100 * np.arange(
            ra_indices_1[1] - ra_indices_1[0]),
        smooth(ra_mean_data_1, 5)[np.arange(ra_indices_1[0], ra_indices_1[1])],
        1
    )

    ra_y1 = ra_a1 * ra_direction * 100 * np.arange(
        ra_indices_1[1] - ra_indices_1[0]
    ) + ra_b1

    ra_indices_2 = [5, 15]

    ra_a2, ra_b2 = np.polyfit(
        ra_direction * 100 * np.arange(
            ra_indices_2[1] - ra_indices_2[0]),
        smooth(ra_mean_data_2, 5)[np.arange(ra_indices_2[0], ra_indices_2[1])],
        1
    )

    ra_y2 = ra_a2 * ra_direction * 100 * np.arange(
        ra_indices_2[1] - ra_indices_2[0]
    ) + ra_b2

    try:
        dec_f = open(dec_filename, encoding='utf-16-le')
    except Exception:
        dec_f = open(dec_filename)

    dec_lines = dec_f.readlines()

    dec_data_lines = [line for line in dec_lines if line[0].isdigit()]

    dec_data = np.loadtxt(dec_data_lines)

    dec_data = dec_data[:, 1:]

    dec_data_1 = dec_data.reshape(
        dec_data.shape[0], dec_data.shape[1] // 2, 2)[:, :, 0]

    dec_data_2 = dec_data.reshape(
        dec_data.shape[0], dec_data.shape[1] // 2, 2)[:, :, 1]

    dec_mean_data_1 = np.mean(dec_data_1, axis=1)

    dec_mean_data_2 = np.mean(dec_data_2, axis=1)

    dec_indices_1 = [0, 8]

    dec_a1, dec_b1 = np.polyfit(
        dec_direction * 100 * np.arange(
            dec_indices_1[1] - dec_indices_1[0]),
        smooth(
            dec_mean_data_1, 5)[np.arange(dec_indices_1[0], dec_indices_1[1])],
        1
    )

    dec_y1 = dec_a1 * dec_direction * 100 * np.arange(
        dec_indices_1[1] - dec_indices_1[0]
    ) + dec_b1

    dec_indices_2 = [5, 25]

    dec_a2, dec_b2 = np.polyfit(
        dec_direction * 100 * np.arange(
            dec_indices_2[1] - dec_indices_2[0]),
        smooth(
            dec_mean_data_2, 5)[np.arange(dec_indices_2[0], dec_indices_2[1])],
        1
    )

    dec_y2 = dec_a2 * dec_direction * 100 * np.arange(
        dec_indices_2[1] - dec_indices_2[0]
    ) + dec_b2

    fig, axs = plt.subplots(2, 2, figsize=(19.2, 10.8), dpi=100)

    axs[0][0].scatter(
        ra_direction * 100 * np.arange(
            ra_indices_1[1] - ra_indices_1[0]),
        smooth(ra_mean_data_1, 5)[np.arange(ra_indices_1[0], ra_indices_1[1])],
        color='#364f6B'
    )

    axs[0][0].plot(
        ra_direction * 100 * np.arange(
            ra_indices_1[1] - ra_indices_1[0]),
        ra_y1, 
        color='#3fC1C9'
    )

    axs[0][1].scatter(
        ra_direction * 100 * np.arange(
            ra_indices_2[1] - ra_indices_2[0]),
        smooth(ra_mean_data_2, 5)[np.arange(ra_indices_2[0], ra_indices_2[1])],
        color='#364f6B'
    )

    axs[0][1].plot(
        ra_direction * 100 * np.arange(
            ra_indices_2[1] - ra_indices_2[0]),
        ra_y2, 
        color='#3fC1C9'
    )

    axs[1][0].scatter(
        dec_direction * 100 * np.arange(
            dec_indices_2[1] - dec_indices_2[0]),
        smooth(
            dec_mean_data_2, 5)[np.arange(dec_indices_2[0], dec_indices_2[1])],
        color='#364f6B'
    )

    axs[1][0].plot(
        dec_direction * 100 * np.arange(
            dec_indices_2[1] - dec_indices_2[0]),
        dec_y2, 
        color='#3fC1C9'
    )

    axs[1][1].scatter(
        dec_direction * 100 * np.arange(
            dec_indices_1[1] - dec_indices_1[0]),
        smooth(
            dec_mean_data_1, 5)[np.arange(dec_indices_1[0], dec_indices_1[1])],
        color='#364f6B'
    )

    axs[1][1].plot(
        dec_direction * 100 * np.arange(
            dec_indices_1[1] - dec_indices_1[0]),
        dec_y1, 
        color='#3fC1C9'
    )

    axs[0][0].set_title(r'$V_{x}$ vs $C_{x}$')
    axs[0][1].set_title(r'$V_{y}$ vs $C_{x}$')
    axs[1][0].set_title(r'$V_{x}$ vs $C_{y}$')
    axs[1][1].set_title(r'$V_{y}$ vs $C_{y}$')

    axs[0][0].set_ylabel('Voltages (mV)')
    axs[1][0].set_ylabel('Voltages (mV)')

    axs[1][0].set_xlabel('Motor Count')
    axs[1][1].set_xlabel('Motor Count')

    fig.tight_layout()

    fig.suptitle('Calibration')

    # plt.savefig('Calibration_plots.eps', format='eps')
    plt.show()

    return np.array([[ra_a1, ra_a2], [dec_a2, dec_a1]])
