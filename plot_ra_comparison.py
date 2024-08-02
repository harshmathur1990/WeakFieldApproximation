import numpy as np
import matplotlib.pyplot as plt


filenames = [
    'DEC calib_200_0102_1031.txt'
]


counts = [
    200
]


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def calibrate(filename, direction=1):

    try:
        f = open(filename, encoding='utf-16-le')
        lines = f.readlines()
    except Exception:
        f = open(filename)

        lines = f.readlines()

    data_lines = [line for line in lines if line[0].isdigit()]

    data = np.loadtxt(data_lines)

    data = data[:, 1:]

    data_1 = data.reshape(data.shape[0], data.shape[1] // 2, 2)[:, :, 0]

    data_2 = data.reshape(data.shape[0], data.shape[1] // 2, 2)[:, :, 1]

    mean_data_1 = np.mean(data_1, axis=1)

    mean_data_2 = np.mean(data_2, axis=1)

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(smooth(mean_data_1, 5))

    plt.title('Pick the left most point and the minima point')

    pts = np.asarray(plt.ginput(2, 10))

    indices = pts[:, 0].astype(np.int64)

    a1, b1 = np.polyfit(
        direction * 100 * np.arange(
            indices[1] - indices[0]),
        smooth(mean_data_1, 5)[np.arange(indices[0], indices[1])],
        1
    )

    y1 = a1 * direction * 100 * np.arange(indices[1] - indices[0]) + b1

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(smooth(mean_data_2, 5))

    plt.title('Pick the left most point and the minima point')

    pts = np.asarray(plt.ginput(2, 10))

    indices = pts[:, 0].astype(np.int64)

    a2, b2 = np.polyfit(
        direction * 100 * np.arange(
            indices[1] - indices[0]),
        smooth(mean_data_2, 5)[np.arange(indices[0], indices[1])],
        1
    )

    y2 = a2 * direction * 100 * np.arange(indices[1] - indices[0]) + b2

    return (y1.min() + y1.max()) / 2, a1, b1, (y2.min() + y2.max()) / 2, a2, b2


def plot_one(filename, count):

    f = open(filename, encoding='utf-16-le')

    lines = f.readlines()

    data_lines = [line for line in lines if line[0].isdigit()]

    data = np.loadtxt(data_lines)

    x = data[:, 0]

    data = data[:, 1:]

    mean_data = np.mean(data, axis=1)

    plt.plot(x, mean_data, label='f={} Hz'.format(count), linewidth=0.5)


def overplot_all():

    plt.close('all')

    plt.clf()

    plt.cla()

    for filename, count in zip(filenames, counts):
        plot_one(filename, count)

    plt.title('Overplot of RA Calibration')

    plt.xlabel('Counts')

    plt.ylabel('mVolts')

    plt.legend()

    plt.tight_layout()

    plt.savefig('overplot.png', format='png', dpi=600)


if __name__ == '__main__':
    for filename, count in zip(filenames, counts):
        print (calibrate(filename, count))
    overplot_all()
