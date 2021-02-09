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


def calibrate(filename, count):

    try:
        f = open(filename, encoding='utf-16-le')
        lines = f.readlines()
    except Exception:
        f = open(filename)

        lines = f.readlines()

    data_lines = [line for line in lines if line[0].isdigit()]

    data = np.loadtxt(data_lines)

    data = data[:, 1:]

    # repeated_x = np.repeat(x, repeats=data.shape[1])

    mean_data = np.mean(data, axis=1)

    plt.close('all')

    plt.clf()

    plt.cla()

    smoothed = smooth(mean_data, 5)

    plt.plot(smooth(mean_data, 5))

    # size = plt.rcParams['lines.markersize']

    # plt.scatter(
    #     repeated_x,
    #     data.reshape(data.shape[0] * data.shape[1]),
    #     label='data',
    #     s=size / 4
    # )

    plt.title('Pick the left most point and the minima point')

    pts = np.asarray(plt.ginput(2, 10))

    indices = pts[:, 0].astype(np.int64)

    a, b = np.polyfit(
        100 * np.arange(
            indices[1] - indices[0]),
        smoothed[np.arange(indices[0], indices[1])],
        1
    )

    y = a * 100 * np.arange(indices[1] - indices[0]) + b

    return (y.min() + y.max()) / 2, a, b


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
