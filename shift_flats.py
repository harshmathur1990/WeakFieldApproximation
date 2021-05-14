shifted_intensity = intensity.copy()

for i in range(16):
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.imshow(intensity[i, 0:480], cmap='gray', origin='lower')
    pts = np.asarray(plt.ginput(30, 30))
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    y = a * np.arange(480) + b
    shifts = y - y[-1]
    shifts = shifts * -1

    for j in range(480):
        shifted_intensity[i][j] = scipy.ndimage.shift(intensity[i][j], shifts[j], mode='nearest')

    plt.close('all')
    plt.clf()
    plt.cla()

    plt.imshow(intensity[i, 500:], cmap='gray', origin='lower')
    pts = np.asarray(plt.ginput(30, 30))
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    y = a * np.arange(524) + b
    shifts = y - y[-1]
    shifts = shifts * -1

    for j in range(500, 1024):
        shifted_intensity[i][j] = scipy.ndimage.shift(intensity[i][j], shifts[j-500], mode='nearest')


x_shifted_intensity = shifted_intensity.copy()

for i in range(16):
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.imshow(shifted_intensity[i, 0:480], cmap='gray', origin='lower')
    pts = np.asarray(plt.ginput(30, 30))
    a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    y = a * np.arange(512) + b
    shifts = y - y[-1]
    shifts = shifts * -1

    for j in range(512):
        x_shifted_intensity[i, 0:480, j] = scipy.ndimage.shift(shifted_intensity[i, 0:480, j], shifts[j], mode='nearest')

    plt.close('all')
    plt.clf()
    plt.cla()

    plt.imshow(shifted_intensity[i, 500:], cmap='gray', origin='lower')
    pts = np.asarray(plt.ginput(30, 30))
    a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    y = a * np.arange(512) + b
    shifts = y - y[-1]
    shifts = shifts * -1

    for j in range(512):
        x_shifted_intensity[i, 500:, j] = scipy.ndimage.shift(shifted_intensity[i, 500:, j], shifts[j], mode='nearest')