import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def compare_weights(weight, profile):
    fig, axs = plt.subplots(1, 1)

    axs.plot(weight)

    axs2 = axs.twinx()

    axs2.plot(profile)

    plt.show()


f = h5py.File('rps_stic_profiles_x_30_y_1.nc', 'r')
wei = f['weights'][()]
ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]
compare_weights(wei[ind, 0], f['profiles'][0, 0, 4, ind, 0])
