import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


falc_file = Path('/home/harsh/CourseworkRepo/stic_copy/run_nagaraju_1/falc_nicole_for_stic.nc')

file_1 = Path('/home/harsh/CourseworkRepo/stic/run_nagaraju_1/falc_profs_ls.nc')

file_2 = Path('/home/harsh/CourseworkRepo/stic_copy/run_nagaraju_1/falc_profs.nc')


def make_profile_plots():
    f_ls = h5py.File(file_1, 'r')

    f_jk = h5py.File(file_2, 'r')

    fig, axs = plt.subplots(2, 2, figsize=(7, 4.66))
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.1, top=0.89, wspace=0.3, hspace=0.3)

    axs[0][0].plot(f_ls['wav'][()] - 8542.09, f_ls['profiles'][0, 0, 0, :, 0], label='LS', color='brown', linewidth=1, linestyle='--')
    axs[0][0].plot(f_jk['wav'][()] - 8542.09, f_jk['profiles'][0, 0, 0, :, 0], label='JK', color='blue', linewidth=0.5, linestyle='--')

    axs[0][1].plot(f_ls['wav'][()] - 8542.09, f_ls['profiles'][0, 0, 0, :, 1] * 100 / f_ls['profiles'][0, 0, 0, :, 0], color='brown', linewidth=1, linestyle='--')
    axs[0][1].plot(f_jk['wav'][()] - 8542.09, f_jk['profiles'][0, 0, 0, :, 1] * 100 / f_jk['profiles'][0, 0, 0, :, 0], color='blue', linewidth=0.5, linestyle='--')

    axs[1][0].plot(f_ls['wav'][()] - 8542.09, f_ls['profiles'][0, 0, 0, :, 2] * 100 / f_ls['profiles'][0, 0, 0, :, 0], color='brown', linewidth=1, linestyle='--')
    axs[1][0].plot(f_jk['wav'][()] - 8542.09, f_jk['profiles'][0, 0, 0, :, 2] * 100 / f_jk['profiles'][0, 0, 0, :, 0], color='blue', linewidth=0.5, linestyle='--')

    axs[1][1].plot(f_ls['wav'][()] - 8542.09, f_ls['profiles'][0, 0, 0, :, 3] * 100 / f_ls['profiles'][0, 0, 0, :, 0], color='brown', linewidth=1, linestyle='--')
    axs[1][1].plot(f_jk['wav'][()] - 8542.09, f_jk['profiles'][0, 0, 0, :, 3] * 100 / f_jk['profiles'][0, 0, 0, :, 0], color='blue', linewidth=0.5, linestyle='--')

    handles, labels = axs[0][0].get_legend_handles_labels()

    axs[0][0].legend(
        handles,
        labels,
        ncol=2,
        bbox_to_anchor=(0., 1.02, 2., .102),
        loc='lower left',
        mode="expand",
        borderaxespad=0.,
        fontsize=10
    )

    axs[1][0].set_xlabel(r'$\Delta (\lambda)$')
    axs[1][1].set_xlabel(r'$\Delta (\lambda)$')
    axs[0][0].set_ylabel(r'$I/I_{c}$')
    axs[0][1].set_ylabel(r'$Q/I\;(\%)$')
    axs[1][0].set_ylabel(r'$U/I\;(\%)$')
    axs[1][1].set_ylabel(r'$V/I\;(\%)$')

    # fig.tight_layout()
    fig.savefig('LSvsJK.pdf', dpi=300, format='pdf')
    plt.show()


def make_temperature_plot():
    f = h5py.File(falc_file, 'r')

    plt.plot(f['ltau500'][0, 0, 0], f['temp'][0, 0, 0] / 1e3)

    plt.xlabel(r'$\log(\tau_{500})$')
    plt.ylabel(r'$T[kK]$')

    plt.gcf().set_size_inches(7, 4.66, forward=True)

    plt.savefig('Temperature.pdf', dpi=300, format='pdf')

    plt.show()


if __name__ == '__main__':
    # make_profile_plots()
    make_temperature_plot()
