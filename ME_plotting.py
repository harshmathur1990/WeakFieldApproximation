plt.close('all')

plt.clf()

plt.cla()

extent = [250, 280, 2, 20]

fig, axs = plt.subplots(2, 3, figsize=(11,4))

axs[0][0].imshow(obs[:, :, 0, 8], cmap='gray', origin='lower', extent=extent)

axs[0][1].imshow(obs[:, :, 0, 60], cmap='gray', origin='lower', extent=extent)

im02 = axs[0][2].imshow(mag_field_fe, cmap='gray', origin='lower', extent=extent)

im10 = axs[1][0].imshow(temperature_map_wave_0, cmap='hot', origin='lower', extent=extent)

im11 = axs[1][1].imshow(temperature_map_wave_61, cmap='hot', origin='lower', extent=extent)

im12 = axs[1][2].imshow(f['mo'][:, :, 3], cmap='bwr', origin='lower', extent=extent, vmin=-4, vmax=4, aspect='equal')

mask = np.loadtxt('/home/harsh/Spinor_2008/Ca_x_30_y_18_2_20_250_280/Fe_ME/mask_pore.txt')

mask = mask.astype(np.int64)

X, Y = np.meshgrid(np.arange(250, 280), np.arange(2, 20))

axs[0][0].contour(X, Y, mask, levels=0)

axs[0][1].contour(X, Y, mask, levels=0)

axs[0][2].contour(X, Y, mask, levels=0)

axs[1][0].contour(X, Y, mask, levels=0)

axs[1][1].contour(X, Y, mask, levels=0)

axs[1][2].contour(X, Y, mask, levels=0)

axs[0][0].set_xlabel('(a)')

axs[0][1].set_xlabel('(b)')

axs[0][2].set_xlabel('(c)')

axs[1][0].set_xlabel('(d)')

axs[1][1].set_xlabel('(e)')

axs[1][2].set_xlabel('(f)')

fig.colorbar(im02, ax=axs[0][2])

fig.colorbar(im10, ax=axs[1][0])

fig.colorbar(im11, ax=axs[1][1])

fig.colorbar(im12, ax=axs[1][2])

fig.suptitle('Inversions Result from Fe I 6569 line using ME')

fig.tight_layout()

plt.tight_layout()

plt.savefig('ME_Results.png', format='png', dpi=600)

plt.show()