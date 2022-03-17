transitions = [
    [3, 0],
    [4, 0],
    [3, 1],
    [4, 1],
    [4, 2],
    [1, 0],
    [2, 0],
    [2, 1],
    [3, 2],
    [4, 3]
]


for i in range(10):
    cularr[i] = out.atoms[0].Cij[:, transitions[i][1], transitions[i][0]]


interested_wave = list(np.array([3933.66, 3968.47, 8498.02, 8542.09, 8662.14]) / 10)

interested_wave = [121.5668237310, 121.5673644608, 656.275181, 656.290944, 102.572182505, 102.572296565, 656.272483, 656.277153, 656.270970, 656.285177, 656.286734]

eta_c = np.zeros((11, 82))
eps_c = np.zeros((11, 82))

for index, wave in enumerate(interested_wave):
    ind = np.argmin(np.abs(out.spectrum.waves - wave))

    out.opacity.read(ind, 4)

    eta_c[index] = out.opacity.chi_c
    eps_c[index] = out.opacity.eta_c