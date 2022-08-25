import scipy.optimize
from lightweaver.fal import Falc82
import lightweaver as lw
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path
from lightweaver_extension import conv_atom
import multiprocessing


base_path = Path(
    '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/Atoms'
)
atoms_no_substructure_list = [
    'H_6.atom',
    'C.atom',
    'O.atom',
    'Si.atom',
    'Al.atom',
    'CaII.atom',
    'Fe.atom',
    'He.atom',
    'Mg.atom',
    'N.atom',
    'Na.atom',
    'S.atom'
]

atoms_with_substructure_list = [
    'H_6_sublevel_strict.atom',
    'C.atom',
    'O.atom',
    'Si.atom',
    'Al.atom',
    'CaII.atom',
    'Fe.atom',
    'He.atom',
    'Mg.atom',
    'N.atom',
    'Na.atom',
    'S.atom'
]


wave = np.arange(6562.8-4, 6562.8 + 4, 0.01)

def synthesize_line(atoms, atmos, conserve, useNe, wave, q):
    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)
    # Configure the set of atomic models to use.
    aSet = lw.RadiativeSet(atoms)
    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    aSet.set_active('H', 'Ca')
    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # Either compute the equilibrium populations at the fixed electron density
    # provided in the model, or iterate an LTE electron density and compute the
    # corresponding equilibrium populations (SpeciesStateTable).
    if useNe:
        eqPops = aSet.compute_eq_pops(atmos)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(atmos)

    # Configure the Context which holds the state of the simulation for the
    # backend, and provides the python interface to the backend.
    # Feel free to increase Nthreads to increase the number of threads the
    # program will use.
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=conserve, Nthreads=16)
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver)
    lw.iterate_ctx_se(ctx)
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    # return ctx, Iwave
    q.put(Iwave)

def cost_function(observation, synth1, synth2):

    observation /= observation[0]

    synth1 /= synth1[0]

    synth2 /= synth2[0]

    chi1 = np.sqrt(
        np.square(
            np.subtract(
                observation,
                synth1
            )
        )
    )

    chi2 = np.sqrt(
        np.square(
            np.subtract(
                observation,
                synth2
            )
        )
    )

    chi3 = np.sqrt(
        np.square(
            np.subtract(
                synth2,
                synth1
            )
        )
    )

    res = np.array(list(chi1) + list(chi2) + list(chi3))

    return res


def get_observation():
    catalog = np.loadtxt('catalog_6563.txt')

    obs_wave, intensity = catalog[:, 0], catalog[:, 1]

    indices = list()

    for w in wave:
        indices.append(np.argmin(np.abs(obs_wave - w)))

    indices = np.array(indices)

    return intensity[indices]


def synthesize(f_values, waver):
    line_indices = [
        (5, 1),
        (5, 3),
        (7, 2),
        (4, 2),
        (6, 1),
        (8, 3),
        (6, 3)
    ]

    atoms_no_substructure = list()

    atoms_with_substructure = list()

    for at in atoms_no_substructure_list:
        atoms_no_substructure.append(conv_atom(base_path / at))

    for at in atoms_with_substructure_list:
        atoms_with_substructure.append(conv_atom(base_path / at))

    total_gf = 0

    h_with_substructure = atoms_with_substructure[0]

    index = 0

    for line_indice in line_indices:
        for line in h_with_substructure.lines:
            if line.j == line_indice[0] and line.i == line_indice[1]:
                line.f = f_values[index]
                total_gf += f_values[index] * line.iLevel.g
                index += 1
                break

    h_with_substructure.recompute_radiative_broadening()
    h_with_substructure.recompute_collisional_rates()

    total_gf /= 8

    h_without_substructure = atoms_no_substructure[0]

    h_without_substructure.lines[4].f = total_gf
    h_without_substructure.recompute_radiative_broadening()
    h_without_substructure.recompute_collisional_rates()

    h_with_substructure.__post_init__()
    h_without_substructure.__post_init__()

    q = multiprocessing.Queue()

    fal = Falc82()

    p1 = multiprocessing.Process(target=synthesize_line, args=(atoms_with_substructure, fal, False, True, waver, q))

    fal = Falc82()

    p2 = multiprocessing.Process(target=synthesize_line, args=(atoms_no_substructure, fal, False, True, waver, q))

    p1.start()

    p2.start()

    p1.join()

    p2.join()

    i_obs_1 = q.get()

    i_obs_2 = q.get()

    return i_obs_1, i_obs_2


def prepare_minimization_func(waver):

    observation = get_observation()
    def minimization_func(f_values):

        print(f_values)

        for ff in f_values:
            if ff <= 0:
                res = np.zeros(waver.size * 3)
                res[:] = np.inf
                return res

        i_obs_1, i_obs_2 = synthesize(f_values, waver)

        return cost_function(observation, i_obs_1, i_obs_2)

    return minimization_func


if __name__ == '__main__':
    obs = get_observation()
    f_values = np.array([1.3596e-2, 1.3599e-2, 2.9005e-1, 1.4503e-1, 6.9614E-1, 6.2654E-1, 6.9616E-2])
    # f_values = np.ones(7) * 0.01
    min_func = prepare_minimization_func(wave)
    res_1 = scipy.optimize.least_squares(min_func, f_values, method='lm')
    np.savetxt('solution.txt', res_1.x)
    obs_1, obs_2 = synthesize(res_1.x, wave)
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs.plot(wave, obs_1 / obs_1[0], color='blue')
    axs.plot(wave, obs_2 / obs_2[0], color='green')
    axs.plot(wave, obs / obs[0], color='orange')
    fig.tight_layout()
    fig.savefig('solution.pdf', format='pdf', dpi=300)
