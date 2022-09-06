import os

import scipy.optimize
from lightweaver.fal import Falc82
import lightweaver as lw
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path
from lightweaver_extension import conv_atom
import multiprocessing
from lightweaver.utils import vac_to_air, air_to_vac
from lmfit import Parameters, minimize


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


wave = np.arange(6564.5-4, 6564.5 + 4, 0.01) / 10

def synthesize_line(atoms, atmos, conserve, useNe, wave):
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
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=conserve, Nthreads=32)
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver)
    lw.iterate_ctx_se(ctx)
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    # return ctx, Iwave
    return ctx, Iwave


def get_observation():
    catalog = np.loadtxt('catalog_6563.txt')

    obs_wave, intensity = catalog[:, 0], catalog[:, 1]

    indices = list()

    for w in vac_to_air(wave):
        indices.append(np.argmin(np.abs(obs_wave - w * 10)))

    indices = np.array(indices)

    return intensity[indices]


def synthesize(f_values, waver, parallel=True):

    line_indices = [
        (5, 1),
        (5, 3),
        (7, 2),
        (4, 2),
        (6, 1),
        (8, 3),
        (6, 3)
    ]

    # atoms_no_substructure = list()

    atoms_with_substructure = list()

    # for at in atoms_no_substructure_list:
    #     atoms_no_substructure.append(conv_atom(base_path / at))

    for at in atoms_with_substructure_list:
        atoms_with_substructure.append(conv_atom(base_path / at))

    # total_gf = 0

    h_with_substructure = atoms_with_substructure[0]

    index = 0

    for line_indice in line_indices:
        for line in h_with_substructure.lines:
            if line.j == line_indice[0] and line.i == line_indice[1]:
                line.f = f_values[index]
                # total_gf += f_values[index] * line.iLevel.g
                index += 1
                break

    h_with_substructure.recompute_radiative_broadening()
    h_with_substructure.recompute_collisional_rates()

    h_with_substructure.__post_init__()

    fal = Falc82()
    _, i_obs_1 = synthesize_line(atoms_with_substructure, fal, False, True, waver)
    _, i_obs_2 = None, None

    return i_obs_1, i_obs_2


def cost_function(synth1, synth2=None, observation=None, weights=None):

    if observation is None:
        observation = np.zeros_like(synth1)
    else:
        observation /= observation[0]

    if weights is None:
        weights = np.ones_like(synth1)

    synth1 /= synth1[0]

    diff = np.subtract(
        synth1,
        observation
    )

    diff /= weights

    return diff


def minimization_func(params, waver, observation=None, weights=None):

    f_values = np.zeros(7, dtype=np.float64)
    f_values[0] = params['f0']
    f_values[1] = params['f1']
    f_values[2] = params['f2']
    f_values[3] = params['f3']
    f_values[4] = params['f4']
    f_values[5] = params['f5']
    f_values[6] = params['f6']

    print(f_values)

    f_this = f_values

    i_obs_1, i_obs_2 = synthesize(f_this, waver, parallel=False)

    wave_vac = vac_to_air(waver)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    axs.plot(wave_vac, i_obs_1 / i_obs_1[0], color='blue')
    # axs.plot(wave_vac, i_obs_2 / i_obs_2[0], color='green')
    axs.plot(wave_vac, observation / observation[0], color='orange')
    fig.tight_layout()
    fig.savefig('solution.pdf', format='pdf', dpi=300)

    return cost_function(synth1=i_obs_1, synth2=i_obs_2, observation=observation, weights=weights)


def synthesize_individual_lines():
    line_indices = [
        (5, 1),
        (5, 3),
        (7, 2),
        (4, 2),
        (6, 1),
        (8, 3),
        (6, 3)
    ]

    syn = list()

    for line_indice in line_indices:

        atoms_with_substructure = list()

        for at in atoms_with_substructure_list:
            atoms_with_substructure.append(conv_atom(base_path / at))

        h_with_substructure = atoms_with_substructure[0]

        new_lines = list()

        for line in h_with_substructure.lines:
            if line.j == line_indice[0] and line.i == line_indice[1]:
                new_lines.append(line)
            if line.i == 0:
                new_lines.append(line)

        h_with_substructure.lines = new_lines

        h_with_substructure.__post_init__()

        fal = Falc82()

        _, i_obs_1 = synthesize_line(
            atoms=atoms_with_substructure,
            atmos=fal,
            conserve=False,
            useNe=True,
            wave=wave
        )

        syn.append(i_obs_1)

    atoms_with_substructure = list()

    for at in atoms_with_substructure_list:
        atoms_with_substructure.append(conv_atom(base_path / at))

    h_with_substructure = atoms_with_substructure[0]

    obs = get_observation()

    wave_vac = vac_to_air(wave)

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 1, figsize=(7, 5))

    for indd, (ssyn, line_indice) in enumerate(zip(syn, line_indices)):

        for line in h_with_substructure.lines:
            if line.j == line_indice[0] and line.i == line_indice[1]:
                gf = np.round(
                    np.log10(
                        line.iLevel.g * line.f
                    ),
                    3
                )
                break

        filename = '{}->{}, log gf={}.txt'.format(
            h_with_substructure.levels[line_indice[0]].label,
            h_with_substructure.levels[line_indice[1]].label,
            gf
        )

        np.savetxt(filename, ssyn)

        ssyn /= ssyn[0]

        axs.plot(
            wave_vac * 10,
            ssyn,
            label='{}->{}, log gf={}'.format(
                h_with_substructure.levels[line_indice[0]].label,
                h_with_substructure.levels[line_indice[1]].label,
                gf
            )
        )

    np.savetxt('catalog.txt', obs)

    obs /= obs[0]

    np.savetxt('wave_grid.txt', wave_vac)

    axs.plot(wave_vac * 10, obs, label='BASS 2000')

    axs.legend(loc="lower left", prop={'size': 6})

    axs.tick_params(labelsize=7)

    fig.savefig('individual_lines.pdf', format='pdf', dpi=300)


if __name__ == '__main__':
    obs = get_observation()
    params = Parameters()
    params.add('f0', value=0.75, min=0.55, max=0.95, brute_step=0.1)
    params.add('f1', value=0.75, min=0.55, max=0.95, brute_step=0.1)
    params.add('f2', value=0.001, min=0.001, max=0.201, brute_step=0.1)
    params.add('f3', value=0.001, min=0.001, max=0.201, brute_step=0.1)
    params.add('f4', value=0.75, min=0.55, max=0.95, brute_step=0.1)
    params.add('f5', value=0.001, min=0.001, max=0.201, brute_step=0.1)
    params.add('f6', value=0.75, min=0.55, max=0.95, brute_step=0.1)
    weights = np.ones_like(wave) * 0.003
    weights[300:500] = 0.002
    weights[350:450] = 0.001
    out = minimize(minimization_func, params, args=(wave, obs, weights), method='brute')
    os.remove('solution.pdf')
    f_this = np.zeros(7, dtype=np.float64)
    f_this[0] = out.params['f0']
    f_this[1] = out.params['f1']
    f_this[2] = out.params['f2']
    f_this[3] = out.params['f3']
    f_this[4] = out.params['f4']
    f_this[5] = out.params['f5']
    f_this[6] = out.params['f6']
    np.savetxt('solution.txt', f_this)
    obs_1, obs_2 = synthesize(f_this, wave, parallel=False)
    wave_vac = vac_to_air(wave)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    axs.plot(wave_vac, obs_1 / obs_1[0], color='blue')
    # axs.plot(wave_vac, obs_2 / obs_2[0], color='green')
    axs.plot(wave_vac, obs / obs[0], color='orange')
    fig.tight_layout()
    fig.savefig('solution.pdf', format='pdf', dpi=300)
    f_values = np.array([1.3596e-2, 1.3599e-2, 2.9005e-1, 1.4503e-1, 6.9614E-1, 6.2654E-1, 6.9616E-2])
    obs_1, obs_2 = synthesize(f_values, wave, parallel=False)
    plt.close('all')
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    axs.plot(wave_vac, obs_1 / obs_1[0], color='blue')
    # axs.plot(wave_vac, obs_2 / obs_2[0], color='green')
    axs.plot(wave_vac, obs / obs[0], color='orange')
    fig.tight_layout()
    fig.savefig('original.pdf', format='pdf', dpi=300)
    # synthesize_individual_lines()
