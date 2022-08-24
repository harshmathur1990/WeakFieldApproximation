from lightweaver.fal import Falc82
import lightweaver as lw
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path
from lightweaver_extension import conv_atom


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


catalog = np.loadtxt('catalog_6563.txt')

def synth_halpha(atoms, atmos, conserve, useNe, wave):
    '''
    Synthesise a spectral line for given atmosphere with different
    conditions.

    Parameters
    ----------
    atmos : lw.Atmosphere
        The atmospheric model in which to synthesise the line.
    conserve : bool
        Whether to start from LTE electron density and conserve charge, or
        simply use from the electron density present in the atomic model.
    useNe : bool
        Whether to use the electron density present in the model as the
        starting solution, or compute the LTE electron density.
    wave : np.ndarray
        Array of wavelengths over which to resynthesise the final line
        profile for muz=1.

    Returns
    -------
    ctx : lw.Context
        The Context object that was used to compute the equilibrium
        populations.
    Iwave : np.ndarray
        The intensity at muz=1 for each wavelength in `wave`.
    '''
    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)
    # Configure the set of atomic models to use.
    aSet = lw.RadiativeSet(atoms)
    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    aSet.set_active('H', 'Ca', 'He')
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
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=conserve, Nthreads=1)
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver)
    lw.iterate_ctx_se(ctx)
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    return ctx, Iwave

def cost_function(observation, synth1, synth2):

    observation /= observation[0]

    synth1 /= synth1[0]

    synth2 /= synth2[0]

    chi1 = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    observation,
                    synth1
                )
            )
        )
    )

    chi2 = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    observation,
                    synth2
                )
            )
        )
    )

    chi3 = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    synth2,
                    synth1
                )
            )
        )
    )

    return chi1 + chi2 + chi3


def get_observation():
    obs_wave, intensity = catalog[:, 0], catalog[:, 1]

    indices = list()

    for w in wave:
        indices.append(np.argmin(np.abs(obs_wave - w)))

    indices = np.array(indices)

    return intensity[indices]

def minimization_func(f_values):
    pass

