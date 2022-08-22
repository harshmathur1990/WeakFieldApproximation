import numpy as np
from dataclasses import dataclass
from lightweaver.atomic_model import AtomicModel, AtomicLevel
from lightweaver.broadening import RadiativeBroadening, LineBroadening
from lightweaver.collisional_rates import *
from lightweaver.constants import QElectron, Epsilon0, MElectron, CLight, ERydberg, RBohr, KBoltzmann,
from typing import List


F_QUADRUPOLE = 0.1
R_AVG_FLAG = True


double phiImpact(double x)
{
  double beta0, beta1, zeta1, sigma_0, sigma_1, righthand;

  x += x0;

  beta0 = sqrt(x/ERkT) * x0/(2*x + x0) * R0;
  sigma_0 = beta0 * Bessel_K0(beta0) * Bessel_K1(beta0);

  righthand = SQ(2*x + x0) / (8 * ERkT * x0 * f);
  beta1 = findbeta1(righthand, &zeta1);
  sigma_1 = 0.5*zeta1 + beta1 * Bessel_K0(beta1) * Bessel_K1(beta1);

  if (VERBOSE >= 1) {
    fprintf(stderr, "sigma_0: %E,  sigma_1: %E\n", sigma_0, sigma_1);
  }
  return 8.0*PIa0square * SQ(ERkT) * (f/x0) * MIN(sigma_0, sigma_1);
}

def GaussLaguerre(function):

    integral = 0.0

    x_Laguerre = [
        0.170279632305,  0.903701776799,  2.251086629866,  4.266700170288,
        7.045905402393, 10.758516010181, 15.740678641278, 22.863131736889
    ]

    w_Laguerre = [
        3.69188589342E-01, 4.18786780814E-01, 1.75794986637E-01,
        3.33434922612E-02, 2.79453623523E-03, 9.07650877336E-05,
        8.48574671627E-07, 1.04800117487E-09
    ]

    N_LAGUERRE = len(x_Laguerre)

    for nl in range(N_LAGUERRE):
        integral += w_Laguerre[nl] * function(x_Laguerre[nl])

    return integral


def impactParam(line, flag, Ntemp, temp):

    i = line.i
    j = line.j

    atom = line.atom

    CE = np.zeros(Ntemp)

    if flag is True:
        n_eff_min = np.min(line.iLevel.n_eff, line.jLevel.n_eff)
        R0 = 0.25 * (5.0 * np.square(n_eff_min) + n_eff_min + 1.0)
    else:
        Ri = 0.5 * (3.0*square(line.iLevel.n_eff) - line.iLevel.L*(line.iLevel.L + 1.0))
        Rj = 0.5 * (3.0*square(n_eff[j]) - l[j]*(l[j] + 1.0))
        R0 = np.min(Ri, Rj)

    deltaE = line.jLevel.E_SI - line.iLevel.E_SI
    for k in range(Ntemp):
        x0 = deltaE / (KBoltzmann * temp[k])
        ERkT  = ERydberg / (KBoltzmann * temp[k]);
        CE[k] = GaussLaguerre(phiImpact)

    return CE


@dataclass
class NewAtomicLevel(AtomicLevel):
    n_eff: int

@dataclass
class NewAtomicModel(AtomicModel):

    def recompute_radiative_broadening(self):
        for line in self.lines:
            elastic_broadening = line.broadening.elastic

            Grad = 0

            for another_line in self.lines:
                if another_line.j == line.j:
                    Grad += another_line.Aji

            line.broadening = LineBroadening(natural=[RadiativeBroadening(Grad)], elastic=elastic_broadening)
    def compute_n_eff(self):
        for i, level in enumerate(self.levels[:-1]):

            ic = i + 1

            while self.levels[ic].stage < level.stage + 1 and ic < len(self.levels):
                ic += 1

            if self.levels[ic].stage == level.stage:
                raise Exception("Found no overlying continuum for level {}".format(level.label))

            z = level.stage + 1

            level.n_eff = z * np.sqrt(ERydberg / self.levels[ic].E_SI - level.E_SI)

    def recompute_collisional_rates(self):

        temp = [
            1264.9, 2249.4, 4000.0, 7113.1, 12649.1, 22493.6, 40000.0,
            71131.1, 126491.0, 224936.0, 400000.0, 711312.0, 1264910.0
        ]

        NTEMP = len(temp)

        C = 2 * np.pi * (QElectron / Epsilon0) * (QElectron / MElectron) / CLight;

        C0 = 1.55E+11

        C1 = ((ERydberg / np.sqrt(MElectron)) * np.pi * np.square(RBohr)) * np.sqrt(8.0 / (np.pi * KBoltzmann))

        C2_atom = 2.15E-6

        C2_ion = 3.96E-6

        C3 = np.sqrt(8 * KBoltzmann / (np.pi * MElectron))

        PIa0square = np.pi * np.square(RBohr)

        for i_level in self.levels:
            for j_level in self.levels[1:]:
                if i_level.stage == j_level.stage:


                    validtransition = False

                    for line in self.lines:
                        if line.i == i_level.i and line.j == j_level.j:
                            validtransition = True
                            f = line.f
                            break

                    if validtransition is False:
                        f = F_QUADRUPOLE

                    if i_level.stage == 0:
                        if validtransition is True:
                            pass

        collisions: List[CollisionalRates] = []



