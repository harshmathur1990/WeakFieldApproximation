import numpy as np
from dataclasses import dataclass
from lightweaver.atomic_model import AtomicModel, AtomicLevel
from lightweaver.broadening import RadiativeBroadening, LineBroadening
from lightweaver.collisional_rates import *
from lightweaver.constants import QElectron, Epsilon0, MElectron, CLight, ERydberg, RBohr, KBoltzmann
from typing import List, Optional


F_QUADRUPOLE = 0.1
R_AVG_FLAG = True
N_MAX_TRY = 20
DELTA = 1.0E-3
BETA_START = 1.0E-5

def Bessel_I0(x):
    ax = np.abs(x)
    if ax < 3.75:
        y = (x * x) / (3.75 * 3.75)
        ans = 1.0 + y * (
            3.5156229 + y * (
                3.0899424 + y * (
                    1.2067492 + y * (
                        0.2659732 + y * (
                            0.360768E-1 + y * 0.45813E-2
                        )
                    )
                )
            )
        )
    else:
        y = 3.75 / ax
        ans = (
            np.exp(ax) / np.sqrt(ax)
        ) * (
            0.39894228 + y * (
                0.1328592E-1 + y * (
                    0.225319E-2 + y * (
                        -0.157565E-2 + y * (
                            0.916281E-2 + y * (
                                -0.2057706E-1 + y * (
                                    0.2635537E-1 + y * (
                                        -0.1647633E-1 + y * 0.392377E-2
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    return ans

def Bessel_I1(x):

    ax = np.abs(x)

    if ax < 3.75:
        y = (x * x) / (3.75 * 3.75)
        ans = ax * (
            0.5 + y * (
                0.87890594 + y * (
                    0.51498869 + y * (
                        0.15084934 + y * (
                            0.2658733E-1 + y * (
                                0.301532E-2 + y * 0.32411E-3
                            )
                        )
                    )
                )
            )
        )
    else:
        y = 3.75 / ax
        ans = 0.2282967E-1 + y * (
            -0.2895312E-1 + y * (
                0.1787654E-1 - y * 0.420059E-2
            )
        )
        ans = 0.39894228 + y * (
            -0.3988024E-1 + y * (
                -0.362018E-2 + y * (
                    0.163801E-2 + y * (
                        -0.1031555E-1 + y * ans
                    )
                )
            )
        )
        ans *= (
                np.exp(ax) / np.sqrt(ax)
        )

    if x < 0.0:
        ans *= -1
    return ans


def Bessel_K0(x):

    if x <= 2.0:
        y = x * x / 4.0
        ans = (
                -np.log(x / 2.0) * Bessel_I0(x)
              ) + (
                -0.57721566 + y * (
                    0.42278420 + y * (
                        0.23069756 + y * (
                            0.3488590E-1 + y * (
                                0.262698E-2 + y * (
                                    0.10750E-3 + y * 0.74E-5
                                )
                            )
                        )
                    )
                )
            )
    else:
        y = 2.0 / x
        ans = (
            np.exp(-x) / np.sqrt(x)
        ) * (
            1.25331414 + y * (
                -0.7832358E-1 + y * (
                    0.2189568E-1 + y * (
                        -0.1062446E-1 + y * (
                            0.587872E-2 + y * (
                                -0.251540E-2 + y * 0.53208E-3
                            )
                        )
                    )
                )
            )
        )

    return ans

def Bessel_K1(x):

    if x <= 2.0:
        y = x * x / 4.0
        ans = (
            np.log(x / 2.0) * Bessel_I1(x)
        ) + (
            1.0 / x
        ) * (
            1.0 + y * (
                0.15443144 + y * (
                    -0.67278579 + y * (
                        -0.18156897 + y * (
                            -0.1919402E-1 + y * (
                                -0.110404E-2 + y * (
                                    -0.4686E-4
                                )
                            )
                        )
                    )
                )
            )
        )
    else:
        y = 2.0 / x
        ans = (
            np.exp(-x) / np.sqrt(x)
        ) * (
            1.25331414 + y * (
                0.23498619 + y * (
                    -0.3655620E-1 + y * (
                        0.1504268E-1 + y * (
                            -0.780353E-2 + y * (
                                0.325614E-2 + y * (
                                    -0.68245E-3
                                )
                            )
                        )
                    )
                )
            )
        )

    return ans


def findbeta1(righthand):

    beta1 = BETA_START
    y = np.square(Bessel_K0(beta1)) + np.square(Bessel_K1(beta1)) - righthand

    if y <= 0.0:
        zeta1 = righthand * np.square(beta1)
        return beta1, zeta1

    Ntry = 0

    while Ntry <= N_MAX_TRY:
        beta2 = 2.0 * beta1
        y = np.square(Bessel_K0(beta2)) + np.square(Bessel_K1(beta2)) - righthand
        if y > 0.0:
            beta1 = beta2
            Ntry += 1
        else:
            break

    if Ntry > N_MAX_TRY:
        zeta1 = np.square(beta2) * righthand
        return beta2, zeta1

    delta = 1.0 - beta1 / beta2
    while delta > DELTA:
        beta = 0.5*(beta1 + beta2)
        y = np.square(Bessel_K0(beta))
        if (y + np.square(Bessel_K1(beta)) - righthand) > 0.0:
          beta1 = beta
        else:
          beta2 = beta

    zeta1 = np.square(beta) * righthand
    return beta, zeta1

def phiImpact(x, x0, ERkT, R0, f, PIa0square):

    x += x0

    beta0 = np.sqrt(x / ERkT) * x0/(2 * x + x0) * R0
    sigma_0 = beta0 * Bessel_K0(beta0) * Bessel_K1(beta0);

    righthand = np.square(2 * x + x0) / (8 * ERkT * x0 * f)
    beta1, zeta1 = findbeta1(righthand)
    sigma_1 = 0.5 * zeta1 + beta1 * Bessel_K0(beta1) * Bessel_K1(beta1)

    return 8.0 * PIa0square * np.square(ERkT) * (f/x0) * np.min(sigma_0, sigma_1)


def GaussLaguerre(x0, ERkT, R0, f, PIa0square):

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
        integral += w_Laguerre[nl] * phiImpact(x_Laguerre[nl], x0, ERkT, R0, f, PIa0square)

    return integral


def impactParam(line, flag, Ntemp, temp, f, PIa0square):

    i = line.i
    j = line.j

    atom = line.atom

    CE = np.zeros(Ntemp)

    if flag is True:
        n_eff_min = np.min(line.iLevel.n_eff, line.jLevel.n_eff)
        R0 = 0.25 * (5.0 * np.square(n_eff_min) + n_eff_min + 1.0)
    else:
        Ri = 0.5 * (3.0 * np.square(line.iLevel.n_eff) - line.iLevel.L * (line.iLevel.L + 1.0))
        Rj = 0.5 * (3.0 * np.square(line.jLevel.n_eff) - line.jLevel.L * (line.jLevel.L + 1.0))
        R0 = np.min(Ri, Rj)

    deltaE = line.jLevel.E_SI - line.iLevel.E_SI
    for k in range(Ntemp):
        x0 = deltaE / (KBoltzmann * temp[k])
        ERkT  = ERydberg / (KBoltzmann * temp[k]);
        CE[k] = GaussLaguerre(x0, ERkT, R0, f, PIa0square)

    return CE


@dataclass
class NewAtomicLevel(AtomicLevel):
    n_eff: Optional[int] = 0

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

        collisions: List[CollisionalRates] = []

        for i_level in self.levels:
            for j_level in self.levels[1:]:
                if i_level.stage == j_level.stage:

                    deltaE = line.jLevel.E_SI - line.iLevel.E_SI

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
                            CE = impactParam(line, R_AVG_FLAG, NTEMP, temp, f, PIa0square)

                            CE *= C3

                        else:
                            CE = C2_atom / np.square(temp) * f * np.power(KBoltzmann * temp / deltaE, 1.68)
