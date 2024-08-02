import traceback

import numpy as np
from dataclasses import dataclass

from lightweaver import PeriodicTable
from lightweaver.atomic_model import AtomicModel, AtomicLevel, ExplicitContinuum, HydrogenicContinuum, AtomicContinuum, \
    VoigtLine, LinearCoreExpWings, LineType
from lightweaver.broadening import RadiativeBroadening, LineBroadening, HydrogenLinearStarkBroadening, \
    QuadraticStarkBroadening, MultiplicativeStarkBroadening, VdwUnsold, VdwBarklem
from lightweaver.collisional_rates import *
from lightweaver.constants import QElectron, Epsilon0, MElectron, CLight, ERydberg, RBohr, KBoltzmann
from typing import List, Optional
from parse import parse
import re
from fractions import Fraction
import colorama
from colorama import Fore, Style


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

        delta = 1.0 - beta1 / beta2

    zeta1 = np.square(beta) * righthand
    return beta, zeta1

def phiImpact(x, x0, ERkT, R0, f, PIa0square):

    x += x0

    beta0 = np.sqrt(x / ERkT) * x0/(2 * x + x0) * R0
    sigma_0 = beta0 * Bessel_K0(beta0) * Bessel_K1(beta0);

    righthand = np.square(2 * x + x0) / (8 * ERkT * x0 * f)
    beta1, zeta1 = findbeta1(righthand)
    sigma_1 = 0.5 * zeta1 + beta1 * Bessel_K0(beta1) * Bessel_K1(beta1)

    return 8.0 * PIa0square * np.square(ERkT) * (f/x0) * np.minimum(sigma_0, sigma_1)


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

    rates = np.zeros(Ntemp)

    if flag is True:
        n_eff_min = np.minimum(line.iLevel.n_eff, line.jLevel.n_eff)
        R0 = 0.25 * (5.0 * np.square(n_eff_min) + n_eff_min + 1.0)
    else:
        Ri = 0.5 * (3.0 * np.square(line.iLevel.n_eff) - line.iLevel.L * (line.iLevel.L + 1.0))
        Rj = 0.5 * (3.0 * np.square(line.jLevel.n_eff) - line.jLevel.L * (line.jLevel.L + 1.0))
        R0 = np.minimum(Ri, Rj)

    deltaE = line.jLevel.E_SI - line.iLevel.E_SI

    for k in range(Ntemp):
        x0 = deltaE / (KBoltzmann * temp[k])
        ERkT  = ERydberg / (KBoltzmann * temp[k]);
        rates[k] = GaussLaguerre(x0, ERkT, R0, f, PIa0square)

    return rates


@dataclass(eq=False)
class NewAtomicLevel(AtomicLevel):
    n_eff: Optional[int] = 0

    def __repr__(self):
        return super(NewAtomicLevel, self).__repr__()


@dataclass(eq=False)
class NewExplicitContinuum(ExplicitContinuum):
    alpha0: Optional[float] = 0.0

    def __repr__(self):
        return super(NewExplicitContinuum, self).__repr__()


@dataclass(eq=False)
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

            level.n_eff = z * np.sqrt(ERydberg / (self.levels[ic].E_SI - level.E_SI))

    def recompute_collisional_rates(self):

        self.compute_n_eff()

        temp = [
            1264.9, 2249.4, 4000.0, 7113.1, 12649.1, 22493.6, 40000.0,
            71131.1, 126491.0, 224936.0, 400000.0, 711312.0, 1264910.0
        ]

        NTEMP = len(temp)

        C0 = 1.55E+11

        C1 = ((ERydberg / np.sqrt(MElectron)) * np.pi * np.square(RBohr)) * np.sqrt(8.0 / (np.pi * KBoltzmann))

        C2_atom = 2.15E-6

        C2_ion = 3.96E-6

        C3 = np.sqrt(8 * KBoltzmann / (np.pi * MElectron))

        PIa0square = np.pi * np.square(RBohr)

        collisions: List[CollisionalRates] = []

        for i, i_level in enumerate(self.levels):
            for j_index, j_level in enumerate(self.levels[i+1:]):

                j = i + 1 + j_index

                if i_level.stage == j_level.stage:

                    delta_e = j_level.E_SI - i_level.E_SI

                    valid_transition = False

                    for line in self.lines:
                        if line.i == i and line.j == j:
                            valid_transition = True
                            f = line.f
                            break

                    if valid_transition is False:
                        f = F_QUADRUPOLE

                    if i_level.stage == 0:
                        if valid_transition is True:
                            rates = impactParam(line, R_AVG_FLAG, NTEMP, temp, f, PIa0square)

                            rates *= C3

                        else:
                            rates = C2_atom / np.square(temp) * f * np.power(KBoltzmann * np.array(temp) / delta_e, 1.68)

                        collisions.append(CE(j=j, i=i, temperature=temp, rates=rates))

                    else:
                        rates = np.zeros(NTEMP)

                        for k in range(NTEMP):
                            rates[k] = line.iLevel.g * C2_ion / (C1 * temp[k]) * f * ((KBoltzmann * temp[k]) / delta_e)

                        collisions.append(Omega(j=j, i=i, temperature=temp, rates=rates))

        for continuum in self.continua:
            i = continuum.i
            ic = continuum.j

            alpha0 = continuum.alpha0

            delta_e = self.levels[ic].E_SI - self.levels[i].E_SI

            if continuum.iLevel.stage == 0:
                gbar_i = 0.1
            elif continuum.iLevel.stage == 1:
                gbar_i = 0.2
            else:
                gbar_i = 0.3

            rates = np.zeros(NTEMP)

            rates = C0 / np.array(temp) * alpha0 * gbar_i * ((KBoltzmann * np.array(temp)) / delta_e)

            collisions.append(CI(j=ic, i=i, temperature=temp, rates=rates))

        self.collisions = collisions

    def __repr__(self):
        return super(NewAtomicModel, self).__repr__()

# https://stackoverflow.com/a/3303361
def clean(s):
    # Replace '.' with '_'
    s = re.sub('[.]', '_', s)
    # Replace '-' with '_'
    s = re.sub('[-]', '_', s)
    # Remove invalid characters
    s = re.sub('[^0-9a-zA-Z_]', '', s)
    # Remove leading characters until we find a letter or underscore
    s = re.sub('^[^a-zA-Z_]+', '', s)
    return s

@dataclass
class PrincipalQuantum:
    J: Fraction
    L: int
    S: Fraction

class CompositeLevelError(Exception):
    pass

def get_oribital_number(orbit: str) -> int:
    orbits = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X']
    return orbits.index(orbit)


def determinate(level: AtomicLevel) -> PrincipalQuantum:
    endIdx = [level.label.upper().rfind(x) for x in ['E', 'O']]
    maxIdx = max(endIdx)
    if maxIdx == -1:
        raise ValueError("Unable to determine parity of level %s" % (repr(level)))
    label = level.label[:maxIdx+1].upper()
    words: List[str] = label.split()

    # _, multiplicity, orbit = parse('{}{:d}{!s}', words[-1])
    match = re.match('[\S-]*(\d)(\S)[EO]$', words[-1])
    if match is None:
        raise ValueError('Unable to parse level label: %s' % level.label)
    else:
        multiplicity = int(match.group(1))
        orbit = match.group(2)
    S = Fraction(int(multiplicity - 1), 2)
    L = get_oribital_number(orbit)
    J = Fraction(int(level.g - 1.0), 2)

    # if J > L + S:
    #     raise CompositeLevelError('J (%f) > L (%d) + S (%f): %s' %(J, L, S, repr(level)))

    return PrincipalQuantum(J=J, L=L, S=S)

def check_barklem_compatible(vals: List[float],
                             iLev: AtomicLevel, jLev: AtomicLevel) -> bool:

    if vals[0] >= 20.0:
        return True

    if iLev.stage > 0:
        return False

    lowerNum = iLev.L
    upperNum = jLev.L
    if upperNum is None or lowerNum is None:
        return False

    if not ((abs(upperNum - lowerNum) == 1)
            and (max(upperNum, lowerNum) <= 3)):
        return False

    # NOTE(cmo): We're not checking the table bounds here, but that should be fine.

    return True

def getNextLine(data):
    if len(data) == 0:
        return None
    for i, d in enumerate(data):
        if d.strip().startswith('#') or d.strip() == '':
            # print('Skipping %s' % d)
            continue
        # print('Accepting %s' % d)
        break
    d = data[i]
    if i == len(data) - 1:
        data[:] = []
        return d.strip()
    data[:] = data[i+1:]
    return d.strip()

def maybe_int(s):
    try:
        v = int(s)
    except:
        v = None
    return v

def conv_atom(inFile):
    with open(inFile, 'r') as fi:
        data = fi.readlines()

    ID = getNextLine(data)
    element = PeriodicTable[ID]
    print('='*40 + '\n' + 'Reading model atom %s from file %s' % (ID, inFile) )
    Ns = [maybe_int(d) for d in getNextLine(data).split()]
    Nlevel = Ns[0]
    Nline = Ns[1]
    Ncont = Ns[2]
    Nfixed = Ns[3]

    if Nfixed != 0:
        raise ValueError("Fixed transitions are not supported")

    levels = []
    # levelNos: List[int] = []
    for n in range(Nlevel):
        line = getNextLine(data)

        res = parse('{:f}{}{:f}{}\'{}\'{}{:d}{}{:d}', line.strip())
        # print(res)
        # print(line)
        E = res[0]
        g = res[2]
        label = res[4].strip()
        stage = res[6]
        # levelNo = int(res[8])
        # if n > 0:
        #     if levelNo < levelNos[-1]:
        #         raise ValueError('Levels are not monotonically increasing (%f < %f)' % (levelNo, levelNos[-1]))
        # levelNos.append(levelNo)
        levels.append(NewAtomicLevel(E=E, g=g, label=label, stage=stage))
        try:
            qNos = determinate(levels[-1])
            levels[-1].J = qNos.J
            levels[-1].L = qNos.L
            levels[-1].S = qNos.S
        except Exception as e:
            print(Fore.BLUE + 'Unable to determine quantum numbers for %s' % repr(levels[-1]))
            print('\t %s' % (repr(e)) + Style.RESET_ALL)


    lines = []
    lineNLambdas = []
    for n in range(Nline):
        line = getNextLine(data)
        line = line.split()

        j = int(line[0])
        i = int(line[1])
        f = float(line[2])
        typ = line[3]
        Nlambda = int(line[4])
        sym = line[5]
        qCore = float(line[6])
        qWing = float(line[7])
        vdw = line[8]
        vdwParams = [float(x) for x in line[9:13]]
        gRad = float(line[13])
        stark = float(line[14])
        if len(line) > 15:
            gLande = float(line[15])
        else:
            gLande = None

        if typ.upper() == 'PRD':
            lineType = LineType.PRD
        elif typ.upper() == 'VOIGT':
            lineType = LineType.CRD
        else:
            raise ValueError('Only PRD and VOIGT lines are supported, found type %s' % typ)

        # if sym.upper() != 'ASYMM':
        #     print('Only Asymmetric lines are supported, doubling Nlambda')
        #     Nlambda *= 2

        # if vdw.upper() == 'PARAMTR':
        #     vdwApprox: VdwApprox = VdwRidderRensbergen(vdwParams)
        if vdw.upper() == 'UNSOLD':
            vdwParams = [vdwParams[0], vdwParams[2]]
            vdwApprox = VdwUnsold(vdwParams)
        elif vdw.upper() == 'BARKLEM':
            vdwParams = [vdwParams[0], vdwParams[2]]
            if check_barklem_compatible(vdwParams, levels[i], levels[j]):
                vdwApprox = VdwBarklem(vdwParams)
            else:
                vdwApprox = VdwUnsold(vdwParams)
        else:
            raise ValueError('Unknown vdw type %s' % vdw)

        if stark <= 0:
            starkBroaden = MultiplicativeStarkBroadening(abs(stark))
        else:
            starkBroaden = QuadraticStarkBroadening(stark)

        broadening = LineBroadening(natural=[RadiativeBroadening(gRad)], elastic=[vdwApprox, starkBroaden])
        if element == PeriodicTable[1]:
            broadening.elastic.append(HydrogenLinearStarkBroadening())

        quadrature = LinearCoreExpWings(qCore=qCore, qWing=qWing, Nlambda=Nlambda)
        lines.append(VoigtLine(j=j, i=i, f=f, type=lineType, quadrature=quadrature, broadening=broadening, gLandeEff=gLande))
        lineNLambdas.append(Nlambda)


    continua: List[AtomicContinuum] = []
    for n in range(Ncont):
        line = getNextLine(data)
        line = line.split()
        j = int(line[0])
        i = int(line[1])
        alpha0 = float(line[2])
        Nlambda = int(line[3])
        wavelengthDep = line[4]
        minLambda = float(line[5])

        if wavelengthDep.upper() == 'EXPLICIT':
            wavelengths = []
            alphas = []
            for _ in range(Nlambda):
                l = getNextLine(data)
                l = l.split()
                wavelengths.append(float(l[0]))
                alphas.append(float(l[1]))
            wavelengthGrid = wavelengths[::-1]
            alphaGrid = alphas[::-1]
            continua.append(NewExplicitContinuum(j=j, i=i, wavelengthGrid=wavelengthGrid, alphaGrid=alphaGrid))
        elif wavelengthDep.upper() == 'HYDROGENIC':
            continua.append(HydrogenicContinuum(j=j, i=i, alpha0=alpha0, minWavelength=minLambda, NlambdaGen=Nlambda))
        else:
            raise ValueError('Unknown Continuum type %s' % wavelengthDep)

    collisions: List[CollisionalRates] = []
    while True:
        line = getNextLine(data)
        if line == 'END' or line is None:
            break

        line = line.split()
        if line[0].upper() == 'TEMP':
            Ntemp = int(line[1])
            temperatureGrid = []
            for i in range(Ntemp):
                temperatureGrid.append(float(line[i+2]))
        elif line[0].upper() == 'OMEGA':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(Omega(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CI':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CI(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CE':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CE(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CP':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CP(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CH':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CH(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CH0':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(ChargeExchangeNeutralH(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CH+':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(ChargeExchangeProton(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'AR85-CDI':
            i1 = int(line[1])
            i2 = int(line[2])
            Nrow = int(line[3])
            Mshell = 5
            j = max(i1, i2)
            i = min(i1, i2)
            cdi = []
            for n in range(Nrow):
                line = getNextLine(data)
                line = line.split()
                cdi.append([])
                for m in range(Mshell):
                    cdi[n].append(float(line[m]))
            collisions.append(Ar85Cdi(j=j, i=i, cdi=cdi))
        elif line[0].upper() == 'BURGESS':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            fudge = float(line[3])
            collisions.append(Burgess(j=j, i=i, fudge=fudge))
        else:
            print(Fore.YELLOW + "Ignoring unknown collisional string %s" % line[0].upper() + Style.RESET_ALL)

    atom = NewAtomicModel(element=element, levels=levels, lines=lines, continua=continua, collisions=collisions)
    return atom
