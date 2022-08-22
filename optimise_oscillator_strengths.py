import numpy as np
from dataclasses import dataclass
from lightweaver.atomic_model import AtomicModel
from lightweaver.broadening import RadiativeBroadening, LineBroadening
from lightweaver.collisional_rates import *
from lightweaver.constants import QElectron, Epsilon0, MElectron, CLight, ERydberg, RBohr, KBoltzmann,
from typing import List


F_QUADRUPOLE = 0.1

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
                    deltaE = j_level.E_SI - i_level.E_SI

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




        collisions: List[CollisionalRates] = []



