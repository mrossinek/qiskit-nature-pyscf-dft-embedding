# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ElectronicDensity property."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import List, Tuple

import re
import logging

import numpy as np

from qiskit_nature import ListOrDictType, QiskitNatureError, settings
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult
from .bases import ElectronicBasis, ElectronicBasisTransform
from .integrals import (
    ElectronicIntegrals,
    IntegralProperty,
    OneBodyElectronicIntegrals,
)
from .particle_number import ParticleNumber

LOGGER = logging.getLogger(__name__)


class ElectronicDensity(IntegralProperty):
    """The ElectronicDensity property.

    This is the central property around which the iterative DFT-Embedding is implemented.
    """

    def __init__(self, electronic_integrals: List[ElectronicIntegrals]) -> None:
        """
        Args:
        """
        super().__init__(self.__class__.__name__, electronic_integrals)
        self._norbs = self.get_electronic_integral(ElectronicBasis.MO, 1)._matrices[0].shape[0]

    def second_q_ops(self) -> ListOrDictType[FermionicOp]:
        """Returns a list containing the Hamiltonian constructed by the stored electronic integrals."""
        if not settings.dict_aux_operators:
            raise QiskitNatureError(
                "The `ElectronicDensity` property only works with dict-based aux_operators."
            )

        aux_ops = {}
        for mo_i, mo_j in product(range(self._norbs), repeat=2):
            aux_ops[f"ElectronicDensity({mo_i}, {mo_j})"] = FermionicOp(
                f"+_{mo_i} -_{mo_j}", register_length=2 * self._norbs, display_format="sparse"
            )

            aux_ops[f"ElectronicDensity({mo_i + self._norbs}, {mo_j + self._norbs})"] = FermionicOp(
                f"+_{mo_i + self._norbs} -_{mo_j + self._norbs}",
                register_length=2 * self._norbs,
                display_format="sparse",
            )

        return aux_ops

    @staticmethod
    def from_particle_number(particle_number: ParticleNumber) -> ElectronicDensity:
        """TODO."""
        return ElectronicDensity(
            [
                OneBodyElectronicIntegrals(
                    ElectronicBasis.MO,
                    (
                        np.diag(particle_number.occupation_alpha),
                        np.diag(particle_number.occupation_beta),
                    ),
                ),
            ]
        )

    def evaluate_particle_number(self, aux_values) -> Tuple[float, float]:
        """TODO."""
        mo_idx_regex = re.compile(r"ElectronicDensity\((\d+), (\d+)\)")

        alpha_sum = 0.0
        beta_sum = 0.0

        for name, aux_value in aux_values.items():
            match = mo_idx_regex.fullmatch(name)
            if match is None:
                continue
            mo_i, mo_j = (int(idx) for idx in match.groups())
            if mo_i != mo_j:
                continue
            if mo_i < self._norbs:
                alpha_sum += aux_value[0]
            else:
                beta_sum += aux_value[0]

        return (alpha_sum, beta_sum)

    def transform_basis(self, transform: ElectronicBasisTransform) -> None:
        """Applies an ElectronicBasisTransform to the internal integrals.

        Args:
            transform: the ElectronicBasisTransform to apply.
        """
        for integral in self._electronic_integrals[transform.initial_basis].values():
            self.add_electronic_integral(integral.transform_basis(transform))
            self._norbs = self.get_electronic_integral(ElectronicBasis.MO, 1)._matrices[0].shape[0]

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Note that in this specific case, the active density information as evaluated during the
        quantum algorithm gets extracted from the result and is used to update this property itself,
        too.

        Args:
            result: the result to add meaning to.
        """
        rho_update_a = np.zeros((self._norbs, self._norbs), dtype=float)
        rho_update_b = np.zeros((self._norbs, self._norbs), dtype=float)

        mo_idx_regex = re.compile(r"ElectronicDensity\((\d+), (\d+)\)")

        for name, aux_value in result.aux_operator_eigenvalues[0].items():
            match = mo_idx_regex.fullmatch(name)
            if match is None:
                continue
            mo_i, mo_j = (int(idx) for idx in match.groups())
            if mo_i < self._norbs and mo_j < self._norbs:
                rho_update_a[mo_i, mo_j] = np.real(aux_value[0])
            elif mo_i >= self._norbs and mo_j >= self._norbs:
                rho_update_b[mo_i - self._norbs, mo_j - self._norbs] = np.real(aux_value[0])

        result.active_density = ElectronicDensity(
            [OneBodyElectronicIntegrals(ElectronicBasis.MO, (rho_update_a, rho_update_b))]
        )

    def __rmul__(self, other: complex) -> ElectronicDensity:
        return ElectronicDensity([other * int for int in iter(self)])

    def __truediv__(self, other: complex) -> ElectronicDensity:
        return ElectronicDensity([(1.0 / other) * int for int in iter(self)])

    def __add__(self, other: ElectronicDensity) -> ElectronicDensity:
        added = deepcopy(self)

        iterator = added.__iter__()
        sum_int = None

        while True:
            try:
                self_int = iterator.send(sum_int)
            except StopIteration:
                break

            other_int = other.get_electronic_integral(self_int.basis, 1)
            if other_int is None:
                sum_int = None
            else:
                sum_int = self_int + other_int

        return added

    def __sub__(self, other: ElectronicDensity) -> ElectronicDensity:
        return self + (-1.0) * other
