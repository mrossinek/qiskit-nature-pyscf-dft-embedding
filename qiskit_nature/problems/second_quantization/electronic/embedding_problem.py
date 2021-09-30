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

"""The Embedding Problem."""

from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Union, cast

import logging
import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult

from qiskit_nature.drivers.second_quantization import ElectronicStructureDriver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.properties.second_quantization.electronic import ElectronicDensity
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasisTransform,
)
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult
from qiskit_nature.transformers.second_quantization import BaseTransformer

from .electronic_structure_problem import ElectronicStructureProblem

LOGGER = logging.getLogger(__name__)


class EmbeddingResult(ElectronicStructureResult):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__()
        self._active_density: ElectronicDensity

    @property
    def active_density(self) -> ElectronicDensity:
        """Return the active density."""
        return self._active_density

    @active_density.setter
    def active_density(self, active_density: ElectronicDensity) -> None:
        """Set the active density."""
        self._active_density = active_density


class EmbeddingProblem(ElectronicStructureProblem):
    """TODO."""

    def __init__(
        self,
        driver: ElectronicStructureDriver,
        transformers: Optional[List[BaseTransformer]] = None,
        density_mixing: Callable[
            [List[ElectronicDensity]], ElectronicDensity
        ] = lambda history: history[-1],
    ):
        super().__init__(driver, transformers)

        self._active_density_history: List[ElectronicDensity] = []
        self._density_mixing = density_mixing

    def _merge_densities(self) -> ElectronicDensity:
        """TODO."""
        basis_trafo = cast(
            ElectronicBasisTransform,
            self.grouped_property_transformed.get_property(ElectronicBasisTransform),
        )
        # 1. get total density
        total = cast(ElectronicDensity, self.grouped_property.get_property(ElectronicDensity))
        # 2. extract active subspace matrix from total density
        subspace = deepcopy(total)
        subspace.transform_basis(basis_trafo)
        subspace.transform_basis(basis_trafo.invert())
        # 3. remove subspace density part from total density
        total -= subspace
        # 4. get active density
        active = self._density_mixing(self._active_density_history)
        self._active_density_history[-1] = active
        # 5. enlarge active density to total space
        superspace = deepcopy(active)
        superspace.transform_basis(basis_trafo.invert())
        # 6. insert active density into total density
        total += superspace
        return total

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """TODO."""
        second_q_ops: List[SecondQuantizedOp]
        if self._grouped_property is None:
            second_q_ops = super().second_q_ops()
            self._active_density_history.append(
                self._grouped_property_transformed.get_property("ElectronicDensity")
            )
        else:
            # update total density based on new active density
            merged_density = self._merge_densities()
            LOGGER.info("Merged density = %s", str(merged_density))
            # evaluate energy at new total density
            self._grouped_property = self.driver.evaluate_energy(merged_density)  # type: ignore
            # reduce new grouped property based with AST
            self._grouped_property_transformed = self._transform(self.grouped_property)
            # finally, get second quantized operators
            second_q_ops = self.grouped_property_transformed.second_q_ops()

        return second_q_ops

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """TODO."""
        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            electronic_density = cast(
                ElectronicDensity, self.grouped_property_transformed.get_property(ElectronicDensity)
            )
            exp_aux_values = electronic_density.evaluate_particle_number(aux_values)
            LOGGER.info("Filter: %s", exp_aux_values)
            return np.allclose(self.num_particles, exp_aux_values)

        return partial(filter_criterion, self)

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> EmbeddingResult:
        """TODO."""
        eigenstate_result = None
        if isinstance(raw_result, EigenstateResult):
            eigenstate_result = raw_result
        elif isinstance(raw_result, EigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = raw_result.eigenvalues
            eigenstate_result.eigenstates = raw_result.eigenstates
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
        elif isinstance(raw_result, MinimumEigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = np.asarray([raw_result.eigenvalue])
            eigenstate_result.eigenstates = [raw_result.eigenstate]
            eigenstate_result.aux_operator_eigenvalues = [raw_result.aux_operator_eigenvalues]
        result = EmbeddingResult()
        result.combine(eigenstate_result)
        self._grouped_property_transformed.interpret(result)
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])
        self._active_density_history.append(result.active_density)
        return result
