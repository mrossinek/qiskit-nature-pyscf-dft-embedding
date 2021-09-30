# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Iterative Embedding solver."""

import logging
from typing import List, Optional, Union

import numpy as np

from qiskit.algorithms import MinimumEigensolver
from qiskit.opflow import PauliSumOp

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.problems.second_quantization.electronic.embedding_problem import EmbeddingResult

from .ground_state_eigensolver import GroundStateEigensolver
from .minimum_eigensolver_factories import MinimumEigensolverFactory

LOGGER = logging.getLogger(__name__)


class IterativeEmbedding(GroundStateEigensolver):
    """TODO."""

    def __init__(
        self,
        qubit_converter: QubitConverter,
        solver: Union[MinimumEigensolver, MinimumEigensolverFactory],
        max_iter: int = 50,
        threshold: float = 1e-6,
    ) -> None:
        """TODO."""
        super().__init__(qubit_converter, solver)
        self._max_iter: int = max_iter
        self._threshold: float = threshold

    @property
    def max_iter(self) -> int:
        """Return the max_iter."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int) -> None:
        """Set the max_iter."""
        self._max_iter = max_iter

    @property
    def threshold(self) -> float:
        """Return the threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """Set the threshold."""
        self._threshold = threshold

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[List[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ) -> "IterativeEmbeddingResult":
        """TODO."""
        e_prev = float("NaN")
        e_next = float("NaN")
        n_iter = 0

        result: EmbeddingResult
        history: List[EmbeddingResult] = []
        converged = False

        while n_iter < self._max_iter and not converged:
            n_iter += 1
            LOGGER.info("========== %s ==========", n_iter)

            result = super().solve(problem, aux_operators)
            LOGGER.info(str(result))

            history.append(result)

            e_prev = e_next
            e_next = result.total_energies[0]

            converged = np.abs(e_prev - e_next) < self._threshold

        final_result = IterativeEmbeddingResult()
        final_result.combine(result)
        final_result.num_iterations = n_iter
        final_result.converged = converged
        final_result.history = history
        return final_result


class IterativeEmbeddingResult(EmbeddingResult):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__()
        self._num_iterations: int = 0
        self._converged: bool = False
        self._history: List[EmbeddingResult] = []

    @property
    def num_iterations(self) -> int:
        """Return number of iterations."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        """Set number of iterations."""
        self._num_iterations = value

    @property
    def converged(self) -> bool:
        """Return whether the solver converged."""
        return self._converged

    @converged.setter
    def converged(self, converged: bool) -> None:
        """Set whether the solver converged."""
        self._converged = converged

    @property
    def history(self) -> List[EmbeddingResult]:
        """Return the history."""
        return self._history

    @history.setter
    def history(self, history: List[EmbeddingResult]) -> None:
        """Set the history."""
        self._history = history
