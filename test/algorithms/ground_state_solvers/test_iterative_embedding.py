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

"""Unittests for the IterativeEmbedding solver."""

import unittest

from test import QiskitNatureTestCase

from qiskit.test import slow_test

from qiskit_nature.settings import settings
from qiskit_nature.algorithms.ground_state_solvers import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers.iterative_embedding import IterativeEmbedding
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers.second_quantization import MethodType
from qiskit_nature.drivers.second_quantization.pyscfd.embedding_pyscfdriver import (
    EmbeddingPySCFDriver,
)
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.problems.second_quantization.electronic.embedding_problem import EmbeddingProblem
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
import qiskit_nature.optionals as _optionals


class TestIterativeEmbedding(QiskitNatureTestCase):
    """TODO."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()

        settings.dict_aux_operators = True

        omega = 500.0
        self.transformer = ActiveSpaceTransformer(2, 2, indirect_computation=True)
        self.driver = EmbeddingPySCFDriver(
            atom="H 0.0 0.0 0.0; H 0.0 0.0 0.735",
            charge=0,
            spin=0,
            basis="sto3g",
            method=MethodType.UKS,
            xc_functional="ldaerf + lr_hf",
            xcf_library="xcfun",
            omega=omega,
        )

        self.problem = EmbeddingProblem(self.driver, [self.transformer])

        self.qubit_converter = QubitConverter(JordanWignerMapper())

    def test_numpy(self):
        """TODO."""
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)

        app = IterativeEmbedding(self.qubit_converter, solver)

        result = app.solve(self.problem)

        self.assertAlmostEqual(result.total_energies[0], -1.13730256, places=7)

    @slow_test
    def test_larger(self):
        """TODO."""
        transformer = ActiveSpaceTransformer((3, 1), 4, indirect_computation=True)
        driver = EmbeddingPySCFDriver(
            atom="O 0.0 0.0 0.609; O 0.0 0.0 -0.609",
            charge=0,
            spin=2,
            basis="sto3g",
            method=MethodType.UKS,
            xc_functional="ldaerf + lr_hf",
            xcf_library="xcfun",
            omega=1.0,
        )

        problem = EmbeddingProblem(driver, [transformer])
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)

        qubit_converter = QubitConverter(JordanWignerMapper())
        app = IterativeEmbedding(qubit_converter, solver)

        result = app.solve(problem)

        self.assertAlmostEqual(result.total_energies[0], -147.33022519, places=7)


if __name__ == "__main__":
    unittest.main()
