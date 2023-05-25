"""TODO."""

import unittest

import numpy as np
from qiskit_nature.second_q.algorithms import (GroundStateEigensolver,
                                               NumPyMinimumEigensolverFactory)
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from dft_embedding_solver import DFTEmbeddingSolver


class TestDFTEmbeddingSolver(unittest.TestCase):
    """TODO."""

    def test_demo_example(self):
        """TODO."""
        omega = 1.0

        driver = PySCFDriver(
            atom="O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
            basis="6-31g*",
            method=MethodType.RKS,
            xc_functional=f"ldaerf + lr_hf({omega})",
            xcf_library="xcfun",
        )

        active_space = ActiveSpaceTransformer(4, 4)

        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        algo = GroundStateEigensolver(converter, solver)

        dft_solver = DFTEmbeddingSolver(active_space, algo)

        result = dft_solver.solve(driver, omega)

        self.assertAlmostEqual(result.total_energies[0], -75.930449640258, places=6)

    def test_hf_limit(self):
        """TODO."""
        omega = 10000.0

        driver = PySCFDriver(
            atom="O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
            basis="6-31g*",
            method=MethodType.RKS,
            xc_functional=f"ldaerf + lr_hf({omega})",
            xcf_library="xcfun",
        )

        active_space = ActiveSpaceTransformer(4, 4)

        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        algo = GroundStateEigensolver(converter, solver)

        dft_solver = DFTEmbeddingSolver(active_space, algo)

        result = dft_solver.solve(driver, omega)

        ref_problem = ActiveSpaceTransformer(4, 4).transform(driver.run())
        ref_result = algo.solve(ref_problem)

        self.assertAlmostEqual(
            result.total_energies[0], ref_result.total_energies[0], places=5
        )

    def test_dft_limit(self):
        """TODO."""
        omega = 0.01

        driver = PySCFDriver(
            atom="O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
            basis="6-31g*",
            method=MethodType.RKS,
            xc_functional=f"ldaerf + lr_hf({omega})",
            xcf_library="xcfun",
        )

        active_space = ActiveSpaceTransformer(4, 4)

        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        algo = GroundStateEigensolver(converter, solver)

        dft_solver = DFTEmbeddingSolver(active_space, algo)

        result = dft_solver.solve(driver, omega)

        self.assertAlmostEqual(
            result.total_energies[0], result.hartree_fock_energy, places=5
        )


if __name__ == "__main__":
    unittest.main()
