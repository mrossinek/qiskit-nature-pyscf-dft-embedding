# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the `DFTEmbeddingSolver`."""

import unittest
from functools import partial

import numpy as np
from ddt import data, ddt, unpack
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.settings import settings

from dft_embedding_solver import DFTEmbeddingSolver

settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True


def filter_criterion(
    eigenstate,
    eigenvalue,
    aux_values,
    expected_num_particles,
    expected_angular_momentum,
):
    eval_num_particles = aux_values.get("ParticleNumber", None)
    if eval_num_particles is None:
        return True
    num_particles_close = np.isclose(eval_num_particles[0], expected_num_particles)

    eval_angular_momentum = aux_values.get("AngularMomentum", None)
    if eval_angular_momentum is None:
        return num_particles_close
    angular_momentum_close = np.isclose(
        eval_angular_momentum[0], expected_angular_momentum
    )

    return num_particles_close and angular_momentum_close


@ddt
class TestDFTEmbeddingSolver(unittest.TestCase):
    """Tests for the `DFTEmbeddingSolver`."""

    def test_demo_example(self):
        """Test the example provided in the `demo.py` file of this repo."""
        omega = 1.0

        driver = PySCFDriver(
            atom="O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
            basis="6-31g*",
            method=MethodType.RKS,
            xc_functional=f"ldaerf + lr_hf({omega})",
            xcf_library="xcfun",
        )

        active_space = ActiveSpaceTransformer(4, 4)

        mapper = ParityMapper(num_particles=(2, 2))
        solver = NumPyMinimumEigensolver()
        solver.filter_criterion = lambda state, val, aux: np.isclose(
            aux["ParticleNumber"][0], 4.0
        )
        algo = GroundStateEigensolver(mapper, solver)

        dft_solver = DFTEmbeddingSolver(active_space, algo)

        result = dft_solver.solve(driver, omega)

        self.assertAlmostEqual(result.total_energies[0], -75.93044878549, places=6)

    def test_hf_limit(self):
        """Test that the embedding converges to the HF value for large omega."""
        omega = 10000.0

        driver = PySCFDriver(
            atom="H 0.0 0.0 0.0; H 0.0 0.0 0.735",
            basis="sto-3g",
            method=MethodType.RKS,
            xc_functional=f"ldaerf + lr_hf({omega})",
            xcf_library="xcfun",
        )

        active_space = ActiveSpaceTransformer(2, 2)

        mapper = ParityMapper(num_particles=(1, 1))
        solver = NumPyMinimumEigensolver()
        solver.filter_criterion = partial(
            filter_criterion,
            expected_num_particles=2,
            expected_angular_momentum=0,
        )
        algo = GroundStateEigensolver(mapper, solver)

        dft_solver = DFTEmbeddingSolver(active_space, algo)

        result = dft_solver.solve(driver, omega)

        ref_problem = ActiveSpaceTransformer(2, 2).transform(driver.run())
        ref_result = algo.solve(ref_problem)

        self.assertAlmostEqual(
            result.total_energies[0], ref_result.total_energies[0], places=5
        )

    def test_dft_limit(self):
        """Test that the embedding converges to the DFT value for small omega."""
        omega = 0.01

        driver = PySCFDriver(
            atom="H 0.0 0.0 0.0; H 0.0 0.0 0.735",
            basis="sto-3g",
            method=MethodType.RKS,
            xc_functional=f"ldaerf + lr_hf({omega})",
            xcf_library="xcfun",
        )

        active_space = ActiveSpaceTransformer(2, 2)

        mapper = ParityMapper(num_particles=(1, 1))
        solver = NumPyMinimumEigensolver()
        solver.filter_criterion = partial(
            filter_criterion,
            expected_num_particles=2,
            expected_angular_momentum=0,
        )
        algo = GroundStateEigensolver(mapper, solver)

        dft_solver = DFTEmbeddingSolver(active_space, algo)

        result = dft_solver.solve(driver, omega)

        self.assertAlmostEqual(
            result.total_energies[0], result.hartree_fock_energy, places=5
        )

    @unpack
    @data(
        (
            "O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
            "sto-3g",
            0,
            MethodType.RKS,
            0.1,
            (2, 2),
            -74.73098435889123,
        ),
        (
            "O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
            "sto-3g",
            0,
            MethodType.RKS,
            1.0,
            (2, 2),
            -74.8430435685246,
        ),
        (
            "O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
            "sto-3g",
            0,
            MethodType.RKS,
            10.0,
            (2, 2),
            -74.99908894924124,
        ),
        (
            "N 0.0 0.0 0.539; N 0.0 0.0 -0.539",
            "sto-3g",
            0,
            MethodType.RKS,
            0.1,
            (2, 2),
            -107.13720604014377,
        ),
        (
            "N 0.0 0.0 0.539; N 0.0 0.0 -0.539",
            "sto-3g",
            0,
            MethodType.RKS,
            1.0,
            (2, 2),
            -107.20511090905522,
        ),
        (
            "N 0.0 0.0 0.539; N 0.0 0.0 -0.539",
            "sto-3g",
            0,
            MethodType.RKS,
            10.0,
            (2, 2),
            -107.54791372130454,
        ),
        (
            "O 0.0 0.0 0.609; O 0.0 0.0 -0.609",
            "sto-3g",
            2,
            MethodType.UKS,
            0.1,
            (2, 3),
            -147.1976331919598,
        ),
        (
            "O 0.0 0.0 0.609; O 0.0 0.0 -0.609",
            "sto-3g",
            2,
            MethodType.UKS,
            1.0,
            (2, 3),
            -147.3192014235415,
        ),
        (
            "O 0.0 0.0 0.609; O 0.0 0.0 -0.609",
            "sto-3g",
            2,
            MethodType.UKS,
            10.0,
            (2, 3),
            -147.7044685785955,
        ),
    )
    def test_references(
        self, atom, basis, spin, method, omega, active_space, expected_value
    ):
        """Some additional molecule tests.

        Args:
            atom: the atomic coordinates.
            basis: the basis set.
            spin: the spin of the system (in PySCF convention this is `2 * S`)
            method: the `MethodType` (should be either `RKS` or `UKS`).
            omega: the range-separation parameter.
            active_space: the active-space specification.
            expected_value: the expected total energy value.
        """
        driver = PySCFDriver(
            atom=atom,
            basis=basis,
            spin=spin,
            method=method,
            xc_functional=f"ldaerf + lr_hf({omega})",
            xcf_library="xcfun",
        )

        trafo = ActiveSpaceTransformer(*active_space)

        expected_num_particles = active_space[0]
        expected_angular_momentum = ((spin + 1) ** 2 - 1.0) / 4.0

        num_elec_total = active_space[0]
        num_beta = num_elec_total // 2 - spin
        if num_beta < 0:
            num_beta = 0
        num_alpha = num_elec_total - num_beta

        mapper = ParityMapper(num_particles=(num_alpha, num_beta))
        solver = NumPyMinimumEigensolver()
        solver.filter_criterion = partial(
            filter_criterion,
            expected_num_particles=expected_num_particles,
            expected_angular_momentum=expected_angular_momentum,
        )
        algo = GroundStateEigensolver(mapper, solver)

        dft_solver = DFTEmbeddingSolver(trafo, algo)

        result = dft_solver.solve(driver, omega)

        self.assertAlmostEqual(result.total_energies[0], expected_value, places=5)


if __name__ == "__main__":
    unittest.main()
