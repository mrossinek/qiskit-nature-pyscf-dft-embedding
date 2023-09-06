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

"""This module provides the `DFTEmbeddingSolver`.

This work is the latest derivation of the work published in J. Chem. Phys. 154, 114105 (2021).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from qiskit_nature.second_q.algorithms import GroundStateSolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureResult
from qiskit_nature.second_q.properties import ElectronicDensity
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BasisTransformer


class DFTEmbeddingSolver:
    """A class to solve the DFT-Embedding problem by means of range-separation.

    See the `demo.py` file in the root of this repository for an example on how to use this class.
    """

    def __init__(
        self,
        active_space: ActiveSpaceTransformer,
        solver: GroundStateSolver,
        *,
        max_iter: int = 100,
        threshold: float = 1e-5,
    ) -> None:
        """
        Args:
            active_space: the `ActiveSpaceTransformer` specifying the active-space decomposition.
            solver: the `GroundStateSolver` used to solve the active-space problem.
            max_iter: the maximum number of iterations before aborting.
            threshold: the energy convergence threshold.
        """
        self.active_space = active_space
        self.solver = solver
        self.max_iter = max_iter
        self.threshold = threshold

    def solve(self, driver: PySCFDriver, omega: float) -> ElectronicStructureResult:
        """Solves the actual DFT-Embedding problem by means of range-separation.

        Args:
            driver: the `PySCFDriver` dealing with the interface to PySCF.
            omega: the range-separation parameter.

        Returns:
            The `ElectronicStructureResult` obtained by solving the active-space problem in the
            final iteration.
        """
        # 1. run the reference calculation
        driver.run_pyscf()

        # 2. build the AO-to-MO basis transformer to ensure constant MOs during entire procedure
        (
            mo_coeff,
            mo_coeff_b,
        ) = driver._expand_mo_object(  # pylint: disable=protected-access
            driver._calc.mo_coeff, array_dimension=3  # pylint: disable=protected-access
        )
        basis_trafo = BasisTransformer(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(mo_coeff, h1_b=mo_coeff_b),
        )

        # 3. generate the problem with range-separated 2-body terms
        with driver._mol.with_range_coulomb(  # pylint: disable=protected-access
            omega=omega
        ):
            problem = driver.to_problem(basis=ElectronicBasis.MO, include_dipole=False)

            total_mo_density = ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations,
                problem.orbital_occupations_b,
                include_rdm2=False,
            )
            problem.properties.electronic_density = total_mo_density

        # 4. prepare the active space by initializing it with the total problem size information
        self.active_space.prepare_active_space(
            problem.num_particles,
            problem.num_spatial_orbitals,
            occupation_alpha=problem.orbital_occupations,
            occupation_beta=problem.orbital_occupations_b,
        )
        # also initialize the history of the active densities
        # NOTE: this list tracks the active densities in their reduced dimension: i.e. the active
        # dimension
        active_density_history = [
            self.active_space.active_basis.transform_electronic_integrals(
                total_mo_density
            )
        ]
        # also initialize the inactive density in the AO basis (which remains constant at all times)
        inactive_ao_density = basis_trafo.invert().transform_electronic_integrals(
            total_mo_density
            - self.active_space.active_basis.invert().transform_electronic_integrals(
                active_density_history[-1]
            )
        )

        # 5. prepare some variables which we need to keep track of
        e_nuc = problem.hamiltonian.nuclear_repulsion_energy
        e_tot = driver._calc.e_tot  # pylint: disable=protected-access
        e_next = float("NaN")
        e_prev = float("NaN")
        converged = False
        n_iter = 0

        # 6. finally run the iterative embedding
        while n_iter < self.max_iter:
            n_iter += 1

            # a) expand the active density into the dimensions of the total system
            active_mo_density = (
                self.active_space.active_basis.invert().transform_electronic_integrals(
                    active_density_history[-1]
                )
            )

            # b) transform the active density into the AO basis
            active_ao_density = basis_trafo.invert().transform_electronic_integrals(
                active_mo_density
            )

            # c) compute the total density in the AO basis
            total_ao_density = inactive_ao_density + active_ao_density

            # d) translate the total density into the form required by PySCF
            if basis_trafo.coefficients.beta.is_empty():
                rho = np.asarray(total_ao_density.trace_spin()["+-"])
            else:
                rho = np.asarray(
                    [total_ao_density.alpha["+-"], total_ao_density.beta["+-"]]
                )

            # e) evaluate the total energy at the new total density
            e_tot = driver._calc.energy_tot(dm=rho)  # pylint: disable=protected-access
            # f) also evaluate the total Fock operator at the new total density
            (
                fock_a,
                fock_b,
            ) = driver._expand_mo_object(  # pylint: disable=protected-access
                driver._calc.get_fock(dm=rho),  # pylint: disable=protected-access
                array_dimension=3,
            )

            # g) update the active space transformer components
            self.active_space.active_density = active_mo_density
            self.active_space.reference_inactive_energy = e_tot - e_nuc
            self.active_space.reference_inactive_fock = (
                basis_trafo.transform_electronic_integrals(
                    ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
                )
            )

            # h) use the updated active space transformer to reduce the problem to the active space
            as_problem = self.active_space.transform(problem)

            # i) solve the active space problem
            result = self.solver.solve(as_problem)

            # j) append the reduced-size active density in the MO basis to the history, taking into
            # account any user-specified damping procedure
            active_density_history.append(
                self.damp_active_density(
                    active_density_history + [result.electronic_density]
                )
            )

            # k) check for convergence
            e_prev = e_next
            e_next = result.total_energies[0]
            converged = np.abs(e_prev - e_next) < self.threshold
            if converged:
                break

        return result

    @staticmethod
    def damp_active_density(density_history: list[ElectronicDensity]) -> ElectronicDensity:
        """The active density damping method.

        An end-user may overwrite this method to implement any damping procedure on the active
        density matrix.

        Args:
            density_history: the history of active densities in the form of `ElectronicDensity`
                objects.

        Returns:
            A single `ElectronicDensity` object.
        """
        return density_history[-1]
