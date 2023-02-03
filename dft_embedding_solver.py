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
    """TODO."""

    def __init__(
        self,
        active_space: ActiveSpaceTransformer,
        solver: GroundStateSolver,
        *,
        max_iter: int = 100,
        threshold: float = 1e-5,
    ) -> None:
        """TODO."""
        self.active_space = active_space
        self.solver = solver
        self.max_iter = max_iter
        self.threshold = threshold
        # TODO: extract this into the iteration method to make this solver stateless
        self.active_density_history: list[ElectronicDensity] = []

    def solve(self, driver: PySCFDriver, omega: float) -> ElectronicStructureResult:
        """TODO."""
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

            density = ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations,
                problem.orbital_occupations_b,
                include_rdm2=False,
            )
            self.active_density_history.append(density)
            problem.properties.electronic_density = density

        # 4. compute the fixed inactive Fock operator and plug it into the active space transformer
        fock_a, fock_b = driver._expand_mo_object(  # pylint: disable=protected-access
            driver._calc.get_fock(),  # pylint: disable=protected-access
            array_dimension=3,
        )
        self.active_space.reference_inactive_fock = (
            basis_trafo.transform_electronic_integrals(
                ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
            )
        )

        # 5. initialize the inactive energy component in the active space transformer
        e_nuc = problem.hamiltonian.nuclear_repulsion_energy
        e_tot = driver._calc.e_tot  # pylint: disable=protected-access
        self.active_space.reference_inactive_energy = e_tot - e_nuc

        # 6. run the iterative algorithm
        e_next = float("NaN")
        e_prev = float("NaN")
        converged = False
        n_iter = 0

        while n_iter < self.max_iter:
            n_iter += 1

            # reduce problem to active space
            as_problem = self.active_space.transform(problem)

            # solve active space problem
            result = self.solver.solve(as_problem)
            self.active_density_history.append(result.electronic_density)

            # check convergence
            e_prev = e_next
            e_next = result.total_energies[0]

            converged = np.abs(e_prev - e_next) < self.threshold
            if converged:
                break

            # merge active space density into total MO density
            total = self.update_total_density(problem)

            # evaluate energy at new density
            e_tot = self.evaluate_energy(total, basis_trafo, driver)

            # update active space transformer
            self.active_space.active_density = (
                self.active_space.get_active_density_component(total)
            )
            self.active_space.reference_inactive_energy = e_tot - e_nuc

        return result

    @staticmethod
    def evaluate_energy(total_density, basis_trafo, driver):
        """TODO."""
        # convert MO density to AO
        ao_density = basis_trafo.invert().transform_electronic_integrals(total_density)

        # construct density in PySCF required form
        if basis_trafo.coefficients.beta.is_empty():
            rho = ao_density.trace_spin()["+-"]
        else:
            rho = np.asarray([ao_density.alpha["+-"], ao_density.beta["+-"]])

        # chop density
        rho[np.abs(rho) < 1e-8] = 0.0

        # evaluate new energy
        e_tot = driver._calc.energy_tot(dm=rho)  # pylint: disable=protected-access

        return e_tot

    def update_total_density(self, problem):
        """TODO."""
        # get total MO density
        total = problem.properties.electronic_density
        # find total MO density component which overlaps with active space
        subspace = self.active_space.get_active_density_component(total)
        # subtract that component from the total
        total -= subspace
        # get active space component
        active = self.damp_active_density(self.active_density_history)
        self.active_density_history[-1] = active
        # expand active component to total system size
        superspace = deepcopy(active)
        superspace = (
            self.active_space.active_basis.invert().transform_electronic_integrals(
                superspace
            )
        )
        # add expanded active space component to the total
        total += superspace

        return total

    @staticmethod
    def damp_active_density(density_history):
        """TODO."""
        return density_history[-1]
